"""
Created on Feb 23, 2026.
post_retrievertraining.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""
import os.path
import pdb
import re
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple
import hashlib
from transformers import AutoTokenizer, AutoModel
from dataclasses import asdict, dataclass
import faiss
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import random
import numpy as np
import yaml

from models.model import BiEncoder
from config.serde import open_experiment, create_experiment, delete_experiment, write_config


import warnings
warnings.filterwarnings('ignore')




class RadiopaediaStep6ReembedAndFaissBuilder:
    """
    Step 6A — Re-embed the frozen evidence snapshot with trained retriever R* and rebuild FAISS.

    Why:
      - Step 4B FAISS was built with a bootstrap embedder for weak labels.
      - After training R* (Step 5), you MUST re-embed all evidence with the trained doc encoder.

    Inputs:
      - chunks.parquet (frozen evidence texts)
      - trained retriever checkpoint from Step 5 (either trained_model.pth or checkpoint.tar)
      - retriever config (model_name, max_doc_tokens, tied_encoders, pooling)

    Outputs (in out_dir):
      - chunk_ids.npy
      - chunk_embeds_Rstar.npy
      - faiss_Rstar.index
      - manifest_step6.json
    """

    def __init__(
        self,
        cfg_path: str | Path,
        chunks_parquet: str | Path,
        model_weights_path: str | Path,
        out_dir: str | Path,
        experiment_name="name",
    # embedding
        batch_size: int = 128,
        use_amp: bool = True,
        device: str = "cuda",
        # columns
        chunk_id_col: str = "chunk_id",
        chunk_text_col: str = "chunk_text",
        # determinism
        seed: int = 13,
        # misc
        tqdm_desc: str = "Step 6: embedding chunks with R* and building FAISS",
    ) -> None:
        self.cfg_path = Path(cfg_path)
        self.chunks_parquet = Path(chunks_parquet)
        self.model_weights_path = Path(model_weights_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.params = open_experiment(experiment_name, self.cfg_path)
        self.cfg_path = self.params['cfg_path']

        self.batch_size = int(batch_size)
        self.use_amp = bool(use_amp)
        self.device = str(device)
        self.seed = int(seed)
        self.tqdm_desc = str(tqdm_desc)

        self.chunk_id_col = str(chunk_id_col)
        self.chunk_text_col = str(chunk_text_col)

        # outputs
        self.chunk_ids_path = self.out_dir / "chunk_ids.npy"
        self.embeds_path = self.out_dir / "chunk_embeds_Rstar.npy"
        self.faiss_path = self.out_dir / "faiss_Rstar.index"
        self.manifest_path = self.out_dir / "manifest_step6.json"

        # runtime
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model = None
        self._torch_device: Optional[torch.device] = None

        self._set_determinism()


    def _set_determinism(self) -> None:
        import random

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


    def _setup_device(self) -> torch.device:
        if self.device == "cuda" and torch.cuda.is_available():
            self._torch_device = torch.device("cuda")
        else:
            self._torch_device = torch.device("cpu")
        return self._torch_device

    def _normalize_text(self, x: str) -> str:
        if x is None:
            return ""
        s = str(x)
        if s.lower() == "nan":
            return ""
        return " ".join(s.split()).strip()

    def _l2_normalize_np(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def _tokenize(self, texts: List[str], max_len: int) -> Dict[str, torch.Tensor]:
        assert self._tokenizer is not None
        assert self._torch_device is not None
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {k: v.to(self._torch_device) for k, v in enc.items()}



    def _load_chunks(self) -> pd.DataFrame:
        if not self.chunks_parquet.exists():
            raise FileNotFoundError(f"chunks.parquet not found: {self.chunks_parquet}")

        cols = [self.chunk_id_col, self.chunk_text_col]
        df = pd.read_parquet(self.chunks_parquet, columns=cols)

        df[self.chunk_id_col] = pd.to_numeric(df[self.chunk_id_col], errors="coerce")
        df = df[~df[self.chunk_id_col].isna()].copy()
        df[self.chunk_id_col] = df[self.chunk_id_col].astype(np.int64)

        df[self.chunk_text_col] = df[self.chunk_text_col].astype(str).map(self._normalize_text)

        # stable order
        df = df.sort_values(self.chunk_id_col).reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("No chunks found in chunks.parquet after filtering.")

        return df



    def _init_model_and_tokenizer(self, params: Dict) -> None:

        model_name = params["retriever"]["model_name"]
        tied = bool(params["retriever"]["tied_encoders"])
        pooling = params["retriever"]["pooling"]
        max_doc_tokens = int(params["retriever"]["max_doc_tokens"])

        self.max_doc_tokens = max_doc_tokens

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._model = BiEncoder(model_name, tied=tied, pooling=pooling)
        self._model.eval()

        assert self._torch_device is not None
        self._model = self._model.to(self._torch_device)

    def _load_weights(self) -> None:
        if not self.model_weights_path.exists():
            raise FileNotFoundError(f"model_weights_path not found: {self.model_weights_path}")

        wpath = str(self.model_weights_path)
        obj = torch.load(wpath, map_location="cpu")

        # support both formats:
        # 1) checkpoint.tar with {"model_state_dict": ...}
        # 2) trained_model.pth that is a pure state_dict
        if isinstance(obj, dict) and "model_state_dict" in obj:
            state = obj["model_state_dict"]
        else:
            state = obj

        self._model.load_state_dict(state, strict=True)
        self._model.eval()




    def _infer_dim(self) -> int:
        with torch.no_grad():
            tok = self._tokenize(["hello world"], self.max_doc_tokens)
            with torch.cuda.amp.autocast(enabled=(self.use_amp and self._torch_device.type == "cuda")):
                emb = self._model.encode_docs(tok["input_ids"], tok["attention_mask"])
            return int(emb.shape[1])


    def _embed_all_chunks(self, texts: List[str], dim: int) -> np.ndarray:
        n = len(texts)
        tmp_memmap = self.out_dir / "chunk_embeds_Rstar.memmap"
        if tmp_memmap.exists():
            tmp_memmap.unlink()

        mm = np.memmap(tmp_memmap, dtype=np.float32, mode="w+", shape=(n, dim))

        pbar = tqdm(total=n, desc=self.tqdm_desc, unit="chunks", miniters=1)

        with torch.no_grad():
            start = 0
            while start < n:
                end = min(start + self.batch_size, n)
                batch_text = texts[start:end]

                tok = self._tokenize(batch_text, self.max_doc_tokens)

                with torch.cuda.amp.autocast(enabled=(self.use_amp and self._torch_device.type == "cuda")):
                    emb = self._model.encode_docs(tok["input_ids"], tok["attention_mask"])

                emb = emb.detach().float().cpu().numpy().astype(np.float32)
                emb = self._l2_normalize_np(emb)

                mm[start:end, :] = emb
                pbar.update(end - start)
                start = end

        pbar.close()
        mm.flush()

        embeds = np.asarray(mm)
        try:
            del mm
            tmp_memmap.unlink(missing_ok=True)
        except Exception:
            pass

        return embeds

    def _build_faiss(self, embeds: np.ndarray) -> faiss.Index:
        dim = int(embeds.shape[1])
        index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        index.add(embeds)
        return index



    def run(self) -> None:
        # if not self.cfg_path.exists():
        #     raise FileNotFoundError(f"cfg_path not found: {self.cfg_path}")

        self._setup_device()

        df = self._load_chunks()
        chunk_ids = df[self.chunk_id_col].to_numpy(dtype=np.int64)
        texts = df[self.chunk_text_col].tolist()

        self._init_model_and_tokenizer(self.params)
        self._load_weights()

        dim = self._infer_dim()
        embeds = self._embed_all_chunks(texts, dim)

        # save artifacts
        np.save(self.chunk_ids_path, chunk_ids)
        np.save(self.embeds_path, embeds)

        index = self._build_faiss(embeds)
        faiss.write_index(index, str(self.faiss_path))

        manifest = {
            "step": "6",
            "inputs": {
                "cfg_path": str(self.cfg_path),
                "chunks_parquet": str(self.chunks_parquet.resolve()),
                "model_weights_path": str(self.model_weights_path),
            },
            "outputs": {
                "chunk_ids_npy": str(self.chunk_ids_path),
                "chunk_embeds_Rstar_npy": str(self.embeds_path),
                "faiss_index": str(self.faiss_path),
            },
            "params": {
                "model_name": self.params["retriever"]["model_name"],
                "tied_encoders": bool(self.params["retriever"]["tied_encoders"]),
                "pooling": self.params["retriever"]["pooling"],
                "max_doc_tokens": int(self.params["retriever"]["max_doc_tokens"]),
                "batch_size": self.batch_size,
                "use_amp": self.use_amp,
                "device": str(self._torch_device),
                "faiss": "IndexFlatIP (cosine via L2-normalized embeddings)",
                "seed": self.seed,
            },
            "stats": {
                "num_chunks": int(len(df)),
                "embed_dim": int(dim),
            },
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.chunk_ids_path}  (N={len(chunk_ids):,})")
        print(f"Wrote: {self.embeds_path}  shape={embeds.shape}")
        print(f"Wrote: {self.faiss_path}")
        print(f"Wrote: {self.manifest_path}")








@dataclass(frozen=True)
class WeakAlignmentRow:
    qa_row_id: int
    report_rel_path: str
    qa_type: str
    claim_text: str
    retrieved_chunk_ids_json: str
    support_weights_json: str
    embed_scores_json: str
    lexical_scores_json: str
    combined_scores_json: str


class RadiopaediaRStarWeakEvidenceAligner:
    """
    Step 6B: Weak claim–evidence alignment using trained retriever R* (no humans)

    Inputs:
      - qa_csv: Step 3 output (mimic_qa_step3.csv)
      - chunks_parquet: Step 4A output (chunks.parquet)  [IDs + text for lexical scoring only]
      - faiss_index: Step 6 output (faiss_Rstar.index)   [built on R* doc embeddings]
      - rstar_query_encoder_ckpt: checkpoint path for query encoder (HF AutoModel state_dict or full model)

    Output:
      - out_parquet: per-QA row: top-k retrieved chunk_ids + soft weights (JSON arrays)

    Notes:
      - This mirrors Step 4B but uses R* query encoder and the Step 6 FAISS index.
      - It does NOT re-embed chunks. Step 6 already did that.
    """

    def __init__(
        self,
        qa_csv: str | Path,
        chunks_parquet: str | Path,
        faiss_index: str | Path,
        out_parquet: str | Path,
        # R* query encoder
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        rstar_query_encoder_ckpt: Optional[str | Path] = None,
        # tokenization/encoding
        max_query_tokens: int = 128,
        qa_batch_size: int = 256,
        # retrieval
        top_k: int = 20,
        # scoring
        alpha: float = 0.85,
        temperature: float = 0.07,
        # columns
        claim_col: str = "claim_text",
        report_path_col: str = "report_rel_path",
        qa_type_col: str = "qa_type",
        chunk_id_col: str = "chunk_id",
        chunk_text_col: str = "chunk_text",
        # determinism/runtime
        seed: int = 13,
        device: str = "cuda",
        tqdm_desc: str = "Step 6B: R* weak evidence alignment",
        qa_chunksize_rows: int = 50_000,
        parquet_compression: str = "zstd",
        # lexical overlap toggle (deterministic)
        use_lexical: bool = True,
    ) -> None:
        self.qa_csv = Path(qa_csv)
        self.chunks_parquet = Path(chunks_parquet)
        self.faiss_index_path = Path(faiss_index)
        self.out_parquet = Path(out_parquet)

        self.model_name = str(model_name)
        self.rstar_query_encoder_ckpt = Path(rstar_query_encoder_ckpt) if rstar_query_encoder_ckpt else None

        self.max_query_tokens = int(max_query_tokens)
        self.qa_batch_size = int(qa_batch_size)

        self.top_k = int(top_k)

        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.claim_col = str(claim_col)
        self.report_path_col = str(report_path_col)
        self.qa_type_col = str(qa_type_col)

        self.chunk_id_col = str(chunk_id_col)
        self.chunk_text_col = str(chunk_text_col)

        self.seed = int(seed)
        self.device = str(device)
        self.tqdm_desc = str(tqdm_desc)
        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_compression = str(parquet_compression)
        self.use_lexical = bool(use_lexical)

        self._word_re = re.compile(r"[a-z0-9]+")
        self._pa_schema: Optional[pa.Schema] = None

        self._tokenizer: Optional[AutoTokenizer] = None
        self._query_model: Optional[AutoModel] = None

        self._set_determinism()

    def _set_determinism(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    def _normalize_text(self, s: str) -> str:
        if s is None:
            return ""
        x = str(s).lower()
        x = re.sub(r"\s+", " ", x).strip()
        return x

    def _tokenize_words(self, s: str) -> List[str]:
        s = self._normalize_text(s)
        if not s:
            return []
        return self._word_re.findall(s)

    def _lexical_overlap(self, query: str, doc: str) -> float:
        q = self._tokenize_words(query)
        if not q:
            return 0.0
        qset = set(q)
        dset = set(self._tokenize_words(doc))
        if not dset:
            return 0.0
        return float(len(qset.intersection(dset))) / float(len(qset))

    def _get_tokenizer_and_query_model(self) -> Tuple[AutoTokenizer, AutoModel]:
        if self._tokenizer is not None and self._query_model is not None:
            return self._tokenizer, self._query_model

        tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model = AutoModel.from_pretrained(self.model_name, ignore_mismatched_sizes=True)

        # load R* weights if provided
        if self.rstar_query_encoder_ckpt is not None:
            ckpt = torch.load(str(self.rstar_query_encoder_ckpt), map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            # allow both full-model and query-only checkpoints
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing:
                print(f"[WARN] Missing keys when loading R* query encoder: {len(missing)}")
            if unexpected:
                print(f"[WARN] Unexpected keys when loading R* query encoder: {len(unexpected)}")

        model.eval()
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        self._tokenizer = tok
        self._query_model = model
        return tok, model

    def _mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _encode_queries(self, texts: List[str], pbar: Optional["tqdm"] = None) -> np.ndarray:
        tok, model = self._get_tokenizer_and_query_model()

        embs: List[np.ndarray] = []
        for i in range(0, len(texts), self.qa_batch_size):
            batch = texts[i : i + self.qa_batch_size]
            batch = [self._normalize_text(x) for x in batch]

            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_query_tokens,
                return_tensors="pt",
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embs.append(pooled.detach().cpu().numpy().astype(np.float32))

            if pbar is not None:
                pbar.update(len(batch))

        if not embs:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack(embs)

    def _load_chunks_table(self) -> pd.DataFrame:
        cols = [self.chunk_id_col, self.chunk_text_col]
        df = pd.read_parquet(self.chunks_parquet, columns=cols)
        df[self.chunk_id_col] = pd.to_numeric(df[self.chunk_id_col], errors="coerce")
        df = df[~df[self.chunk_id_col].isna()].copy()
        df[self.chunk_id_col] = df[self.chunk_id_col].astype(int)
        df[self.chunk_text_col] = df[self.chunk_text_col].astype(str)
        df = df.sort_values(self.chunk_id_col).reset_index(drop=True)
        return df

    def _load_faiss(self) -> faiss.Index:
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")
        return faiss.read_index(str(self.faiss_index_path))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x)
        ex = np.exp(x)
        s = float(np.sum(ex))
        if s <= 0:
            return np.ones_like(x, dtype=np.float32) / float(len(x))
        return ex / s

    def _weights_from_scores(self, combined: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            w = np.zeros_like(combined, dtype=np.float32)
            if len(w) > 0:
                w[int(np.argmax(combined))] = 1.0
            return w
        return self._softmax(combined / float(self.temperature)).astype(np.float32)

    def _init_writer(self) -> pq.ParquetWriter:
        self.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        self._pa_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("claim_text", pa.string()),
                ("retrieved_chunk_ids_json", pa.string()),
                ("support_weights_json", pa.string()),
                ("embed_scores_json", pa.string()),
                ("lexical_scores_json", pa.string()),
                ("combined_scores_json", pa.string()),
            ]
        )
        return pq.ParquetWriter(
            str(self.out_parquet),
            schema=self._pa_schema,
            compression=self.parquet_compression,
        )

    def _coerce_batch_df_for_parquet(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        batch_df["qa_row_id"] = pd.to_numeric(batch_df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        for c in self._pa_schema.names:
            if c != "qa_row_id":
                batch_df[c] = batch_df[c].astype(str)
        return batch_df

    def run(self) -> None:
        if not self.qa_csv.exists():
            raise FileNotFoundError(f"QA CSV not found: {self.qa_csv}")
        if not self.chunks_parquet.exists():
            raise FileNotFoundError(f"chunks.parquet not found: {self.chunks_parquet}")

        chunks_df = self._load_chunks_table()
        chunk_ids = chunks_df[self.chunk_id_col].to_numpy(dtype=np.int64)
        chunk_texts = chunks_df[self.chunk_text_col].tolist()

        index = self._load_faiss()

        total_rows = sum(1 for _ in open(self.qa_csv, "r", encoding="utf-8", errors="ignore")) - 1
        pbar = tqdm(total=max(total_rows, 0), desc=self.tqdm_desc, unit="rows", miniters=1)

        writer = self._init_writer()
        assert self._pa_schema is not None

        qa_row_id = 0
        try:
            for df in pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows):
                for c in [self.report_path_col, self.qa_type_col, self.claim_col]:
                    if c not in df.columns:
                        raise ValueError(f"QA CSV missing required column: {c}")

                df[self.report_path_col] = df[self.report_path_col].astype(str)
                df[self.qa_type_col] = df[self.qa_type_col].astype(str)
                df[self.claim_col] = df[self.claim_col].astype(str)

                claims = df[self.claim_col].tolist()
                q_emb = self._encode_queries(claims, pbar=pbar)

                D, I = index.search(q_emb, self.top_k)  # (B,k)

                out_rows: List[Dict[str, object]] = []
                for i in range(len(df)):
                    report_rel_path = df.iloc[i][self.report_path_col]
                    qa_type = df.iloc[i][self.qa_type_col]
                    claim_text = df.iloc[i][self.claim_col]
                    claim_norm = self._normalize_text(claim_text)

                    if not claim_norm:
                        out_rows.append(
                            asdict(
                                WeakAlignmentRow(
                                    qa_row_id=qa_row_id,
                                    report_rel_path=report_rel_path,
                                    qa_type=qa_type,
                                    claim_text=claim_text,
                                    retrieved_chunk_ids_json="[]",
                                    support_weights_json="[]",
                                    embed_scores_json="[]",
                                    lexical_scores_json="[]",
                                    combined_scores_json="[]",
                                )
                            )
                        )
                        qa_row_id += 1
                        continue

                    faiss_idxs = I[i].tolist()
                    embed_scores = D[i].astype(np.float32).tolist()

                    retrieved_ids: List[int] = []
                    retrieved_texts: List[str] = []
                    valid_embed_scores: List[float] = []

                    for j, sc in zip(faiss_idxs, embed_scores):
                        if j < 0:
                            continue
                        retrieved_ids.append(int(chunk_ids[j]))
                        retrieved_texts.append(chunk_texts[j])
                        valid_embed_scores.append(float(sc))

                    e = np.asarray(valid_embed_scores, dtype=np.float32)
                    if self.use_lexical:
                        l_list = [float(self._lexical_overlap(claim_norm, t)) for t in retrieved_texts]
                        l = np.asarray(l_list, dtype=np.float32)
                    else:
                        l = np.zeros_like(e, dtype=np.float32)

                    combined = self.alpha * e + (1.0 - self.alpha) * l
                    w = self._weights_from_scores(combined)

                    out_rows.append(
                        asdict(
                            WeakAlignmentRow(
                                qa_row_id=qa_row_id,
                                report_rel_path=report_rel_path,
                                qa_type=qa_type,
                                claim_text=claim_text,
                                retrieved_chunk_ids_json=json.dumps(retrieved_ids),
                                support_weights_json=json.dumps([float(x) for x in w.tolist()]),
                                embed_scores_json=json.dumps([float(x) for x in e.tolist()]),
                                lexical_scores_json=json.dumps([float(x) for x in l.tolist()]),
                                combined_scores_json=json.dumps([float(x) for x in combined.tolist()]),
                            )
                        )
                    )
                    qa_row_id += 1

                batch_df = pd.DataFrame(out_rows)
                batch_df = self._coerce_batch_df_for_parquet(batch_df)
                table = pa.Table.from_pandas(batch_df, schema=self._pa_schema, preserve_index=False)
                writer.write_table(table)

        finally:
            pbar.close()
            writer.close()

        manifest = {
            "step": "6B",
            "qa_csv": str(self.qa_csv.resolve()),
            "chunks_parquet": str(self.chunks_parquet.resolve()),
            "faiss_index": str(self.faiss_index_path.resolve()),
            "out_parquet": str(self.out_parquet.resolve()),
            "query_encoder": {
                "model_name": self.model_name,
                "checkpoint": str(self.rstar_query_encoder_ckpt) if self.rstar_query_encoder_ckpt else None,
            },
            "params": {
                "max_query_tokens": self.max_query_tokens,
                "top_k": self.top_k,
                "alpha": self.alpha,
                "temperature": self.temperature,
                "seed": self.seed,
                "device": self.device,
                "qa_chunksize_rows": self.qa_chunksize_rows,
                "qa_batch_size": self.qa_batch_size,
                "use_lexical": self.use_lexical,
            },
        }
        (self.out_parquet.parent / "manifest_step6b.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {self.out_parquet.parent / 'manifest_step6b.json'}")






@dataclass(frozen=True)
class RetrieverSupervisionRow:
    qa_row_id: int
    subject_id: int
    split: str  # "train" / "valid" / "test"
    report_rel_path: str
    qa_type: str
    claim_text: str
    pos_chunk_ids_json: str
    pos_weights_json: str


@dataclass(frozen=True)
class VerifierSupervisionRow:
    qa_row_id: int
    subject_id: int
    split: str
    report_rel_path: str
    qa_type: str
    case_context: str
    question: str
    answer_target: str
    claim_text: str
    evidence_chunk_ids_json: str
    evidence_weights_json: str
    soft_support_score: float


class RadiopaediaStep6CSupervisionBuilder:
    """
    Step 6C: Convert Step 3 + Step 6B (R* weak alignment) into final supervision artifacts,
             while REUSING the existing patient-wise split map created earlier in Step 4C.

    Key requirement:
      - DO NOT generate a new split.
      - Use patient_split_map.json produced by the original Step 4C.

    Inputs:
      - qa_csv: mimic_qa_step3.csv (must include subject_id + fields needed for verifier)
      - weak_alignment_parquet: weak_alignment_step6b_Rstar.parquet (Step 6B)
      - patient_split_map_json: step4c_supervision/patient_split_map.json (from the original Step 4C)

    Outputs (in out_dir):
      - retriever_supervision.parquet
      - verifier_supervision.parquet
      - manifest_step6c.json

    Notes:
      - Streams both QA CSV and alignment parquet in the same qa_row_id order.
      - Assumes Step 6B wrote rows in strict qa_row_id increasing order.
      - Writes split as provided by patient_split_map_json.
      - By default, writes ALL rows (train/valid/test).
    """

    def __init__(
        self,
        qa_csv: str | Path,
        weak_alignment_parquet: str | Path,
        patient_split_map_json: str | Path,
        out_dir: str | Path,
        # columns in qa_csv
        subject_id_col: str = "subject_id",
        report_path_col: str = "report_rel_path",
        qa_type_col: str = "qa_type",
        claim_col: str = "claim_text",
        case_context_col: str = "case_context",
        question_col: str = "question",
        answer_col: str = "answer_target",
        # streaming
        qa_chunksize_rows: int = 50_000,
        parquet_batch_rows: int = 50_000,
        # supervision shaping
        top_k: Optional[int] = None,  # if set, truncate to first k from 6B
        min_weight: float = 0.0,      # drop items below this weight after truncation
        # optional filtering
        allowed_splits: Optional[List[str]] = None,  # e.g. ["train","valid"] to write only train/valid
        # progress
        tqdm_desc: str = "Step 6C: building supervision from R* alignments",
        parquet_compression: str = "zstd",
    ) -> None:
        self.qa_csv = Path(qa_csv)
        self.weak_alignment_parquet = Path(weak_alignment_parquet)
        self.patient_split_map_json = Path(patient_split_map_json)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.subject_id_col = str(subject_id_col)
        self.report_path_col = str(report_path_col)
        self.qa_type_col = str(qa_type_col)
        self.claim_col = str(claim_col)
        self.case_context_col = str(case_context_col)
        self.question_col = str(question_col)
        self.answer_col = str(answer_col)

        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_batch_rows = int(parquet_batch_rows)

        self.top_k = int(top_k) if top_k is not None else None
        self.min_weight = float(min_weight)

        self.allowed_splits = allowed_splits[:] if allowed_splits is not None else None

        self.tqdm_desc = str(tqdm_desc)
        self.parquet_compression = str(parquet_compression)

        self.retriever_out = self.out_dir / "retriever_supervision.parquet"
        self.verifier_out = self.out_dir / "verifier_supervision.parquet"
        self.manifest_out = self.out_dir / "manifest_step6c.json"

        self._retr_schema: Optional[pa.Schema] = None
        self._ver_schema: Optional[pa.Schema] = None

        self._patient_split_map: Optional[Dict[int, str]] = None

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _filter_and_truncate(self, ids: List[int], w: List[float]) -> Tuple[List[int], List[float]]:
        if self.top_k is not None:
            ids = ids[: self.top_k]
            w = w[: self.top_k]
        if self.min_weight > 0:
            keep = [(i, ww) for i, ww in zip(ids, w) if float(ww) >= self.min_weight]
            ids = [int(i) for i, _ in keep]
            w = [float(ww) for _, ww in keep]
        return ids, w

    def _soft_support_from_combined(self, combined_scores: List[float]) -> float:
        if not combined_scores:
            return 0.0
        try:
            return float(np.max(np.asarray(combined_scores, dtype=np.float32)))
        except Exception:
            return 0.0

    def _load_patient_split_map(self) -> Dict[int, str]:
        if not self.patient_split_map_json.exists():
            raise FileNotFoundError(f"patient_split_map_json not found: {self.patient_split_map_json}")

        obj = json.loads(self.patient_split_map_json.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("patient_split_map_json must contain a JSON object mapping subject_id -> split")

        # keys may be strings in JSON; coerce to int
        split_map: Dict[int, str] = {}
        for k, v in obj.items():
            try:
                sid = int(k)
            except Exception:
                continue
            split_map[sid] = str(v)

        if len(split_map) == 0:
            raise ValueError("Loaded patient_split_map_json but it contained no usable subject_id entries.")
        return split_map

    def _init_retriever_writer(self) -> pq.ParquetWriter:
        self._retr_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("subject_id", pa.int64()),
                ("split", pa.string()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("claim_text", pa.string()),
                ("pos_chunk_ids_json", pa.string()),
                ("pos_weights_json", pa.string()),
            ]
        )
        return pq.ParquetWriter(
            str(self.retriever_out),
            schema=self._retr_schema,
            compression=self.parquet_compression,
        )

    def _init_verifier_writer(self) -> pq.ParquetWriter:
        self._ver_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("subject_id", pa.int64()),
                ("split", pa.string()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("case_context", pa.string()),
                ("question", pa.string()),
                ("answer_target", pa.string()),
                ("claim_text", pa.string()),
                ("evidence_chunk_ids_json", pa.string()),
                ("evidence_weights_json", pa.string()),
                ("soft_support_score", pa.float32()),
            ]
        )
        return pq.ParquetWriter(
            str(self.verifier_out),
            schema=self._ver_schema,
            compression=self.parquet_compression,
        )

    def _to_table(self, df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
        for c in schema.names:
            if c not in df.columns:
                df[c] = None

        df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").fillna(-1).astype(np.int64)

        for c in [
            "split",
            "report_rel_path",
            "qa_type",
            "claim_text",
            "pos_chunk_ids_json",
            "pos_weights_json",
            "case_context",
            "question",
            "answer_target",
            "evidence_chunk_ids_json",
            "evidence_weights_json",
        ]:
            if c in df.columns:
                df[c] = df[c].astype(str)

        if "soft_support_score" in df.columns:
            df["soft_support_score"] = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).astype(
                np.float32
            )

        return pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    def run(self) -> None:
        if not self.qa_csv.exists():
            raise FileNotFoundError(f"QA CSV not found: {self.qa_csv}")
        if not self.weak_alignment_parquet.exists():
            raise FileNotFoundError(f"Weak alignment parquet not found: {self.weak_alignment_parquet}")

        self._patient_split_map = self._load_patient_split_map()

        # progress
        total_rows = sum(1 for _ in open(self.qa_csv, "r", encoding="utf-8", errors="ignore")) - 1
        pbar = tqdm(total=max(total_rows, 0), desc=self.tqdm_desc, unit="rows", miniters=1)

        retr_writer = self._init_retriever_writer()
        ver_writer = self._init_verifier_writer()
        assert self._retr_schema is not None
        assert self._ver_schema is not None

        pf = pq.ParquetFile(str(self.weak_alignment_parquet))

        align_cols = [
            "qa_row_id",
            "report_rel_path",
            "qa_type",
            "claim_text",
            "retrieved_chunk_ids_json",
            "support_weights_json",
            "combined_scores_json",
        ]

        needed_qa_cols = [
            self.subject_id_col,
            self.report_path_col,
            self.qa_type_col,
            self.claim_col,
            self.case_context_col,
            self.question_col,
            self.answer_col,
        ]

        qa_reader = pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows)

        qa_row_id_expected = 0
        align_row_id_expected = 0

        # iterate alignment parquet batches
        align_iter = pf.iter_batches(batch_size=self.parquet_batch_rows, columns=align_cols)

        written_rows = 0
        dropped_by_split = 0

        try:
            for qa_chunk in qa_reader:
                for c in needed_qa_cols:
                    if c not in qa_chunk.columns:
                        raise ValueError(f"QA CSV missing required column: {c}")

                qa_chunk = qa_chunk.copy()

                # subject_id must exist; keep rows with valid subject_id only
                qa_chunk[self.subject_id_col] = pd.to_numeric(qa_chunk[self.subject_id_col], errors="coerce")
                qa_chunk = qa_chunk[~qa_chunk[self.subject_id_col].isna()]
                qa_chunk[self.subject_id_col] = qa_chunk[self.subject_id_col].astype(np.int64)

                qa_chunk[self.report_path_col] = qa_chunk[self.report_path_col].astype(str)
                qa_chunk[self.qa_type_col] = qa_chunk[self.qa_type_col].astype(str)
                qa_chunk[self.claim_col] = qa_chunk[self.claim_col].astype(str)
                qa_chunk[self.case_context_col] = qa_chunk[self.case_context_col].astype(str)
                qa_chunk[self.question_col] = qa_chunk[self.question_col].astype(str)
                qa_chunk[self.answer_col] = qa_chunk[self.answer_col].astype(str)

                n = len(qa_chunk)
                if n == 0:
                    continue

                # Pull exactly n alignment rows (may span multiple parquet batches)
                align_rows: List[pd.DataFrame] = []
                need = n
                while need > 0:
                    batch = next(align_iter, None)
                    if batch is None:
                        raise RuntimeError(f"Alignment parquet ended early. Still needed {need} rows.")
                    bdf = batch.to_pandas()
                    align_rows.append(bdf)
                    need -= len(bdf)

                align_df = pd.concat(align_rows, ignore_index=True)
                if len(align_df) > n:
                    align_df = align_df.iloc[:n].copy()

                if "qa_row_id" not in align_df.columns:
                    raise ValueError("Alignment parquet missing qa_row_id column.")

                first_id = int(align_df.iloc[0]["qa_row_id"])
                if first_id != align_row_id_expected:
                    raise RuntimeError(
                        f"Alignment qa_row_id mismatch: expected {align_row_id_expected}, got {first_id}. "
                        f"(This means alignment parquet is not in the same order as QA CSV.)"
                    )

                retr_rows: List[Dict[str, object]] = []
                ver_rows: List[Dict[str, object]] = []

                assert self._patient_split_map is not None

                for i in range(n):
                    qa_row_id = qa_row_id_expected
                    al_row = align_df.iloc[i]

                    ids = self._loads_json_list(self._safe_str(al_row.get("retrieved_chunk_ids_json", "[]")))
                    w = self._loads_json_list(self._safe_str(al_row.get("support_weights_json", "[]")))
                    combined = self._loads_json_list(self._safe_str(al_row.get("combined_scores_json", "[]")))

                    ids = [int(x) for x in ids if str(x).strip() != ""]
                    w = [float(x) for x in w]
                    combined = [float(x) for x in combined]

                    ids, w = self._filter_and_truncate(ids, w)
                    if len(w) != len(ids):
                        m = min(len(w), len(ids))
                        ids, w = ids[:m], w[:m]

                    subject_id = int(qa_chunk.iloc[i][self.subject_id_col])
                    split = self._patient_split_map.get(subject_id, None)
                    if split is None:
                        # if subject_id was missing from map, this is usually a pipeline mismatch
                        # defaulting to "train" keeps code running but you probably want to fix upstream
                        split = "train"

                    if self.allowed_splits is not None and split not in self.allowed_splits:
                        dropped_by_split += 1
                        qa_row_id_expected += 1
                        align_row_id_expected += 1
                        continue

                    report_rel_path = self._safe_str(qa_chunk.iloc[i][self.report_path_col])
                    qa_type = self._safe_str(qa_chunk.iloc[i][self.qa_type_col])
                    claim_text = self._safe_str(qa_chunk.iloc[i][self.claim_col])
                    case_context = self._safe_str(qa_chunk.iloc[i][self.case_context_col])
                    question = self._safe_str(qa_chunk.iloc[i][self.question_col])
                    answer_target = self._safe_str(qa_chunk.iloc[i][self.answer_col])

                    retr_rows.append(
                        asdict(
                            RetrieverSupervisionRow(
                                qa_row_id=qa_row_id,
                                subject_id=subject_id,
                                split=split,
                                report_rel_path=report_rel_path,
                                qa_type=qa_type,
                                claim_text=claim_text,
                                pos_chunk_ids_json=json.dumps(ids),
                                pos_weights_json=json.dumps([float(x) for x in w]),
                            )
                        )
                    )

                    ver_rows.append(
                        asdict(
                            VerifierSupervisionRow(
                                qa_row_id=qa_row_id,
                                subject_id=subject_id,
                                split=split,
                                report_rel_path=report_rel_path,
                                qa_type=qa_type,
                                case_context=case_context,
                                question=question,
                                answer_target=answer_target,
                                claim_text=claim_text,
                                evidence_chunk_ids_json=json.dumps(ids),
                                evidence_weights_json=json.dumps([float(x) for x in w]),
                                soft_support_score=self._soft_support_from_combined(combined),
                            )
                        )
                    )

                    qa_row_id_expected += 1
                    align_row_id_expected += 1

                if retr_rows:
                    retr_df = pd.DataFrame(retr_rows)
                    ver_df = pd.DataFrame(ver_rows)

                    retr_writer.write_table(self._to_table(retr_df, self._retr_schema))
                    ver_writer.write_table(self._to_table(ver_df, self._ver_schema))

                    written_rows += len(retr_rows)

                pbar.update(n)

        finally:
            pbar.close()
            retr_writer.close()
            ver_writer.close()

        manifest = {
            "step": "6C",
            "inputs": {
                "qa_csv": str(self.qa_csv.resolve()),
                "weak_alignment_parquet": str(self.weak_alignment_parquet.resolve()),
                "patient_split_map_json": str(self.patient_split_map_json.resolve()),
            },
            "outputs": {
                "retriever_supervision": str(self.retriever_out.resolve()),
                "verifier_supervision": str(self.verifier_out.resolve()),
            },
            "split": {
                "source": "patient_split_map_json (reused, no recomputation)",
                "allowed_splits": self.allowed_splits if self.allowed_splits is not None else ["train", "valid", "test"],
            },
            "params": {
                "qa_chunksize_rows": self.qa_chunksize_rows,
                "parquet_batch_rows": self.parquet_batch_rows,
                "top_k": self.top_k,
                "min_weight": self.min_weight,
                "soft_support_scalar": "max(combined_scores)",
                "parquet_compression": self.parquet_compression,
            },
            "stats": {
                "written_rows": int(written_rows),
                "dropped_by_split": int(dropped_by_split),
            },
        }
        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.retriever_out}")
        print(f"Wrote: {self.verifier_out}")
        print(f"Wrote: {self.manifest_out}")
        print(f"Rows written: {written_rows:,} | Dropped by split filter: {dropped_by_split:,}")






@dataclass(frozen=True)
class VerifierMaterializedRow:
    qa_row_id: int
    subject_id: int
    split: str
    report_rel_path: str
    qa_type: str
    case_context: str
    question: str
    answer_target: str
    claim_text: str

    evidence_chunk_ids_json: str
    evidence_weights_json: str

    evidence_texts_json: str          # list[str] aligned with ids
    evidence_text_concat: str         # concatenated evidence text for DeBERTa input

    soft_support_score: float


class RadiopaediaStep7VerifierMaterializer:
    """
    Step 7 — Prepare verifier training examples (materialize evidence text)

    Why:
      - verifier_supervision.parquet stores only chunk IDs + weights.
      - verifier training needs actual evidence text.

    Inputs:
      - verifier_supervision.parquet  (from Step 6C)
      - chunks.parquet                (frozen evidence snapshot)

    Output (in out_dir):
      - verifier_train_materialized.parquet
      - manifest_step7.json

    Notes:
      - Does NOT change split. Uses split already present in verifier_supervision.
      - Builds a chunk_id -> chunk_text lookup from chunks.parquet (in memory).
      - Streams the supervision parquet to keep memory stable.
    """

    def __init__(
        self,
        verifier_supervision_parquet: str | Path,
        chunks_parquet: str | Path,
        out_dir: str | Path,
        # columns
        chunk_id_col: str = "chunk_id",
        chunk_text_col: str = "chunk_text",
        # how to build evidence string
        max_evidence_chunks: Optional[int] = None,   # if set, truncate to first N evidence chunks
        min_evidence_weight: float = 0.0,            # drop evidence items below this weight (after truncation)
        evidence_sep: str = "\n\n---\n\n",           # separator between evidence chunks
        # streaming
        parquet_batch_rows: int = 50_000,
        # progress
        tqdm_desc: str = "Step 7: materializing verifier evidence text",
        parquet_compression: str = "zstd",
    ) -> None:
        self.verifier_supervision_parquet = Path(verifier_supervision_parquet)
        self.chunks_parquet = Path(chunks_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_id_col = str(chunk_id_col)
        self.chunk_text_col = str(chunk_text_col)

        self.max_evidence_chunks = int(max_evidence_chunks) if max_evidence_chunks is not None else None
        self.min_evidence_weight = float(min_evidence_weight)
        self.evidence_sep = str(evidence_sep)

        self.parquet_batch_rows = int(parquet_batch_rows)
        self.tqdm_desc = str(tqdm_desc)
        self.parquet_compression = str(parquet_compression)

        self.out_parquet = self.out_dir / "verifier_train_materialized.parquet"
        self.manifest_out = self.out_dir / "manifest_step7.json"

        self._schema: Optional[pa.Schema] = None
        self._chunk_map: Optional[Dict[int, str]] = None

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _normalize_text(self, s: str) -> str:
        if s is None:
            return ""
        x = str(s)
        if x.lower() == "nan":
            return ""
        return " ".join(x.split()).strip()

    def _filter_and_truncate(self, ids: List[int], w: List[float]) -> Tuple[List[int], List[float]]:
        if self.max_evidence_chunks is not None:
            ids = ids[: self.max_evidence_chunks]
            w = w[: self.max_evidence_chunks]

        if self.min_evidence_weight > 0:
            keep = [(i, ww) for i, ww in zip(ids, w) if float(ww) >= self.min_evidence_weight]
            ids = [int(i) for i, _ in keep]
            w = [float(ww) for _, ww in keep]

        return ids, w

    def _load_chunk_map(self) -> Dict[int, str]:
        if not self.chunks_parquet.exists():
            raise FileNotFoundError(f"chunks.parquet not found: {self.chunks_parquet}")

        cols = [self.chunk_id_col, self.chunk_text_col]
        df = pd.read_parquet(self.chunks_parquet, columns=cols)

        df[self.chunk_id_col] = pd.to_numeric(df[self.chunk_id_col], errors="coerce")
        df = df[~df[self.chunk_id_col].isna()].copy()
        df[self.chunk_id_col] = df[self.chunk_id_col].astype(np.int64)

        df[self.chunk_text_col] = df[self.chunk_text_col].astype(str).map(self._normalize_text)

        # build dict
        chunk_map: Dict[int, str] = {}
        for cid, txt in zip(df[self.chunk_id_col].tolist(), df[self.chunk_text_col].tolist()):
            chunk_map[int(cid)] = txt

        if len(chunk_map) == 0:
            raise RuntimeError("chunk_map is empty after loading chunks.parquet.")
        return chunk_map

    def _init_writer(self) -> pq.ParquetWriter:
        self._schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("subject_id", pa.int64()),
                ("split", pa.string()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("case_context", pa.string()),
                ("question", pa.string()),
                ("answer_target", pa.string()),
                ("claim_text", pa.string()),
                ("evidence_chunk_ids_json", pa.string()),
                ("evidence_weights_json", pa.string()),
                ("evidence_texts_json", pa.string()),
                ("evidence_text_concat", pa.string()),
                ("soft_support_score", pa.float32()),
            ]
        )
        return pq.ParquetWriter(
            str(self.out_parquet),
            schema=self._schema,
            compression=self.parquet_compression,
        )

    def _coerce_df(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._schema is not None

        for c in self._schema.names:
            if c not in df.columns:
                df[c] = None

        df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").fillna(-1).astype(np.int64)

        for c in [
            "split",
            "report_rel_path",
            "qa_type",
            "case_context",
            "question",
            "answer_target",
            "claim_text",
            "evidence_chunk_ids_json",
            "evidence_weights_json",
            "evidence_texts_json",
            "evidence_text_concat",
        ]:
            df[c] = df[c].astype(str)

        df["soft_support_score"] = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).astype(
            np.float32
        )
        return df

    def run(self) -> None:
        if not self.verifier_supervision_parquet.exists():
            raise FileNotFoundError(f"verifier_supervision.parquet not found: {self.verifier_supervision_parquet}")

        self._chunk_map = self._load_chunk_map()

        pf = pq.ParquetFile(str(self.verifier_supervision_parquet))

        # required columns from verifier supervision
        in_cols = [
            "qa_row_id",
            "subject_id",
            "split",
            "report_rel_path",
            "qa_type",
            "case_context",
            "question",
            "answer_target",
            "claim_text",
            "evidence_chunk_ids_json",
            "evidence_weights_json",
            "soft_support_score",
        ]

        writer = self._init_writer()
        assert self._schema is not None
        assert self._chunk_map is not None

        total_rows = pf.metadata.num_rows if pf.metadata is not None else 0
        pbar = tqdm(total=total_rows, desc=self.tqdm_desc, unit="rows", miniters=1)

        written = 0

        try:
            for batch in pf.iter_batches(batch_size=self.parquet_batch_rows, columns=in_cols):
                bdf = batch.to_pandas()

                out_rows: List[Dict[str, object]] = []

                for _, row in bdf.iterrows():
                    qa_row_id = int(row["qa_row_id"])
                    subject_id = int(row["subject_id"])
                    split = self._safe_str(row["split"])

                    report_rel_path = self._safe_str(row["report_rel_path"])
                    qa_type = self._safe_str(row["qa_type"])
                    case_context = self._safe_str(row["case_context"])
                    question = self._safe_str(row["question"])
                    answer_target = self._safe_str(row["answer_target"])
                    claim_text = self._safe_str(row["claim_text"])

                    ids = self._loads_json_list(self._safe_str(row["evidence_chunk_ids_json"]))
                    w = self._loads_json_list(self._safe_str(row["evidence_weights_json"]))

                    ids = [int(x) for x in ids if str(x).strip() != ""]
                    w = [float(x) for x in w]

                    # keep aligned lengths
                    if len(ids) != len(w):
                        m = min(len(ids), len(w))
                        ids, w = ids[:m], w[:m]

                    ids, w = self._filter_and_truncate(ids, w)
                    if len(ids) != len(w):
                        m = min(len(ids), len(w))
                        ids, w = ids[:m], w[:m]

                    # materialize texts
                    texts: List[str] = []
                    for cid in ids:
                        txt = self._chunk_map.get(int(cid), "")
                        texts.append(self._normalize_text(txt))

                    # concat (even if some are empty, keep them; the verifier sees what the retriever produced)
                    evidence_concat = self.evidence_sep.join([t for t in texts if t != ""])
                    soft_support_score = float(row.get("soft_support_score", 0.0))

                    out_rows.append(
                        asdict(
                            VerifierMaterializedRow(
                                qa_row_id=qa_row_id,
                                subject_id=subject_id,
                                split=split,
                                report_rel_path=report_rel_path,
                                qa_type=qa_type,
                                case_context=case_context,
                                question=question,
                                answer_target=answer_target,
                                claim_text=claim_text,
                                evidence_chunk_ids_json=json.dumps(ids),
                                evidence_weights_json=json.dumps([float(x) for x in w]),
                                evidence_texts_json=json.dumps(texts),
                                evidence_text_concat=evidence_concat,
                                soft_support_score=soft_support_score,
                            )
                        )
                    )

                out_df = pd.DataFrame(out_rows)
                out_df = self._coerce_df(out_df)

                table = pa.Table.from_pandas(out_df, schema=self._schema, preserve_index=False)
                writer.write_table(table)

                written += len(out_df)
                pbar.update(len(out_df))

        finally:
            pbar.close()
            writer.close()

        manifest = {
            "step": "7",
            "inputs": {
                "verifier_supervision_parquet": str(self.verifier_supervision_parquet.resolve()),
                "chunks_parquet": str(self.chunks_parquet.resolve()),
            },
            "outputs": {
                "verifier_train_materialized": str(self.out_parquet.resolve()),
            },
            "params": {
                "max_evidence_chunks": self.max_evidence_chunks,
                "min_evidence_weight": self.min_evidence_weight,
                "evidence_sep": self.evidence_sep,
                "parquet_batch_rows": self.parquet_batch_rows,
                "parquet_compression": self.parquet_compression,
            },
            "stats": {
                "rows_written": int(written),
                "num_chunks_in_map": int(len(self._chunk_map)),
            },
        }
        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.out_parquet}  (rows={written:,})")
        print(f"Wrote: {self.manifest_out}")






if __name__ == "__main__":
    global_config_path = "/PATH/grounding/config/config.yaml"
    model_weights_path = "/PATH/trained_model.pth"
    experiment_name = "name"


    with open(global_config_path, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    base_path_mimicData = params['Preprocessing']['base_path_mimicData']
    base_path_evidenceData = params['Preprocessing']['base_path_evidenceData']




    ''' step 6A '''
    # builder = RadiopaediaStep6ReembedAndFaissBuilder(
    #     cfg_path=global_config_path,
    #     chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
    #     model_weights_path=model_weights_path,
    #     out_dir=os.path.join(base_path_evidenceData, "step6_faiss_rstar"),
    #     batch_size=128,
    #     experiment_name=experiment_name,
    #     use_amp=True,
    #     device="cuda",
    #     seed=13,
    # )
    # builder.run()



    ''' step 6B '''
    # builder = RadiopaediaRStarWeakEvidenceAligner(
    #     qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
    #     chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
    #     faiss_index=os.path.join(base_path_evidenceData, "step6_faiss_rstar/faiss_Rstar.index"),
    #     out_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step6b_Rstar.parquet"),
    #     model_name=params['Preprocessing']['tokenizer_name'],
    #     rstar_query_encoder_ckpt=model_weights_path,
    #     top_k=20,
    #     alpha=0.85,
    #     temperature=0.07,
    #     seed=13,
    #     device="cuda",
    #     use_lexical=True,
    # )
    # builder.run()



    ''' step 6C '''
    # builder = RadiopaediaStep6CSupervisionBuilder(
    #     qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
    #     weak_alignment_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step6b_Rstar.parquet"),
    #     patient_split_map_json=os.path.join(base_path_evidenceData, "step4c_supervision/patient_split_map.json"),
    #     out_dir=os.path.join(base_path_evidenceData, "step6c_supervision_rstar"),
    #     top_k=20,
    #     min_weight=0.0,
    #     qa_chunksize_rows=50_000,
    #     parquet_batch_rows=50_000,
    #     # allowed_splits=["train", "valid"],
    # )
    # builder.run()



    ''' step 7 '''
    builder = RadiopaediaStep7VerifierMaterializer(
        verifier_supervision_parquet=os.path.join(base_path_evidenceData, "step6c_supervision_rstar/verifier_supervision.parquet"),
        chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "step7_verifier_materialized_rstar"),
        max_evidence_chunks=20,      # keep consistent with top_k
        min_evidence_weight=0.0,
        parquet_batch_rows=50_000,
        parquet_compression="zstd",
    )
    builder.run()
