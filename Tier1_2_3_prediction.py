"""
Created on Feb 24, 2026.
Tier1_2_3_prediction.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/


Tier1:
  - End2End R* retrieval -> V* predictions
  - Regression metrics (MSE/MAE/Pearson)
  - Calibration (ECE/Brier)
  - Selective risk (risk@coverage)

Tier2:
  - Retrieval metrics (Recall@1/5/10/20, MRR@10) against phase1 evidence_chunk_ids_json
  - Faithfulness stress tests:
      (a) random evidence
      (b) shuffled evidence from other samples (proxy "topically similar but wrong")
      (c) truncate to top-1 / top-3
      (d) shuffle chunk order
    For each: compute ΔMSE/ΔPearson/ΔECE relative to baseline S4.

Tier3: Distribution shift slices on the fixed test set:
    (a) report_length_bucket (short/medium/long)
    (b) device_present (0/1)
    (c) uncertainty_present (0/1)
    (d) label_vs_extractive (label/extractive)
    For each slice: report n, per-system metrics, and bootstrap CIs for key deltas (ΔMSE/ΔECE/ΔAURC) vs S4.


Cross-system:
    Align S1–S4 by qa_row_id
    Per-system metrics (MSE/MAE/Pearson/Spearman/ECE/Brier/NLL/AURC)
    Pairwise comparisons (all pairs):
    (a) paired bootstrap 95% CI for metric deltas
    (b) example-level win rates:
    - abs error win rate: |y−ŷ_A| < |y−ŷ_B|
    - calibration win rate: |p_A−y| < |p_B−y|

    faithfulness severity index (S2–S4 if tier2 faithfulness predictions exist):
    relative MSE degradation per perturbation
    fraction of examples with worse absolute error under corruption
"""

import os.path
import pdb
import re
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
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
import torch.nn as nn
import math
import pickle
import heapq
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from openai import OpenAI
import time
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.model import BiEncoder
from config.serde import open_experiment, create_experiment, delete_experiment, write_config


import warnings
warnings.filterwarnings('ignore')

openai.api_key = "PLACEHOLDER"
os.environ["OPENAI_API_KEY"] = "PLACEHOLDER"





class S4Tier123End2EndEvaluatorTestOnly:
    """
    S4 evaluator

    Input:
      phase1_fixed_test_eval/mimic_test_eval.parquet

    Runtime behavior:
      - loads R* and V*
      - builds evidence embeddings + FAISS index in-memory
      - retrieves for each claim_text
      - runs verifier
      - writes a single predictions parquet
      - writes json reports + one csv summary
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        model_weights_path: str,       # retriever trained weights
        verifier_weights_path: str,    # verifier trained weights
        device: str = "cuda",
        top_k: int = 20,
        verifier_top_p: int = 5,
        verifier_max_tokens: int = 1024,
        evidence_sep: str = "\n\n---\n\n",
        input_test_data: str = "phase1_fixed_test_eval/mimic_test_eval.parquet",
        batch_size_retrieve: int = 256,
        batch_size_verify: int = 4,
        batch_size_embed_docs: int = 256,
        ece_bins: int = 15,
        out_dir: Optional[str] = None,
        seed: int = 13,
        # Tier2
        mrr_k: int = 10,
        recall_ks: Tuple[int, ...] = (1, 5, 10, 20),
        faithfulness_random_trials: int = 1,
        faithfulness_max_eval: Optional[int] = None,
        # FAISS
        faiss_index_factory: str = "Flat",
        # Evidence corpus
        chunks_parquet_rel: str = "chunks.parquet",  # under base_path_evidenceData
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.model_weights_path = model_weights_path
        self.verifier_weights_path = verifier_weights_path

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)  # retrieval top-k stays 20
        self.verifier_top_p = int(verifier_top_p)
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.evidence_sep = evidence_sep

        self.batch_size_retrieve = int(batch_size_retrieve)
        self.batch_size_verify = int(batch_size_verify)
        self.batch_size_embed_docs = int(batch_size_embed_docs)

        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.mrr_k = int(mrr_k)
        self.recall_ks = tuple(int(x) for x in recall_ks)
        self.faithfulness_random_trials = int(faithfulness_random_trials)
        self.faithfulness_max_eval = int(faithfulness_max_eval) if faithfulness_max_eval is not None else None

        self.faiss_index_factory = str(faiss_index_factory)

        # Phase-1 fixed test
        self.test_parquet = os.path.join(
            self.base_path_evidenceData, input_test_data
        )

        # Evidence chunks
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, chunks_parquet_rel)

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier123_results_s4_end2end_testonly")
        os.makedirs(self.out_dir, exist_ok=True)

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        self.retriever_tok = None
        self.retriever_model = None
        self.verifier_tok = None
        self.verifier_model = None

        # Built on the fly
        self.faiss_index = None
        self.faiss_chunk_ids = None
        self.all_chunk_ids_universe = None
        self.chunk_map = None

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

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _load_chunk_map(self) -> Dict[int, str]:
        if not os.path.exists(self.chunks_parquet):
            raise FileNotFoundError(self.chunks_parquet)

        df = pd.read_parquet(self.chunks_parquet, columns=["chunk_id", "chunk_text"])
        df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
        df = df[~df["chunk_id"].isna()].copy()
        df["chunk_id"] = df["chunk_id"].astype(np.int64)
        df["chunk_text"] = df["chunk_text"].astype(str)

        m: Dict[int, str] = {}
        for cid, txt in zip(df["chunk_id"].tolist(), df["chunk_text"].tolist()):
            t = " ".join(str(txt).split()).strip()
            m[int(cid)] = t
        if not m:
            raise RuntimeError("chunk_map is empty after loading chunks.parquet.")
        return m

    def _load_retriever(self) -> None:
        model_name = self.params["retriever"]["model_name"]
        tied = bool(self.params["retriever"]["tied_encoders"])
        pooling = self.params["retriever"]["pooling"]
        self.max_query_tokens = int(self.params["retriever"]["max_query_tokens"])
        self.max_doc_tokens = int(self.params["retriever"].get("max_doc_tokens", 512))

        self.retriever_tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.retriever_model = BiEncoder(model_name, tied=tied, pooling=pooling).to(self.device)
        self.retriever_model.eval()

        obj = torch.load(self.model_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj
        self.retriever_model.load_state_dict(state, strict=True)

    def _load_verifier(self) -> None:
        vname = self.params["Verifier"]["model_name"]
        self.verifier_tok = AutoTokenizer.from_pretrained(vname, use_fast=True)

        self.verifier_model = AutoModelForSequenceClassification.from_pretrained(
            vname,
            num_labels=1,
            problem_type="regression",
        ).to(self.device)
        self.verifier_model.eval()

        obj = torch.load(self.verifier_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj
        self.verifier_model.load_state_dict(state, strict=True)
        self.verifier_model.eval()


    def _loads_json_list(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _normalize_claim(self, x: str) -> str:
        return " ".join(str(x).split()).strip().lower()

    # Build FAISS on the fly
    def _sanitize_faiss_factory(self, s: str) -> str:
        raw = str(s).strip()
        if raw.lower() in {"flatip", "flat_ip", "flat-ип", "flatinnerproduct"}:
            return "Flat"
        if raw.lower() == "flatl2":
            return "Flat"
        if raw.endswith("IP") and raw[:-2] == "Flat":
            return "Flat"
        return raw

    def _build_faiss_on_the_fly(self) -> None:
        assert self.retriever_model is not None
        assert self.retriever_tok is not None

        if not os.path.exists(self.chunks_parquet):
            raise FileNotFoundError(self.chunks_parquet)

        pf = pq.ParquetFile(self.chunks_parquet)
        n_rows = pf.metadata.num_rows if pf.metadata is not None else 0

        chunk_ids: List[int] = []
        doc_embs: List[np.ndarray] = []

        desc = "S4: embedding evidence with R* (doc encoder)"
        with torch.no_grad():
            for batch in tqdm(
                pf.iter_batches(batch_size=50_000, columns=["chunk_id", "chunk_text"]),
                total=max(1, math.ceil(n_rows / 50_000)) if n_rows else None,
                desc=desc,
                unit="batch",
            ):
                df = batch.to_pandas()
                df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
                df = df[~df["chunk_id"].isna()].copy()
                df["chunk_id"] = df["chunk_id"].astype(np.int64)
                df["chunk_text"] = df["chunk_text"].astype(str)

                ids = df["chunk_id"].tolist()
                texts = [" ".join(t.split()).strip() for t in df["chunk_text"].tolist()]
                if not texts:
                    continue

                for i in range(0, len(texts), self.batch_size_embed_docs):
                    t_batch = texts[i:i + self.batch_size_embed_docs]
                    id_batch = ids[i:i + self.batch_size_embed_docs]

                    enc = self.retriever_tok(
                        t_batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_doc_tokens,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}

                    d_emb = self.retriever_model.encode_docs(enc["input_ids"], enc["attention_mask"])
                    d_emb = torch.nn.functional.normalize(d_emb, p=2, dim=1).detach().cpu().numpy().astype(np.float32)

                    doc_embs.append(d_emb)
                    chunk_ids.extend([int(x) for x in id_batch])

        if not chunk_ids:
            raise RuntimeError("No evidence chunks were embedded. chunks.parquet may be empty or malformed.")

        X = np.concatenate(doc_embs, axis=0)
        self.faiss_chunk_ids = np.asarray(chunk_ids, dtype=np.int64)
        self.all_chunk_ids_universe = self.faiss_chunk_ids.copy()

        dim = int(X.shape[1])

        factory = self._sanitize_faiss_factory(self.faiss_index_factory)
        try:
            index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
        except Exception as e:
            raise RuntimeError(
                f"FAISS index_factory failed. factory='{factory}' (from '{self.faiss_index_factory}'), dim={dim}. "
                f"Use 'Flat' or 'IVF4096,Flat' style strings."
            ) from e

        index.add(X)
        self.faiss_index = index

    # Retrieval (R*)
    def _tokenize_queries(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.retriever_tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_query_tokens,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _retrieve_topk_chunk_ids(self, claim_texts: List[str], desc: str) -> Tuple[List[List[int]], List[List[float]]]:
        assert self.retriever_model is not None
        assert self.faiss_index is not None
        assert self.faiss_chunk_ids is not None

        all_ids: List[List[int]] = []
        all_scores: List[List[float]] = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(claim_texts), self.batch_size_retrieve),
                total=math.ceil(len(claim_texts) / self.batch_size_retrieve),
                desc=desc,
                unit="batch",
            ):
                batch = [self._normalize_claim(x) for x in claim_texts[i:i + self.batch_size_retrieve]]
                tok = self._tokenize_queries(batch)
                q_emb = self.retriever_model.encode_queries(tok["input_ids"], tok["attention_mask"])
                q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).detach().cpu().numpy().astype(np.float32)

                D, I = self.faiss_index.search(q_emb, self.top_k)
                for j in range(I.shape[0]):
                    idxs = I[j].tolist()
                    scrs = D[j].astype(np.float32).tolist()
                    cids = [int(self.faiss_chunk_ids[ii]) for ii in idxs if ii >= 0]
                    all_ids.append(cids)
                    all_scores.append([float(s) for s in scrs[:len(cids)]])

        return all_ids, all_scores

    def _materialize_evidence(self, chunk_ids: List[int]) -> str:
        assert self.chunk_map is not None
        texts = []
        for cid in chunk_ids[: self.verifier_top_p]:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)

    # Verifier inference
    def _build_verifier_inputs(
        self,
        case_context: str,
        question: str,
        answer_target: str,
        claim_text: str,
        evidence_text: str,
    ) -> str:
        sep = "\n\n"
        return sep.join(
            [
                "CASE:",
                self._safe_str(case_context).strip(),
                "QUESTION:",
                self._safe_str(question).strip(),
                "CLAIM:",
                self._safe_str(claim_text).strip(),
                "EVIDENCE:",
                self._safe_str(evidence_text).strip(),
            ]
        )

    def _predict_support(self, texts: List[str], desc: str) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size_verify),
                total=math.ceil(len(texts) / self.batch_size_verify),
                desc=desc,
                unit="batch",
            ):
                batch = texts[i:i + self.batch_size_verify]
                enc = self.verifier_tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.verifier_max_tokens,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.verifier_model(**enc)
                y = out.logits.squeeze(-1)
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

    def _predict_with_ids(self, df_sub: pd.DataFrame, ids_per_sample: List[List[int]], desc: str) -> np.ndarray:
        inputs: List[str] = []
        for i in range(len(df_sub)):
            q = df_sub.iloc[i]["question"]
            c = df_sub.iloc[i]["case_context"]

            if "answer_target" in df_sub.columns:
                a = df_sub.iloc[i]["answer_target"]
            elif "answer" in df_sub.columns:
                a = df_sub.iloc[i]["answer"]
            else:
                a = ""

            if "claim_text" in df_sub.columns:
                cl = df_sub.iloc[i]["claim_text"]
            elif "claim" in df_sub.columns:
                cl = df_sub.iloc[i]["claim"]
            else:
                cl = ""

            ev = self._materialize_evidence(ids_per_sample[i])
            inputs.append(
                self._build_verifier_inputs(
                    case_context=c,
                    question=q,
                    answer_target=a,
                    claim_text=cl,
                    evidence_text=ev,
                )
            )
        return self._predict_support(inputs, desc=desc)

    def _mse(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean((y - yhat) ** 2))

    def _mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean(np.abs(y - yhat)))

    def _pearson(self, y: np.ndarray, yhat: np.ndarray) -> float:
        y = y.astype(np.float64)
        yhat = yhat.astype(np.float64)
        y = y - y.mean()
        yhat = yhat - yhat.mean()
        denom = (np.sqrt((y ** 2).sum()) * np.sqrt((yhat ** 2).sum()))
        if denom <= 0:
            return 0.0
        return float((y * yhat).sum() / denom)

    def _rankdata_avg_ties(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)

        xs = x[order]
        i = 0
        while i < len(xs):
            j = i + 1
            while j < len(xs) and xs[j] == xs[i]:
                j += 1
            if j - i > 1:
                avg = (i + (j - 1)) / 2.0
                ranks[order[i:j]] = avg
            i = j
        return ranks

    def _spearman(self, y: np.ndarray, yhat: np.ndarray) -> float:
        ry = self._rankdata_avg_ties(y)
        rx = self._rankdata_avg_ties(yhat)
        return self._pearson(ry.astype(np.float32), rx.astype(np.float32))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _brier(self, y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean((p - y) ** 2))

    def _ece(self, y: np.ndarray, p: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(y)
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(y[mask].mean())
            conf = float(p[mask].mean())
            ece += (m / n) * abs(acc - conf)
        return float(ece)

    def _risk_coverage_from_prob(self, y: np.ndarray, yhat: np.ndarray, p: np.ndarray) -> Dict[str, object]:
        conf = np.maximum(p, 1.0 - p)
        order = np.argsort(-conf)
        y_o = y[order]
        yhat_o = yhat[order]

        coverages = np.linspace(0.1, 1.0, 10)
        fixed = [0.6, 0.8, 0.9]

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(y_o))))
            return float(np.mean((y_o[:k] - yhat_o[:k]) ** 2))

        curve = [{"coverage": float(c), "risk_mse": risk_at(float(c))} for c in coverages]
        fixed_out = {f"risk_mse@{int(c*100)}": risk_at(c) for c in fixed}
        return {"curve": curve, **fixed_out}

    def _eval_preds_metrics(self, y01: np.ndarray, yhat: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        return {
            "MSE": self._mse(y01, yhat),
            "MAE": self._mae(y01, yhat),
            "Pearson": self._pearson(y01, yhat),
            "Spearman": self._spearman(y01, yhat),
            "ECE": self._ece(y01, p, self.ece_bins),
            "Brier": self._brier(y01, p),
        }

    # Tier2 retrieval
    def _recall_at_k(self, retrieved: List[int], gold: List[int], k: int) -> float:
        if not gold:
            return 0.0
        r = retrieved[:k]
        g = set(gold)
        return 1.0 if any((x in g) for x in r) else 0.0

    def _mrr_at_k(self, retrieved: List[int], gold: List[int], k: int) -> float:
        if not gold:
            return 0.0
        g = set(gold)
        r = retrieved[:k]
        for i, x in enumerate(r):
            if x in g:
                return 1.0 / float(i + 1)
        return 0.0

    def _compute_retrieval_metrics(self, df_eval: pd.DataFrame, retrieved_topk_ids: List[List[int]]) -> Dict[str, float]:
        gold_lists: List[List[int]] = []
        for s in df_eval["evidence_chunk_ids_json"].astype(str).tolist():
            ids = self._loads_json_list(s)
            ids = [int(x) for x in ids if str(x).strip() != ""]
            gold_lists.append(ids)

        out: Dict[str, float] = {}
        for k in self.recall_ks:
            vals = [self._recall_at_k(r, g, k) for r, g in zip(retrieved_topk_ids, gold_lists)]
            out[f"Recall@{k}"] = float(np.mean(vals)) if vals else 0.0

        k_mrr = self.mrr_k
        vals_mrr = [self._mrr_at_k(r, g, k_mrr) for r, g in zip(retrieved_topk_ids, gold_lists)]
        out[f"MRR@{k_mrr}"] = float(np.mean(vals_mrr)) if vals_mrr else 0.0
        return out

    # Tier2 faithfulness stress
    def _stress_random_ids(self, n: int) -> List[List[int]]:
        assert self.all_chunk_ids_universe is not None
        U = self.all_chunk_ids_universe
        out: List[List[int]] = []
        for _ in range(n):
            idx = np.random.choice(len(U), size=min(self.top_k, len(U)), replace=False)
            out.append([int(x) for x in U[idx].tolist()])
        return out

    def _stress_swap_between_samples(self, ids_per_sample: List[List[int]]) -> List[List[int]]:
        if len(ids_per_sample) <= 1:
            return ids_per_sample
        out = [ids_per_sample[(i + 1) % len(ids_per_sample)] for i in range(len(ids_per_sample))]
        return [list(x[: self.top_k]) for x in out]

    def _stress_truncate(self, ids_per_sample: List[List[int]], k: int) -> List[List[int]]:
        kk = max(1, int(k))
        return [list(x[:kk]) for x in ids_per_sample]

    def _stress_shuffle_order(self, ids_per_sample: List[List[int]]) -> List[List[int]]:
        out: List[List[int]] = []
        for x in ids_per_sample:
            y = list(x[: self.top_k])
            random.shuffle(y)
            out.append(y)
        return out

    # Tier3 slices
    def _compute_slice_metrics(
        self,
        df_test: pd.DataFrame,
        y01: np.ndarray,
        yhat: np.ndarray,
        p: np.ndarray,
        prefix: str = "test_slice",
    ) -> Dict[str, object]:
        out: Dict[str, object] = {}

        def metrics_for_mask(mask: np.ndarray) -> Dict[str, object]:
            if mask.sum() == 0:
                return {"n": 0, "MSE": None, "ECE": None, "risk_mse@80": None}
            yy = y01[mask]
            yh = yhat[mask]
            pp = p[mask]
            rc = self._risk_coverage_from_prob(yy, yh, pp)
            return {
                "n": int(mask.sum()),
                "MSE": self._mse(yy, yh),
                "ECE": self._ece(yy, pp, self.ece_bins),
                "risk_mse@80": float(rc["risk_mse@80"]),
            }

        if "report_length_bucket" in df_test.columns:
            for v in ["short", "medium", "long"]:
                mask = (df_test["report_length_bucket"].astype(str).values == v)
                out[f"{prefix}.report_length_bucket.{v}"] = metrics_for_mask(mask)

        if "device_present" in df_test.columns:
            mask1 = (pd.to_numeric(df_test["device_present"], errors="coerce").fillna(0).astype(int).values == 1)
            out[f"{prefix}.device_present.1"] = metrics_for_mask(mask1)
            out[f"{prefix}.device_present.0"] = metrics_for_mask(~mask1)

        if "uncertainty_present" in df_test.columns:
            mask1 = (pd.to_numeric(df_test["uncertainty_present"], errors="coerce").fillna(0).astype(int).values == 1)
            out[f"{prefix}.uncertainty_present.1"] = metrics_for_mask(mask1)
            out[f"{prefix}.uncertainty_present.0"] = metrics_for_mask(~mask1)

        if "label_vs_extractive" in df_test.columns:
            for v in ["closed_form", "other"]:
                mask = (df_test["label_vs_extractive"].astype(str).values == v)
                out[f"{prefix}.label_vs_extractive.{v}"] = metrics_for_mask(mask)

        return out

    def _run_end2end_test(self, df_test: pd.DataFrame) -> Dict[str, object]:
        claim_texts = df_test["claim_text"].astype(str).tolist()
        topk_ids, _ = self._retrieve_topk_chunk_ids(claim_texts, desc="S4 Retrieval (R*) [test]")

        inputs: List[str] = []
        for i in range(len(df_test)):
            q = df_test.iloc[i]["question"]
            c = df_test.iloc[i]["case_context"]

            if "answer_target" in df_test.columns:
                a = df_test.iloc[i]["answer_target"]
            elif "answer" in df_test.columns:
                a = df_test.iloc[i]["answer"]
            else:
                a = ""

            if "claim_text" in df_test.columns:
                cl = df_test.iloc[i]["claim_text"]
            elif "claim" in df_test.columns:
                cl = df_test.iloc[i]["claim"]
            else:
                cl = ""

            ev = self._materialize_evidence(topk_ids[i])   # now only top-5 is materialized for verifier
            inputs.append(
                self._build_verifier_inputs(
                    case_context=c,
                    question=q,
                    answer_target=a,
                    claim_text=cl,
                    evidence_text=ev,
                )
            )

        yhat = self._predict_support(inputs, desc="S4 Verification (V*) [test]")
        y = pd.to_numeric(df_test["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01 = np.clip(y, 0.0, 1.0)
        p = np.clip(yhat, 0.0, 1.0)

        out_df = df_test.copy()
        out_df["s4_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        out_df["s4_pred_support_raw"] = yhat.astype(np.float32)
        out_df["s4_pred_prob_sigmoid"] = p.astype(np.float32)

        pred_path = os.path.join(self.out_dir, "s4_test_predictions.parquet")
        out_df.to_parquet(pred_path, index=False)

        return {
            "n": int(len(df_test)),
            "y01": y01,
            "yhat": yhat,
            "p": p,
            "topk_ids": topk_ids,
            "predictions_parquet": pred_path,
        }

    def _write_csv_summary(self) -> str:
        tier1_path = os.path.join(self.out_dir, "s4_tier1_report.json")
        tier2_faith_path = os.path.join(self.out_dir, "s4_tier2_faithfulness_report.json")
        tier2_ret_path = os.path.join(self.out_dir, "s4_tier2_retrieval_metrics.json")
        tier3_path = os.path.join(self.out_dir, "s4_tier3_distribution_slices.json")

        out_csv = os.path.join(self.out_dir, "S4_full_results.csv")

        def load_json(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        def safe_get(d, keys, default=None):
            cur = d
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        rows = []
        t1 = load_json(tier1_path) if os.path.exists(tier1_path) else {}
        t2f = load_json(tier2_faith_path) if os.path.exists(tier2_faith_path) else {}
        t2r = load_json(tier2_ret_path) if os.path.exists(tier2_ret_path) else {}
        t3 = load_json(tier3_path) if os.path.exists(tier3_path) else {}

        t1m = t1.get("metrics", {}) or {}
        for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]:
            if k in t1m:
                rows.append(["Tier1", "overall", k, t1m[k]])
        risk80 = safe_get(t1m, ["SelectiveRisk", "risk_mse@80"])
        if risk80 is not None:
            rows.append(["Tier1", "overall", "risk_mse@80", risk80])

        for k, val in (t2r.get("metrics", {}) or {}).items():
            rows.append(["Tier2_Retrieval", "overall", k, val])

        base = t2f.get("baseline_metrics", {}) or {}
        for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]:
            if k in base:
                rows.append(["Tier2_Faithfulness", "baseline", k, base[k]])

        for name, vals in (t2f.get("perturbations", {}) or {}).items():
            d = vals.get("delta_vs_base", {}) or {}
            for dk in ["ΔMSE", "ΔECE", "ΔBrier", "ΔMAE", "ΔPearson", "ΔSpearman"]:
                if dk in d:
                    rows.append(["Tier2_Faithfulness", name, f"{dk}_vs_base", d[dk]])

        for slice_name, vals in (t3.get("metrics", {}) or {}).items():
            n = vals.get("n")
            if n is not None:
                rows.append(["Tier3", slice_name, "n", n])
            for k in ["MSE", "ECE", "Brier"]:
                if k in vals and vals[k] is not None:
                    rows.append(["Tier3", slice_name, k, vals[k]])
            r80 = vals.get("risk_mse@80")
            if r80 is not None:
                rows.append(["Tier3", slice_name, "risk_mse@80", r80])

        df_csv = pd.DataFrame(rows, columns=["Tier", "Subgroup", "Metric", "Value"])
        df_csv["Value"] = pd.to_numeric(df_csv["Value"], errors="coerce")

        mask_n = df_csv["Metric"].astype(str).str.lower().eq("n")
        df_csv.loc[~mask_n, "Value"] = df_csv.loc[~mask_n, "Value"] * 100.0
        df_csv["Value"] = df_csv["Value"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        df_csv.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        return out_csv

    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)

        df_test = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        df_test = df_test.iloc[: int(len(df_test) * 0.5)].reset_index(drop=True)

        if "split" in df_test.columns:
            df_test = df_test[df_test["split"].astype(str) == "test"].copy()

        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df_test = df_test.sample(
                n=min(len(df_test), self.faithfulness_max_eval),
                random_state=self.seed
            ).reset_index(drop=True)

        self.chunk_map = self._load_chunk_map()
        self._load_retriever()
        self._load_verifier()
        self._build_faiss_on_the_fly()

        pack = self._run_end2end_test(df_test)
        y01 = pack["y01"]
        yhat = pack["yhat"]
        p = pack["p"]

        # Tier1
        tier1_metrics = self._eval_preds_metrics(y01, yhat, p)
        rc = self._risk_coverage_from_prob(y01, yhat, p)

        tier1_report = {
            "system": "S4 (R* retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(pack["n"]),
            "retriever": {
                "weights": self.model_weights_path,
                "faiss_factory": self._sanitize_faiss_factory(self.faiss_index_factory),
                "top_k": self.top_k,
            },
            "verifier": {
                "weights": self.verifier_weights_path,
                "top_p": self.verifier_top_p,        # NEW
                "max_tokens": self.verifier_max_tokens,
            },
            "metrics": {**tier1_metrics, "SelectiveRisk": rc},
            "artifacts": {"predictions_parquet": pack["predictions_parquet"]},
        }

        tier1_path = os.path.join(self.out_dir, "s4_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        # Tier2A retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics(df_test, pack["topk_ids"])
        retrieval_path = os.path.join(self.out_dir, "s4_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S4 (R* retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(pack["n"]),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )

        # Tier2B faithfulness stress tests
        base_metrics = self._eval_preds_metrics(y01, yhat, p)
        topk_ids = pack["topk_ids"]

        ids_top1 = self._stress_truncate(topk_ids, 1)
        ids_top3 = self._stress_truncate(topk_ids, 3)
        ids_shuffle = self._stress_shuffle_order(topk_ids)
        ids_swapped = self._stress_swap_between_samples(topk_ids)

        random_trial_metrics: List[Dict[str, float]] = []
        random_trial_preds: List[np.ndarray] = []

        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df_test))
            yhat_rand = self._predict_with_ids(df_test, ids_rand, desc=f"S4 Faithfulness random (trial {t+1}) [test]")
            random_trial_preds.append(yhat_rand)
            p_rand = np.clip(yhat_rand, 0.0, 1.0)
            random_trial_metrics.append(self._eval_preds_metrics(y01, yhat_rand, p_rand))

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {k: 0.0 for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]}
            keys = list(ms[0].keys())
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        yhat_top1 = self._predict_with_ids(df_test, ids_top1, desc="S4 Faithfulness top-1 [test]")
        yhat_top3 = self._predict_with_ids(df_test, ids_top3, desc="S4 Faithfulness top-3 [test]")
        yhat_shuffle = self._predict_with_ids(df_test, ids_shuffle, desc="S4 Faithfulness shuffle-order [test]")
        yhat_swapped = self._predict_with_ids(df_test, ids_swapped, desc="S4 Faithfulness swapped-evidence [test]")

        def pack_metrics(yhat_x: np.ndarray) -> Dict[str, float]:
            return self._eval_preds_metrics(y01, yhat_x, np.clip(yhat_x, 0.0, 1.0))

        met_top1 = pack_metrics(yhat_top1)
        met_top3 = pack_metrics(yhat_top3)
        met_shuffle = pack_metrics(yhat_shuffle)
        met_swapped = pack_metrics(yhat_swapped)
        met_random = avg_metrics(random_trial_metrics)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S4 (R* retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(pack["n"]),
            "baseline_metrics": base_metrics,
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "metrics": met_random,
                    "delta_vs_base": delta(met_random, base_metrics),
                },
                "swapped_evidence": {
                    "metrics": met_swapped,
                    "delta_vs_base": delta(met_swapped, base_metrics),
                },
                "truncate_top1": {
                    "metrics": met_top1,
                    "delta_vs_base": delta(met_top1, base_metrics),
                },
                "truncate_top3": {
                    "metrics": met_top3,
                    "delta_vs_base": delta(met_top3, base_metrics),
                },
                "shuffle_order": {
                    "metrics": met_shuffle,
                    "delta_vs_base": delta(met_shuffle, base_metrics),
                },
            },
        }

        faith_df = df_test.copy()
        faith_df["s4_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        faith_df["s4_pred_base_raw"] = yhat.astype(np.float32)
        faith_df["s4_pred_top1_raw"] = yhat_top1.astype(np.float32)
        faith_df["s4_pred_top3_raw"] = yhat_top3.astype(np.float32)
        faith_df["s4_pred_shuffle_raw"] = yhat_shuffle.astype(np.float32)
        faith_df["s4_pred_swapped_raw"] = yhat_swapped.astype(np.float32)
        if random_trial_preds:
            yhat_rand_avg = np.mean(np.stack(random_trial_preds, axis=0), axis=0).astype(np.float32)
            faith_df["s4_pred_random_avg_raw"] = yhat_rand_avg

        faith_pred_path = os.path.join(self.out_dir, "s4_tier2_faithfulness_predictions.parquet")
        faith_df.to_parquet(faith_pred_path, index=False)

        faith_path = os.path.join(self.out_dir, "s4_tier2_faithfulness_report.json")
        with open(faith_path, "w", encoding="utf-8") as f:
            json.dump({**faith_report, "artifacts": {"faithfulness_predictions_parquet": faith_pred_path}}, f, indent=2)

        # Tier3 slices
        slice_metrics = self._compute_slice_metrics(df_test, y01, yhat, p, prefix="test_slice")
        slice_path = os.path.join(self.out_dir, "s4_tier3_distribution_slices.json")
        with open(slice_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S4 (R* retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(pack["n"]),
                    "metrics": slice_metrics,
                },
                f,
                indent=2,
            )

        summary = {
            "test": {
                "tier1_report": tier1_path,
                "tier2_retrieval_metrics": retrieval_path,
                "tier2_faithfulness_report": faith_path,
                "tier2_faithfulness_predictions": faith_pred_path,
                "tier3_distribution_slices": slice_path,
                "predictions_parquet": pack["predictions_parquet"],
            }
        }
        summary_path = os.path.join(self.out_dir, "s4_tier123_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_path}")
        print(f"Wrote: {slice_path}")
        print(f"Wrote: {summary_path}")

        self._write_csv_summary()

        return summary_path




class S2Tier123End2EndEvaluatorTestOnly:
    """
    S2 — Frozen dense MPNet retrieval -> V*
    Test-only, end-to-end:
      - Reads phase1_fixed_test_eval/mimic_test_eval.parquet
      - Off-the-shelf MPNet retriever (no finetuning)
      - Embeds evidence on-the-fly (in memory) and builds FAISS in memory
      - Runs tiers 1-3 on test
      - Writes one main predictions parquet + json reports + one csv summary
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,  # verifier trained_model.pth
        device: str = "cuda",
        top_k: int = 20,
        verifier_top_p: int = 5,
        verifier_max_tokens: int = 1024,
        input_test_data: str = "phase1_fixed_test_eval/mimic_test_eval.parquet",
        evidence_sep: str = "\n\n---\n\n",
        batch_size_embed_chunks: int = 128,
        batch_size_retrieve: int = 256,
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        out_dir: Optional[str] = None,
        seed: int = 13,
        # Tier2 knobs
        mrr_k: int = 10,
        recall_ks: Tuple[int, ...] = (1, 5, 10, 20),
        faithfulness_random_trials: int = 1,
        faithfulness_max_eval: Optional[int] = None,  # e.g. 2000 for fast debug
        # Output knobs
        write_faithfulness_predictions_parquet: bool = False,
        # Evidence corpus
        chunks_parquet_rel: str = "chunks.parquet",
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)
        self.verifier_top_p = int(verifier_top_p)   # NEW
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.evidence_sep = evidence_sep

        self.batch_size_embed_chunks = int(batch_size_embed_chunks)
        self.batch_size_retrieve = int(batch_size_retrieve)
        self.batch_size_verify = int(batch_size_verify)
        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.mrr_k = int(mrr_k)
        self.recall_ks = tuple(int(x) for x in recall_ks)
        self.faithfulness_random_trials = int(faithfulness_random_trials)
        self.faithfulness_max_eval = int(faithfulness_max_eval) if faithfulness_max_eval is not None else None

        self.write_faithfulness_predictions_parquet = bool(write_faithfulness_predictions_parquet)

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, input_test_data
        )
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, chunks_parquet_rel)

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier123_results_s2_end2end_testonly")
        os.makedirs(self.out_dir, exist_ok=True)

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        # Retriever (MPNet)
        self.retriever_tok = None
        self.retriever_model = None
        self.max_query_tokens = None
        self.max_doc_tokens = None

        # Verifier (V*)
        self.verifier_tok = None
        self.verifier_model = None

        # Evidence
        self.chunk_map = None  # Dict[int,str]
        self.faiss_index = None
        self.faiss_chunk_ids = None
        self.all_chunk_ids_universe = None  # np.ndarray int64

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

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _load_chunk_map(self) -> Dict[int, str]:
        if not os.path.exists(self.chunks_parquet):
            raise FileNotFoundError(self.chunks_parquet)

        df = pd.read_parquet(self.chunks_parquet, columns=["chunk_id", "chunk_text"])
        df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
        df = df[~df["chunk_id"].isna()].copy()
        df["chunk_id"] = df["chunk_id"].astype(np.int64)
        df["chunk_text"] = df["chunk_text"].astype(str)

        m: Dict[int, str] = {}
        for cid, txt in zip(df["chunk_id"].tolist(), df["chunk_text"].tolist()):
            t = " ".join(str(txt).split()).strip()
            m[int(cid)] = t
        if not m:
            raise RuntimeError("chunk_map is empty after loading chunks.parquet.")
        return m

    # Load off-the-shelf MPNet retriever
    def _load_offtheshelf_mpnet(self) -> None:
        model_name = self.params["retriever"]["model_name"]  # sentence-transformers/all-mpnet-base-v2
        self.max_query_tokens = int(self.params["retriever"]["max_query_tokens"])
        self.max_doc_tokens = int(self.params["retriever"]["max_doc_tokens"])

        self.retriever_tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.retriever_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.retriever_model.eval()

    # Build FAISS in memory on the fly
    def _mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _normalize_claim(self, x: str) -> str:
        return " ".join(str(x).split()).strip().lower()

    def _encode_texts(self, texts: List[str], max_len: int, batch_size: int, desc: str) -> np.ndarray:
        assert self.retriever_tok is not None
        assert self.retriever_model is not None

        embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), batch_size),
                total=math.ceil(len(texts) / batch_size),
                desc=desc,
                unit="batch",
            ):
                batch = [self._normalize_claim(x) for x in texts[i:i + batch_size]]
                enc = self.retriever_tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.retriever_model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embs.append(pooled.detach().float().cpu().numpy().astype(np.float32))

        dim = int(embs[0].shape[1]) if embs else 768
        return np.vstack(embs) if embs else np.zeros((0, dim), dtype=np.float32)

    def _build_faiss_on_the_fly(self) -> None:
        """
        Stream chunks.parquet, embed chunk_text using MPNet, build IndexFlatIP in memory.
        """
        if not os.path.exists(self.chunks_parquet):
            raise FileNotFoundError(self.chunks_parquet)

        assert self.max_doc_tokens is not None

        pf = pq.ParquetFile(self.chunks_parquet)
        n_rows = pf.metadata.num_rows if pf.metadata is not None else 0

        chunk_ids: List[int] = []
        doc_embs: List[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(
                pf.iter_batches(batch_size=50_000, columns=["chunk_id", "chunk_text"]),
                total=max(1, math.ceil(n_rows / 50_000)) if n_rows else None,
                desc="S2: embedding evidence with MPNet (doc encoder)",
                unit="batch",
            ):
                df = batch.to_pandas()
                df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
                df = df[~df["chunk_id"].isna()].copy()
                df["chunk_id"] = df["chunk_id"].astype(np.int64)
                df["chunk_text"] = df["chunk_text"].astype(str).map(lambda x: " ".join(str(x).split()).strip())

                ids = df["chunk_id"].tolist()
                texts = df["chunk_text"].tolist()
                if not texts:
                    continue

                for i in range(0, len(texts), self.batch_size_embed_chunks):
                    t_batch = texts[i:i + self.batch_size_embed_chunks]
                    id_batch = ids[i:i + self.batch_size_embed_chunks]

                    embeds = self._encode_texts(
                        texts=t_batch,
                        max_len=self.max_doc_tokens,
                        batch_size=self.batch_size_embed_chunks,
                        desc="S2 embed chunks (MPNet)",
                    )
                    doc_embs.append(embeds)
                    chunk_ids.extend([int(x) for x in id_batch])

        if not chunk_ids:
            raise RuntimeError("No chunks embedded. chunks.parquet empty or wrong schema?")

        X = np.concatenate(doc_embs, axis=0).astype(np.float32)
        chunk_ids_np = np.asarray(chunk_ids, dtype=np.int64)

        dim = int(X.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(X)

        self.faiss_index = index
        self.faiss_chunk_ids = chunk_ids_np
        self.all_chunk_ids_universe = chunk_ids_np.copy()

    # Load verifier
    def _load_verifier(self) -> None:
        vname = self.params["Verifier"]["model_name"]
        self.verifier_tok = AutoTokenizer.from_pretrained(vname, use_fast=True)
        self.verifier_model = AutoModelForSequenceClassification.from_pretrained(
            vname,
            num_labels=1,
            problem_type="regression",
        ).to(self.device)
        self.verifier_model.eval()

        obj = torch.load(self.verifier_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj
        self.verifier_model.load_state_dict(state, strict=True)
        self.verifier_model.eval()


    # Retrieval
    def _retrieve_topk_chunk_ids(self, claim_texts: List[str]) -> List[List[int]]:
        assert self.faiss_index is not None
        assert self.faiss_chunk_ids is not None
        assert self.max_query_tokens is not None

        q_emb = self._encode_texts(
            texts=claim_texts,
            max_len=self.max_query_tokens,
            batch_size=self.batch_size_retrieve,
            desc="S2 Retrieval (MPNet off-the-shelf) [test]",
        )
        D, I = self.faiss_index.search(q_emb.astype(np.float32), self.top_k)

        all_ids: List[List[int]] = []
        for j in range(I.shape[0]):
            idxs = I[j].tolist()
            cids = [int(self.faiss_chunk_ids[ii]) for ii in idxs if ii >= 0]
            all_ids.append(cids)
        return all_ids

    def _materialize_evidence(self, chunk_ids: List[int]) -> str:
        assert self.chunk_map is not None
        texts = []
        for cid in chunk_ids[: self.verifier_top_p]:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)

    # ===== Verifier inference =====
    def _build_verifier_inputs(
        self,
        case_context: str,
        question: str,
        answer_target: str,
        claim_text: str,
        evidence_text: str,
    ) -> str:
        sep = "\n\n"
        return sep.join(
            [
                "CASE:",
                self._safe_str(case_context).strip(),
                "QUESTION:",
                self._safe_str(question).strip(),
                "CLAIM:",
                self._safe_str(claim_text).strip(),
                "EVIDENCE:",
                self._safe_str(evidence_text).strip(),
            ]
        )

    def _predict_support(self, texts: List[str], desc: str) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size_verify),
                total=math.ceil(len(texts) / self.batch_size_verify),
                desc=desc,
                unit="batch",
            ):
                batch = texts[i:i + self.batch_size_verify]
                enc = self.verifier_tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.verifier_max_tokens,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.verifier_model(**enc)
                y = out.logits.squeeze(-1)
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

    # ===== Metrics =====
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _mse(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean((y - yhat) ** 2))

    def _mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean(np.abs(y - yhat)))

    def _pearson(self, y: np.ndarray, yhat: np.ndarray) -> float:
        y = y.astype(np.float64)
        yhat = yhat.astype(np.float64)
        y = y - y.mean()
        yhat = yhat - yhat.mean()
        denom = (np.sqrt((y ** 2).sum()) * np.sqrt((yhat ** 2).sum()))
        if denom <= 0:
            return 0.0
        return float((y * yhat).sum() / denom)

    def _rankdata_avg(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        n = x.size
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    def _spearman(self, y: np.ndarray, yhat: np.ndarray) -> float:
        ry = self._rankdata_avg(y.astype(np.float64))
        rh = self._rankdata_avg(yhat.astype(np.float64))
        return self._pearson(ry, rh)

    def _brier(self, y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean((p - y) ** 2))

    def _ece(self, y: np.ndarray, p: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(y)
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(y[mask].mean())
            conf = float(p[mask].mean())
            ece += (m / n) * abs(acc - conf)
        return float(ece)

    def _risk_coverage(self, y01: np.ndarray, yhat: np.ndarray, p_for_conf: np.ndarray) -> Dict[str, object]:
        conf = np.maximum(p_for_conf, 1.0 - p_for_conf)
        order = np.argsort(-conf)
        y_o = y01[order]
        yhat_o = yhat[order]

        coverages = np.linspace(0.1, 1.0, 10)
        fixed = [0.6, 0.8, 0.9]

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(y_o))))
            return float(np.mean((y_o[:k] - yhat_o[:k]) ** 2))

        curve = [{"coverage": float(c), "risk_mse": risk_at(float(c))} for c in coverages]
        fixed_out = {f"risk_mse@{int(c*100)}": risk_at(c) for c in fixed}
        return {"curve": curve, **fixed_out}

    def _eval_preds_metrics(self, y01: np.ndarray, yhat: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        return {
            "MSE": self._mse(y01, yhat),
            "MAE": self._mae(y01, yhat),
            "Pearson": self._pearson(y01, yhat),
            "Spearman": self._spearman(y01, yhat),
            "ECE": self._ece(y01, p, self.ece_bins),
            "Brier": self._brier(y01, p),
        }

    # ===== Tier2 retrieval metrics =====
    def _loads_json_list(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _recall_at_k(self, retrieved: List[int], gold: List[int], k: int) -> float:
        if not gold:
            return 0.0
        r = retrieved[:k]
        g = set(gold)
        return 1.0 if any((x in g) for x in r) else 0.0

    def _mrr_at_k(self, retrieved: List[int], gold: List[int], k: int) -> float:
        if not gold:
            return 0.0
        g = set(gold)
        r = retrieved[:k]
        for i, x in enumerate(r):
            if x in g:
                return 1.0 / float(i + 1)
        return 0.0

    def _compute_retrieval_metrics(self, df_eval: pd.DataFrame, retrieved_topk_ids: List[List[int]]) -> Dict[str, float]:
        gold_lists: List[List[int]] = []
        for s in df_eval["evidence_chunk_ids_json"].astype(str).tolist():
            ids = self._loads_json_list(s)
            ids = [int(x) for x in ids if str(x).strip() != ""]
            gold_lists.append(ids)

        out: Dict[str, float] = {}
        for k in self.recall_ks:
            vals = [self._recall_at_k(r, g, k) for r, g in zip(retrieved_topk_ids, gold_lists)]
            out[f"Recall@{k}"] = float(np.mean(vals)) if vals else 0.0

        k_mrr = self.mrr_k
        vals_mrr = [self._mrr_at_k(r, g, k_mrr) for r, g in zip(retrieved_topk_ids, gold_lists)]
        out[f"MRR@{k_mrr}"] = float(np.mean(vals_mrr)) if vals_mrr else 0.0
        return out

    # ===== Tier2 faithfulness stress tests =====
    def _stress_random_ids(self, n: int) -> List[List[int]]:
        assert self.all_chunk_ids_universe is not None
        U = self.all_chunk_ids_universe
        out: List[List[int]] = []
        for _ in range(n):
            k = min(self.top_k, len(U))
            idx = np.random.choice(len(U), size=k, replace=False)
            out.append([int(x) for x in U[idx].tolist()])
        return out

    def _stress_swap_between_samples(self, ids_per_sample: List[List[int]]) -> List[List[int]]:
        if len(ids_per_sample) <= 1:
            return ids_per_sample
        out = [ids_per_sample[(i + 1) % len(ids_per_sample)] for i in range(len(ids_per_sample))]
        return [list(x[: self.top_k]) for x in out]

    def _stress_truncate(self, ids_per_sample: List[List[int]], k: int) -> List[List[int]]:
        kk = max(1, int(k))
        return [list(x[:kk]) for x in ids_per_sample]

    def _stress_shuffle_order(self, ids_per_sample: List[List[int]]) -> List[List[int]]:
        out: List[List[int]] = []
        for x in ids_per_sample:
            y = list(x[: self.top_k])
            random.shuffle(y)
            out.append(y)
        return out

    def _predict_with_ids(self, df_sub: pd.DataFrame, ids_per_sample: List[List[int]], desc: str) -> np.ndarray:
        inputs: List[str] = []
        for i in range(len(df_sub)):
            q = df_sub.iloc[i]["question"]
            c = df_sub.iloc[i]["case_context"]

            if "answer_target" in df_sub.columns:
                a = df_sub.iloc[i]["answer_target"]
            elif "answer" in df_sub.columns:
                a = df_sub.iloc[i]["answer"]
            else:
                a = ""

            if "claim_text" in df_sub.columns:
                cl = df_sub.iloc[i]["claim_text"]
            elif "claim" in df_sub.columns:
                cl = df_sub.iloc[i]["claim"]
            else:
                cl = ""

            ev = self._materialize_evidence(ids_per_sample[i])
            inputs.append(
                self._build_verifier_inputs(
                    case_context=c,
                    question=q,
                    answer_target=a,
                    claim_text=cl,
                    evidence_text=ev,
                )
            )
        return self._predict_support(inputs, desc=desc)

    # ===== Tier3 slices =====
    def _compute_slice_metrics(self, df_test: pd.DataFrame, y01: np.ndarray, yhat: np.ndarray, p: np.ndarray) -> Dict[str, object]:
        def pack(mask: np.ndarray) -> Dict[str, object]:
            if mask.sum() == 0:
                return {"n": 0, "MSE": None, "ECE": None, "risk_mse@80": None}
            yy = y01[mask]
            yh = yhat[mask]
            pp = p[mask]
            rc = self._risk_coverage(yy, yh, pp)
            return {
                "n": int(mask.sum()),
                "MSE": self._mse(yy, yh),
                "ECE": self._ece(yy, pp, self.ece_bins),
                "risk_mse@80": float(rc["risk_mse@80"]),
            }

        out: Dict[str, object] = {}
        if "report_length_bucket" in df_test.columns:
            for v in ["short", "medium", "long"]:
                out[f"test_slice.report_length_bucket.{v}"] = pack(df_test["report_length_bucket"].astype(str).values == v)

        if "device_present" in df_test.columns:
            vals = pd.to_numeric(df_test["device_present"], errors="coerce").fillna(0).astype(int).values
            out["test_slice.device_present.1"] = pack(vals == 1)
            out["test_slice.device_present.0"] = pack(vals == 0)

        if "uncertainty_present" in df_test.columns:
            vals = pd.to_numeric(df_test["uncertainty_present"], errors="coerce").fillna(0).astype(int).values
            out["test_slice.uncertainty_present.1"] = pack(vals == 1)
            out["test_slice.uncertainty_present.0"] = pack(vals == 0)

        if "label_vs_extractive" in df_test.columns:
            vals = df_test["label_vs_extractive"].astype(str).values
            if "label_vs_extractive" in df_test.columns:
                vals = df_test["label_vs_extractive"].astype(str).values
                out["test_slice.label_vs_extractive.closed_form"] = pack(vals == "closed_form")
                out["test_slice.label_vs_extractive.other"] = pack(vals == "other")
        return out

    # ===== Single CSV summary =====
    def _write_csv_summary(self) -> str:
        tier1_path = os.path.join(self.out_dir, "s2_tier1_report.json")
        tier2_faith_path = os.path.join(self.out_dir, "s2_tier2_faithfulness_report.json")
        tier2_ret_path = os.path.join(self.out_dir, "s2_tier2_retrieval_metrics.json")
        tier3_path = os.path.join(self.out_dir, "s2_tier3_distribution_slices.json")

        out_csv = os.path.join(self.out_dir, "S2_full_results.csv")

        def load_json(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        rows = []

        t1 = load_json(tier1_path) if os.path.exists(tier1_path) else {}
        t2f = load_json(tier2_faith_path) if os.path.exists(tier2_faith_path) else {}
        t2r = load_json(tier2_ret_path) if os.path.exists(tier2_ret_path) else {}
        t3 = load_json(tier3_path) if os.path.exists(tier3_path) else {}

        t1m = t1.get("metrics", {}) or {}
        for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]:
            if k in t1m:
                rows.append(["Tier1", "overall", k, t1m[k]])
        r80 = (t1m.get("SelectiveRisk", {}) or {}).get("risk_mse@80", None)
        if r80 is not None:
            rows.append(["Tier1", "overall", "risk_mse@80", r80])

        for k, v in (t2r.get("metrics", {}) or {}).items():
            rows.append(["Tier2_Retrieval", "overall", k, v])

        base = t2f.get("baseline_metrics", {}) or {}
        for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]:
            if k in base:
                rows.append(["Tier2_Faithfulness", "baseline", k, base[k]])

        for name, vals in (t2f.get("perturbations", {}) or {}).items():
            d = (vals.get("delta_vs_base", {}) or {})
            for dk in ["ΔMSE", "ΔECE", "ΔBrier", "ΔMAE", "ΔPearson", "ΔSpearman"]:
                if dk in d:
                    rows.append(["Tier2_Faithfulness", name, f"{dk}_vs_base", d[dk]])

        for slice_name, vals in (t3.get("metrics", {}) or {}).items():
            n = vals.get("n")
            if n is not None:
                rows.append(["Tier3", slice_name, "n", n])
            for k in ["MSE", "ECE", "Brier"]:
                if k in vals and vals[k] is not None:
                    rows.append(["Tier3", slice_name, k, vals[k]])
            r80 = vals.get("risk_mse@80")
            if r80 is not None:
                rows.append(["Tier3", slice_name, "risk_mse@80", r80])

        df = pd.DataFrame(rows, columns=["Tier", "Subgroup", "Metric", "Value"])
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        mask_n = df["Metric"].astype(str).str.lower().eq("n")

        df.loc[~mask_n, "Value"] = df.loc[~mask_n, "Value"] * 100.0
        df["Value"] = df["Value"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        return out_csv

    # ===== Main run =====
    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)

        df_test = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        df_test = df_test.iloc[: int(len(df_test) * 0.002)].reset_index(drop=True)

        if "split" in df_test.columns:
            df_test = df_test[df_test["split"].astype(str) == "test"].copy()

        # Optional speed knob
        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df_test = df_test.sample(
                n=min(len(df_test), self.faithfulness_max_eval),
                random_state=self.seed
            ).reset_index(drop=True)

        # Load models and evidence
        self.chunk_map = self._load_chunk_map()
        self._load_offtheshelf_mpnet()
        self._build_faiss_on_the_fly()
        self._load_verifier()

        # End2end test: retrieve -> verify
        claim_texts = df_test["claim_text"].astype(str).tolist()
        topk_test = self._retrieve_topk_chunk_ids(claim_texts)

        inputs: List[str] = []
        for i in range(len(df_test)):
            q = df_test.iloc[i]["question"]
            c = df_test.iloc[i]["case_context"]

            if "answer_target" in df_test.columns:
                a = df_test.iloc[i]["answer_target"]
            elif "answer" in df_test.columns:
                a = df_test.iloc[i]["answer"]
            else:
                a = ""

            if "claim_text" in df_test.columns:
                cl = df_test.iloc[i]["claim_text"]
            elif "claim" in df_test.columns:
                cl = df_test.iloc[i]["claim"]
            else:
                cl = ""

            ev = self._materialize_evidence(topk_test[i])  # now only top-5 goes to verifier
            inputs.append(
                self._build_verifier_inputs(
                    case_context=c,
                    question=q,
                    answer_target=a,
                    claim_text=cl,
                    evidence_text=ev,
                )
            )

        yhat = self._predict_support(inputs, desc="S2 Verification (V*) [test]")
        y = pd.to_numeric(df_test["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01 = np.clip(y, 0.0, 1.0)
        p = np.clip(yhat, 0.0, 1.0)

        # Write ONE main predictions parquet
        out_df = df_test.copy()
        out_df["s2_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_test]
        out_df["s2_pred_support_raw"] = yhat.astype(np.float32)
        out_df["s2_pred_prob_sigmoid"] = p.astype(np.float32)

        test_pred_path = os.path.join(self.out_dir, "s2_test_predictions.parquet")
        out_df.to_parquet(test_pred_path, index=False)

        # Tier1 report
        tier1_metrics = self._eval_preds_metrics(y01, yhat, p)
        rc = self._risk_coverage(y01, yhat, p)

        tier1_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df_test)),
            "retriever": {
                "mpnet_model": self.params["retriever"]["model_name"],
                "top_k": self.top_k,
                "faiss": "IndexFlatIP (in-memory, not saved)",
            },
            "verifier": {
                "weights": self.verifier_weights_path,
                "top_p": self.verifier_top_p,  # NEW
                "max_tokens": self.verifier_max_tokens
            },
            "metrics": {**tier1_metrics, "SelectiveRisk": rc},
            "artifacts": {"predictions_parquet": test_pred_path},
        }

        tier1_path = os.path.join(self.out_dir, "s2_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        # Tier2A retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics(df_test, topk_test)
        retrieval_path = os.path.join(self.out_dir, "s2_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df_test)),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )

        # Tier2B faithfulness
        base_metrics = tier1_metrics

        ids_top1 = self._stress_truncate(topk_test, 1)
        ids_top3 = self._stress_truncate(topk_test, 3)
        ids_shuffle = self._stress_shuffle_order(topk_test)
        ids_swapped = self._stress_swap_between_samples(topk_test)

        rand_metrics_list: List[Dict[str, float]] = []
        rand_preds: List[np.ndarray] = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df_test))
            yhat_rand = self._predict_with_ids(df_test, ids_rand, desc=f"S2 Faithfulness random (trial {t+1}) [test]")
            rand_preds.append(yhat_rand)
            p_rand = np.clip(yhat_rand, 0.0, 1.0)
            rand_metrics_list.append(self._eval_preds_metrics(y01, yhat_rand, p_rand))

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {k: 0.0 for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]}
            keys = list(ms[0].keys())
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        yhat_top1 = self._predict_with_ids(df_test, ids_top1, desc="S2 Faithfulness top-1 [test]")
        yhat_top3 = self._predict_with_ids(df_test, ids_top3, desc="S2 Faithfulness top-3 [test]")
        yhat_shuffle = self._predict_with_ids(df_test, ids_shuffle, desc="S2 Faithfulness shuffle-order [test]")
        yhat_swapped = self._predict_with_ids(df_test, ids_swapped, desc="S2 Faithfulness swapped-evidence [test]")

        def met(yhat_x: np.ndarray) -> Dict[str, float]:
            return self._eval_preds_metrics(y01, yhat_x, np.clip(yhat_x, 0.0, 1.0))

        met_top1 = met(yhat_top1)
        met_top3 = met(yhat_top3)
        met_shuffle = met(yhat_shuffle)
        met_swapped = met(yhat_swapped)
        met_random = avg_metrics(rand_metrics_list)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df_test)),
            "baseline_metrics": base_metrics,
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "metrics": met_random,
                    "delta_vs_base": delta(met_random, base_metrics),
                },
                "swapped_evidence": {
                    "metrics": met_swapped,
                    "delta_vs_base": delta(met_swapped, base_metrics),
                },
                "truncate_top1": {
                    "metrics": met_top1,
                    "delta_vs_base": delta(met_top1, base_metrics),
                },
                "truncate_top3": {
                    "metrics": met_top3,
                    "delta_vs_base": delta(met_top3, base_metrics),
                },
                "shuffle_order": {
                    "metrics": met_shuffle,
                    "delta_vs_base": delta(met_shuffle, base_metrics),
                },
            },
        }

        faith_pred_path = None
        if self.write_faithfulness_predictions_parquet:
            faith_df = df_test.copy()
            faith_df["s2_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_test]
            faith_df["s2_pred_base_raw"] = yhat.astype(np.float32)
            faith_df["s2_pred_top1_raw"] = yhat_top1.astype(np.float32)
            faith_df["s2_pred_top3_raw"] = yhat_top3.astype(np.float32)
            faith_df["s2_pred_shuffle_raw"] = yhat_shuffle.astype(np.float32)
            faith_df["s2_pred_swapped_raw"] = yhat_swapped.astype(np.float32)
            if rand_preds:
                yhat_rand_avg = np.mean(np.stack(rand_preds, axis=0), axis=0).astype(np.float32)
                faith_df["s2_pred_random_avg_raw"] = yhat_rand_avg

            faith_pred_path = os.path.join(self.out_dir, "s2_tier2_faithfulness_predictions.parquet")
            faith_df.to_parquet(faith_pred_path, index=False)

        faith_path = os.path.join(self.out_dir, "s2_tier2_faithfulness_report.json")
        with open(faith_path, "w", encoding="utf-8") as f:
            payload = dict(faith_report)
            if faith_pred_path is not None:
                payload["artifacts"] = {"faithfulness_predictions_parquet": faith_pred_path}
            json.dump(payload, f, indent=2)

        # Tier3 slices
        slice_metrics = self._compute_slice_metrics(df_test, y01, yhat, p)
        slice_path = os.path.join(self.out_dir, "s2_tier3_distribution_slices.json")
        with open(slice_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df_test)),
                    "metrics": slice_metrics,
                },
                f,
                indent=2,
            )

        # Summary json
        summary = {
            "test": {
                "tier1_report": tier1_path,
                "tier2_retrieval_metrics": retrieval_path,
                "tier2_faithfulness_report": faith_path,
                "tier2_faithfulness_predictions": faith_pred_path,
                "tier3_distribution_slices": slice_path,
                "predictions_parquet": test_pred_path,
            }
        }
        summary_path = os.path.join(self.out_dir, "s2_tier123_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {test_pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_path}")
        if faith_pred_path is not None:
            print(f"Wrote: {faith_pred_path}")
        print(f"Wrote: {slice_path}")
        print(f"Wrote: {summary_path}")

        # Single CSV
        self._write_csv_summary()

        return summary_path






class S3Tier123End2EndEvaluatorTestOnly:
    """
    S3 — BM25 retrieval -> V*
    Test-only, end-to-end:
      - Reads phase1_fixed_test_eval/mimic_test_eval.parquet
      - Builds BM25 index in memory on the fly
      - Runs tiers 1-3 on test
      - Writes one main predictions parquet + json reports + one csv summary
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,
        device: str = "cuda",
        top_k: int = 20,
        verifier_top_p: int = 5,          # NEW
        verifier_max_tokens: int = 1024,  # NEW
        evidence_sep: str = "\n\n---\n\n",
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        out_dir: Optional[str] = None,
        seed: int = 13,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        # Tier2 knobs
        mrr_k: int = 10,
        recall_ks: Tuple[int, ...] = (1, 5, 10, 20),
        faithfulness_random_trials: int = 1,
        input_test_data: str = "phase1_fixed_test_eval/mimic_test_eval.parquet",
        faithfulness_max_eval: Optional[int] = None,
        # Output knobs
        write_faithfulness_predictions_parquet: bool = False,
        # Evidence corpus
        chunks_parquet_rel: str = "chunks.parquet",
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)
        self.verifier_top_p = int(verifier_top_p)   # NEW
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.evidence_sep = evidence_sep
        self.batch_size_verify = int(batch_size_verify)
        self.ece_bins = int(ece_bins)
        self.seed = int(seed)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        self.mrr_k = int(mrr_k)
        self.recall_ks = tuple(int(x) for x in recall_ks)
        self.faithfulness_random_trials = int(faithfulness_random_trials)
        self.faithfulness_max_eval = int(faithfulness_max_eval) if faithfulness_max_eval is not None else None

        self.write_faithfulness_predictions_parquet = bool(write_faithfulness_predictions_parquet)

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, input_test_data
        )
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, chunks_parquet_rel)

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier123_results_s3_end2end_testonly")
        os.makedirs(self.out_dir, exist_ok=True)

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        # Verifier
        self.verifier_tok = None
        self.verifier_model = None

        # Evidence
        self.chunk_map = None  # Dict[int,str]

        # BM25 runtime state
        self._bm25 = None
        self.all_chunk_ids_universe = None  # np.ndarray int64

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

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    # ===== Loading chunks into a map for evidence materialization =====
    def _load_chunk_map(self) -> Dict[int, str]:
        if not os.path.exists(self.chunks_parquet):
            raise FileNotFoundError(self.chunks_parquet)

        df = pd.read_parquet(self.chunks_parquet, columns=["chunk_id", "chunk_text"])
        df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
        df = df[~df["chunk_id"].isna()].copy()
        df["chunk_id"] = df["chunk_id"].astype(np.int64)
        df["chunk_text"] = df["chunk_text"].astype(str)

        m: Dict[int, str] = {}
        for cid, txt in zip(df["chunk_id"].tolist(), df["chunk_text"].tolist()):
            t = " ".join(str(txt).split()).strip()
            m[int(cid)] = t
        if not m:
            raise RuntimeError("chunk_map is empty after loading chunks.parquet.")
        return m

    # Verifier
    def _load_verifier(self) -> None:
        vname = self.params["Verifier"]["model_name"]
        self.verifier_tok = AutoTokenizer.from_pretrained(vname, use_fast=True)
        self.verifier_model = AutoModelForSequenceClassification.from_pretrained(
            vname,
            num_labels=1,
            problem_type="regression",
        ).to(self.device)
        self.verifier_model.eval()

        obj = torch.load(self.verifier_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj
        self.verifier_model.load_state_dict(state, strict=True)
        self.verifier_model.eval()


    # BM25 (in memory, on the fly)
    def _tokenize_bm25(self, text: str) -> List[str]:
        t = " ".join(str(text).lower().split()).strip()
        out = []
        cur = []
        for ch in t:
            if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
                cur.append(ch)
            else:
                if cur:
                    out.append("".join(cur))
                    cur = []
        if cur:
            out.append("".join(cur))
        return out

    def _build_bm25_on_the_fly(self) -> None:
        """
        Stream chunks.parquet, build postings/idf/doclen for BM25 in memory.
        """
        if not os.path.exists(self.chunks_parquet):
            raise FileNotFoundError(self.chunks_parquet)

        pf = pq.ParquetFile(self.chunks_parquet)
        n_rows = pf.metadata.num_rows if pf.metadata is not None else 0

        chunk_ids: List[int] = []
        texts: List[str] = []

        for batch in tqdm(
            pf.iter_batches(batch_size=50_000, columns=["chunk_id", "chunk_text"]),
            total=max(1, math.ceil(n_rows / 50_000)) if n_rows else None,
            desc="S3: loading evidence for BM25 (stream)",
            unit="batch",
        ):
            df = batch.to_pandas()
            df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
            df = df[~df["chunk_id"].isna()].copy()
            df["chunk_id"] = df["chunk_id"].astype(np.int64)
            df["chunk_text"] = df["chunk_text"].astype(str).map(lambda x: " ".join(str(x).split()).strip())

            chunk_ids.extend(df["chunk_id"].tolist())
            texts.extend(df["chunk_text"].tolist())

        if not chunk_ids:
            raise RuntimeError("No chunks loaded from chunks.parquet.")

        # Sort by chunk_id for determinism
        order = np.argsort(np.asarray(chunk_ids, dtype=np.int64))
        chunk_ids_np = np.asarray(chunk_ids, dtype=np.int64)[order]
        texts_sorted = [texts[i] for i in order.tolist()]

        N = len(texts_sorted)
        doc_len = np.zeros((N,), dtype=np.int32)

        postings: Dict[str, List[Tuple[int, int]]] = {}
        dfreq: Dict[str, int] = {}

        for docid in tqdm(range(N), desc="S3 BM25: tokenizing chunks", unit="doc"):
            toks = self._tokenize_bm25(texts_sorted[docid])
            doc_len[docid] = len(toks)
            if not toks:
                continue
            tf: Dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            for w, c in tf.items():
                postings.setdefault(w, []).append((docid, c))
                dfreq[w] = dfreq.get(w, 0) + 1

        avgdl = float(doc_len.mean()) if N > 0 else 0.0

        idf: Dict[str, float] = {}
        for w, dfw in dfreq.items():
            idf[w] = math.log(1.0 + (N - dfw + 0.5) / (dfw + 0.5))

        self._bm25 = {
            "chunk_ids": chunk_ids_np,
            "postings": postings,
            "idf": idf,
            "doc_len": doc_len,
            "avgdl": avgdl,
            "k1": float(self.bm25_k1),
            "b": float(self.bm25_b),
        }
        self.all_chunk_ids_universe = chunk_ids_np.copy()

    def _retrieve_topk_chunk_ids(self, claim_texts: List[str], desc: str) -> List[List[int]]:
        assert self._bm25 is not None
        chunk_ids = self._bm25["chunk_ids"]
        postings = self._bm25["postings"]
        idf = self._bm25["idf"]
        doc_len = self._bm25["doc_len"]
        avgdl = float(self._bm25["avgdl"])
        k1 = float(self._bm25["k1"])
        b = float(self._bm25["b"])

        out: List[List[int]] = []

        for q in tqdm(claim_texts, desc=desc, unit="query"):
            toks = self._tokenize_bm25(q)
            if not toks:
                out.append([])
                continue

            scores: Dict[int, float] = {}
            for t in toks:
                plist = postings.get(t, None)
                if not plist:
                    continue
                idft = float(idf.get(t, 0.0))
                for docid, tf in plist:
                    dl = float(doc_len[docid])
                    denom = tf + k1 * (1.0 - b + b * (dl / (avgdl + 1e-12)))
                    s = idft * (tf * (k1 + 1.0)) / (denom + 1e-12)
                    scores[docid] = scores.get(docid, 0.0) + float(s)

            if not scores:
                out.append([])
                continue

            best = heapq.nlargest(self.top_k, scores.items(), key=lambda x: x[1])
            docids = [int(d) for d, _ in best]
            out.append([int(chunk_ids[d]) for d in docids])

        return out

    def _materialize_evidence(self, chunk_ids: List[int]) -> str:
        assert self.chunk_map is not None
        texts = []
        for cid in chunk_ids[: self.verifier_top_p]:  # NEW
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)

    # Verifier inference
    def _build_verifier_inputs(
        self,
        case_context: str,
        question: str,
        answer_target: str,
        claim_text: str,
        evidence_text: str,
    ) -> str:
        sep = "\n\n"
        return sep.join(
            [
                "CASE:",
                self._safe_str(case_context).strip(),
                "QUESTION:",
                self._safe_str(question).strip(),
                # "ANSWER_TARGET:",
                # self._safe_str(answer_target).strip(),
                "CLAIM:",
                self._safe_str(claim_text).strip(),
                "EVIDENCE:",
                self._safe_str(evidence_text).strip(),
            ]
        )

    def _predict_support(self, texts: List[str], desc: str) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size_verify),
                total=math.ceil(len(texts) / self.batch_size_verify),
                desc=desc,
                unit="batch",
            ):
                batch = texts[i:i + self.batch_size_verify]
                enc = self.verifier_tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.verifier_max_tokens,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.verifier_model(**enc)
                y = out.logits.squeeze(-1)
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _mse(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean((y - yhat) ** 2))

    def _mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean(np.abs(y - yhat)))

    def _pearson(self, y: np.ndarray, yhat: np.ndarray) -> float:
        y = y.astype(np.float64)
        yhat = yhat.astype(np.float64)
        y = y - y.mean()
        yhat = yhat - yhat.mean()
        denom = (np.sqrt((y ** 2).sum()) * np.sqrt((yhat ** 2).sum()))
        if denom <= 0:
            return 0.0
        return float((y * yhat).sum() / denom)

    def _rankdata_avg(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        n = x.size
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    def _spearman(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return self._pearson(self._rankdata_avg(y.astype(np.float64)), self._rankdata_avg(yhat.astype(np.float64)))

    def _brier(self, y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean((p - y) ** 2))

    def _ece(self, y: np.ndarray, p: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(y)
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(y[mask].mean())
            conf = float(p[mask].mean())
            ece += (m / n) * abs(acc - conf)
        return float(ece)

    def _risk_coverage(self, y01: np.ndarray, yhat: np.ndarray, p_for_conf: np.ndarray) -> Dict[str, object]:
        conf = np.maximum(p_for_conf, 1.0 - p_for_conf)
        order = np.argsort(-conf)
        y_o = y01[order]
        yhat_o = yhat[order]

        coverages = np.linspace(0.1, 1.0, 10)
        fixed = [0.6, 0.8, 0.9]

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(y_o))))
            return float(np.mean((y_o[:k] - yhat_o[:k]) ** 2))

        curve = [{"coverage": float(c), "risk_mse": risk_at(float(c))} for c in coverages]
        fixed_out = {f"risk_mse@{int(c*100)}": risk_at(c) for c in fixed}
        return {"curve": curve, **fixed_out}

    def _eval_preds_metrics(self, y01: np.ndarray, yhat: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        return {
            "MSE": self._mse(y01, yhat),
            "MAE": self._mae(y01, yhat),
            "Pearson": self._pearson(y01, yhat),
            "Spearman": self._spearman(y01, yhat),
            "ECE": self._ece(y01, p, self.ece_bins),
            "Brier": self._brier(y01, p),
        }

    # Tier2 retrieval metrics
    def _loads_json_list(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _recall_at_k(self, retrieved: List[int], gold: List[int], k: int) -> float:
        if not gold:
            return 0.0
        r = retrieved[:k]
        g = set(gold)
        return 1.0 if any((x in g) for x in r) else 0.0

    def _mrr_at_k(self, retrieved: List[int], gold: List[int], k: int) -> float:
        if not gold:
            return 0.0
        g = set(gold)
        r = retrieved[:k]
        for i, x in enumerate(r):
            if x in g:
                return 1.0 / float(i + 1)
        return 0.0

    def _compute_retrieval_metrics(self, df_eval: pd.DataFrame, retrieved_topk_ids: List[List[int]]) -> Dict[str, float]:
        gold_lists: List[List[int]] = []
        for s in df_eval["evidence_chunk_ids_json"].astype(str).tolist():
            ids = self._loads_json_list(s)
            ids = [int(x) for x in ids if str(x).strip() != ""]
            gold_lists.append(ids)

        out: Dict[str, float] = {}
        for k in self.recall_ks:
            vals = [self._recall_at_k(r, g, k) for r, g in zip(retrieved_topk_ids, gold_lists)]
            out[f"Recall@{k}"] = float(np.mean(vals)) if vals else 0.0

        k_mrr = self.mrr_k
        vals_mrr = [self._mrr_at_k(r, g, k_mrr) for r, g in zip(retrieved_topk_ids, gold_lists)]
        out[f"MRR@{k_mrr}"] = float(np.mean(vals_mrr)) if vals_mrr else 0.0
        return out

    # Tier2 faithfulness helpers
    def _stress_random_ids(self, n: int) -> List[List[int]]:
        assert self.all_chunk_ids_universe is not None
        U = self.all_chunk_ids_universe
        out: List[List[int]] = []
        for _ in range(n):
            size = min(self.top_k, len(U))
            idx = np.random.choice(len(U), size=size, replace=False)
            out.append([int(x) for x in U[idx].tolist()])
        return out

    def _stress_swap_between_samples(self, ids_per_sample: List[List[int]]) -> List[List[int]]:
        if len(ids_per_sample) <= 1:
            return ids_per_sample
        out = [ids_per_sample[(i + 1) % len(ids_per_sample)] for i in range(len(ids_per_sample))]
        return [list(x[: self.top_k]) for x in out]

    def _stress_truncate(self, ids_per_sample: List[List[int]], k: int) -> List[List[int]]:
        kk = max(1, int(k))
        return [list(x[:kk]) for x in ids_per_sample]

    def _stress_shuffle_order(self, ids_per_sample: List[List[int]]) -> List[List[int]]:
        out: List[List[int]] = []
        for x in ids_per_sample:
            y = list(x[: self.top_k])
            random.shuffle(y)
            out.append(y)
        return out

    def _predict_with_ids(self, df_sub: pd.DataFrame, ids_per_sample: List[List[int]], desc: str) -> np.ndarray:
        inputs: List[str] = []
        for i in range(len(df_sub)):
            q = df_sub.iloc[i]["question"]
            c = df_sub.iloc[i]["case_context"]

            if "answer_target" in df_sub.columns:
                a = df_sub.iloc[i]["answer_target"]
            elif "answer" in df_sub.columns:
                a = df_sub.iloc[i]["answer"]
            else:
                a = ""

            if "claim_text" in df_sub.columns:
                cl = df_sub.iloc[i]["claim_text"]
            elif "claim" in df_sub.columns:
                cl = df_sub.iloc[i]["claim"]
            else:
                cl = ""

            ev = self._materialize_evidence(ids_per_sample[i])
            inputs.append(
                self._build_verifier_inputs(
                    case_context=c,
                    question=q,
                    answer_target=a,
                    claim_text=cl,
                    evidence_text=ev,
                )
            )
        return self._predict_support(inputs, desc=desc)

    # Tier3 slices
    def _compute_slice_metrics(self, df_test: pd.DataFrame, y01: np.ndarray, yhat: np.ndarray, p: np.ndarray) -> Dict[str, object]:
        def pack(mask: np.ndarray) -> Dict[str, object]:
            if mask.sum() == 0:
                return {"n": 0, "MSE": None, "ECE": None, "risk_mse@80": None}
            yy = y01[mask]
            yh = yhat[mask]
            pp = p[mask]
            rc = self._risk_coverage(yy, yh, pp)
            return {
                "n": int(mask.sum()),
                "MSE": self._mse(yy, yh),
                "ECE": self._ece(yy, pp, self.ece_bins),
                "risk_mse@80": float(rc["risk_mse@80"]),
            }

        out: Dict[str, object] = {}
        if "report_length_bucket" in df_test.columns:
            for v in ["short", "medium", "long"]:
                out[f"test_slice.report_length_bucket.{v}"] = pack(df_test["report_length_bucket"].astype(str).values == v)

        if "device_present" in df_test.columns:
            vals = pd.to_numeric(df_test["device_present"], errors="coerce").fillna(0).astype(int).values
            out["test_slice.device_present.1"] = pack(vals == 1)
            out["test_slice.device_present.0"] = pack(vals == 0)

        if "uncertainty_present" in df_test.columns:
            vals = pd.to_numeric(df_test["uncertainty_present"], errors="coerce").fillna(0).astype(int).values
            out["test_slice.uncertainty_present.1"] = pack(vals == 1)
            out["test_slice.uncertainty_present.0"] = pack(vals == 0)

        if "label_vs_extractive" in df_test.columns:
            vals = df_test["label_vs_extractive"].astype(str).values
            if "label_vs_extractive" in df_test.columns:
                vals = df_test["label_vs_extractive"].astype(str).values
                out["test_slice.label_vs_extractive.closed_form"] = pack(vals == "closed_form")
                out["test_slice.label_vs_extractive.other"] = pack(vals == "other")
        return out

    def _write_csv_summary(self) -> str:
        tier1_path = os.path.join(self.out_dir, "s3_tier1_report.json")
        tier2_faith_path = os.path.join(self.out_dir, "s3_tier2_faithfulness_report.json")
        tier2_ret_path = os.path.join(self.out_dir, "s3_tier2_retrieval_metrics.json")
        tier3_path = os.path.join(self.out_dir, "s3_tier3_distribution_slices.json")

        out_csv = os.path.join(self.out_dir, "S3_full_results.csv")

        def load_json(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        rows = []
        t1 = load_json(tier1_path) if os.path.exists(tier1_path) else {}
        t2f = load_json(tier2_faith_path) if os.path.exists(tier2_faith_path) else {}
        t2r = load_json(tier2_ret_path) if os.path.exists(tier2_ret_path) else {}
        t3 = load_json(tier3_path) if os.path.exists(tier3_path) else {}

        t1m = t1.get("metrics", {}) or {}
        for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]:
            if k in t1m:
                rows.append(["Tier1", "overall", k, t1m[k]])
        r80 = (t1m.get("SelectiveRisk", {}) or {}).get("risk_mse@80", None)
        if r80 is not None:
            rows.append(["Tier1", "overall", "risk_mse@80", r80])

        for k, v in (t2r.get("metrics", {}) or {}).items():
            rows.append(["Tier2_Retrieval", "overall", k, v])

        base = t2f.get("baseline_metrics", {}) or {}
        for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]:
            if k in base:
                rows.append(["Tier2_Faithfulness", "baseline", k, base[k]])

        for name, vals in (t2f.get("perturbations", {}) or {}).items():
            d = (vals.get("delta_vs_base", {}) or {})
            for dk in ["ΔMSE", "ΔECE", "ΔBrier", "ΔMAE", "ΔPearson", "ΔSpearman"]:
                if dk in d:
                    rows.append(["Tier2_Faithfulness", name, f"{dk}_vs_base", d[dk]])

        for slice_name, vals in (t3.get("metrics", {}) or {}).items():
            n = vals.get("n")
            if n is not None:
                rows.append(["Tier3", slice_name, "n", n])
            for k in ["MSE", "ECE", "Brier"]:
                if k in vals and vals[k] is not None:
                    rows.append(["Tier3", slice_name, k, vals[k]])
            r80 = vals.get("risk_mse@80")
            if r80 is not None:
                rows.append(["Tier3", slice_name, "risk_mse@80", r80])

        df = pd.DataFrame(rows, columns=["Tier", "Subgroup", "Metric", "Value"])
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        mask_n = df["Metric"].astype(str).str.lower().eq("n")

        df.loc[~mask_n, "Value"] = df.loc[~mask_n, "Value"] * 100.0
        df["Value"] = df["Value"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        return out_csv

    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)

        df_test = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        df_test = df_test.iloc[: int(len(df_test) * 0.5)].reset_index(drop=True)

        if "split" in df_test.columns:
            df_test = df_test[df_test["split"].astype(str) == "test"].copy()

        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df_test = df_test.sample(
                n=min(len(df_test), self.faithfulness_max_eval),
                random_state=self.seed
            ).reset_index(drop=True)

        self.chunk_map = self._load_chunk_map()
        self._build_bm25_on_the_fly()
        self._load_verifier()

        # End2end: retrieve -> verify
        claim_test = df_test["claim_text"].astype(str).tolist()
        topk_test = self._retrieve_topk_chunk_ids(claim_test, desc="S3 Retrieval (BM25) [test]")

        inputs: List[str] = []
        for i in range(len(df_test)):
            q = df_test.iloc[i]["question"]
            c = df_test.iloc[i]["case_context"]

            if "answer_target" in df_test.columns:
                a = df_test.iloc[i]["answer_target"]
            elif "answer" in df_test.columns:
                a = df_test.iloc[i]["answer"]
            else:
                a = ""

            if "claim_text" in df_test.columns:
                cl = df_test.iloc[i]["claim_text"]
            elif "claim" in df_test.columns:
                cl = df_test.iloc[i]["claim"]
            else:
                cl = ""

            ev = self._materialize_evidence(topk_test[i])
            inputs.append(
                self._build_verifier_inputs(
                    case_context=c,
                    question=q,
                    answer_target=a,
                    claim_text=cl,
                    evidence_text=ev,
                )
            )

        yhat = self._predict_support(inputs, desc="S3 Verification (V*) [test]")
        y = pd.to_numeric(df_test["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01 = np.clip(y, 0.0, 1.0)
        p = np.clip(yhat, 0.0, 1.0)

        out_df = df_test.copy()
        out_df["s3_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_test]
        out_df["s3_pred_support_raw"] = yhat.astype(np.float32)
        out_df["s3_pred_prob_sigmoid"] = p.astype(np.float32)

        test_pred_path = os.path.join(self.out_dir, "s3_test_predictions.parquet")
        out_df.to_parquet(test_pred_path, index=False)

        # Tier1 report
        tier1_metrics = self._eval_preds_metrics(y01, yhat, p)
        rc = self._risk_coverage(y01, yhat, p)

        tier1_report = {
            "system": "S3 (BM25 retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df_test)),
            "retriever": {
                "bm25": "in-memory (not saved)",
                "top_k": self.top_k,
                "k1": self.bm25_k1,
                "b": self.bm25_b,
            },
            "verifier": {
                "weights": self.verifier_weights_path,
                "top_p": self.verifier_top_p,  # NEW
                "max_tokens": self.verifier_max_tokens
            },
            "metrics": {**tier1_metrics, "SelectiveRisk": rc},
            "artifacts": {"predictions_parquet": test_pred_path},
        }

        tier1_path = os.path.join(self.out_dir, "s3_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        # Tier2A retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics(df_test, topk_test)
        retrieval_path = os.path.join(self.out_dir, "s3_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S3 (BM25 retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df_test)),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )

        # Tier2B faithfulness
        base_metrics = tier1_metrics

        ids_top1 = self._stress_truncate(topk_test, 1)
        ids_top3 = self._stress_truncate(topk_test, 3)
        ids_shuffle = self._stress_shuffle_order(topk_test)
        ids_swapped = self._stress_swap_between_samples(topk_test)

        rand_metrics_list: List[Dict[str, float]] = []
        rand_preds: List[np.ndarray] = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df_test))
            yhat_rand = self._predict_with_ids(df_test, ids_rand, desc=f"S3 Faithfulness random (trial {t+1}) [test]")
            rand_preds.append(yhat_rand)
            rand_metrics_list.append(self._eval_preds_metrics(y01, yhat_rand, np.clip(yhat_rand, 0.0, 1.0)))

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {k: 0.0 for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]}
            keys = list(ms[0].keys())
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        yhat_top1 = self._predict_with_ids(df_test, ids_top1, desc="S3 Faithfulness top-1 [test]")
        yhat_top3 = self._predict_with_ids(df_test, ids_top3, desc="S3 Faithfulness top-3 [test]")
        yhat_shuffle = self._predict_with_ids(df_test, ids_shuffle, desc="S3 Faithfulness shuffle-order [test]")
        yhat_swapped = self._predict_with_ids(df_test, ids_swapped, desc="S3 Faithfulness swapped-evidence [test]")

        def met(yhat_x: np.ndarray) -> Dict[str, float]:
            return self._eval_preds_metrics(y01, yhat_x, np.clip(yhat_x, 0.0, 1.0))

        met_top1 = met(yhat_top1)
        met_top3 = met(yhat_top3)
        met_shuffle = met(yhat_shuffle)
        met_swapped = met(yhat_swapped)
        met_random = avg_metrics(rand_metrics_list)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S3 (BM25 retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df_test)),
            "baseline_metrics": base_metrics,
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "metrics": met_random,
                    "delta_vs_base": delta(met_random, base_metrics),
                },
                "swapped_evidence": {
                    "metrics": met_swapped,
                    "delta_vs_base": delta(met_swapped, base_metrics),
                },
                "truncate_top1": {
                    "metrics": met_top1,
                    "delta_vs_base": delta(met_top1, base_metrics),
                },
                "truncate_top3": {
                    "metrics": met_top3,
                    "delta_vs_base": delta(met_top3, base_metrics),
                },
                "shuffle_order": {
                    "metrics": met_shuffle,
                    "delta_vs_base": delta(met_shuffle, base_metrics),
                },
            },
        }

        faith_pred_path = None
        if self.write_faithfulness_predictions_parquet:
            faith_df = df_test.copy()
            faith_df["s3_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_test]
            faith_df["s3_pred_base_raw"] = yhat.astype(np.float32)
            faith_df["s3_pred_top1_raw"] = yhat_top1.astype(np.float32)
            faith_df["s3_pred_top3_raw"] = yhat_top3.astype(np.float32)
            faith_df["s3_pred_shuffle_raw"] = yhat_shuffle.astype(np.float32)
            faith_df["s3_pred_swapped_raw"] = yhat_swapped.astype(np.float32)
            if rand_preds:
                yhat_rand_avg = np.mean(np.stack(rand_preds, axis=0), axis=0).astype(np.float32)
                faith_df["s3_pred_random_avg_raw"] = yhat_rand_avg

            faith_pred_path = os.path.join(self.out_dir, "s3_tier2_faithfulness_predictions.parquet")
            faith_df.to_parquet(faith_pred_path, index=False)

        faith_path = os.path.join(self.out_dir, "s3_tier2_faithfulness_report.json")
        with open(faith_path, "w", encoding="utf-8") as f:
            payload = dict(faith_report)
            if faith_pred_path is not None:
                payload["artifacts"] = {"faithfulness_predictions_parquet": faith_pred_path}
            json.dump(payload, f, indent=2)

        # Tier3 slices
        slice_metrics = self._compute_slice_metrics(df_test, y01, yhat, p)
        slice_path = os.path.join(self.out_dir, "s3_tier3_distribution_slices.json")
        with open(slice_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S3 (BM25 retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df_test)),
                    "metrics": slice_metrics,
                },
                f,
                indent=2,
            )

        summary = {
            "test": {
                "tier1_report": tier1_path,
                "tier2_retrieval_metrics": retrieval_path,
                "tier2_faithfulness_report": faith_path,
                "tier2_faithfulness_predictions": faith_pred_path,
                "tier3_distribution_slices": slice_path,
                "predictions_parquet": test_pred_path,
            }
        }
        summary_path = os.path.join(self.out_dir, "s3_tier123_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {test_pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_path}")
        if faith_pred_path is not None:
            print(f"Wrote: {faith_pred_path}")
        print(f"Wrote: {slice_path}")
        print(f"Wrote: {summary_path}")

        self._write_csv_summary()

        return summary_path





class S1Tier123CaseOnlyEvaluatorTestOnly:
    """
    S1 - Case-only verifier V* with the same verifier architecture and
    the same input formatting style used during verifier training, except
    that the evidence field is intentionally left empty.

    Test-only, end-to-end:
      - Reads phase1_fixed_test_eval/mimic_test_eval.parquet
      - Runs verifier on case/question/answer/claim with empty evidence
      - Tier1: regression + calibration-style metrics computed from sigmoid
      - Tier3: distribution shift slices on test
      - Writes one predictions parquet + json reports + one csv summary
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,
        device: str = "cuda",
        verifier_top_p: int = 5,
        verifier_max_tokens: int = 1024,
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        out_dir: Optional[str] = None,
        seed: int = 13,
        input_test_data: str = "phase1_fixed_test_eval/mimic_test_eval.parquet"
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.verifier_top_p = int(verifier_top_p)
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.batch_size_verify = int(batch_size_verify)
        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, input_test_data
        )

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier123_results_s1_testonly")
        os.makedirs(self.out_dir, exist_ok=True)

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        self.verifier_tok = None
        self.verifier_model = None

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

    def _load_verifier(self) -> None:
        vname = self.params["Verifier"]["model_name"]
        self.verifier_tok = AutoTokenizer.from_pretrained(vname, use_fast=True)

        self.verifier_model = AutoModelForSequenceClassification.from_pretrained(
            vname,
            num_labels=1,
            problem_type="regression",
        ).to(self.device)
        self.verifier_model.eval()

        obj = torch.load(self.verifier_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj

        self.verifier_model.load_state_dict(state, strict=True)
        self.verifier_model.eval()


    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    # Text building
    def _build_verifier_inputs(
        self,
        case_context: str,
        question: str,
        answer_target: str,
        claim_text: str,
        evidence_text: str = "",
    ) -> str:
        sep = "\n\n"
        return sep.join(
            [
                "CASE:",
                self._safe_str(case_context).strip(),
                "QUESTION:",
                self._safe_str(question).strip(),
                "CLAIM:",
                self._safe_str(claim_text).strip(),
            ]
        )

    # Inference
    def _predict_support(self, texts: List[str], desc: str) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size_verify),
                total=math.ceil(len(texts) / self.batch_size_verify),
                desc=desc,
                unit="batch",
            ):
                batch = texts[i:i + self.batch_size_verify]
                enc = self.verifier_tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.verifier_max_tokens,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}


                out = self.verifier_model(**enc)
                y = out.logits.squeeze(-1)

                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

    # Metrics
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _mse(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean((y - yhat) ** 2))

    def _mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean(np.abs(y - yhat)))

    def _pearson(self, y: np.ndarray, yhat: np.ndarray) -> float:
        y = y.astype(np.float64)
        yhat = yhat.astype(np.float64)
        y = y - y.mean()
        yhat = yhat - yhat.mean()
        denom = (np.sqrt((y ** 2).sum()) * np.sqrt((yhat ** 2).sum()))
        if denom <= 0:
            return 0.0
        return float((y * yhat).sum() / denom)

    def _rankdata_avg(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        n = x.size
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    def _spearman(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return self._pearson(self._rankdata_avg(y.astype(np.float64)), self._rankdata_avg(yhat.astype(np.float64)))

    def _brier(self, y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean((p - y) ** 2))

    def _ece(self, y: np.ndarray, p: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(y)
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(y[mask].mean())
            conf = float(p[mask].mean())
            ece += (m / n) * abs(acc - conf)
        return float(ece)

    def _reliability_bins(self, y01: np.ndarray, p: np.ndarray, bins: int) -> Dict[str, object]:
        edges = np.linspace(0.0, 1.0, bins + 1)
        out = {"bin_lo": [], "bin_hi": [], "count": [], "acc": [], "conf": []}
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            out["bin_lo"].append(lo)
            out["bin_hi"].append(hi)
            out["count"].append(m)
            if m == 0:
                out["acc"].append(None)
                out["conf"].append(None)
            else:
                out["acc"].append(float(y01[mask].mean()))
                out["conf"].append(float(p[mask].mean()))
        return out

    def _risk_coverage(self, y01: np.ndarray, yhat: np.ndarray, p_for_conf: np.ndarray) -> Dict[str, object]:
        conf = np.maximum(p_for_conf, 1.0 - p_for_conf)
        order = np.argsort(-conf)
        y_o = y01[order]
        yhat_o = yhat[order]

        coverages = np.linspace(0.1, 1.0, 10)
        fixed = [0.6, 0.8, 0.9]

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(y_o))))
            return float(np.mean((y_o[:k] - yhat_o[:k]) ** 2))

        curve = [{"coverage": float(c), "risk_mse": risk_at(float(c))} for c in coverages]
        fixed_out = {f"risk_mse@{int(c*100)}": risk_at(c) for c in fixed}
        return {"curve": curve, **fixed_out}

    def _eval_preds_metrics(self, y01: np.ndarray, yhat: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        return {
            "MSE": self._mse(y01, yhat),
            "MAE": self._mae(y01, yhat),
            "Pearson": self._pearson(y01, yhat),
            "Spearman": self._spearman(y01, yhat),
            "ECE": self._ece(y01, p, self.ece_bins),
            "Brier": self._brier(y01, p),
        }

    # Tier3 slices
    def _compute_slice_metrics(self, df_test: pd.DataFrame, y01: np.ndarray, yhat: np.ndarray, p: np.ndarray) -> Dict[str, object]:
        def pack(mask: np.ndarray) -> Dict[str, object]:
            if mask.sum() == 0:
                return {"n": 0, "MSE": None, "ECE": None, "risk_mse@80": None}
            yy = y01[mask]
            yh = yhat[mask]
            pp = p[mask]
            rc = self._risk_coverage(yy, yh, pp)
            return {
                "n": int(mask.sum()),
                "MSE": self._mse(yy, yh),
                "ECE": self._ece(yy, pp, self.ece_bins),
                "risk_mse@80": float(rc["risk_mse@80"]),
            }

        out: Dict[str, object] = {}

        if "report_length_bucket" in df_test.columns:
            for v in ["short", "medium", "long"]:
                out[f"test_slice.report_length_bucket.{v}"] = pack(df_test["report_length_bucket"].astype(str).values == v)

        if "device_present" in df_test.columns:
            vals = pd.to_numeric(df_test["device_present"], errors="coerce").fillna(0).astype(int).values
            out["test_slice.device_present.1"] = pack(vals == 1)
            out["test_slice.device_present.0"] = pack(vals == 0)

        if "uncertainty_present" in df_test.columns:
            vals = pd.to_numeric(df_test["uncertainty_present"], errors="coerce").fillna(0).astype(int).values
            out["test_slice.uncertainty_present.1"] = pack(vals == 1)
            out["test_slice.uncertainty_present.0"] = pack(vals == 0)

        if "label_vs_extractive" in df_test.columns:
            vals = df_test["label_vs_extractive"].astype(str).values
            if "label_vs_extractive" in df_test.columns:
                vals = df_test["label_vs_extractive"].astype(str).values
                out["test_slice.label_vs_extractive.closed_form"] = pack(vals == "closed_form")
                out["test_slice.label_vs_extractive.other"] = pack(vals == "other")
        return out

    def _write_csv_summary(self) -> str:
        tier1_path = os.path.join(self.out_dir, "s1_tier1_report.json")
        tier3_path = os.path.join(self.out_dir, "s1_tier3_distribution_slices.json")
        out_csv = os.path.join(self.out_dir, "S1_full_results.csv")

        def load_json(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        def fmt3(x):
            if isinstance(x, (int, float)) and pd.notna(x):
                return f"{x:.3f}"
            return x

        rows = []
        t1 = load_json(tier1_path) if os.path.exists(tier1_path) else {}
        t3 = load_json(tier3_path) if os.path.exists(tier3_path) else {}

        t1m = t1.get("metrics", {}) or {}
        for k in ["MSE", "MAE", "Pearson", "Spearman", "ECE", "Brier"]:
            if k in t1m:
                rows.append(["Tier1", "overall", k, t1m[k]])
        r80 = (t1m.get("SelectiveRisk", {}) or {}).get("risk_mse@80", None)
        if r80 is not None:
            rows.append(["Tier1", "overall", "risk_mse@80", r80])

        for slice_name, vals in (t3.get("metrics", {}) or {}).items():
            n = vals.get("n")
            if n is not None:
                rows.append(["Tier3", slice_name, "n", n])
            for k in ["MSE", "ECE", "Brier"]:
                if k in vals and vals[k] is not None:
                    rows.append(["Tier3", slice_name, k, vals[k]])
            r80 = vals.get("risk_mse@80")
            if r80 is not None:
                rows.append(["Tier3", slice_name, "risk_mse@80", r80])

        df = pd.DataFrame(rows, columns=["Tier", "Subgroup", "Metric", "Value"])
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        mask_n = df["Metric"].astype(str).str.lower().eq("n")

        df.loc[~mask_n, "Value"] = df.loc[~mask_n, "Value"] * 100.0
        df["Value"] = df["Value"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        return out_csv


    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)

        df_test = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        df_test = df_test.iloc[: int(len(df_test) * 0.002)].reset_index(drop=True)

        if "split" in df_test.columns:
            df_test = df_test[df_test["split"].astype(str) == "test"].copy()

        self._load_verifier()

        inputs: List[str] = []
        for i in range(len(df_test)):
            q = df_test.iloc[i]["question"]
            c = df_test.iloc[i]["case_context"]

            if "answer_target" in df_test.columns:
                a = df_test.iloc[i]["answer_target"]
            elif "answer" in df_test.columns:
                a = df_test.iloc[i]["answer"]
            else:
                a = ""

            if "claim_text" in df_test.columns:
                cl = df_test.iloc[i]["claim_text"]
            elif "claim" in df_test.columns:
                cl = df_test.iloc[i]["claim"]
            else:
                cl = ""

            inputs.append(
                self._build_verifier_inputs(
                    case_context=c,
                    question=q,
                    answer_target=a,
                    claim_text=cl,
                    evidence_text="",
                )
            )

        yhat = self._predict_support(inputs, desc="S1 Verification (case-only, empty evidence) [test]")
        y = pd.to_numeric(df_test["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        y01 = np.clip(y, 0.0, 1.0)
        p = np.clip(yhat, 0.0, 1.0)

        # One predictions parquet
        out_df = df_test.copy()
        out_df["s1_pred_support_raw"] = yhat.astype(np.float32)
        out_df["s1_pred_prob_sigmoid"] = p.astype(np.float32)

        pred_path = os.path.join(self.out_dir, "s1_test_predictions.parquet")
        out_df.to_parquet(pred_path, index=False)

        # Tier1 report
        tier1_metrics = self._eval_preds_metrics(y01, yhat, p)
        rc = self._risk_coverage(y01, yhat, p)
        bins = self._reliability_bins(y01, p, self.ece_bins)

        tier1_report = {
            "system": "S1 (case-only verifier)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df_test)),
            "verifier": {
                "weights": self.verifier_weights_path,
                "top_p": self.verifier_top_p,   # NEW
                "max_tokens": self.verifier_max_tokens
            },
            "metrics": {**tier1_metrics, "SelectiveRisk": rc},
            "reliability_bins": bins,
            "artifacts": {"predictions_parquet": pred_path},
        }
        tier1_path = os.path.join(self.out_dir, "s1_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        # Tier3 slices
        slice_metrics = self._compute_slice_metrics(df_test, y01, yhat, p)
        slice_path = os.path.join(self.out_dir, "s1_tier3_distribution_slices.json")
        with open(slice_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S1 (case-only verifier)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df_test)),
                    "metrics": slice_metrics,
                },
                f,
                indent=2,
            )

        summary = {
            "test": {
                "tier1_report": tier1_path,
                "tier3_distribution_slices": slice_path,
                "predictions_parquet": pred_path,
            }
        }
        summary_path = os.path.join(self.out_dir, "s1_tier123_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {slice_path}")
        print(f"Wrote: {summary_path}")

        self._write_csv_summary()

        return summary_path




class CrossSystemComparatorPredOnlyTestOnly:
    """
    Cross-system comparison on the TEST split only, using the intersection of qa_row_id
    across all provided systems.

    Inputs
    ------
    system_test_predictions:
        {
            "S1": ".../s1_test_predictions.parquet",
            "S2": ".../s2_test_predictions.parquet",
            ...
        }

    system_full_results_csvs:
        Optional mapping used to build a final mega-csv:
        {
            "S1": ".../S1_full_results.csv",
            ...
        }

    system_faithfulness_predictions:
        Optional mapping:
        {
            "S2": ".../s2_tier2_faithfulness_predictions.parquet",
            ...
        }

    Outputs
    -------
    out_path JSON with:
      - per_system_metrics
      - pairwise_all_pairs
      - faithfulness_severity
      - alignment metadata

    and also:
      - ALL_systems_plus_cross_summary.csv
        written in the same directory as out_path
    """

    def __init__(
        self,
        system_test_predictions: Dict[str, str],
        out_path: str,
        *,
        system_full_results_csvs: Optional[Dict[str, str]] = None,
        system_faithfulness_predictions: Optional[Dict[str, str]] = None,
        ece_bins: int = 15,
        n_boot: int = 2000,
        seed: int = 13,
        verbose: bool = True,
        use_unordered_pairs_only: bool = True,
    ) -> None:
        self.system_test_predictions = dict(system_test_predictions)
        self.system_full_results_csvs = dict(system_full_results_csvs or {})
        self.system_faithfulness_predictions = dict(system_faithfulness_predictions or {})
        self.out_path = out_path

        self.ece_bins = int(ece_bins)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.use_unordered_pairs_only = bool(use_unordered_pairs_only)

        self.rng = np.random.RandomState(self.seed)


    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _rankdata(self, x: np.ndarray) -> np.ndarray:
        return pd.Series(np.asarray(x)).rank(method="average").to_numpy(dtype=np.float64)

    def _pearson(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        a_centered = a - a.mean()
        b_centered = b - b.mean()

        denom = np.sqrt((a_centered * a_centered).sum()) * np.sqrt((b_centered * b_centered).sum())
        if denom <= 0.0:
            return 0.0
        return float((a_centered * b_centered).sum() / denom)

    def _spearman(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._pearson(self._rankdata(a), self._rankdata(b))

    def _mse(self, y: np.ndarray, yhat: np.ndarray) -> float:
        d = np.asarray(y, dtype=np.float64) - np.asarray(yhat, dtype=np.float64)
        return float(np.mean(d * d))

    def _mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        d = np.asarray(y, dtype=np.float64) - np.asarray(yhat, dtype=np.float64)
        return float(np.mean(np.abs(d)))

    def _brier(self, y01: np.ndarray, p: np.ndarray) -> float:
        d = np.asarray(p, dtype=np.float64) - np.asarray(y01, dtype=np.float64)
        return float(np.mean(d * d))

    def _nll_soft_bernoulli(self, y01: np.ndarray, p: np.ndarray) -> float:
        y01 = np.asarray(y01, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        return float(np.mean(-(y01 * np.log(p) + (1.0 - y01) * np.log(1.0 - p))))

    def _ece(self, y01: np.ndarray, p: np.ndarray, bins: int) -> float:
        y01 = np.asarray(y01, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)

        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(y01)

        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            if b < bins - 1:
                mask = (p >= lo) & (p < hi)
            else:
                mask = (p >= lo) & (p <= hi)

            m = int(mask.sum())
            if m == 0:
                continue

            acc = float(y01[mask].mean())
            conf = float(p[mask].mean())
            ece += (m / n) * abs(acc - conf)

        return float(ece)

    def _aurc(self, y01: np.ndarray, yhat: np.ndarray) -> float:
        """
        Fast AURC implementation.
        This replaces the O(n^2) prefix-recompute version with O(n log n).

        Original logic:
          sort by confidence descending
          compute prefix MSE at every k
          average all prefix MSE values
        """
        y01 = np.asarray(y01, dtype=np.float64)
        yhat = np.asarray(yhat, dtype=np.float64)

        p = np.clip(yhat, 0.0, 1.0)
        conf = np.maximum(p, 1.0 - p)
        order = np.argsort(-conf, kind="stable")

        sq_err = (y01[order] - yhat[order]) ** 2
        prefix_sums = np.cumsum(sq_err, dtype=np.float64)
        ks = np.arange(1, len(sq_err) + 1, dtype=np.float64)
        prefix_mse = prefix_sums / ks
        return float(prefix_mse.mean())


    # Bootstrap
    def _precompute_boot_indices(self, n: int) -> np.ndarray:
        self._log(f"[bootstrap] precomputing {self.n_boot} resamples of length {n}")
        return self.rng.randint(0, n, size=(self.n_boot, n), dtype=np.int64)

    def _paired_bootstrap_delta_precomputed(
        self,
        y01: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        fn: Callable[[np.ndarray, np.ndarray], float],
        boot_indices: np.ndarray,
    ) -> Dict[str, float]:
        deltas = np.empty((boot_indices.shape[0],), dtype=np.float64)

        for t, idx in enumerate(boot_indices):
            deltas[t] = fn(y01[idx], a[idx]) - fn(y01[idx], b[idx])

        return {
            "mean_delta": float(deltas.mean()),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
        }

    def _infer_cols(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        cols = list(df.columns)

        pred_raw = None
        for c in cols:
            if c.endswith("_pred_support_raw") or ("pred_support_raw" in c):
                pred_raw = c
                break

        if pred_raw is None:
            for c in cols:
                if c.endswith("_pred_raw") or c.endswith("pred_raw"):
                    pred_raw = c
                    break

        pred_prob = None
        for c in cols:
            if ("pred_prob" in c) and ("sigmoid" in c):
                pred_prob = c
                break

        if pred_prob is None:
            for c in cols:
                if c.endswith("_pred_prob_sigmoid") or c.endswith("pred_prob_sigmoid"):
                    pred_prob = c
                    break

        return {"pred_raw": pred_raw, "pred_prob": pred_prob}


    def _load_and_align_systems(self):
        systems_raw: Dict[str, pd.DataFrame] = {}

        for sys_name, path in self.system_test_predictions.items():
            self._log(f"[load] {sys_name}: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            df = pd.read_parquet(path)

            if "qa_row_id" not in df.columns:
                raise RuntimeError(f"{sys_name} predictions parquet missing qa_row_id")
            if "soft_support_score" not in df.columns:
                raise RuntimeError(f"{sys_name} predictions parquet missing soft_support_score")

            df = df.copy()
            df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
            df = df[df["qa_row_id"] >= 0].copy()
            df = df.sort_values("qa_row_id").reset_index(drop=True)

            systems_raw[sys_name] = df

        # Intersection of ids across all systems
        sys_ids = {k: set(v["qa_row_id"].tolist()) for k, v in systems_raw.items()}
        eval_ids = None
        for ids in sys_ids.values():
            eval_ids = set(ids) if eval_ids is None else (eval_ids & ids)

        eval_ids = np.array(sorted(list(eval_ids or [])), dtype=np.int64)
        if len(eval_ids) == 0:
            raise RuntimeError("No overlapping qa_row_id across provided systems.")

        # Filter and align
        systems: Dict[str, pd.DataFrame] = {}
        dropped = {}

        for sys_name, df in systems_raw.items():
            df_f = df[df["qa_row_id"].isin(eval_ids)].copy()
            df_f = df_f.sort_values("qa_row_id").reset_index(drop=True)
            systems[sys_name] = df_f
            dropped[f"{sys_name}_dropped"] = int(len(df) - len(df_f))

        # Sanity check aligned id order
        ref_ids = next(iter(systems.values()))["qa_row_id"].to_numpy()
        for sys_name, df in systems.items():
            if not np.array_equal(df["qa_row_id"].to_numpy(), ref_ids):
                raise RuntimeError(f"Internal alignment failure for {sys_name}")

        # Gold y: prefer S4 if present, else first system
        prefer = "S4" if "S4" in systems else list(systems.keys())[0]
        y = pd.to_numeric(systems[prefer]["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01 = np.clip(y, 0.0, 1.0)

        # Verify identical target across all systems
        for sys_name, df in systems.items():
            y_other = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if not np.allclose(y_other, y, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    f"soft_support_score mismatch between {prefer} and {sys_name} on aligned eval ids."
                )

        meta = {
            "n_eval": int(len(eval_ids)),
            "eval_id_source": "intersection_of_system_test_parquets",
            "dropped_counts": dropped,
            "systems": list(systems.keys()),
        }
        return systems, y01, meta


    def _faithfulness_severity(
        self,
        y01: np.ndarray,
        faith_df: pd.DataFrame,
        sys_prefix: str,
    ) -> Dict[str, Any]:
        base_col = f"{sys_prefix}_pred_base_raw"
        if base_col not in faith_df.columns:
            return {"available": False}

        yhat_base = pd.to_numeric(faith_df[base_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        base_err = np.abs(y01 - yhat_base)

        def summarize(pert_col: str) -> Dict[str, float]:
            if pert_col not in faith_df.columns:
                return {}

            yhat_p = pd.to_numeric(faith_df[pert_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            err_p = np.abs(y01 - yhat_p)

            mse_base = self._mse(y01, yhat_base)
            mse_p = self._mse(y01, yhat_p)
            rel = float((mse_p - mse_base) / (mse_base + 1e-12))
            frac_worse = float(np.mean(err_p > base_err))

            return {
                "mse_base": float(mse_base),
                "mse_pert": float(mse_p),
                "relative_mse_degradation": rel,
                "frac_examples_worse_abs_error": frac_worse,
            }

        out = {"available": True, "perturbations": {}}
        out["perturbations"]["truncate_top1"] = summarize(f"{sys_prefix}_pred_top1_raw")
        out["perturbations"]["truncate_top3"] = summarize(f"{sys_prefix}_pred_top3_raw")
        out["perturbations"]["shuffle_order"] = summarize(f"{sys_prefix}_pred_shuffle_raw")
        out["perturbations"]["swapped_evidence"] = summarize(f"{sys_prefix}_pred_swapped_raw")
        out["perturbations"]["random_evidence"] = summarize(f"{sys_prefix}_pred_random_avg_raw")
        return out


    def _ensure_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = ["Tier", "Subgroup", "Metric", "Value"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in a system CSV: {missing}. Found: {list(df.columns)}")
        return df

    def _write_mega_csv(self, cross_json_path: str) -> str:
        out_dir = os.path.dirname(cross_json_path)
        out_csv = os.path.join(out_dir, "ALL_systems_plus_cross_summary.csv")

        dfs: List[pd.DataFrame] = []
        for sys_name in sorted(self.system_test_predictions.keys()):
            if sys_name not in self.system_full_results_csvs:
                continue

            path = self.system_full_results_csvs[sys_name]
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            df = pd.read_csv(path)
            df = self._ensure_cols(df)
            df.insert(0, "Source", sys_name)
            dfs.append(df)

        if dfs:
            systems_df = pd.concat(dfs, ignore_index=True)
        else:
            systems_df = pd.DataFrame(columns=["Source", "Tier", "Subgroup", "Metric", "Value"])

        with open(cross_json_path, "r", encoding="utf-8") as f:
            cross = json.load(f)

        cross_rows = []

        # per-system metrics
        per_sys = cross.get("per_system_metrics", {})
        for sys_name, metrics in per_sys.items():
            if not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    cross_rows.append(["cross", "Cross_PerSystem", sys_name, k, v])

        # pairwise
        pairwise = cross.get("pairwise_all_pairs", {})
        for pair_name, block in pairwise.items():
            win = (block or {}).get("win_rates", {}) or {}
            for k, v in win.items():
                if isinstance(v, (int, float)):
                    cross_rows.append(["cross", "Cross_Pairwise_WinRates", pair_name, k, v])

            deltas = (block or {}).get("bootstrap_deltas_A_minus_B", {}) or {}
            for metric_name, stats in deltas.items():
                if not isinstance(stats, dict):
                    continue
                for stat_k in ["mean_delta", "ci_low", "ci_high"]:
                    if stat_k in stats and isinstance(stats[stat_k], (int, float)):
                        cross_rows.append(
                            ["cross", "Cross_Pairwise_Deltas", pair_name, f"{metric_name}.{stat_k}", stats[stat_k]]
                        )

        # faithfulness
        faith_sev = cross.get("faithfulness_severity", {}) or {}
        for sys_name, sys_block in faith_sev.items():
            if not isinstance(sys_block, dict) or not sys_block.get("available", False):
                continue

            perts = sys_block.get("perturbations", {}) or {}
            for pert_name, pert_metrics in perts.items():
                if not isinstance(pert_metrics, dict):
                    continue

                subgroup = f"{sys_name}/{pert_name}"
                for k, v in pert_metrics.items():
                    if isinstance(v, (int, float)):
                        cross_rows.append(["cross", "Cross_FaithfulnessSeverity", subgroup, k, v])

        cross_df = pd.DataFrame(cross_rows, columns=["Source", "Tier", "Subgroup", "Metric", "Value"])

        final_df = pd.concat([systems_df, cross_df], ignore_index=True)
        final_df["Value"] = pd.to_numeric(final_df["Value"], errors="coerce")
        final_df = final_df.sort_values(by=["Source", "Tier", "Subgroup", "Metric"], kind="stable").reset_index(drop=True)
        final_df.to_csv(out_csv, index=False, float_format="%.6f")

        self._log(f"[write] mega csv: {out_csv}")
        return out_csv


    def run(self) -> str:
        systems, y01, meta = self._load_and_align_systems()
        n = len(y01)
        self._log(f"[align] n_eval={n}")

        boot_indices = self._precompute_boot_indices(n)

        per_system = {}
        series = {}

        self._log("[metrics] per-system metrics")
        for sys_name, df in systems.items():
            cols = self._infer_cols(df)

            if cols["pred_raw"] is None:
                raise RuntimeError(f"Could not infer pred raw prediction column for {sys_name}")

            yhat = pd.to_numeric(df[cols["pred_raw"]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

            if cols["pred_prob"] is not None:
                p = pd.to_numeric(df[cols["pred_prob"]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                p = np.clip(p, 0.0, 1.0)
            else:
                p = np.clip(yhat, 0.0, 1.0).astype(np.float32)

            per_system[sys_name] = {
                "MSE": self._mse(y01, yhat),
                "MAE": self._mae(y01, yhat),
                "Pearson": self._pearson(y01, yhat),
                "Spearman": self._spearman(y01, yhat),
                "ECE": self._ece(y01, p, self.ece_bins),
                "Brier": self._brier(y01, p),
                "NLL": self._nll_soft_bernoulli(y01, p),
                "AURC": self._aurc(y01, yhat),
            }

            series[sys_name] = {"yhat": yhat, "p": p}
            self._log(f"  done: {sys_name}")

        def ece_fn(yy: np.ndarray, pp: np.ndarray) -> float:
            return self._ece(yy, pp, self.ece_bins)

        def brier_fn(yy: np.ndarray, pp: np.ndarray) -> float:
            return self._brier(yy, pp)

        def nll_fn(yy: np.ndarray, pp: np.ndarray) -> float:
            return self._nll_soft_bernoulli(yy, pp)

        pairwise = {}
        names = list(systems.keys())

        self._log("[pairwise] bootstrap comparisons")
        if self.use_unordered_pairs_only:
            pair_iter = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pair_iter.append((names[i], names[j]))
        else:
            pair_iter = []
            for a in names:
                for b in names:
                    if a != b:
                        pair_iter.append((a, b))

        for a, b in pair_iter:
            self._log(f"  pair: {a} vs {b}")

            yhat_a = series[a]["yhat"]
            yhat_b = series[b]["yhat"]
            p_a = series[a]["p"]
            p_b = series[b]["p"]

            abs_err_a = np.abs(y01 - yhat_a)
            abs_err_b = np.abs(y01 - yhat_b)
            win_abs = float(np.mean(abs_err_a < abs_err_b))

            cal_err_a = np.abs(p_a - y01)
            cal_err_b = np.abs(p_b - y01)
            win_cal = float(np.mean(cal_err_a < cal_err_b))

            pairwise[f"{a}_vs_{b}"] = {
                "win_rates": {
                    "frac_examples_abs_error_A_better": win_abs,
                    "frac_examples_calibration_error_A_better": win_cal,
                },
                "bootstrap_deltas_A_minus_B": {
                    "ΔMSE": self._paired_bootstrap_delta_precomputed(y01, yhat_a, yhat_b, self._mse, boot_indices),
                    "ΔMAE": self._paired_bootstrap_delta_precomputed(y01, yhat_a, yhat_b, self._mae, boot_indices),
                    "ΔPearson": self._paired_bootstrap_delta_precomputed(
                        y01, yhat_a, yhat_b, lambda yy, aa: self._pearson(yy, aa), boot_indices
                    ),
                    "ΔSpearman": self._paired_bootstrap_delta_precomputed(
                        y01, yhat_a, yhat_b, lambda yy, aa: self._spearman(yy, aa), boot_indices
                    ),
                    "ΔAURC": self._paired_bootstrap_delta_precomputed(y01, yhat_a, yhat_b, self._aurc, boot_indices),
                    "ΔECE": self._paired_bootstrap_delta_precomputed(y01, p_a, p_b, ece_fn, boot_indices),
                    "ΔBrier": self._paired_bootstrap_delta_precomputed(y01, p_a, p_b, brier_fn, boot_indices),
                    "ΔNLL": self._paired_bootstrap_delta_precomputed(y01, p_a, p_b, nll_fn, boot_indices),
                },
            }

        faithfulness = {}
        if self.system_faithfulness_predictions:
            self._log("[faithfulness] summaries")
            ref_ids = systems[next(iter(systems))]["qa_row_id"].to_numpy(dtype=np.int64)
            ref_ids_df = pd.DataFrame({"qa_row_id": ref_ids})

            for sys_name, path in self.system_faithfulness_predictions.items():
                self._log(f"  system: {sys_name}")

                if not os.path.exists(path):
                    faithfulness[sys_name] = {"available": False, "reason": "file_not_found"}
                    continue

                fdf = pd.read_parquet(path)
                if "qa_row_id" not in fdf.columns:
                    faithfulness[sys_name] = {"available": False, "reason": "missing_qa_row_id"}
                    continue

                fdf = fdf.copy()
                fdf["qa_row_id"] = pd.to_numeric(fdf["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
                fdf = fdf.merge(ref_ids_df, on="qa_row_id", how="inner").sort_values("qa_row_id").reset_index(drop=True)

                base_cols = [c for c in fdf.columns if c.endswith("_pred_base_raw")]
                if not base_cols:
                    faithfulness[sys_name] = {"available": False, "reason": "missing_base_prediction_column"}
                    continue

                prefix = base_cols[0].replace("_pred_base_raw", "")

                if len(fdf) != len(y01):
                    faithfulness[sys_name] = {
                        "available": False,
                        "reason": "faithfulness_file_did_not_align_to_eval_ids",
                    }
                    continue

                faithfulness[sys_name] = self._faithfulness_severity(y01, fdf, prefix)

        final = {
            "n_eval": int(meta["n_eval"]),
            "alignment_mode": meta["eval_id_source"],
            "dropped_counts": meta["dropped_counts"],
            "per_system_metrics": per_system,
            "pairwise_all_pairs": pairwise,
            "faithfulness_severity": faithfulness,
            "params": {
                "ece_bins": int(self.ece_bins),
                "n_boot": int(self.n_boot),
                "seed": int(self.seed),
                "use_unordered_pairs_only": bool(self.use_unordered_pairs_only),
            },
        }

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)

        self._log(f"[write] json: {self.out_path}")
        self._log(f"[done] n_eval={final['n_eval']} dropped={final['dropped_counts']}")

        self._write_mega_csv(self.out_path)
        return self.out_path








def analyze_retrieval_error_coupling(
    parquet_path: str,
    system_prefix: str = "s4",
    top_k: int = 20,
    make_plot: bool = True,
):
    df = pd.read_parquet(parquet_path).copy()

    retrieved_col = f"{system_prefix}_retrieved_chunk_ids_json"
    pred_col = f"{system_prefix}_pred_support_raw"
    proto_col = "evidence_chunk_ids_json"
    target_col = "soft_support_score"

    def loads_ids(x):
        if isinstance(x, list):
            return x
        return json.loads(x)

    df["proto_ids"] = df[proto_col].apply(loads_ids)
    df["retrieved_ids"] = df[retrieved_col].apply(loads_ids)

    def rr_at_k(retrieved, proto, k):
        proto = set(proto)
        for rank, cid in enumerate(retrieved[:k], start=1):
            if cid in proto:
                return 1.0 / rank
        return 0.0

    df["rr"] = [
        rr_at_k(r, p, top_k) for r, p in zip(df["retrieved_ids"], df["proto_ids"])
    ]

    df["abs_err"] = (df[pred_col] - df[target_col]).abs()
    df["sq_err"] = (df[pred_col] - df[target_col]) ** 2

    rho, pval = spearmanr(df["rr"], df["abs_err"])
    r_pearson, p_pearson = pearsonr(df["rr"], df["abs_err"])

    bins = [-0.01, 0.0, 0.1, 0.2, 0.34, 0.51, 1.0]
    labels = ["no hit", "rank10", "rank6-9", "rank3-5", "rank2", "rank1"]

    df["rr_bin"] = pd.cut(df["rr"], bins=bins, labels=labels)

    bin_stats = (
        df.groupby("rr_bin")
        .agg(
            n=("qa_row_id", "count"),
            mean_abs_err=("abs_err", "mean"),
            mean_sq_err=("sq_err", "mean"),
        )
        .reset_index()
    )

    output_dir = os.path.dirname(parquet_path)

    os.makedirs(os.path.join(output_dir, "Mechanistic_results"), exist_ok=True)

    bin_stats.to_csv(os.path.join(output_dir, "Mechanistic_results", "retrieval_error_coupling_bins.csv"), index=False)

    summary = {
        "spearman_rr_vs_abs_err": float(rho),
        "spearman_p": float(pval),
        "pearson_rr_vs_abs_err": float(r_pearson),
        "pearson_p": float(p_pearson),
        "mean_rr": float(df["rr"].mean()),
        "mean_abs_error": float(df["abs_err"].mean()),
        "n_samples": int(len(df)),
        "meaning_of_analysis": "This analysis tests whether retrieval quality actually affects downstream prediction quality in the evidence-grounded system. For each test example, we compute a retrieval alignment score (e.g., reciprocal rank RR@k measuring how highly a protocol-aligned evidence chunk appears in the retrieved list) and the verifier prediction error (e.g., | yhat_i - y_i |). We then measure the correlation between these two quantities across all examples and also examine prediction error across bins of retrieval quality (e.g., no hit, rank 6-10, rank 3-5, rank 2, rank 1). If the correlation is negative, it means that examples with better retrieved evidence tend to have lower verifier error, indicating that the model's predictions are genuinely conditioned on the retrieved evidence. In other words, improved retrieval alignment translates into improved prediction quality, providing mechanistic evidence that the system is truly evidence-grounded rather than relying mainly on the question or case context",
    }

    with open(os.path.join(output_dir, "Mechanistic_results", "retrieval_error_coupling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if make_plot:
        plt.figure(figsize=(7,4))

        order = labels
        stats = bin_stats.set_index("rr_bin").reindex(order)

        plt.bar(order, stats["mean_abs_err"])
        plt.ylabel("Mean Absolute Error")
        plt.xlabel(f"Retrieval quality bin (RR@{top_k})")
        plt.title("Retrieval quality vs verifier error")

        plt.xticks(rotation=30)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Mechanistic_results", "retrieval_error_coupling_plot.png"))
        plt.close()

    return summary, df








if __name__ == "__main__":

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]


    # results, df_out = analyze_retrieval_error_coupling(
    #     # parquet_path=os.path.join(base_path_evidenceData, "CheXpertPlus_Results/S4", "s4_test_predictions.parquet"),
    #     parquet_path=os.path.join(base_path_evidenceData, "MIMIC_Results/S4", "s4_test_predictions.parquet"),
    #     system_prefix="s4",
    #     top_k=20,
    #     make_plot=True,
    # )



    ''' S1 '''
    s1 = S1Tier123CaseOnlyEvaluatorTestOnly(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        verifier_top_p=5,
        verifier_max_tokens=1024,
        out_dir=os.path.join(base_path_evidenceData, "MIMIC_Results/S1"),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        input_test_data="phase1_fixed_test_eval/mimic_test_eval.parquet",
    )
    s1.run()

    # ''' S2 '''
    s2 = S2Tier123End2EndEvaluatorTestOnly(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_top_p=5,
        verifier_max_tokens=1024,
        out_dir=os.path.join(base_path_evidenceData, "MIMIC_Results/S2"),
        batch_size_embed_chunks=128,
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        input_test_data="phase1_fixed_test_eval/mimic_test_eval.parquet",
        seed=13,
    )
    s2.run()
    #
    #
    # ''' S3 '''
    # s3 = S3Tier123End2EndEvaluatorTestOnly(
    #     cfg_path=global_config_path,
    #     experiment_name=experiment_name,
    #     base_path_evidenceData=base_path_evidenceData,
    #     verifier_weights_path=verifier_weights_path,
    #     device="cuda",
    #     top_k=20,
    #     verifier_top_p=5,
    #     verifier_max_tokens=1024,
    #     out_dir=os.path.join(base_path_evidenceData, "MIMIC_Results/S3"),
    #     batch_size_verify=16,
    #     ece_bins=15,
    #     seed=13,
    #     input_test_data="phase1_fixed_test_eval/mimic_test_eval.parquet",
    #     bm25_k1=1.2,
    #     bm25_b=0.75,
    # )
    # s3.run()
    #
    #
    # ''' S4 '''
    # runner = S4Tier123End2EndEvaluatorTestOnly(
    #     cfg_path=global_config_path,
    #     experiment_name=experiment_name,
    #     base_path_evidenceData=base_path_evidenceData,
    #     model_weights_path=model_weights_path,
    #     verifier_weights_path=verifier_weights_path,
    #     device="cuda",
    #     top_k=20,
    #     verifier_top_p=5,
    #     verifier_max_tokens=1024,
    #     out_dir=os.path.join(base_path_evidenceData, "MIMIC_Results/S4"),
    #     batch_size_retrieve=256,
    #     batch_size_verify=4,
    #     input_test_data="phase1_fixed_test_eval/mimic_test_eval.parquet",
    #     batch_size_embed_docs=256,
    #     ece_bins=15,
    # )
    # runner.run()
    #
    #
    #
    #
    # ''' Cross comparison '''
    # base = params["Preprocessing"]["base_path_evidenceData"]
    #
    # comparator = CrossSystemComparatorPredOnlyTestOnly(
    #     system_test_predictions={
    #         "S1": os.path.join(base, "MIMIC_Results/S1", "s1_test_predictions.parquet"),
    #         "S2": os.path.join(base, "MIMIC_Results/S2", "s2_test_predictions.parquet"),
    #         "S3": os.path.join(base, "MIMIC_Results/S3", "s3_test_predictions.parquet"),
    #         "S4": os.path.join(base, "MIMIC_Results/S4", "s4_test_predictions.parquet"),
    #     },
    #     system_full_results_csvs={
    #         "S1": os.path.join(base, "MIMIC_Results/S1", "S1_full_results.csv"),
    #         "S2": os.path.join(base, "MIMIC_Results/S2", "S2_full_results.csv"),
    #         "S3": os.path.join(base, "MIMIC_Results/S3", "S3_full_results.csv"),
    #         "S4": os.path.join(base, "MIMIC_Results/S4", "S4_full_results.csv"),
    #     },
    #     system_faithfulness_predictions={
    #         "S2": os.path.join(base, "MIMIC_Results/S2", "s2_tier2_faithfulness_predictions.parquet"),
    #         "S3": os.path.join(base, "MIMIC_Results/S3", "s3_tier2_faithfulness_predictions.parquet"),
    #         "S4": os.path.join(base, "MIMIC_Results/S4", "s4_tier2_faithfulness_predictions.parquet"),
    #     },
    #     out_path=os.path.join(
    #         base,
    #         "MIMIC_Results",
    #         "cross_system_results",
    #         "s1_s4_cross_system_comparison.json",
    #     ),
    #     ece_bins=15,
    #     n_boot=1000,
    #     seed=13,
    #     verbose=True,
    #     use_unordered_pairs_only=True,
    # )
    #
    # comparator.run()

