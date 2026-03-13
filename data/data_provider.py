"""
Created on Feb 23, 2026.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import json
import os.path
import pdb
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import math
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional


from config.serde import read_config







@dataclass(frozen=True)
class RetrieverBatch:
    queries: List[str]                       # B
    docs: List[str]                          # B*K
    pos_mask: torch.Tensor                   # (B, B*K) bool, True where doc is positive for query
    pos_weights: torch.Tensor                # (B, B*K) float, weights only on positives (0 elsewhere)


class RadiopaediaChunksStore:
    """
    Loads chunk_id -> chunk_text into memory.
    If your chunks table is huge, replace this with an Arrow-backed lookup,
    but for typical Radiopaedia chunk counts this is fine.
    """
    def __init__(self, chunks_parquet: str, chunk_id_col: str = "chunk_id", chunk_text_col: str = "chunk_text"):
        self.chunks_parquet = chunks_parquet
        self.chunk_id_col = chunk_id_col
        self.chunk_text_col = chunk_text_col

        df = pd.read_parquet(chunks_parquet, columns=[chunk_id_col, chunk_text_col])
        df[chunk_id_col] = pd.to_numeric(df[chunk_id_col], errors="coerce")
        df = df.dropna(subset=[chunk_id_col]).copy()
        df[chunk_id_col] = df[chunk_id_col].astype(int)
        df[chunk_text_col] = df[chunk_text_col].astype(str)

        self.id2text: Dict[int, str] = dict(zip(df[chunk_id_col].tolist(), df[chunk_text_col].tolist()))

    def get(self, chunk_id: int) -> str:
        return self.id2text.get(int(chunk_id), "")




class RetrieverSupervisionDataset(Dataset):
    """
    Reads retriever_supervision.parquet and returns:
      claim_text, pos_chunk_ids(list), pos_weights(list)

    Requires columns:
      - split
      - claim_text
      - pos_chunk_ids_json
      - pos_weights_json
    """
    def __init__(
        self,
        cfg_path,
        chunks_store: RadiopaediaChunksStore,
        split_name: str,
        top_k: int = 20,
        min_pos_weight: float = 0.0,
        claim_col: str = "claim_text",
        split_col: str = "split",
        pos_ids_col: str = "pos_chunk_ids_json",
        pos_w_col: str = "pos_weights_json",
    ):
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)

        self.chunks_store = chunks_store
        self.top_k = int(top_k)
        self.min_pos_weight = float(min_pos_weight)
        self.base_path = self.params["Preprocessing"]["base_path_evidenceData"]

        retriever_supervision_path = os.path.join(self.base_path, "step4c_supervision/retriever_supervision.parquet")
        df = pd.read_parquet(retriever_supervision_path)

        for c in [split_col, claim_col, pos_ids_col, pos_w_col]:
            if c not in df.columns:
                raise ValueError(f"Missing required column in retriever supervision: {c}")

        df[split_col] = df[split_col].astype(str)
        df = df[df[split_col] == str(split_name)].reset_index(drop=True)

        df[claim_col] = df[claim_col].astype(str)

        self.df = df
        self.claim_col = claim_col
        self.pos_ids_col = pos_ids_col
        self.pos_w_col = pos_w_col



    def __len__(self) -> int:
        return len(self.df)



    def _loads(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []



    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        claim = str(row[self.claim_col]).strip()

        ids = self._loads(str(row[self.pos_ids_col]))
        w = self._loads(str(row[self.pos_w_col]))

        ids = [int(x) for x in ids][: self.top_k]
        w = [float(x) for x in w][: self.top_k]

        # length align
        m = min(len(ids), len(w))
        ids, w = ids[:m], w[:m]

        # filter low-weight positives
        if self.min_pos_weight > 0:
            keep = [(i, ww) for i, ww in zip(ids, w) if ww >= self.min_pos_weight]
            ids = [i for i, _ in keep]
            w = [ww for _, ww in keep]

        # normalize weights per query (sum to 1)
        if len(w) > 0:
            s = float(np.sum(w))
            if s > 0:
                w = [float(x / s) for x in w]
            else:
                w = [1.0 / len(w) for _ in w]

        # materialize doc texts now (fast dict lookup)
        docs = [self.chunks_store.get(i) for i in ids]

        # drop empty docs
        ids2, docs2, w2 = [], [], []
        for i, d, ww in zip(ids, docs, w):
            if d and d.strip():
                ids2.append(i)
                docs2.append(d)
                w2.append(float(ww))

        # if everything collapsed, return empty positives (trainer will skip)
        return {
            "claim_text": claim,
            "pos_docs": docs2,
            "pos_weights": w2,
        }





def retriever_collate_fn(batch: List[Dict], top_k: int) -> Optional[RetrieverBatch]:
    """
    Makes fixed K positives per query by:
      - if fewer than K, pad with empty strings and 0 weight
      - if more than K, truncate
    Creates (B, B*K) mask and weights for weighted multi-positive InfoNCE.
    """
    B = len(batch)
    K = int(top_k)

    queries: List[str] = []
    docs: List[str] = []

    pos_mask = torch.zeros((B, B * K), dtype=torch.bool)
    pos_weights = torch.zeros((B, B * K), dtype=torch.float32)

    # if any item has no positives, we still keep it but it will produce zero numerator
    for i, item in enumerate(batch):
        q = str(item["claim_text"]).strip()
        queries.append(q)

        pos_docs = list(item.get("pos_docs", []))[:K]
        w = list(item.get("pos_weights", []))[:K]

        # pad
        if len(pos_docs) < K:
            pad_n = K - len(pos_docs)
            pos_docs = pos_docs + [""] * pad_n
            w = w + [0.0] * pad_n

        # renormalize over nonzero weights (for stability)
        nz = [x for x in w if x > 0]
        if len(nz) > 0:
            s = float(np.sum(nz))
            if s > 0:
                w = [float(x / s) if x > 0 else 0.0 for x in w]

        # place docs in flat doc list
        base = i * K
        for j in range(K):
            docs.append(pos_docs[j])

            if pos_docs[j].strip() and w[j] > 0:
                pos_mask[i, base + j] = True
                pos_weights[i, base + j] = float(w[j])

    return RetrieverBatch(queries=queries, docs=docs, pos_mask=pos_mask, pos_weights=pos_weights)




@dataclass(frozen=True)
class VerifierBatch:
    inputs: List[str]
    targets: torch.Tensor
    qa_row_ids: Optional[List[int]]


class EvidenceTextStore:
    """
    Deterministic chunk_id -> chunk_text lookup.

    Assumption:
      chunk_id is the row index in the evidence parquet (0-based).
    If chunk ids are 1-based, set id_base=1.
    """

    def __init__(
        self,
        evidence_parquet_path: str,
        text_col: str = "chunk_text",
        id_base: int = 0,
        preload: bool = False,
        cache_row_groups: int = 2,
    ) -> None:
        if not os.path.exists(evidence_parquet_path):
            raise FileNotFoundError(f"Evidence parquet not found: {evidence_parquet_path}")

        self.path = evidence_parquet_path
        self.text_col = str(text_col)
        self.id_base = int(id_base)
        self.preload = bool(preload)
        self.cache_row_groups = int(cache_row_groups)

        self.pf = pq.ParquetFile(self.path)
        if self.text_col not in set(self.pf.schema.names):
            raise ValueError(f"Evidence parquet missing required text column: {self.text_col}")

        self._rg_num_rows = [self.pf.metadata.row_group(i).num_rows for i in range(self.pf.num_row_groups)]
        self._rg_prefix = np.cumsum([0] + self._rg_num_rows).tolist()
        self._total_rows = self._rg_prefix[-1]

        self._rg_cache: Dict[int, List[str]] = {}
        self._rg_cache_order: List[int] = []

        self._texts: Optional[List[str]] = None
        if self.preload:
            self._texts = self.pf.read(columns=[self.text_col]).column(0).to_pylist()

    def _cache_put(self, rg: int, texts: List[str]) -> None:
        self._rg_cache[rg] = texts
        self._rg_cache_order.append(rg)
        while len(self._rg_cache_order) > self.cache_row_groups:
            old = self._rg_cache_order.pop(0)
            self._rg_cache.pop(old, None)

    def _row_to_rg(self, row_idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self._rg_prefix) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._rg_prefix[mid] <= row_idx < self._rg_prefix[mid + 1]:
                return mid, row_idx - self._rg_prefix[mid]
            if row_idx < self._rg_prefix[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(f"row_idx out of range: {row_idx}")

    def get(self, chunk_id: int) -> str:
        if chunk_id is None:
            return ""
        try:
            row_idx = int(chunk_id) - self.id_base
        except Exception:
            return ""

        if row_idx < 0 or row_idx >= self._total_rows:
            return ""

        if self._texts is not None:
            x = self._texts[row_idx]
            return "" if x is None or str(x).lower() == "nan" else str(x)

        rg, off = self._row_to_rg(row_idx)
        cached = self._rg_cache.get(rg)
        if cached is None:
            t = self.pf.read_row_group(rg, columns=[self.text_col])
            cached = t.column(0).to_pylist()
            self._cache_put(rg, cached)

        x = cached[off]
        return "" if x is None or str(x).lower() == "nan" else str(x)


class VerifierSupervisionDataset(Dataset):
    """
    Reads Step 4C verifier_supervision.parquet and materializes evidence text on-the-fly
    using a frozen evidence chunk store.

    Updated for the new pipeline:
      - supports supporting_span / retrieval_query columns
      - sorts evidence by evidence_weights_json before truncating to top-k
      - keeps verifier input as CASE + QUESTION + CLAIM + EVIDENCE
    """

    def __init__(
        self,
        cfg_path: str,
        split_name: str,
        supervision_relpath: str = "step4c_supervision/verifier_supervision.parquet",
        evidence_parquet_relpath: str = "chunks.parquet",
        evidence_text_col: str = "chunk_text",
        evidence_id_base: int = 0,
        preload_evidence_texts: bool = False,
        evidence_cache_row_groups: int = 2,
        split_col: str = "split",
        qa_row_id_col: str = "qa_row_id",
        case_context_col: str = "case_context",
        question_col: str = "question",
        answer_col: str = "answer_target",
        claim_col: str = "claim_text",
        supporting_span_col: str = "supporting_span",
        retrieval_query_col: str = "retrieval_query",
        evidence_ids_col: str = "evidence_chunk_ids_json",
        evidence_weights_col: str = "evidence_weights_json",
        target_col: str = "soft_support_score",
        evidence_separator: str = "\n\n---\n\n",
        max_evidence_chars: int = 20_000,
        verifier_top_k_evidence: int = 5,
        join_with_newlines: bool = True,
        cache_row_groups: int = 2,
        assume_split_is_string: bool = True,
    ) -> None:
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.base_path = self.params["Preprocessing"]["base_path_evidenceData"]

        self.split_name = str(split_name)

        self.split_col = str(split_col)
        self.qa_row_id_col = str(qa_row_id_col)
        self.case_context_col = str(case_context_col)
        self.question_col = str(question_col)
        self.answer_col = str(answer_col)
        self.claim_col = str(claim_col)
        self.supporting_span_col = str(supporting_span_col)
        self.retrieval_query_col = str(retrieval_query_col)
        self.evidence_ids_col = str(evidence_ids_col)
        self.evidence_weights_col = str(evidence_weights_col)
        self.target_col = str(target_col)

        self.evidence_separator = str(evidence_separator)
        self.max_evidence_chars = int(max_evidence_chars)
        self.verifier_top_k_evidence = int(verifier_top_k_evidence)
        self.join_with_newlines = bool(join_with_newlines)

        self.cache_row_groups = int(cache_row_groups)
        self.assume_split_is_string = bool(assume_split_is_string)

        self.supervision_path = os.path.join(self.base_path, supervision_relpath)
        if not os.path.exists(self.supervision_path):
            raise FileNotFoundError(f"Verifier supervision parquet not found: {self.supervision_path}")

        self.pf = pq.ParquetFile(self.supervision_path)

        schema_names = set(self.pf.schema.names)
        required = [
            self.split_col,
            self.case_context_col,
            self.question_col,
            self.claim_col,
            self.evidence_ids_col,
            self.evidence_weights_col,
            self.target_col,
        ]
        for c in required:
            if c not in schema_names:
                raise ValueError(f"Missing required column in verifier_supervision.parquet: {c}")

        self.has_qa_row_id = self.qa_row_id_col in schema_names
        self.has_answer = self.answer_col in schema_names
        self.has_supporting_span = self.supporting_span_col in schema_names
        self.has_retrieval_query = self.retrieval_query_col in schema_names

        evidence_path = os.path.join(self.base_path, evidence_parquet_relpath)
        self.evidence_store = EvidenceTextStore(
            evidence_parquet_path=evidence_path,
            text_col=evidence_text_col,
            id_base=evidence_id_base,
            preload=preload_evidence_texts,
            cache_row_groups=evidence_cache_row_groups,
        )

        self._rg_row_index: List[Tuple[int, int]] = []
        self._build_split_index()

        self._rg_cache: Dict[int, Dict[str, List]] = {}
        self._rg_cache_order: List[int] = []

    def _build_split_index(self) -> None:
        for rg in range(self.pf.num_row_groups):
            t = self.pf.read_row_group(rg, columns=[self.split_col])
            split_arr = t.column(0)
            split_list = split_arr.to_pylist()

            for i, s in enumerate(split_list):
                if str(s) == self.split_name:
                    self._rg_row_index.append((rg, i))

        if len(self._rg_row_index) == 0:
            print(f"[WARN] No rows found for split='{self.split_name}' in {self.supervision_path}")

    def __len__(self) -> int:
        return len(self._rg_row_index)

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

    def _truncate_evidence(self, s: str) -> str:
        s = self._safe_str(s)
        if self.max_evidence_chars > 0 and len(s) > self.max_evidence_chars:
            return s[: self.max_evidence_chars]
        return s

    def _materialize_evidence(self, evidence_ids_json: str, evidence_weights_json: str) -> str:
        ids = self._loads_json_list(self._safe_str(evidence_ids_json))
        weights = self._loads_json_list(self._safe_str(evidence_weights_json))

        pairs: List[Tuple[int, float]] = []
        m = min(len(ids), len(weights))
        for i in range(m):
            try:
                cid = int(ids[i])
                ww = float(weights[i])
                pairs.append((cid, ww))
            except Exception:
                continue

        if len(ids) > m:
            for i in range(m, len(ids)):
                try:
                    pairs.append((int(ids[i]), 0.0))
                except Exception:
                    continue

        pairs.sort(key=lambda x: x[1], reverse=True)

        if self.verifier_top_k_evidence > 0:
            pairs = pairs[: self.verifier_top_k_evidence]

        out_chunks: List[str] = []
        for cid, _ in pairs:
            try:
                txt = self.evidence_store.get(int(cid))
                if txt:
                    out_chunks.append(txt)
            except Exception:
                pass

        joined = self.evidence_separator.join(out_chunks)
        return self._truncate_evidence(joined)

    def _build_input_text(
        self,
        case_context: str,
        question: str,
        claim_text: str,
        evidence_text: str,
    ) -> str:
        sep = "\n\n" if self.join_with_newlines else " "
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

    def _cache_put(self, rg: int, payload: Dict[str, List]) -> None:
        self._rg_cache[rg] = payload
        self._rg_cache_order.append(rg)
        while len(self._rg_cache_order) > self.cache_row_groups:
            old = self._rg_cache_order.pop(0)
            self._rg_cache.pop(old, None)

    def _get_row_group_columns(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached

        cols = [
            self.case_context_col,
            self.question_col,
            self.claim_col,
            self.evidence_ids_col,
            self.evidence_weights_col,
            self.target_col,
        ]
        if self.has_qa_row_id:
            cols.append(self.qa_row_id_col)
        if self.has_answer:
            cols.append(self.answer_col)
        if self.has_supporting_span:
            cols.append(self.supporting_span_col)
        if self.has_retrieval_query:
            cols.append(self.retrieval_query_col)

        t = self.pf.read_row_group(rg, columns=cols)
        payload: Dict[str, List] = {c: t.column(c).to_pylist() for c in cols}

        self._cache_put(rg, payload)
        return payload

    def __getitem__(self, idx: int) -> Dict:
        rg, row_in_group = self._rg_row_index[idx]
        payload = self._get_row_group_columns(rg)

        if self.has_qa_row_id:
            try:
                qa_row_id = int(payload[self.qa_row_id_col][row_in_group])
            except Exception:
                qa_row_id = -1
        else:
            qa_row_id = int(idx)

        case_context = payload[self.case_context_col][row_in_group]
        question = payload[self.question_col][row_in_group]
        claim_text = payload[self.claim_col][row_in_group]
        evidence_ids_json = payload[self.evidence_ids_col][row_in_group]
        evidence_weights_json = payload[self.evidence_weights_col][row_in_group]

        evidence_text = self._materialize_evidence(
            evidence_ids_json=evidence_ids_json,
            evidence_weights_json=evidence_weights_json,
        )

        input_text = self._build_input_text(
            case_context=case_context,
            question=question,
            claim_text=claim_text,
            evidence_text=evidence_text,
        )

        try:
            target = float(payload[self.target_col][row_in_group])
        except Exception:
            target = 0.0

        return {
            "qa_row_id": qa_row_id,
            "input_text": input_text,
            "target": target,
        }


class NonRepeatingChunkSampler(Sampler[int]):
    def __init__(
        self,
        data_source,
        chunk_size: int = 100_000,
        seed: int = 13,
        drop_last: bool = True,
    ):
        self.data_source = data_source
        self.N = len(data_source)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.N <= 0:
            raise ValueError("Dataset is empty")

        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(self.N, generator=g).tolist()
        self._cursor = 0
        self._round = 0

    def set_round(self, round_idx: int) -> None:
        round_idx = int(round_idx)
        if round_idx < 0:
            round_idx = 0
        self._round = round_idx
        self._cursor = round_idx * self.chunk_size
        while self._cursor >= self.N:
            self._reshuffle()
            self._cursor = max(0, self._cursor - self.N)

    def _reshuffle(self) -> None:
        self._round += 1
        g = torch.Generator()
        g.manual_seed(self.seed + self._round)
        self._perm = torch.randperm(self.N, generator=g).tolist()
        self._cursor = 0

    def __iter__(self) -> Iterator[int]:
        if self._cursor >= self.N:
            self._reshuffle()

        end = self._cursor + self.chunk_size
        if end <= self.N:
            out = self._perm[self._cursor:end]
            self._cursor = end
            return iter(out)

        remainder = self._perm[self._cursor:self.N]
        self._cursor = self.N

        if self.drop_last:
            self._reshuffle()
            return iter(remainder)
        else:
            return iter(remainder)

    def __len__(self) -> int:
        return min(self.chunk_size, self.N)


def verifier_collate_fn(batch: List[Dict]) -> VerifierBatch:
    inputs: List[str] = [str(x.get("input_text", "")).strip() for x in batch]
    targets = torch.tensor([float(x.get("target", 0.0)) for x in batch], dtype=torch.float32)
    qa_row_ids = [int(x.get("qa_row_id", -1)) for x in batch]
    return VerifierBatch(inputs=inputs, targets=targets, qa_row_ids=qa_row_ids)