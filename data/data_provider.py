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
        self.base_path = self.params["paths"]["base_path"]

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
    inputs: List[str]                 # B (full text fed to DeBERTa)
    targets: torch.Tensor             # (B,) float32 soft_support_score
    qa_row_ids: Optional[List[int]]   # for debugging / analysis


class VerifierSupervisionDataset(Dataset):
    """
    Memory-safe, fast map-style Dataset for large materialized verifier parquet.

    Works with Step 7 output (updated):
      - evidence_text exists (alias of evidence_text_concat)
      - evidence_text_concat may also exist (kept for debugging)

    Strategy:
      1) Build an index of rows belonging to split_name WITHOUT converting to pandas:
         stores (row_group_id, row_in_group) for each selected row.
      2) In __getitem__, load the needed row-group (cached), fetch row fields, build input text.

    Avoids:
      - pd.read_parquet() loading the entire file into RAM
      - huge Arrow->pandas string object blowups
      - per-sample random disk reads (we read row groups, not single rows)

    Notes:
      - Each DataLoader worker has its own cache.
      - Keep num_workers moderate (2-8). More workers means more caches.
    """

    def __init__(
        self,
        cfg_path: str,
        split_name: str,
        # paths
        materialized_relpath: str = "step7_verifier_materialized_rstar/verifier_train_materialized.parquet",
        # columns
        split_col: str = "split",
        qa_row_id_col: str = "qa_row_id",
        case_context_col: str = "case_context",
        question_col: str = "question",
        answer_col: str = "answer_target",
        claim_col: str = "claim_text",
        evidence_text_col: str = "evidence_text",           # preferred (new Step 7)
        evidence_text_fallback_col: str = "evidence_text_concat",  # fallback if older file
        target_col: str = "soft_support_score",
        # formatting
        max_evidence_chars: int = 20_000,
        join_with_newlines: bool = True,
        # performance knobs
        cache_row_groups: int = 2,   # how many row groups to keep in-memory per worker
        assume_split_is_string: bool = True,
    ) -> None:
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.base_path = self.params["paths"]["base_path"]

        self.split_name = str(split_name)

        self.split_col = str(split_col)
        self.qa_row_id_col = str(qa_row_id_col)
        self.case_context_col = str(case_context_col)
        self.question_col = str(question_col)
        self.answer_col = str(answer_col)
        self.claim_col = str(claim_col)
        self.evidence_text_col = str(evidence_text_col)
        self.evidence_text_fallback_col = str(evidence_text_fallback_col)
        self.target_col = str(target_col)

        self.max_evidence_chars = int(max_evidence_chars)
        self.join_with_newlines = bool(join_with_newlines)

        self.cache_row_groups = int(cache_row_groups)
        self.assume_split_is_string = bool(assume_split_is_string)

        self.materialized_path = os.path.join(self.base_path, materialized_relpath)
        if not os.path.exists(self.materialized_path):
            raise FileNotFoundError(f"Materialized verifier parquet not found: {self.materialized_path}")

        # Open parquet lazily
        self.pf = pq.ParquetFile(self.materialized_path)

        # Validate columns exist without loading data
        schema_names = set(self.pf.schema.names)
        required_base = [
            self.split_col,
            self.case_context_col,
            self.question_col,
            self.answer_col,
            self.claim_col,
            self.target_col,
        ]
        for c in required_base:
            if c not in schema_names:
                raise ValueError(f"Missing required column in verifier materialized parquet: {c}")

        # Evidence column selection:
        # - prefer evidence_text (new Step 7)
        # - fallback to evidence_text_concat (older Step 7)
        if self.evidence_text_col in schema_names:
            self._evidence_col_in_file = self.evidence_text_col
        elif self.evidence_text_fallback_col in schema_names:
            self._evidence_col_in_file = self.evidence_text_fallback_col
        else:
            raise ValueError(
                f"Missing evidence text column. Expected '{self.evidence_text_col}' or "
                f"fallback '{self.evidence_text_fallback_col}'."
            )

        self.has_qa_row_id = self.qa_row_id_col in schema_names

        # Build split index: list of (row_group_id, row_in_group)
        self._rg_row_index: List[Tuple[int, int]] = []
        self._build_split_index()

        # Row-group cache (per-process/worker)
        self._rg_cache: Dict[int, Dict[str, List]] = {}
        self._rg_cache_order: List[int] = []

    def _build_split_index(self) -> None:
        for rg in range(self.pf.num_row_groups):
            t = self.pf.read_row_group(rg, columns=[self.split_col])
            split_arr = t.column(0)

            if self.assume_split_is_string:
                split_list = split_arr.to_pylist()
                for i, s in enumerate(split_list):
                    if str(s) == self.split_name:
                        self._rg_row_index.append((rg, i))
            else:
                split_list = [str(x) for x in split_arr.to_pylist()]
                for i, s in enumerate(split_list):
                    if s == self.split_name:
                        self._rg_row_index.append((rg, i))

        if len(self._rg_row_index) == 0:
            print(f"[WARN] No rows found for split='{self.split_name}' in {self.materialized_path}")

    def __len__(self) -> int:
        return len(self._rg_row_index)

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _truncate_evidence(self, s: str) -> str:
        s = self._safe_str(s)
        if self.max_evidence_chars > 0 and len(s) > self.max_evidence_chars:
            return s[: self.max_evidence_chars]
        return s

    def _build_input_text(
        self,
        case_context: str,
        question: str,
        answer_target: str,
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
                "ANSWER_TARGET:",
                self._safe_str(answer_target).strip(),
                "CLAIM:",
                self._safe_str(claim_text).strip(),
                "EVIDENCE:",
                self._truncate_evidence(evidence_text).strip(),
            ]
        )

    def _cache_put(self, rg: int, payload: Dict[str, List]) -> None:
        self._rg_cache[rg] = payload
        self._rg_cache_order.append(rg)

        while len(self._rg_cache_order) > self.cache_row_groups:
            old = self._rg_cache_order.pop(0)
            if old in self._rg_cache:
                del self._rg_cache[old]

    def _get_row_group_columns(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached

        cols = [
            self.case_context_col,
            self.question_col,
            self.answer_col,
            self.claim_col,
            self._evidence_col_in_file,  # selected evidence column
            self.target_col,
        ]
        if self.has_qa_row_id:
            cols.append(self.qa_row_id_col)

        t = self.pf.read_row_group(rg, columns=cols)

        payload: Dict[str, List] = {}
        for c in cols:
            payload[c] = t.column(c).to_pylist()

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
        answer_target = payload[self.answer_col][row_in_group]
        claim_text = payload[self.claim_col][row_in_group]
        evidence_text = payload[self._evidence_col_in_file][row_in_group]

        input_text = self._build_input_text(
            case_context=case_context,
            question=question,
            answer_target=answer_target,
            claim_text=claim_text,
            evidence_text=evidence_text,
        )

        try:
            target = float(payload[self.target_col][row_in_group])
        except Exception:
            target = 0.0

        return {"qa_row_id": qa_row_id, "input_text": input_text, "target": target}


def verifier_collate_fn(batch: List[Dict]) -> VerifierBatch:
    inputs: List[str] = [str(x.get("input_text", "")).strip() for x in batch]
    targets = torch.tensor([float(x.get("target", 0.0)) for x in batch], dtype=torch.float32)
    qa_row_ids = [int(x.get("qa_row_id", -1)) for x in batch]
    return VerifierBatch(inputs=inputs, targets=targets, qa_row_ids=qa_row_ids)