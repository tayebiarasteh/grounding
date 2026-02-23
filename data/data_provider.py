"""
Created on Feb 23, 2026.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import json
import os.path
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
