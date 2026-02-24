"""
Created on Feb 24, 2026.
Tier1_prediction.py

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
import torch.nn as nn
import math
import pickle
import heapq

from models.model import BiEncoder
from config.serde import open_experiment, create_experiment, delete_experiment, write_config


import warnings
warnings.filterwarnings('ignore')



"""
S4 Tier1 + Tier2 end-to-end evaluator

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
"""



class S4Tier12End2EndEvaluator:
    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        model_weights_path: str,       # retriever trained_model.pth
        verifier_weights_path: str,    # verifier trained_model.pth
        device: str = "cuda",
        top_k: int = 20,
        verifier_max_tokens: int = 512,
        evidence_sep: str = "\n\n---\n\n",
        batch_size_retrieve: int = 256,
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        out_dir: Optional[str] = None,
        seed: int = 13,
        # Tier2 knobs
        mrr_k: int = 10,
        recall_ks: Tuple[int, ...] = (1, 5, 10, 20),
        faithfulness_random_trials: int = 1,   # keep 1 for speed; increase later
        faithfulness_max_eval: Optional[int] = None,  # e.g. 2000 for fast debug
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.model_weights_path = model_weights_path
        self.verifier_weights_path = verifier_weights_path
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.evidence_sep = evidence_sep
        self.batch_size_retrieve = int(batch_size_retrieve)
        self.batch_size_verify = int(batch_size_verify)
        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.mrr_k = int(mrr_k)
        self.recall_ks = tuple(int(x) for x in recall_ks)
        self.faithfulness_random_trials = int(faithfulness_random_trials)
        self.faithfulness_max_eval = int(faithfulness_max_eval) if faithfulness_max_eval is not None else None

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet"
        )

        self.step6_dir = os.path.join(self.base_path_evidenceData, "step6_faiss_rstar")
        self.faiss_index_path = os.path.join(self.step6_dir, "faiss_Rstar.index")
        self.chunk_ids_path = os.path.join(self.step6_dir, "chunk_ids.npy")
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier12_results_s4_end2end")
        os.makedirs(self.out_dir, exist_ok=True)

        self._set_determinism()

        # Loaded later
        self.params = open_experiment(self.experiment_name, self.cfg_path)
        self.retriever_tok = None
        self.retriever_model = None
        self.verifier_tok = None
        self.verifier_model = None
        self.faiss_index = None
        self.faiss_chunk_ids = None
        self.chunk_map = None
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

    # -------------------------
    # Verifier wrapper
    # -------------------------
    class _DebertaRegressor(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.reg_head = nn.Linear(hidden, 1)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            cls = cls.to(self.reg_head.weight.dtype)
            y = self.reg_head(cls).squeeze(-1)
            return y

    # -------------------------
    # Loading
    # -------------------------
    def _load_chunk_map(self) -> Dict[int, str]:
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

    def _load_faiss(self) -> None:
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(self.faiss_index_path)
        if not os.path.exists(self.chunk_ids_path):
            raise FileNotFoundError(self.chunk_ids_path)
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        self.faiss_chunk_ids = np.load(self.chunk_ids_path).astype(np.int64)
        self.all_chunk_ids_universe = self.faiss_chunk_ids.copy()

    def _load_retriever(self) -> None:
        model_name = self.params["retriever"]["model_name"]
        tied = bool(self.params["retriever"]["tied_encoders"])
        pooling = self.params["retriever"]["pooling"]
        self.max_query_tokens = int(self.params["retriever"]["max_query_tokens"])

        self.retriever_tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.retriever_model = BiEncoder(model_name, tied=tied, pooling=pooling).to(self.device)
        self.retriever_model.eval()

        obj = torch.load(self.model_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj
        self.retriever_model.load_state_dict(state, strict=True)

    def _load_verifier(self) -> None:
        vname = self.params["Verifier"]["model_name"]
        self.verifier_tok = AutoTokenizer.from_pretrained(vname, use_fast=True)
        self.verifier_model = self._DebertaRegressor(vname).to(self.device)
        self.verifier_model.eval()

        obj = torch.load(self.verifier_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj

        missing, unexpected = self.verifier_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] Verifier load_state_dict strict=False. missing={len(missing)} unexpected={len(unexpected)}")

    # -------------------------
    # Helpers
    # -------------------------
    def _loads_json_list(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _normalize_claim(self, x: str) -> str:
        return " ".join(str(x).split()).strip().lower()

    # -------------------------
    # Retrieval (R*)
    # -------------------------
    def _tokenize_queries(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.retriever_tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_query_tokens,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _retrieve_topk_chunk_ids(self, claim_texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        assert self.retriever_model is not None
        assert self.faiss_index is not None
        assert self.faiss_chunk_ids is not None

        all_ids: List[List[int]] = []
        all_scores: List[List[float]] = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(claim_texts), self.batch_size_retrieve),
                total=math.ceil(len(claim_texts) / self.batch_size_retrieve),
                desc="S4 Retrieval (R*)",
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
        for cid in chunk_ids:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)

    # -------------------------
    # Verifier inference
    # -------------------------
    def _build_verifier_inputs(self, question: str, case_context: str, evidence_text: str) -> str:
        q = " ".join(str(question).split()).strip()
        c = " ".join(str(case_context).split()).strip()
        e = " ".join(str(evidence_text).split()).strip()
        return f"Question: {q}\nCase: {c}\nEvidence: {e}"

    def _predict_support(self, texts: List[str], desc: str = "S4 Verification (V*)") -> np.ndarray:
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
                y = self.verifier_model(enc["input_ids"], enc["attention_mask"])
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

    # -------------------------
    # Metrics
    # -------------------------
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

    def _risk_coverage(self, y: np.ndarray, yhat: np.ndarray) -> Dict[str, object]:
        p = 1.0 / (1.0 + np.exp(-np.clip(yhat, -20, 20)))
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

    # -------------------------
    # Tier2: Retrieval metrics
    # -------------------------
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

    def _compute_retrieval_metrics(
        self,
        df_test: pd.DataFrame,
        retrieved_topk_ids: List[List[int]],
    ) -> Dict[str, float]:
        # gold evidence from phase1
        gold_lists: List[List[int]] = []
        for s in df_test["evidence_chunk_ids_json"].astype(str).tolist():
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

    # -------------------------
    # Tier2: Faithfulness stress tests
    # -------------------------
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _eval_preds_metrics(self, y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
        mse = self._mse(y, yhat)
        mae = self._mae(y, yhat)
        pr = self._pearson(y, yhat)
        p = self._sigmoid(yhat)
        y01 = np.clip(y, 0.0, 1.0)
        ece = self._ece(y01, p, self.ece_bins)
        brier = self._brier(y01, p)
        return {"MSE": mse, "MAE": mae, "Pearson": pr, "ECE": ece, "Brier": brier}

    def _stress_random_ids(self, n: int) -> List[List[int]]:
        assert self.all_chunk_ids_universe is not None
        U = self.all_chunk_ids_universe
        out: List[List[int]] = []
        for _ in range(n):
            idx = np.random.choice(len(U), size=self.top_k, replace=False)
            out.append([int(x) for x in U[idx].tolist()])
        return out

    def _stress_swap_between_samples(self, ids_per_sample: List[List[int]]) -> List[List[int]]:
        # shift by 1: evidence from neighboring sample (proxy "topically similar but wrong")
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

    def _predict_with_ids(
        self,
        df_sub: pd.DataFrame,
        ids_per_sample: List[List[int]],
        desc: str,
    ) -> np.ndarray:
        inputs: List[str] = []
        for i in range(len(df_sub)):
            q = df_sub.iloc[i]["question"]
            c = df_sub.iloc[i]["case_context"]
            ev = self._materialize_evidence(ids_per_sample[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))
        return self._predict_support(inputs, desc=desc)

    # -------------------------
    # Run
    # -------------------------
    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)
        df = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        # df = df.sample(frac=0.01, random_state=self.seed).reset_index(drop=True)

        if "split" in df.columns:
            df = df[df["split"].astype(str) == "test"].copy()

        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df = df.sample(n=min(len(df), self.faithfulness_max_eval), random_state=self.seed).reset_index(drop=True)

        # Load artifacts/models
        self.chunk_map = self._load_chunk_map()
        self._load_faiss()
        self._load_retriever()
        self._load_verifier()

        # -------------------------
        # Tier1: baseline S4 end2end
        # -------------------------
        claim_texts = df["claim_text"].astype(str).tolist()
        topk_ids, _ = self._retrieve_topk_chunk_ids(claim_texts)

        inputs: List[str] = []
        for i in range(len(df)):
            q = df.iloc[i]["question"]
            c = df.iloc[i]["case_context"]
            ev = self._materialize_evidence(topk_ids[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))

        yhat = self._predict_support(inputs, desc="S4 Verification (V*)")
        y = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        mse = self._mse(y, yhat)
        mae = self._mae(y, yhat)
        pearson = self._pearson(y, yhat)

        p = self._sigmoid(yhat)
        y01 = np.clip(y, 0.0, 1.0)
        ece = self._ece(y01, p, self.ece_bins)
        brier = self._brier(y01, p)
        rc = self._risk_coverage(y01, yhat)

        out_df = df.copy()
        out_df["s4_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        out_df["s4_pred_support_raw"] = yhat.astype(np.float32)
        out_df["s4_pred_prob_sigmoid"] = p.astype(np.float32)

        pred_path = os.path.join(self.out_dir, "s4_test_predictions.parquet")
        out_df.to_parquet(pred_path, index=False)

        tier1_report = {
            "system": "S4 (R* retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "retriever": {
                "weights": self.model_weights_path,
                "faiss_index": self.faiss_index_path,
                "chunk_ids_npy": self.chunk_ids_path,
                "top_k": self.top_k,
            },
            "verifier": {
                "weights": self.verifier_weights_path,
                "max_tokens": self.verifier_max_tokens,
            },
            "metrics": {
                "MSE": mse,
                "MAE": mae,
                "Pearson": pearson,
                "ECE": ece,
                "Brier": brier,
                "SelectiveRisk": rc,
            },
            "artifacts": {"predictions_parquet": pred_path},
            "n_test": int(len(df)),
        }

        tier1_path = os.path.join(self.out_dir, "s4_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        # -------------------------
        # Tier2A: retrieval metrics (against phase1 evidence ids)
        # -------------------------
        retrieval_metrics = self._compute_retrieval_metrics(df, topk_ids)
        retrieval_path = os.path.join(self.out_dir, "s4_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S4 (R* retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df)),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )

        # -------------------------
        # Tier2B: faithfulness stress tests
        # -------------------------
        base_metrics = self._eval_preds_metrics(y01, yhat)

        # build stress id sets
        ids_top1 = self._stress_truncate(topk_ids, 1)
        ids_top3 = self._stress_truncate(topk_ids, 3)
        ids_shuffle = self._stress_shuffle_order(topk_ids)
        ids_swapped = self._stress_swap_between_samples(topk_ids)

        # random can be multiple trials (average)
        random_trial_metrics: List[Dict[str, float]] = []
        random_trial_preds: List[np.ndarray] = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df))
            yhat_rand = self._predict_with_ids(df, ids_rand, desc=f"S4 Faithfulness random (trial {t+1})")
            random_trial_preds.append(yhat_rand)
            random_trial_metrics.append(self._eval_preds_metrics(y01, yhat_rand))

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {"MSE": 0.0, "MAE": 0.0, "Pearson": 0.0, "ECE": 0.0, "Brier": 0.0}
            keys = ms[0].keys()
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        # run other perturbations once
        yhat_top1 = self._predict_with_ids(df, ids_top1, desc="S4 Faithfulness top-1")
        yhat_top3 = self._predict_with_ids(df, ids_top3, desc="S4 Faithfulness top-3")
        yhat_shuffle = self._predict_with_ids(df, ids_shuffle, desc="S4 Faithfulness shuffle-order")
        yhat_swapped = self._predict_with_ids(df, ids_swapped, desc="S4 Faithfulness swapped-evidence")

        met_top1 = self._eval_preds_metrics(y01, yhat_top1)
        met_top3 = self._eval_preds_metrics(y01, yhat_top3)
        met_shuffle = self._eval_preds_metrics(y01, yhat_shuffle)
        met_swapped = self._eval_preds_metrics(y01, yhat_swapped)
        met_random = avg_metrics(random_trial_metrics)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S4 (R* retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df)),
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

        # Save per-row faithfulness preds
        faith_df = df.copy()
        faith_df["s4_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        faith_df["s4_pred_base_raw"] = yhat.astype(np.float32)
        faith_df["s4_pred_top1_raw"] = yhat_top1.astype(np.float32)
        faith_df["s4_pred_top3_raw"] = yhat_top3.astype(np.float32)
        faith_df["s4_pred_shuffle_raw"] = yhat_shuffle.astype(np.float32)
        faith_df["s4_pred_swapped_raw"] = yhat_swapped.astype(np.float32)
        if random_trial_preds:
            # store avg over trials
            yhat_rand_avg = np.mean(np.stack(random_trial_preds, axis=0), axis=0).astype(np.float32)
            faith_df["s4_pred_random_avg_raw"] = yhat_rand_avg

        faith_pred_path = os.path.join(self.out_dir, "s4_tier2_faithfulness_predictions.parquet")
        faith_df.to_parquet(faith_pred_path, index=False)

        faith_path = os.path.join(self.out_dir, "s4_tier2_faithfulness_report.json")
        with open(faith_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **faith_report,
                    "artifacts": {"faithfulness_predictions_parquet": faith_pred_path},
                },
                f,
                indent=2,
            )

        # final combined summary
        summary = {
            "tier1_report": tier1_path,
            "tier2_retrieval_metrics": retrieval_path,
            "tier2_faithfulness_report": faith_path,
            "tier2_faithfulness_predictions": faith_pred_path,
        }
        summary_path = os.path.join(self.out_dir, "s4_tier12_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(tier1_report["metrics"], indent=2))
        print(json.dumps({"Retrieval": retrieval_metrics}, indent=2))
        print(json.dumps({"Faithfulness_deltas": faith_report["perturbations"]}, indent=2))
        print(f"Wrote: {pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_pred_path}")
        print(f"Wrote: {faith_path}")
        print(f"Wrote: {summary_path}")
        return summary_path





class S2Tier12End2EndEvaluator:
    """
    S2 — Frozen dense MPNet retrieval -> V*
    - Off-the-shelf MPNet (no finetuning)
    - Builds its own FAISS + chunk embeddings inside out_dir
    - Loads the fixed test parquet everywhere
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,  # verifier trained_model.pth
        device: str = "cuda",
        top_k: int = 20,
        verifier_max_tokens: int = 512,
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
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)
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

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet"
        )
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier12_results_s2_end2end")
        os.makedirs(self.out_dir, exist_ok=True)

        # S2-specific artifacts ONLY inside its out_dir
        self.faiss_index_path = os.path.join(self.out_dir, "faiss_mpnet_offtheshelf.index")
        self.chunk_ids_path = os.path.join(self.out_dir, "chunk_ids.npy")
        self.chunk_embeds_path = os.path.join(self.out_dir, "chunk_embeds_mpnet_offtheshelf.npy")

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        self.retriever_tok = None
        self.retriever_model = None
        self.max_query_tokens = None
        self.max_doc_tokens = None

        self.verifier_tok = None
        self.verifier_model = None

        self.faiss_index = None
        self.faiss_chunk_ids = None
        self.chunk_map = None
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

    class _DebertaRegressor(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.reg_head = nn.Linear(hidden, 1)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            cls = cls.to(self.reg_head.weight.dtype)
            y = self.reg_head(cls).squeeze(-1)
            return y

    # -------------------------
    # Loading
    # -------------------------
    def _load_chunk_map(self) -> Dict[int, str]:
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

    def _load_offtheshelf_mpnet(self) -> None:
        model_name = self.params["retriever"]["model_name"]  # sentence-transformers/all-mpnet-base-v2
        self.max_query_tokens = int(self.params["retriever"]["max_query_tokens"])
        self.max_doc_tokens = int(self.params["retriever"]["max_doc_tokens"])

        self.retriever_tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.retriever_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.retriever_model.eval()

    # -------------------------
    # Encoding + FAISS
    # -------------------------
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
        return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)

    def _build_or_load_faiss_s2(self) -> None:
        if (
            os.path.exists(self.faiss_index_path)
            and os.path.exists(self.chunk_ids_path)
            and os.path.exists(self.chunk_embeds_path)
        ):
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            self.faiss_chunk_ids = np.load(self.chunk_ids_path).astype(np.int64)
            self.all_chunk_ids_universe = self.faiss_chunk_ids.copy()
            return

        df = pd.read_parquet(self.chunks_parquet, columns=["chunk_id", "chunk_text"])
        df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
        df = df[~df["chunk_id"].isna()].copy()
        df["chunk_id"] = df["chunk_id"].astype(np.int64)
        df["chunk_text"] = df["chunk_text"].astype(str).map(lambda x: " ".join(str(x).split()).strip())

        df = df.sort_values("chunk_id").reset_index(drop=True)
        chunk_ids = df["chunk_id"].to_numpy(dtype=np.int64)
        texts = df["chunk_text"].tolist()

        embeds = self._encode_texts(
            texts=texts,
            max_len=self.max_doc_tokens,
            batch_size=self.batch_size_embed_chunks,
            desc="S2 Build FAISS: embedding chunks (MPNet off-the-shelf)",
        )

        np.save(self.chunk_ids_path, chunk_ids)
        np.save(self.chunk_embeds_path, embeds)

        dim = int(embeds.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeds)
        faiss.write_index(index, self.faiss_index_path)

        self.faiss_index = index
        self.faiss_chunk_ids = chunk_ids
        self.all_chunk_ids_universe = chunk_ids.copy()

    # -------------------------
    # Verifier
    # -------------------------
    def _load_verifier(self) -> None:
        vname = self.params["Verifier"]["model_name"]
        self.verifier_tok = AutoTokenizer.from_pretrained(vname, use_fast=True)
        self.verifier_model = self._DebertaRegressor(vname).to(self.device)
        self.verifier_model.eval()

        obj = torch.load(self.verifier_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj
        missing, unexpected = self.verifier_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] Verifier load_state_dict strict=False. missing={len(missing)} unexpected={len(unexpected)}")

    # -------------------------
    # Retrieval
    # -------------------------
    def _retrieve_topk_chunk_ids(self, claim_texts: List[str]) -> List[List[int]]:
        assert self.faiss_index is not None
        assert self.faiss_chunk_ids is not None

        q_emb = self._encode_texts(
            texts=claim_texts,
            max_len=self.max_query_tokens,
            batch_size=self.batch_size_retrieve,
            desc="S2 Retrieval (MPNet off-the-shelf)",
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
        for cid in chunk_ids:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)

    # -------------------------
    # Verifier inference
    # -------------------------
    def _build_verifier_inputs(self, question: str, case_context: str, evidence_text: str) -> str:
        q = " ".join(str(question).split()).strip()
        c = " ".join(str(case_context).split()).strip()
        e = " ".join(str(evidence_text).split()).strip()
        return f"Question: {q}\nCase: {c}\nEvidence: {e}"

    def _predict_support(self, texts: List[str], desc: str = "S2 Verification (V*)") -> np.ndarray:
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
                y = self.verifier_model(enc["input_ids"], enc["attention_mask"])
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

    # -------------------------
    # Metrics
    # -------------------------
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

    def _risk_coverage(self, y: np.ndarray, yhat: np.ndarray) -> Dict[str, object]:
        p = self._sigmoid(yhat)
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

    # -------------------------
    # Tier2: Retrieval metrics vs phase1 gold
    # -------------------------
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

    def _compute_retrieval_metrics(
        self,
        df_test: pd.DataFrame,
        retrieved_topk_ids: List[List[int]],
    ) -> Dict[str, float]:
        gold_lists: List[List[int]] = []
        for s in df_test["evidence_chunk_ids_json"].astype(str).tolist():
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

    # -------------------------
    # Tier2: Faithfulness stress tests (same protocol as S4)
    # -------------------------
    def _eval_preds_metrics(self, y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
        mse = self._mse(y, yhat)
        mae = self._mae(y, yhat)
        pr = self._pearson(y, yhat)
        p = self._sigmoid(yhat)
        y01 = np.clip(y, 0.0, 1.0)
        ece = self._ece(y01, p, self.ece_bins)
        brier = self._brier(y01, p)
        return {"MSE": mse, "MAE": mae, "Pearson": pr, "ECE": ece, "Brier": brier}

    def _stress_random_ids(self, n: int) -> List[List[int]]:
        assert self.all_chunk_ids_universe is not None
        U = self.all_chunk_ids_universe
        out: List[List[int]] = []
        for _ in range(n):
            idx = np.random.choice(len(U), size=self.top_k, replace=False)
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

    def _predict_with_ids(
        self,
        df_sub: pd.DataFrame,
        ids_per_sample: List[List[int]],
        desc: str,
    ) -> np.ndarray:
        inputs: List[str] = []
        for i in range(len(df_sub)):
            q = df_sub.iloc[i]["question"]
            c = df_sub.iloc[i]["case_context"]
            ev = self._materialize_evidence(ids_per_sample[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))
        return self._predict_support(inputs, desc=desc)

    # -------------------------
    # Run
    # -------------------------
    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)
        df = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        # df = df.sample(frac=0.01, random_state=self.seed).reset_index(drop=True)

        if "split" in df.columns:
            df = df[df["split"].astype(str) == "test"].copy()

        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df = df.sample(n=min(len(df), self.faithfulness_max_eval), random_state=self.seed).reset_index(drop=True)

        self.chunk_map = self._load_chunk_map()
        self._load_offtheshelf_mpnet()
        self._build_or_load_faiss_s2()
        self._load_verifier()

        # -------------------------
        # Tier1: baseline S2 end2end
        # -------------------------
        claim_texts = df["claim_text"].astype(str).tolist()
        topk_ids = self._retrieve_topk_chunk_ids(claim_texts)

        inputs: List[str] = []
        for i in range(len(df)):
            q = df.iloc[i]["question"]
            c = df.iloc[i]["case_context"]
            ev = self._materialize_evidence(topk_ids[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))

        yhat = self._predict_support(inputs, desc="S2 Verification (V*)")
        y = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        mse = self._mse(y, yhat)
        mae = self._mae(y, yhat)
        pearson = self._pearson(y, yhat)

        p = self._sigmoid(yhat)
        y01 = np.clip(y, 0.0, 1.0)
        ece = self._ece(y01, p, self.ece_bins)
        brier = self._brier(y01, p)

        rc = self._risk_coverage(y01, yhat)

        out_df = df.copy()
        out_df["s2_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        out_df["s2_pred_support_raw"] = yhat.astype(np.float32)
        out_df["s2_pred_prob_sigmoid"] = p.astype(np.float32)

        pred_path = os.path.join(self.out_dir, "s2_test_predictions.parquet")
        out_df.to_parquet(pred_path, index=False)

        tier1_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "retriever": {
                "mpnet_model": self.params["retriever"]["model_name"],
                "built_faiss_index": self.faiss_index_path,
                "chunk_ids_npy": self.chunk_ids_path,
                "chunk_embeds_npy": self.chunk_embeds_path,
                "top_k": self.top_k,
            },
            "verifier": {
                "weights": self.verifier_weights_path,
                "max_tokens": self.verifier_max_tokens,
            },
            "metrics": {
                "MSE": mse,
                "MAE": mae,
                "Pearson": pearson,
                "ECE": ece,
                "Brier": brier,
                "SelectiveRisk": rc,
            },
            "artifacts": {"predictions_parquet": pred_path},
            "n_test": int(len(df)),
        }

        tier1_path = os.path.join(self.out_dir, "s2_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        # -------------------------
        # Tier2A: retrieval metrics vs phase1 evidence ids
        # -------------------------
        retrieval_metrics = self._compute_retrieval_metrics(df, topk_ids)
        retrieval_path = os.path.join(self.out_dir, "s2_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df)),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )

        # -------------------------
        # Tier2B: faithfulness stress tests
        # -------------------------
        base_metrics = self._eval_preds_metrics(y01, yhat)

        ids_top1 = self._stress_truncate(topk_ids, 1)
        ids_top3 = self._stress_truncate(topk_ids, 3)
        ids_shuffle = self._stress_shuffle_order(topk_ids)
        ids_swapped = self._stress_swap_between_samples(topk_ids)

        random_trial_metrics: List[Dict[str, float]] = []
        random_trial_preds: List[np.ndarray] = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df))
            yhat_rand = self._predict_with_ids(df, ids_rand, desc=f"S2 Faithfulness random (trial {t+1})")
            random_trial_preds.append(yhat_rand)
            random_trial_metrics.append(self._eval_preds_metrics(y01, yhat_rand))

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {"MSE": 0.0, "MAE": 0.0, "Pearson": 0.0, "ECE": 0.0, "Brier": 0.0}
            keys = ms[0].keys()
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        yhat_top1 = self._predict_with_ids(df, ids_top1, desc="S2 Faithfulness top-1")
        yhat_top3 = self._predict_with_ids(df, ids_top3, desc="S2 Faithfulness top-3")
        yhat_shuffle = self._predict_with_ids(df, ids_shuffle, desc="S2 Faithfulness shuffle-order")
        yhat_swapped = self._predict_with_ids(df, ids_swapped, desc="S2 Faithfulness swapped-evidence")

        met_top1 = self._eval_preds_metrics(y01, yhat_top1)
        met_top3 = self._eval_preds_metrics(y01, yhat_top3)
        met_shuffle = self._eval_preds_metrics(y01, yhat_shuffle)
        met_swapped = self._eval_preds_metrics(y01, yhat_swapped)
        met_random = avg_metrics(random_trial_metrics)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df)),
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

        faith_df = df.copy()
        faith_df["s2_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        faith_df["s2_pred_base_raw"] = yhat.astype(np.float32)
        faith_df["s2_pred_top1_raw"] = yhat_top1.astype(np.float32)
        faith_df["s2_pred_top3_raw"] = yhat_top3.astype(np.float32)
        faith_df["s2_pred_shuffle_raw"] = yhat_shuffle.astype(np.float32)
        faith_df["s2_pred_swapped_raw"] = yhat_swapped.astype(np.float32)
        if random_trial_preds:
            yhat_rand_avg = np.mean(np.stack(random_trial_preds, axis=0), axis=0).astype(np.float32)
            faith_df["s2_pred_random_avg_raw"] = yhat_rand_avg

        faith_pred_path = os.path.join(self.out_dir, "s2_tier2_faithfulness_predictions.parquet")
        faith_df.to_parquet(faith_pred_path, index=False)

        faith_path = os.path.join(self.out_dir, "s2_tier2_faithfulness_report.json")
        with open(faith_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **faith_report,
                    "artifacts": {"faithfulness_predictions_parquet": faith_pred_path},
                },
                f,
                indent=2,
            )

        summary = {
            "tier1_report": tier1_path,
            "tier2_retrieval_metrics": retrieval_path,
            "tier2_faithfulness_report": faith_path,
            "tier2_faithfulness_predictions": faith_pred_path,
        }
        summary_path = os.path.join(self.out_dir, "s2_tier12_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(tier1_report["metrics"], indent=2))
        print(json.dumps({"Retrieval": retrieval_metrics}, indent=2))
        print(json.dumps({"Faithfulness_deltas": faith_report["perturbations"]}, indent=2))
        print(f"Wrote: {pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_pred_path}")
        print(f"Wrote: {faith_path}")
        print(f"Wrote: {summary_path}")
        return summary_path






class S3Tier12End2EndEvaluator:
    """
    S3 — BM25 retrieval -> V*
    - Lexical BM25 baseline (no FAISS)
    - Stores BM25 intermediate index ONLY inside out_dir
    - Loads the fixed test parquet everywhere
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,  # verifier trained_model.pth
        device: str = "cuda",
        top_k: int = 20,
        verifier_max_tokens: int = 512,
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
        faithfulness_max_eval: Optional[int] = None,  # e.g. 2000 for fast debug
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)
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

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet"
        )
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier12_results_s3_end2end")
        os.makedirs(self.out_dir, exist_ok=True)

        # S3-specific artifacts ONLY inside its out_dir
        self.bm25_index_path = os.path.join(self.out_dir, "bm25_index.pkl")

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        self.verifier_tok = None
        self.verifier_model = None
        self.chunk_map = None

        # BM25 runtime
        self._bm25 = None  # dict with postings/idf/doclen/chunk_ids/etc.
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

    class _DebertaRegressor(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.reg_head = nn.Linear(hidden, 1)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            cls = cls.to(self.reg_head.weight.dtype)
            y = self.reg_head(cls).squeeze(-1)
            return y

    # -------------------------
    # Loading
    # -------------------------
    def _load_chunk_map(self) -> Dict[int, str]:
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

    def _load_verifier(self) -> None:
        vname = self.params["Verifier"]["model_name"]
        self.verifier_tok = AutoTokenizer.from_pretrained(vname, use_fast=True)
        self.verifier_model = self._DebertaRegressor(vname).to(self.device)
        self.verifier_model.eval()

        obj = torch.load(self.verifier_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj
        missing, unexpected = self.verifier_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] Verifier load_state_dict strict=False. missing={len(missing)} unexpected={len(unexpected)}")

    # -------------------------
    # BM25
    # -------------------------
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

    def _build_or_load_bm25(self) -> None:
        if os.path.exists(self.bm25_index_path):
            with open(self.bm25_index_path, "rb") as f:
                self._bm25 = pickle.load(f)
            self.all_chunk_ids_universe = self._bm25["chunk_ids"].astype(np.int64)
            return

        df = pd.read_parquet(self.chunks_parquet, columns=["chunk_id", "chunk_text"])
        df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
        df = df[~df["chunk_id"].isna()].copy()
        df["chunk_id"] = df["chunk_id"].astype(np.int64)
        df["chunk_text"] = df["chunk_text"].astype(str).map(lambda x: " ".join(str(x).split()).strip())

        df = df.sort_values("chunk_id").reset_index(drop=True)
        chunk_ids = df["chunk_id"].to_numpy(dtype=np.int64)
        texts = df["chunk_text"].tolist()

        N = len(texts)
        doc_len = np.zeros((N,), dtype=np.int32)

        postings: Dict[str, List[Tuple[int, int]]] = {}
        dfreq: Dict[str, int] = {}

        for docid in tqdm(range(N), desc="S3 Build BM25: tokenizing chunks", unit="doc"):
            toks = self._tokenize_bm25(texts[docid])
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
            "chunk_ids": chunk_ids,
            "postings": postings,
            "idf": idf,
            "doc_len": doc_len,
            "avgdl": avgdl,
            "k1": self.bm25_k1,
            "b": self.bm25_b,
        }
        self.all_chunk_ids_universe = chunk_ids.copy()

        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(self._bm25, f)

    def _retrieve_topk_chunk_ids(self, claim_texts: List[str]) -> List[List[int]]:
        assert self._bm25 is not None
        chunk_ids = self._bm25["chunk_ids"]
        postings = self._bm25["postings"]
        idf = self._bm25["idf"]
        doc_len = self._bm25["doc_len"]
        avgdl = self._bm25["avgdl"]
        k1 = self._bm25["k1"]
        b = self._bm25["b"]

        out: List[List[int]] = []

        for q in tqdm(claim_texts, desc="S3 Retrieval (BM25)", unit="query"):
            toks = self._tokenize_bm25(q)
            if not toks:
                out.append([])
                continue

            scores: Dict[int, float] = {}
            for t in toks:
                plist = postings.get(t, None)
                if not plist:
                    continue
                idft = idf.get(t, 0.0)
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
        for cid in chunk_ids:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)

    # -------------------------
    # Verifier inference
    # -------------------------
    def _build_verifier_inputs(self, question: str, case_context: str, evidence_text: str) -> str:
        q = " ".join(str(question).split()).strip()
        c = " ".join(str(case_context).split()).strip()
        e = " ".join(str(evidence_text).split()).strip()
        return f"Question: {q}\nCase: {c}\nEvidence: {e}"

    def _predict_support(self, texts: List[str], desc: str = "S3 Verification (V*)") -> np.ndarray:
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
                y = self.verifier_model(enc["input_ids"], enc["attention_mask"])
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

    # -------------------------
    # Metrics
    # -------------------------
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

    def _risk_coverage(self, y: np.ndarray, yhat: np.ndarray) -> Dict[str, object]:
        p = self._sigmoid(yhat)
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

    # -------------------------
    # Tier2: Retrieval metrics vs phase1 gold
    # -------------------------
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

    def _compute_retrieval_metrics(
        self,
        df_test: pd.DataFrame,
        retrieved_topk_ids: List[List[int]],
    ) -> Dict[str, float]:
        gold_lists: List[List[int]] = []
        for s in df_test["evidence_chunk_ids_json"].astype(str).tolist():
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

    # -------------------------
    # Tier2: Faithfulness stress tests
    # -------------------------
    def _eval_preds_metrics(self, y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
        mse = self._mse(y, yhat)
        mae = self._mae(y, yhat)
        pr = self._pearson(y, yhat)
        p = self._sigmoid(yhat)
        y01 = np.clip(y, 0.0, 1.0)
        ece = self._ece(y01, p, self.ece_bins)
        brier = self._brier(y01, p)
        return {"MSE": mse, "MAE": mae, "Pearson": pr, "ECE": ece, "Brier": brier}

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

    def _predict_with_ids(self, df_sub: pd.DataFrame, ids_per_sample: List[List[int]], desc: str) -> np.ndarray:
        inputs: List[str] = []
        for i in range(len(df_sub)):
            q = df_sub.iloc[i]["question"]
            c = df_sub.iloc[i]["case_context"]
            ev = self._materialize_evidence(ids_per_sample[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))
        return self._predict_support(inputs, desc=desc)

    # -------------------------
    # Run
    # -------------------------
    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)
        df = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        # df = df.sample(frac=0.01, random_state=self.seed).reset_index(drop=True)


        if "split" in df.columns:
            df = df[df["split"].astype(str) == "test"].copy()

        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df = df.sample(n=min(len(df), self.faithfulness_max_eval), random_state=self.seed).reset_index(drop=True)

        self.chunk_map = self._load_chunk_map()
        self._build_or_load_bm25()
        self._load_verifier()

        # -------------------------
        # Tier1 baseline
        # -------------------------
        claim_texts = df["claim_text"].astype(str).tolist()
        topk_ids = self._retrieve_topk_chunk_ids(claim_texts)

        inputs: List[str] = []
        for i in range(len(df)):
            q = df.iloc[i]["question"]
            c = df.iloc[i]["case_context"]
            ev = self._materialize_evidence(topk_ids[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))

        yhat = self._predict_support(inputs, desc="S3 Verification (V*)")
        y = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        mse = self._mse(y, yhat)
        mae = self._mae(y, yhat)
        pearson = self._pearson(y, yhat)

        p = self._sigmoid(yhat)
        y01 = np.clip(y, 0.0, 1.0)
        ece = self._ece(y01, p, self.ece_bins)
        brier = self._brier(y01, p)

        rc = self._risk_coverage(y01, yhat)

        out_df = df.copy()
        out_df["s3_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        out_df["s3_pred_support_raw"] = yhat.astype(np.float32)
        out_df["s3_pred_prob_sigmoid"] = p.astype(np.float32)

        pred_path = os.path.join(self.out_dir, "s3_test_predictions.parquet")
        out_df.to_parquet(pred_path, index=False)

        tier1_report = {
            "system": "S3 (BM25 retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "retriever": {
                "bm25_index_pkl": self.bm25_index_path,
                "top_k": self.top_k,
                "k1": self.bm25_k1,
                "b": self.bm25_b,
            },
            "verifier": {
                "weights": self.verifier_weights_path,
                "max_tokens": self.verifier_max_tokens,
            },
            "metrics": {
                "MSE": mse,
                "MAE": mae,
                "Pearson": pearson,
                "ECE": ece,
                "Brier": brier,
                "SelectiveRisk": rc,
            },
            "artifacts": {"predictions_parquet": pred_path},
            "n_test": int(len(df)),
        }

        tier1_path = os.path.join(self.out_dir, "s3_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        # -------------------------
        # Tier2A: retrieval metrics vs phase1 evidence ids
        # -------------------------
        retrieval_metrics = self._compute_retrieval_metrics(df, topk_ids)
        retrieval_path = os.path.join(self.out_dir, "s3_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S3 (BM25 retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df)),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )

        # -------------------------
        # Tier2B: faithfulness stress tests
        # -------------------------
        base_metrics = self._eval_preds_metrics(y01, yhat)

        ids_top1 = self._stress_truncate(topk_ids, 1)
        ids_top3 = self._stress_truncate(topk_ids, 3)
        ids_shuffle = self._stress_shuffle_order(topk_ids)
        ids_swapped = self._stress_swap_between_samples(topk_ids)

        random_trial_metrics: List[Dict[str, float]] = []
        random_trial_preds: List[np.ndarray] = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df))
            yhat_rand = self._predict_with_ids(df, ids_rand, desc=f"S3 Faithfulness random (trial {t+1})")
            random_trial_preds.append(yhat_rand)
            random_trial_metrics.append(self._eval_preds_metrics(y01, yhat_rand))

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {"MSE": 0.0, "MAE": 0.0, "Pearson": 0.0, "ECE": 0.0, "Brier": 0.0}
            keys = ms[0].keys()
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        yhat_top1 = self._predict_with_ids(df, ids_top1, desc="S3 Faithfulness top-1")
        yhat_top3 = self._predict_with_ids(df, ids_top3, desc="S3 Faithfulness top-3")
        yhat_shuffle = self._predict_with_ids(df, ids_shuffle, desc="S3 Faithfulness shuffle-order")
        yhat_swapped = self._predict_with_ids(df, ids_swapped, desc="S3 Faithfulness swapped-evidence")

        met_top1 = self._eval_preds_metrics(y01, yhat_top1)
        met_top3 = self._eval_preds_metrics(y01, yhat_top3)
        met_shuffle = self._eval_preds_metrics(y01, yhat_shuffle)
        met_swapped = self._eval_preds_metrics(y01, yhat_swapped)
        met_random = avg_metrics(random_trial_metrics)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S3 (BM25 retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df)),
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

        faith_df = df.copy()
        faith_df["s3_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        faith_df["s3_pred_base_raw"] = yhat.astype(np.float32)
        faith_df["s3_pred_top1_raw"] = yhat_top1.astype(np.float32)
        faith_df["s3_pred_top3_raw"] = yhat_top3.astype(np.float32)
        faith_df["s3_pred_shuffle_raw"] = yhat_shuffle.astype(np.float32)
        faith_df["s3_pred_swapped_raw"] = yhat_swapped.astype(np.float32)
        if random_trial_preds:
            yhat_rand_avg = np.mean(np.stack(random_trial_preds, axis=0), axis=0).astype(np.float32)
            faith_df["s3_pred_random_avg_raw"] = yhat_rand_avg

        faith_pred_path = os.path.join(self.out_dir, "s3_tier2_faithfulness_predictions.parquet")
        faith_df.to_parquet(faith_pred_path, index=False)

        faith_path = os.path.join(self.out_dir, "s3_tier2_faithfulness_report.json")
        with open(faith_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **faith_report,
                    "artifacts": {"faithfulness_predictions_parquet": faith_pred_path},
                },
                f,
                indent=2,
            )

        summary = {
            "tier1_report": tier1_path,
            "tier2_retrieval_metrics": retrieval_path,
            "tier2_faithfulness_report": faith_path,
            "tier2_faithfulness_predictions": faith_pred_path,
        }
        summary_path = os.path.join(self.out_dir, "s3_tier12_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(tier1_report["metrics"], indent=2))
        print(json.dumps({"Retrieval": retrieval_metrics}, indent=2))
        print(json.dumps({"Faithfulness_deltas": faith_report["perturbations"]}, indent=2))
        print(f"Wrote: {pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_pred_path}")
        print(f"Wrote: {faith_path}")
        print(f"Wrote: {summary_path}")
        return summary_path





class S1Tier1CaseOnlyEvaluator:
    """
    S1 — Case-only verifier (DeBERTa-v3-large), no evidence segment.
    Loads the fixed test set:
      <base_path_evidenceData>/phase1_fixed_test_eval/mimic_test_eval.parquet
    Uses the same trained verifier weights we already have.
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,  # verifier trained_model.pth
        device: str = "cuda",
        verifier_max_tokens: int = 512,
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        out_dir: Optional[str] = None,
        seed: int = 13,
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.batch_size_verify = int(batch_size_verify)
        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet"
        )

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier1_results_s1_caseonly")
        os.makedirs(self.out_dir, exist_ok=True)

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        self.verifier_tok = None
        self.verifier_model = None

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

    class _DebertaRegressor(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.reg_head = nn.Linear(hidden, 1)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            cls = cls.to(self.reg_head.weight.dtype)
            y = self.reg_head(cls).squeeze(-1)
            return y

    def _load_verifier(self) -> None:
        vname = self.params["Verifier"]["model_name"]
        self.verifier_tok = AutoTokenizer.from_pretrained(vname, use_fast=True)
        self.verifier_model = self._DebertaRegressor(vname).to(self.device)
        self.verifier_model.eval()

        obj = torch.load(self.verifier_weights_path, map_location="cpu")
        state = obj["model_state_dict"] if (isinstance(obj, dict) and "model_state_dict" in obj) else obj
        missing, unexpected = self.verifier_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] Verifier load_state_dict strict=False. missing={len(missing)} unexpected={len(unexpected)}")

    def _build_verifier_inputs(self, question: str, case_context: str) -> str:
        q = " ".join(str(question).split()).strip()
        c = " ".join(str(case_context).split()).strip()
        return f"Question: {q}\nCase: {c}"

    def _predict_support(self, texts: List[str]) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size_verify),
                total=math.ceil(len(texts) / self.batch_size_verify),
                desc="S1 Verification (case-only)",
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
                y = self.verifier_model(enc["input_ids"], enc["attention_mask"])
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

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

    def _risk_coverage(self, y: np.ndarray, yhat: np.ndarray) -> Dict[str, object]:
        p = 1.0 / (1.0 + np.exp(-np.clip(yhat, -20, 20)))
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

    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)
        df = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        # df = df.sample(frac=0.01, random_state=self.seed).reset_index(drop=True)

        if "split" in df.columns:
            df = df[df["split"].astype(str) == "test"].copy()

        self._load_verifier()

        inputs: List[str] = []
        for i in range(len(df)):
            q = df.iloc[i]["question"]
            c = df.iloc[i]["case_context"]
            inputs.append(self._build_verifier_inputs(q, c))

        yhat = self._predict_support(inputs)
        y = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        mse = self._mse(y, yhat)
        mae = self._mae(y, yhat)
        pearson = self._pearson(y, yhat)

        p = 1.0 / (1.0 + np.exp(-np.clip(yhat, -20, 20)))
        y01 = np.clip(y, 0.0, 1.0)
        ece = self._ece(y01, p, self.ece_bins)
        brier = self._brier(y01, p)

        rc = self._risk_coverage(y01, yhat)

        out_df = df.copy()
        out_df["s1_pred_support_raw"] = yhat.astype(np.float32)
        out_df["s1_pred_prob_sigmoid"] = p.astype(np.float32)

        pred_path = os.path.join(self.out_dir, "s1_test_predictions.parquet")
        out_df.to_parquet(pred_path, index=False)

        report = {
            "system": "S1 (case-only verifier)",
            "test_parquet": self.test_parquet,
            "verifier": {
                "weights": self.verifier_weights_path,
                "max_tokens": self.verifier_max_tokens,
            },
            "metrics": {
                "MSE": mse,
                "MAE": mae,
                "Pearson": pearson,
                "ECE": ece,
                "Brier": brier,
                "SelectiveRisk": rc,
            },
            "artifacts": {
                "predictions_parquet": pred_path,
            },
            "n_test": int(len(df)),
        }

        report_path = os.path.join(self.out_dir, "s1_tier1_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(json.dumps(report["metrics"], indent=2))
        print(f"Wrote: {pred_path}")
        print(f"Wrote: {report_path}")
        return report_path







if __name__ == "__main__":

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]



    ''' S4 '''
    # runner = S4Tier12End2EndEvaluator(
    #     cfg_path=global_config_path,
    #     experiment_name=experiment_name,
    #     base_path_evidenceData=base_path_evidenceData,
    #     model_weights_path=model_weights_path,
    #     verifier_weights_path=verifier_weights_path,
    #     device="cuda",
    #     top_k=20,
    #     verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
    #     out_dir=os.path.join(base_path_evidenceData, "tier12_results_s4_end2end"),
    #     batch_size_retrieve=256,
    #     batch_size_verify=4,
    #     ece_bins=15,
    #     seed=13,
    #     # Tier2 speed knobs (optional)
    #     # faithfulness_max_eval=2000,
    #     # faithfulness_random_trials=1,
    # )
    # runner.run()



    ''' S2 '''
    # s2 = S2Tier12End2EndEvaluator(
    #     cfg_path=global_config_path,
    #     experiment_name=experiment_name,
    #     base_path_evidenceData=base_path_evidenceData,
    #     verifier_weights_path=verifier_weights_path,
    #     device="cuda",
    #     top_k=20,
    #     verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
    #     out_dir=os.path.join(base_path_evidenceData, "tier12_results_s2_end2end"),
    #     batch_size_embed_chunks=128,
    #     batch_size_retrieve=256,
    #     batch_size_verify=4,
    #     ece_bins=15,
    #     seed=13,
    #     # Tier2 speed knobs (optional)
    #     # faithfulness_max_eval=2000,
    #     # faithfulness_random_trials=1,
    # )
    # s2.run()


    ''' S3 '''
    s3 = S3Tier12End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier12_results_s3_end2end"),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        bm25_k1=1.2,
        bm25_b=0.75,
        # Tier2 speed knobs (optional)
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
    )
    s3.run()


    ''' S1 '''
    # s1 = S1Tier1CaseOnlyEvaluator(
    #     cfg_path=global_config_path,
    #     experiment_name=experiment_name,
    #     base_path_evidenceData=base_path_evidenceData,
    #     verifier_weights_path=verifier_weights_path,
    #     device="cuda",
    #     verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
    #     out_dir=os.path.join(base_path_evidenceData, "tier1_results_s1_caseonly"),
    #     batch_size_verify=4,
    #     ece_bins=15,
    #     seed=13,
    # )
    # s1.run()

