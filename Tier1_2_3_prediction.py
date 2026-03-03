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






class S4Tier123End2EndEvaluator:
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
        faithfulness_random_trials: int = 1,
        faithfulness_max_eval: Optional[int] = None,
        # Tier3 knobs (calibration fit)
        calibration_method: str = "temperature",   # "temperature" (implemented), "isotonic" (not implemented here)
        calibration_max_eval: Optional[int] = None,  # e.g. 10000 for speed; None = full valid
        calibration_opt_steps: int = 200,
        calibration_lr: float = 0.05,
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

        self.calibration_method = str(calibration_method).strip().lower()
        self.calibration_max_eval = int(calibration_max_eval) if calibration_max_eval is not None else None
        self.calibration_opt_steps = int(calibration_opt_steps)
        self.calibration_lr = float(calibration_lr)

        # Fixed protocol parquets (Phase 1)
        self.test_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet"
        )
        self.valid_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_valid_eval", "mimic_valid_eval.parquet"
        )

        # S4 retrieval artifacts (Step6)
        self.step6_dir = os.path.join(self.base_path_evidenceData, "step6_faiss_rstar")
        self.faiss_index_path = os.path.join(self.step6_dir, "faiss_Rstar.index")
        self.chunk_ids_path = os.path.join(self.step6_dir, "chunk_ids.npy")
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        # Output
        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier123_results_s4_end2end")
        os.makedirs(self.out_dir, exist_ok=True)

        self.valid_out_dir = os.path.join(self.out_dir, "valid_results")
        os.makedirs(self.valid_out_dir, exist_ok=True)

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

        # Calibration state
        self.temperature_T = 1.0  # learned on valid, applied to test

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


    # Verifier wrapper
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

    def _loads_json_list(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _normalize_claim(self, x: str) -> str:
        return " ".join(str(x).split()).strip().lower()


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
        for cid in chunk_ids:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)


    # Verifier inference
    def _build_verifier_inputs(self, question: str, case_context: str, evidence_text: str) -> str:
        q = " ".join(str(question).split()).strip()
        c = " ".join(str(case_context).split()).strip()
        e = " ".join(str(evidence_text).split()).strip()
        return f"Question: {q}\nCase: {c}\nEvidence: {e}"

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
                y = self.verifier_model(enc["input_ids"], enc["attention_mask"])
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)

    def _predict_with_ids(self, df_sub: pd.DataFrame, ids_per_sample: List[List[int]], desc: str) -> np.ndarray:
        inputs: List[str] = []
        for i in range(len(df_sub)):
            q = df_sub.iloc[i]["question"]
            c = df_sub.iloc[i]["case_context"]
            ev = self._materialize_evidence(ids_per_sample[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))
        return self._predict_support(inputs, desc=desc)


    # Metrics
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

        # average ties
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

    def _sigmoid_temp(self, x: np.ndarray, T: float) -> np.ndarray:
        T = float(max(T, 1e-6))
        return 1.0 / (1.0 + np.exp(-np.clip(x / T, -20, 20)))

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

    def _reliability_bins(self, y: np.ndarray, p: np.ndarray, bins: int) -> Dict[str, object]:
        edges = np.linspace(0.0, 1.0, bins + 1)
        out = {"bin_lo": [], "bin_hi": [], "count": [], "acc": [], "conf": []}
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            if m == 0:
                out["bin_lo"].append(lo)
                out["bin_hi"].append(hi)
                out["count"].append(0)
                out["acc"].append(None)
                out["conf"].append(None)
                continue
            out["bin_lo"].append(lo)
            out["bin_hi"].append(hi)
            out["count"].append(m)
            out["acc"].append(float(y[mask].mean()))
            out["conf"].append(float(p[mask].mean()))
        return out

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

    def _eval_preds_metrics(self, y01: np.ndarray, yhat: np.ndarray, p_for_cal: np.ndarray) -> Dict[str, float]:
        mse = self._mse(y01, yhat)
        mae = self._mae(y01, yhat)
        pr = self._pearson(y01, yhat)
        sp = self._spearman(y01, yhat)
        ece = self._ece(y01, p_for_cal, self.ece_bins)
        brier = self._brier(y01, p_for_cal)
        return {"MSE": mse, "MAE": mae, "Pearson": pr, "Spearman": sp, "ECE": ece, "Brier": brier}


    # Tier2: Retrieval metrics
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


    # Tier2: Faithfulness stress tests
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


    # Tier3: Calibration fitting on VALID
    def _fit_temperature(self, y01: np.ndarray, logits: np.ndarray) -> float:
        # Fit T > 0 minimizing BCE(y, sigmoid(logits / T))
        y_t = torch.tensor(y01.astype(np.float32), device="cpu")
        z_t = torch.tensor(logits.astype(np.float32), device="cpu")

        logT = torch.tensor([0.0], requires_grad=True, device="cpu")  # T = exp(logT)
        opt = torch.optim.Adam([logT], lr=self.calibration_lr)

        bce = torch.nn.BCELoss()

        for _ in range(self.calibration_opt_steps):
            opt.zero_grad(set_to_none=True)
            T = torch.exp(logT).clamp(min=1e-6, max=1e6)
            p = torch.sigmoid(z_t / T)
            loss = bce(p, y_t)
            loss.backward()
            opt.step()

        T_final = float(torch.exp(logT).detach().cpu().item())
        return float(max(T_final, 1e-6))

    def _compute_slice_metrics(
        self,
        df_test: pd.DataFrame,
        y01: np.ndarray,
        yhat: np.ndarray,
        p_cal: np.ndarray,
        prefix: str = "slice",
    ) -> Dict[str, object]:
        # Uses only tags already in phase1 parquet.
        # Reports MSE, ECE (calibrated), and risk@80 (calibrated confidence).
        out: Dict[str, object] = {}

        def metrics_for_mask(mask: np.ndarray) -> Dict[str, object]:
            if mask.sum() == 0:
                return {"n": 0, "MSE": None, "ECE": None, "risk_mse@80": None}
            yy = y01[mask]
            yh = yhat[mask]
            pp = p_cal[mask]
            rc = self._risk_coverage_from_prob(yy, yh, pp)
            return {
                "n": int(mask.sum()),
                "MSE": self._mse(yy, yh),
                "ECE": self._ece(yy, pp, self.ece_bins),
                "risk_mse@80": float(rc["risk_mse@80"]),
            }

        # report_length_bucket
        if "report_length_bucket" in df_test.columns:
            for v in ["short", "medium", "long"]:
                mask = (df_test["report_length_bucket"].astype(str).values == v)
                out[f"{prefix}.report_length_bucket.{v}"] = metrics_for_mask(mask)

        # device_present / uncertainty_present
        if "device_present" in df_test.columns:
            mask1 = (pd.to_numeric(df_test["device_present"], errors="coerce").fillna(0).astype(int).values == 1)
            out[f"{prefix}.device_present.1"] = metrics_for_mask(mask1)
            out[f"{prefix}.device_present.0"] = metrics_for_mask(~mask1)

        if "uncertainty_present" in df_test.columns:
            mask1 = (pd.to_numeric(df_test["uncertainty_present"], errors="coerce").fillna(0).astype(int).values == 1)
            out[f"{prefix}.uncertainty_present.1"] = metrics_for_mask(mask1)
            out[f"{prefix}.uncertainty_present.0"] = metrics_for_mask(~mask1)

        # label_vs_extractive
        if "label_vs_extractive" in df_test.columns:
            for v in ["label", "extractive"]:
                mask = (df_test["label_vs_extractive"].astype(str).values == v)
                out[f"{prefix}.label_vs_extractive.{v}"] = metrics_for_mask(mask)

        return out


    # End2End run for a split
    def _run_end2end_split(
        self,
        df: pd.DataFrame,
        split_name: str,
        out_dir: str,
        write_predictions: bool = True,
    ) -> Dict[str, object]:
        claim_texts = df["claim_text"].astype(str).tolist()
        topk_ids, _ = self._retrieve_topk_chunk_ids(claim_texts, desc=f"S4 Retrieval (R*) [{split_name}]")

        inputs: List[str] = []
        for i in range(len(df)):
            q = df.iloc[i]["question"]
            c = df.iloc[i]["case_context"]
            ev = self._materialize_evidence(topk_ids[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))

        yhat = self._predict_support(inputs, desc=f"S4 Verification (V*) [{split_name}]")
        y = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01 = np.clip(y, 0.0, 1.0)

        p_uncal = self._sigmoid(yhat)

        pred_path = None
        if write_predictions:
            out_df = df.copy()
            out_df[f"s4_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
            out_df[f"s4_pred_support_raw"] = yhat.astype(np.float32)
            out_df[f"s4_pred_prob_sigmoid"] = p_uncal.astype(np.float32)

            pred_path = os.path.join(out_dir, f"s4_{split_name}_predictions.parquet")
            out_df.to_parquet(pred_path, index=False)

        return {
            "split": split_name,
            "n": int(len(df)),
            "y01": y01,
            "yhat": yhat,
            "p_uncal": p_uncal,
            "topk_ids": topk_ids,
            "predictions_parquet": pred_path,
        }


    # Run
    def run(self) -> str:
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)
        if not os.path.exists(self.valid_parquet):
            raise FileNotFoundError(self.valid_parquet)

        df_test = pd.read_parquet(self.test_parquet)
        df_valid = pd.read_parquet(self.valid_parquet)

        # TODO; remove this
        # df_test = df_test.iloc[: int(len(df_test) * 0.005)].reset_index(drop=True)
        # df_valid = df_valid.iloc[: int(len(df_valid) * 0.005)].reset_index(drop=True)


        if "split" in df_test.columns:
            df_test = df_test[df_test["split"].astype(str) == "test"].copy()
        if "split" in df_valid.columns:
            df_valid = df_valid[df_valid["split"].astype(str) == "valid"].copy()

        # Optional speed knobs
        df_test_full = df_test.copy()
        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df_test = df_test.sample(n=min(len(df_test), self.faithfulness_max_eval), random_state=self.seed).reset_index(drop=True)

        if self.calibration_max_eval is not None and self.calibration_max_eval > 0:
            df_valid = df_valid.sample(n=min(len(df_valid), self.calibration_max_eval), random_state=self.seed).reset_index(drop=True)


        # Load artifacts/models (once)
        self.chunk_map = self._load_chunk_map()
        self._load_faiss()
        self._load_retriever()
        self._load_verifier()


        # VALID: end2end preds (for calibration fitting)
        valid_pack = self._run_end2end_split(
            df=df_valid,
            split_name="valid",
            out_dir=self.valid_out_dir,
            write_predictions=True,
        )
        y01_v = valid_pack["y01"]
        yhat_v = valid_pack["yhat"]

        # Fit calibration on valid
        calib = {"method": self.calibration_method}
        if self.calibration_method == "temperature":
            self.temperature_T = self._fit_temperature(y01_v, yhat_v)
            calib["T"] = float(self.temperature_T)
        else:
            raise RuntimeError(f"Unsupported calibration_method={self.calibration_method}. Use 'temperature'.")

        # Save valid calibration report (and uncal metrics on valid)
        p_uncal_v = valid_pack["p_uncal"]
        p_cal_v = self._sigmoid_temp(yhat_v, self.temperature_T)
        valid_metrics_uncal = self._eval_preds_metrics(y01_v, yhat_v, p_uncal_v)
        valid_metrics_cal = self._eval_preds_metrics(y01_v, yhat_v, p_cal_v)

        valid_reliab_uncal = self._reliability_bins(y01_v, p_uncal_v, self.ece_bins)
        valid_reliab_cal = self._reliability_bins(y01_v, p_cal_v, self.ece_bins)

        valid_report = {
            "system": "S4 (R* retrieval -> V*)",
            "valid_parquet": self.valid_parquet,
            "n_valid": int(valid_pack["n"]),
            "calibration_fit": calib,
            "valid_metrics_uncalibrated": valid_metrics_uncal,
            "valid_metrics_calibrated": valid_metrics_cal,
            "valid_reliability_bins_uncalibrated": valid_reliab_uncal,
            "valid_reliability_bins_calibrated": valid_reliab_cal,
            "artifacts": {
                "valid_predictions_parquet": valid_pack["predictions_parquet"],
            },
        }

        valid_report_path = os.path.join(self.valid_out_dir, "s4_valid_calibration_report.json")
        with open(valid_report_path, "w", encoding="utf-8") as f:
            json.dump(valid_report, f, indent=2)


        # TEST: Tier1 baseline + calibrated metrics
        test_pack = self._run_end2end_split(
            df=df_test,
            split_name="test",
            out_dir=self.out_dir,
            write_predictions=True,
        )
        y01_t = test_pack["y01"]
        yhat_t = test_pack["yhat"]
        p_uncal_t = test_pack["p_uncal"]
        p_cal_t = self._sigmoid_temp(yhat_t, self.temperature_T)

        # Tier1 metrics (uncalibrated and calibrated calibration metrics)
        # Note: regression metrics use raw yhat; calibration metrics use p_uncal / p_cal
        tier1_metrics_uncal = self._eval_preds_metrics(y01_t, yhat_t, p_uncal_t)
        tier1_metrics_cal = self._eval_preds_metrics(y01_t, yhat_t, p_cal_t)

        # Selective risk curves (confidence from probs)
        rc_uncal = self._risk_coverage_from_prob(y01_t, yhat_t, p_uncal_t)
        rc_cal = self._risk_coverage_from_prob(y01_t, yhat_t, p_cal_t)

        # Reliability bins to plot later
        reliab_uncal = self._reliability_bins(y01_t, p_uncal_t, self.ece_bins)
        reliab_cal = self._reliability_bins(y01_t, p_cal_t, self.ece_bins)

        # Save updated test predictions with calibrated prob too
        if test_pack["predictions_parquet"] is not None:
            df_pred = pd.read_parquet(test_pack["predictions_parquet"])
            df_pred["s4_pred_prob_calibrated"] = p_cal_t.astype(np.float32)
            pred_path_v2 = os.path.join(self.out_dir, "s4_test_predictions.parquet")
            df_pred.to_parquet(pred_path_v2, index=False)
            test_pred_path = pred_path_v2
        else:
            test_pred_path = None

        tier1_report = {
            "system": "S4 (R* retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "valid_parquet_for_calibration": self.valid_parquet,
            "calibration_fit": calib,
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
            "metrics_uncalibrated": {
                **tier1_metrics_uncal,
                "SelectiveRisk": rc_uncal,
            },
            "metrics_calibrated": {
                **tier1_metrics_cal,
                "SelectiveRisk": rc_cal,
            },
            "reliability_bins_uncalibrated": reliab_uncal,
            "reliability_bins_calibrated": reliab_cal,
            "artifacts": {"predictions_parquet": test_pred_path},
            "n_test": int(test_pack["n"]),
        }

        tier1_path = os.path.join(self.out_dir, "s4_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)


        # Tier2A: retrieval metrics (against phase1 evidence ids)
        retrieval_metrics = self._compute_retrieval_metrics(df_test, test_pack["topk_ids"])
        retrieval_path = os.path.join(self.out_dir, "s4_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S4 (R* retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(test_pack["n"]),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )


        # Tier2B: faithfulness stress tests (on df_test subset if we set faithfulness_max_eval)
        base_metrics_uncal = self._eval_preds_metrics(y01_t, yhat_t, p_uncal_t)
        base_metrics_cal = self._eval_preds_metrics(y01_t, yhat_t, p_cal_t)

        topk_ids = test_pack["topk_ids"]
        ids_top1 = self._stress_truncate(topk_ids, 1)
        ids_top3 = self._stress_truncate(topk_ids, 3)
        ids_shuffle = self._stress_shuffle_order(topk_ids)
        ids_swapped = self._stress_swap_between_samples(topk_ids)

        random_trial_metrics_uncal: List[Dict[str, float]] = []
        random_trial_metrics_cal: List[Dict[str, float]] = []
        random_trial_preds: List[np.ndarray] = []

        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df_test))
            yhat_rand = self._predict_with_ids(df_test, ids_rand, desc=f"S4 Faithfulness random (trial {t+1}) [test]")
            random_trial_preds.append(yhat_rand)

            p_uncal_r = self._sigmoid(yhat_rand)
            p_cal_r = self._sigmoid_temp(yhat_rand, self.temperature_T)

            random_trial_metrics_uncal.append(self._eval_preds_metrics(y01_t, yhat_rand, p_uncal_r))
            random_trial_metrics_cal.append(self._eval_preds_metrics(y01_t, yhat_rand, p_cal_r))

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {"MSE": 0.0, "MAE": 0.0, "Pearson": 0.0, "Spearman": 0.0, "ECE": 0.0, "Brier": 0.0}
            keys = list(ms[0].keys())
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        yhat_top1 = self._predict_with_ids(df_test, ids_top1, desc="S4 Faithfulness top-1 [test]")
        yhat_top3 = self._predict_with_ids(df_test, ids_top3, desc="S4 Faithfulness top-3 [test]")
        yhat_shuffle = self._predict_with_ids(df_test, ids_shuffle, desc="S4 Faithfulness shuffle-order [test]")
        yhat_swapped = self._predict_with_ids(df_test, ids_swapped, desc="S4 Faithfulness swapped-evidence [test]")

        def pack_metrics(yhat_x: np.ndarray) -> Dict[str, Dict[str, float]]:
            pu = self._sigmoid(yhat_x)
            pc = self._sigmoid_temp(yhat_x, self.temperature_T)
            return {
                "uncalibrated": self._eval_preds_metrics(y01_t, yhat_x, pu),
                "calibrated": self._eval_preds_metrics(y01_t, yhat_x, pc),
            }

        met_top1 = pack_metrics(yhat_top1)
        met_top3 = pack_metrics(yhat_top3)
        met_shuffle = pack_metrics(yhat_shuffle)
        met_swapped = pack_metrics(yhat_swapped)

        met_random_uncal = avg_metrics(random_trial_metrics_uncal)
        met_random_cal = avg_metrics(random_trial_metrics_cal)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S4 (R* retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(test_pack["n"]),
            "calibration_fit": calib,
            "baseline_metrics_uncalibrated": base_metrics_uncal,
            "baseline_metrics_calibrated": base_metrics_cal,
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "metrics_uncalibrated": met_random_uncal,
                    "metrics_calibrated": met_random_cal,
                    "delta_vs_base_uncalibrated": delta(met_random_uncal, base_metrics_uncal),
                    "delta_vs_base_calibrated": delta(met_random_cal, base_metrics_cal),
                },
                "swapped_evidence": {
                    "metrics_uncalibrated": met_swapped["uncalibrated"],
                    "metrics_calibrated": met_swapped["calibrated"],
                    "delta_vs_base_uncalibrated": delta(met_swapped["uncalibrated"], base_metrics_uncal),
                    "delta_vs_base_calibrated": delta(met_swapped["calibrated"], base_metrics_cal),
                },
                "truncate_top1": {
                    "metrics_uncalibrated": met_top1["uncalibrated"],
                    "metrics_calibrated": met_top1["calibrated"],
                    "delta_vs_base_uncalibrated": delta(met_top1["uncalibrated"], base_metrics_uncal),
                    "delta_vs_base_calibrated": delta(met_top1["calibrated"], base_metrics_cal),
                },
                "truncate_top3": {
                    "metrics_uncalibrated": met_top3["uncalibrated"],
                    "metrics_calibrated": met_top3["calibrated"],
                    "delta_vs_base_uncalibrated": delta(met_top3["uncalibrated"], base_metrics_uncal),
                    "delta_vs_base_calibrated": delta(met_top3["calibrated"], base_metrics_cal),
                },
                "shuffle_order": {
                    "metrics_uncalibrated": met_shuffle["uncalibrated"],
                    "metrics_calibrated": met_shuffle["calibrated"],
                    "delta_vs_base_uncalibrated": delta(met_shuffle["uncalibrated"], base_metrics_uncal),
                    "delta_vs_base_calibrated": delta(met_shuffle["calibrated"], base_metrics_cal),
                },
            },
        }

        faith_df = df_test.copy()
        faith_df["s4_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        faith_df["s4_pred_base_raw"] = yhat_t.astype(np.float32)
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
            json.dump(
                {
                    **faith_report,
                    "artifacts": {"faithfulness_predictions_parquet": faith_pred_path},
                },
                f,
                indent=2,
            )


        # Tier3C: Distribution shift slices (computed on test, no new training)
        slice_metrics = self._compute_slice_metrics(df_test, y01_t, yhat_t, p_cal_t, prefix="test_slice")
        slice_path = os.path.join(self.out_dir, "s4_tier3_distribution_slices.json")
        with open(slice_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S4 (R* retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "calibration_fit": calib,
                    "n_test": int(test_pack["n"]),
                    "metrics": slice_metrics,
                },
                f,
                indent=2,
            )


        # Final summary
        summary = {
            "valid": {
                "calibration_report": valid_report_path,
                "valid_predictions": valid_pack["predictions_parquet"],
            },
            "test": {
                "tier1_report": tier1_path,
                "tier2_retrieval_metrics": retrieval_path,
                "tier2_faithfulness_report": faith_path,
                "tier2_faithfulness_predictions": faith_pred_path,
                "tier3_distribution_slices": slice_path,
            },
        }

        summary_path = os.path.join(self.out_dir, "s4_tier123_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {valid_report_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_path}")
        print(f"Wrote: {slice_path}")
        print(f"Wrote: {summary_path}")
        return summary_path





class S2Tier123End2EndEvaluator:
    """
    S2 — Frozen dense MPNet retrieval -> V*
    - Off-the-shelf MPNet (no finetuning)
    - Builds its own FAISS + chunk embeddings inside out_dir
    - Loads fixed valid/test parquets everywhere
    - Fits calibration on valid, applies to test (temperature scaling)
    - Runs tiers 1-3 on test
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
        # Calibration knobs
        calib_max_steps: int = 500,
        calib_lr: float = 0.05,
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

        self.calib_max_steps = int(calib_max_steps)
        self.calib_lr = float(calib_lr)

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet"
        )
        self.valid_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_valid_eval", "mimic_valid_eval.parquet"
        )
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier123_results_s2_end2end")
        os.makedirs(self.out_dir, exist_ok=True)

        self.valid_out_dir = os.path.join(self.out_dir, "valid_results")
        os.makedirs(self.valid_out_dir, exist_ok=True)

        # S2-specific artifacts ONLY inside its out_dir
        self.faiss_index_path = os.path.join(self.out_dir, "faiss_mpnet_offtheshelf.index")
        self.chunk_ids_path = os.path.join(self.out_dir, "chunk_ids.npy")
        self.chunk_embeds_path = os.path.join(self.out_dir, "chunk_embeds_mpnet_offtheshelf.npy")

        self._set_determinism()

        # Loaded later
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

        # Calibration param (temperature)
        self.calib_T = None  # float

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


    # Loading
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


    # Encoding + FAISS
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
        index.add(embeds.astype(np.float32))
        faiss.write_index(index, self.faiss_index_path)

        self.faiss_index = index
        self.faiss_chunk_ids = chunk_ids
        self.all_chunk_ids_universe = chunk_ids.copy()


    # Verifier
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


    # Retrieval
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


    # Verifier inference
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


    # Metrics core
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _sigmoid_calibrated(self, x: np.ndarray, T: float) -> np.ndarray:
        T = float(max(T, 1e-6))
        return 1.0 / (1.0 + np.exp(-np.clip(x / T, -20, 20)))

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
        # Average ranks for ties, 1..n
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

    def _reliability_bins(self, y01: np.ndarray, p: np.ndarray, bins: int) -> Dict[str, object]:
        edges = np.linspace(0.0, 1.0, bins + 1)
        bin_lo, bin_hi, counts, accs, confs = [], [], [], [], []
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            bin_lo.append(lo)
            bin_hi.append(hi)
            counts.append(m)
            if m == 0:
                accs.append(None)
                confs.append(None)
            else:
                accs.append(float(y01[mask].mean()))
                confs.append(float(p[mask].mean()))
        return {"bin_lo": bin_lo, "bin_hi": bin_hi, "count": counts, "acc": accs, "conf": confs}

    def _risk_coverage(self, y01: np.ndarray, yhat: np.ndarray, p_for_conf: Optional[np.ndarray] = None) -> Dict[str, object]:
        # risk always on raw yhat; confidence from p
        if p_for_conf is None:
            p_for_conf = self._sigmoid(yhat)
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


    # Tier2: Retrieval metrics
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
        df_eval: pd.DataFrame,
        retrieved_topk_ids: List[List[int]],
    ) -> Dict[str, float]:
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


    # Tier2: Faithfulness stress tests
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

    def _eval_metrics_bundle(self, y01: np.ndarray, yhat: np.ndarray, T: Optional[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        # returns (uncalibrated, calibrated) with same mse/mae/corr; calibration affects p-based metrics
        mse = self._mse(y01, yhat)
        mae = self._mae(y01, yhat)
        pr = self._pearson(y01, yhat)
        sp = self._spearman(y01, yhat)

        p_unc = self._sigmoid(yhat)
        ece_unc = self._ece(y01, p_unc, self.ece_bins)
        brier_unc = self._brier(y01, p_unc)

        uncal = {"MSE": mse, "MAE": mae, "Pearson": pr, "Spearman": sp, "ECE": ece_unc, "Brier": brier_unc}

        if T is None:
            return uncal, dict(uncal)

        p_cal = self._sigmoid_calibrated(yhat, float(T))
        ece_cal = self._ece(y01, p_cal, self.ece_bins)
        brier_cal = self._brier(y01, p_cal)
        cal = {"MSE": mse, "MAE": mae, "Pearson": pr, "Spearman": sp, "ECE": ece_cal, "Brier": brier_cal}
        return uncal, cal


    # Calibration (temperature scaling) on valid
    def _fit_temperature(self, yhat_valid: np.ndarray, y01_valid: np.ndarray) -> float:
        # Minimize soft-label BCE: -[y log(sigmoid(z/T)) + (1-y)log(1-sigmoid(z/T))]
        z = torch.tensor(yhat_valid.astype(np.float32), device=self.device)
        y = torch.tensor(y01_valid.astype(np.float32), device=self.device)

        # parametrize T = softplus(t) + eps (keeps positive)
        t = torch.zeros((), device=self.device, requires_grad=True)
        opt = torch.optim.Adam([t], lr=self.calib_lr)

        best_T = 1.0
        best_loss = float("inf")

        for _ in range(self.calib_max_steps):
            T = torch.nn.functional.softplus(t) + 1e-6
            p = torch.sigmoid(torch.clamp(z / T, -20, 20))
            loss = -(y * torch.log(p + 1e-12) + (1.0 - y) * torch.log(1.0 - p + 1e-12)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            lv = float(loss.detach().cpu().item())
            Tv = float(T.detach().cpu().item())
            if lv < best_loss:
                best_loss = lv
                best_T = Tv

        return float(best_T)


    # Tier3: test slices
    def _slice_report(
        self,
        df_test: pd.DataFrame,
        yhat: np.ndarray,
        y01: np.ndarray,
        T: float,
    ) -> Dict[str, Dict[str, float]]:
        p_cal = self._sigmoid_calibrated(yhat, T)

        def pack(mask: np.ndarray) -> Dict[str, float]:
            idx = np.where(mask)[0]
            if idx.size == 0:
                return {"n": 0, "MSE": 0.0, "ECE": 0.0, "risk_mse@80": 0.0}
            y_s = y01[idx]
            yhat_s = yhat[idx]
            p_s = p_cal[idx]
            mse = self._mse(y_s, yhat_s)
            ece = self._ece(y_s, p_s, self.ece_bins)
            rc = self._risk_coverage(y_s, yhat_s, p_for_conf=p_s)
            return {"n": int(idx.size), "MSE": float(mse), "ECE": float(ece), "risk_mse@80": float(rc["risk_mse@80"])}

        out: Dict[str, Dict[str, float]] = {}

        # report_length_bucket: short/medium/long
        if "report_length_bucket" in df_test.columns:
            for v in ["short", "medium", "long"]:
                mask = (df_test["report_length_bucket"].astype(str).values == v)
                out[f"test_slice.report_length_bucket.{v}"] = pack(mask)

        # device_present: 0/1
        if "device_present" in df_test.columns:
            vals = df_test["device_present"].astype(int).values
            out["test_slice.device_present.1"] = pack(vals == 1)
            out["test_slice.device_present.0"] = pack(vals == 0)

        # uncertainty_present: 0/1
        if "uncertainty_present" in df_test.columns:
            vals = df_test["uncertainty_present"].astype(int).values
            out["test_slice.uncertainty_present.1"] = pack(vals == 1)
            out["test_slice.uncertainty_present.0"] = pack(vals == 0)

        # label_vs_extractive: label/extractive
        if "label_vs_extractive" in df_test.columns:
            vals = df_test["label_vs_extractive"].astype(str).values
            out["test_slice.label_vs_extractive.label"] = pack(vals == "label")
            out["test_slice.label_vs_extractive.extractive"] = pack(vals == "extractive")

        return out


    # One eval pass (retrieve -> verify) for a df
    def _run_end2end_on_df(self, df: pd.DataFrame, sys_prefix: str, desc_retrieve: str, desc_verify: str) -> Tuple[List[List[int]], np.ndarray]:
        claim_texts = df["claim_text"].astype(str).tolist()
        topk_ids = self._retrieve_topk_chunk_ids(claim_texts)

        inputs: List[str] = []
        for i in range(len(df)):
            q = df.iloc[i]["question"]
            c = df.iloc[i]["case_context"]
            ev = self._materialize_evidence(topk_ids[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))

        yhat = self._predict_support(inputs, desc=desc_verify)
        return topk_ids, yhat


    # Run
    def run(self) -> str:
        if not os.path.exists(self.valid_parquet):
            raise FileNotFoundError(self.valid_parquet)
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)

        df_valid = pd.read_parquet(self.valid_parquet)
        df_test = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        # df_test = df_test.iloc[: int(len(df_test) * 0.005)].reset_index(drop=True)
        # df_valid = df_valid.iloc[: int(len(df_valid) * 0.005)].reset_index(drop=True)


        if "split" in df_valid.columns:
            df_valid = df_valid[df_valid["split"].astype(str) == "valid"].copy()
        if "split" in df_test.columns:
            df_test = df_test[df_test["split"].astype(str) == "test"].copy()

        # Optional speed limit for faithfulness (test only)
        df_test_eval = df_test
        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df_test_eval = df_test.sample(n=min(len(df_test), self.faithfulness_max_eval), random_state=self.seed).reset_index(drop=True)


        # Load artifacts/models (end-to-end, no assumptions)
        self.chunk_map = self._load_chunk_map()
        self._load_offtheshelf_mpnet()
        self._build_or_load_faiss_s2()
        self._load_verifier()


        # VALID: run end-to-end for calibration fitting + valid reporting
        topk_valid, yhat_valid = self._run_end2end_on_df(
            df_valid,
            sys_prefix="s2",
            desc_retrieve="S2 Retrieval (MPNet off-the-shelf) [valid]",
            desc_verify="S2 Verification (V*) [valid]",
        )
        y_valid = pd.to_numeric(df_valid["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01_valid = np.clip(y_valid, 0.0, 1.0)

        # fit temperature on valid
        T = self._fit_temperature(yhat_valid, y01_valid)
        self.calib_T = float(T)

        # valid metrics
        valid_unc, valid_cal = self._eval_metrics_bundle(y01_valid, yhat_valid, T=self.calib_T)
        p_valid_unc = self._sigmoid(yhat_valid)
        p_valid_cal = self._sigmoid_calibrated(yhat_valid, self.calib_T)
        valid_bins_unc = self._reliability_bins(y01_valid, p_valid_unc, self.ece_bins)
        valid_bins_cal = self._reliability_bins(y01_valid, p_valid_cal, self.ece_bins)

        # save valid predictions parquet
        valid_out_df = df_valid.copy()
        valid_out_df["s2_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_valid]
        valid_out_df["s2_pred_support_raw"] = yhat_valid.astype(np.float32)
        valid_out_df["s2_pred_prob_sigmoid_uncal"] = p_valid_unc.astype(np.float32)
        valid_out_df["s2_pred_prob_sigmoid_cal"] = p_valid_cal.astype(np.float32)

        valid_pred_path = os.path.join(self.valid_out_dir, "s2_valid_predictions.parquet")
        valid_out_df.to_parquet(valid_pred_path, index=False)

        valid_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
            "valid_parquet": self.valid_parquet,
            "n_valid": int(len(df_valid)),
            "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
            "valid_metrics_uncalibrated": valid_unc,
            "valid_metrics_calibrated": valid_cal,
            "valid_reliability_bins_uncalibrated": valid_bins_unc,
            "valid_reliability_bins_calibrated": valid_bins_cal,
            "artifacts": {"valid_predictions_parquet": valid_pred_path},
        }
        valid_report_path = os.path.join(self.valid_out_dir, "s2_valid_report.json")
        with open(valid_report_path, "w", encoding="utf-8") as f:
            json.dump(valid_report, f, indent=2)


        # TEST: Tier 1 (uncal + cal, with selective risk + bins)
        topk_test, yhat_test = self._run_end2end_on_df(
            df_test_eval,
            sys_prefix="s2",
            desc_retrieve="S2 Retrieval (MPNet off-the-shelf)",
            desc_verify="S2 Verification (V*)",
        )
        y_test = pd.to_numeric(df_test_eval["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01_test = np.clip(y_test, 0.0, 1.0)

        test_unc, test_cal = self._eval_metrics_bundle(y01_test, yhat_test, T=self.calib_T)

        p_test_unc = self._sigmoid(yhat_test)
        p_test_cal = self._sigmoid_calibrated(yhat_test, self.calib_T)

        rc_unc = self._risk_coverage(y01_test, yhat_test, p_for_conf=p_test_unc)
        rc_cal = self._risk_coverage(y01_test, yhat_test, p_for_conf=p_test_cal)

        bins_unc = self._reliability_bins(y01_test, p_test_unc, self.ece_bins)
        bins_cal = self._reliability_bins(y01_test, p_test_cal, self.ece_bins)

        # save test predictions parquet
        test_out_df = df_test_eval.copy()
        test_out_df["s2_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_test]
        test_out_df["s2_pred_support_raw"] = yhat_test.astype(np.float32)
        test_out_df["s2_pred_prob_sigmoid_uncal"] = p_test_unc.astype(np.float32)
        test_out_df["s2_pred_prob_sigmoid_cal"] = p_test_cal.astype(np.float32)

        test_pred_path = os.path.join(self.out_dir, "s2_test_predictions.parquet")
        test_out_df.to_parquet(test_pred_path, index=False)

        tier1_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "valid_parquet_for_calibration": self.valid_parquet,
            "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
            "retriever": {
                "mpnet_model": self.params["retriever"]["model_name"],
                "built_faiss_index": self.faiss_index_path,
                "chunk_ids_npy": self.chunk_ids_path,
                "chunk_embeds_npy": self.chunk_embeds_path,
                "top_k": self.top_k,
            },
            "verifier": {"weights": self.verifier_weights_path, "max_tokens": self.verifier_max_tokens},
            "metrics_uncalibrated": {**test_unc, "SelectiveRisk": rc_unc},
            "metrics_calibrated": {**test_cal, "SelectiveRisk": rc_cal},
            "reliability_bins_uncalibrated": bins_unc,
            "reliability_bins_calibrated": bins_cal,
            "artifacts": {"predictions_parquet": test_pred_path},
            "n_test": int(len(df_test_eval)),
        }

        tier1_path = os.path.join(self.out_dir, "s2_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)


        # TEST: Tier 2A retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics(df_test_eval, topk_test)
        retrieval_path = os.path.join(self.out_dir, "s2_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df_test_eval)),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )


        # TEST: Tier 2B faithfulness stress tests (uncal + cal)
        base_unc, base_cal = self._eval_metrics_bundle(y01_test, yhat_test, T=self.calib_T)

        ids_top1 = self._stress_truncate(topk_test, 1)
        ids_top3 = self._stress_truncate(topk_test, 3)
        ids_shuffle = self._stress_shuffle_order(topk_test)
        ids_swapped = self._stress_swap_between_samples(topk_test)

        # random trials
        rand_unc_ms: List[Dict[str, float]] = []
        rand_cal_ms: List[Dict[str, float]] = []
        rand_preds: List[np.ndarray] = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df_test_eval))
            yhat_rand = self._predict_with_ids(df_test_eval, ids_rand, desc=f"S2 Faithfulness random (trial {t+1})")
            rand_preds.append(yhat_rand)
            u, c = self._eval_metrics_bundle(y01_test, yhat_rand, T=self.calib_T)
            rand_unc_ms.append(u)
            rand_cal_ms.append(c)

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {"MSE": 0.0, "MAE": 0.0, "Pearson": 0.0, "Spearman": 0.0, "ECE": 0.0, "Brier": 0.0}
            keys = ms[0].keys()
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        # one-shot perturbations
        yhat_top1 = self._predict_with_ids(df_test_eval, ids_top1, desc="S2 Faithfulness top-1")
        yhat_top3 = self._predict_with_ids(df_test_eval, ids_top3, desc="S2 Faithfulness top-3")
        yhat_shuffle = self._predict_with_ids(df_test_eval, ids_shuffle, desc="S2 Faithfulness shuffle-order")
        yhat_swapped = self._predict_with_ids(df_test_eval, ids_swapped, desc="S2 Faithfulness swapped-evidence")

        top1_unc, top1_cal = self._eval_metrics_bundle(y01_test, yhat_top1, T=self.calib_T)
        top3_unc, top3_cal = self._eval_metrics_bundle(y01_test, yhat_top3, T=self.calib_T)
        shuf_unc, shuf_cal = self._eval_metrics_bundle(y01_test, yhat_shuffle, T=self.calib_T)
        swap_unc, swap_cal = self._eval_metrics_bundle(y01_test, yhat_swapped, T=self.calib_T)

        rand_unc = avg_metrics(rand_unc_ms)
        rand_cal = avg_metrics(rand_cal_ms)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df_test_eval)),
            "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
            "baseline_metrics_uncalibrated": base_unc,
            "baseline_metrics_calibrated": base_cal,
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "metrics_uncalibrated": rand_unc,
                    "metrics_calibrated": rand_cal,
                    "delta_vs_base_uncalibrated": delta(rand_unc, base_unc),
                    "delta_vs_base_calibrated": delta(rand_cal, base_cal),
                },
                "swapped_evidence": {
                    "metrics_uncalibrated": swap_unc,
                    "metrics_calibrated": swap_cal,
                    "delta_vs_base_uncalibrated": delta(swap_unc, base_unc),
                    "delta_vs_base_calibrated": delta(swap_cal, base_cal),
                },
                "truncate_top1": {
                    "metrics_uncalibrated": top1_unc,
                    "metrics_calibrated": top1_cal,
                    "delta_vs_base_uncalibrated": delta(top1_unc, base_unc),
                    "delta_vs_base_calibrated": delta(top1_cal, base_cal),
                },
                "truncate_top3": {
                    "metrics_uncalibrated": top3_unc,
                    "metrics_calibrated": top3_cal,
                    "delta_vs_base_uncalibrated": delta(top3_unc, base_unc),
                    "delta_vs_base_calibrated": delta(top3_cal, base_cal),
                },
                "shuffle_order": {
                    "metrics_uncalibrated": shuf_unc,
                    "metrics_calibrated": shuf_cal,
                    "delta_vs_base_uncalibrated": delta(shuf_unc, base_unc),
                    "delta_vs_base_calibrated": delta(shuf_cal, base_cal),
                },
            },
        }

        # Save per-row faithfulness preds
        faith_df = df_test_eval.copy()
        faith_df["s2_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_test]
        faith_df["s2_pred_base_raw"] = yhat_test.astype(np.float32)
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
            json.dump(
                {**faith_report, "artifacts": {"faithfulness_predictions_parquet": faith_pred_path}},
                f,
                indent=2,
            )


        # TEST: Tier 3 slices (on the evaluated df_test_eval)
        tier3_metrics = self._slice_report(df_test_eval, yhat_test, y01_test, T=float(self.calib_T))
        tier3_path = os.path.join(self.out_dir, "s2_tier3_slices_report.json")
        with open(tier3_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S2 (MPNet off-the-shelf retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
                    "n_test": int(len(df_test_eval)),
                    "metrics": tier3_metrics,
                },
                f,
                indent=2,
            )

        # -------------------------
        # Summary
        # -------------------------
        summary = {
            "valid_report": valid_report_path,
            "tier1_report": tier1_path,
            "tier2_retrieval_metrics": retrieval_path,
            "tier2_faithfulness_report": faith_path,
            "tier2_faithfulness_predictions": faith_pred_path,
            "tier3_slices_report": tier3_path,
        }
        summary_path = os.path.join(self.out_dir, "s2_tier123_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[S2] calib_T={self.calib_T:.6f}")
        print(json.dumps(tier1_report["metrics_uncalibrated"], indent=2))
        print(json.dumps(tier1_report["metrics_calibrated"], indent=2))
        print(json.dumps({"Retrieval": retrieval_metrics}, indent=2))
        print(f"Wrote: {valid_pred_path}")
        print(f"Wrote: {valid_report_path}")
        print(f"Wrote: {test_pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_pred_path}")
        print(f"Wrote: {faith_path}")
        print(f"Wrote: {tier3_path}")
        print(f"Wrote: {summary_path}")
        return summary_path




class S3Tier123End2EndEvaluator:
    """
    S3 — BM25 retrieval -> V*
    - Lexical BM25 baseline (no FAISS)
    - Stores BM25 index ONLY inside out_dir
    - Loads fixed valid/test parquets everywhere
    - Fits calibration on valid, applies to test (temperature scaling)
    - Runs tiers 1-3 on test
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
        # Calibration knobs
        calib_max_steps: int = 500,
        calib_lr: float = 0.05,
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

        self.calib_max_steps = int(calib_max_steps)
        self.calib_lr = float(calib_lr)

        self.test_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet"
        )
        self.valid_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_valid_eval", "mimic_valid_eval.parquet"
        )
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier123_results_s3_end2end")
        os.makedirs(self.out_dir, exist_ok=True)

        self.valid_out_dir = os.path.join(self.out_dir, "valid_results")
        os.makedirs(self.valid_out_dir, exist_ok=True)

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

        # Calibration param
        self.calib_T = None

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


    # Loading
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


    # BM25
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

    def _retrieve_topk_chunk_ids(self, claim_texts: List[str], desc: str) -> List[List[int]]:
        assert self._bm25 is not None
        chunk_ids = self._bm25["chunk_ids"]
        postings = self._bm25["postings"]
        idf = self._bm25["idf"]
        doc_len = self._bm25["doc_len"]
        avgdl = self._bm25["avgdl"]
        k1 = self._bm25["k1"]
        b = self._bm25["b"]

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


    # Verifier inference
    def _build_verifier_inputs(self, question: str, case_context: str, evidence_text: str) -> str:
        q = " ".join(str(question).split()).strip()
        c = " ".join(str(case_context).split()).strip()
        e = " ".join(str(evidence_text).split()).strip()
        return f"Question: {q}\nCase: {c}\nEvidence: {e}"

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
                y = self.verifier_model(enc["input_ids"], enc["attention_mask"])
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)


    # Metrics + calibration utilities (match S2/S4)
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _sigmoid_calibrated(self, x: np.ndarray, T: float) -> np.ndarray:
        T = float(max(T, 1e-6))
        return 1.0 / (1.0 + np.exp(-np.clip(x / T, -20, 20)))

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

    def _reliability_bins(self, y01: np.ndarray, p: np.ndarray, bins: int) -> Dict[str, object]:
        edges = np.linspace(0.0, 1.0, bins + 1)
        bin_lo, bin_hi, counts, accs, confs = [], [], [], [], []
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            bin_lo.append(lo)
            bin_hi.append(hi)
            counts.append(m)
            if m == 0:
                accs.append(None)
                confs.append(None)
            else:
                accs.append(float(y01[mask].mean()))
                confs.append(float(p[mask].mean()))
        return {"bin_lo": bin_lo, "bin_hi": bin_hi, "count": counts, "acc": accs, "conf": confs}

    def _risk_coverage(self, y01: np.ndarray, yhat: np.ndarray, p_for_conf: Optional[np.ndarray] = None) -> Dict[str, object]:
        if p_for_conf is None:
            p_for_conf = self._sigmoid(yhat)
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

    def _eval_metrics_bundle(self, y01: np.ndarray, yhat: np.ndarray, T: Optional[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        mse = self._mse(y01, yhat)
        mae = self._mae(y01, yhat)
        pr = self._pearson(y01, yhat)
        sp = self._spearman(y01, yhat)

        p_unc = self._sigmoid(yhat)
        ece_unc = self._ece(y01, p_unc, self.ece_bins)
        brier_unc = self._brier(y01, p_unc)
        uncal = {"MSE": mse, "MAE": mae, "Pearson": pr, "Spearman": sp, "ECE": ece_unc, "Brier": brier_unc}

        if T is None:
            return uncal, dict(uncal)

        p_cal = self._sigmoid_calibrated(yhat, float(T))
        ece_cal = self._ece(y01, p_cal, self.ece_bins)
        brier_cal = self._brier(y01, p_cal)
        cal = {"MSE": mse, "MAE": mae, "Pearson": pr, "Spearman": sp, "ECE": ece_cal, "Brier": brier_cal}
        return uncal, cal

    def _fit_temperature(self, yhat_valid: np.ndarray, y01_valid: np.ndarray) -> float:
        z = torch.tensor(yhat_valid.astype(np.float32), device=self.device)
        y = torch.tensor(y01_valid.astype(np.float32), device=self.device)

        t = torch.zeros((), device=self.device, requires_grad=True)
        opt = torch.optim.Adam([t], lr=float(self.calib_lr))

        best_T = 1.0
        best_loss = float("inf")

        for _ in range(int(self.calib_max_steps)):
            T = torch.nn.functional.softplus(t) + 1e-6
            p = torch.sigmoid(torch.clamp(z / T, -20, 20))
            loss = -(y * torch.log(p + 1e-12) + (1.0 - y) * torch.log(1.0 - p + 1e-12)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            lv = float(loss.detach().cpu().item())
            Tv = float(T.detach().cpu().item())
            if lv < best_loss:
                best_loss = lv
                best_T = Tv

        return float(best_T)


    # Tier2: Retrieval metrics
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


    # Tier2: Faithfulness stress tests helpers
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
            ev = self._materialize_evidence(ids_per_sample[i])
            inputs.append(self._build_verifier_inputs(q, c, ev))
        return self._predict_support(inputs, desc=desc)


    # Tier3: test slices
    def _slice_report(self, df_test: pd.DataFrame, yhat: np.ndarray, y01: np.ndarray, T: float) -> Dict[str, Dict[str, float]]:
        p_cal = self._sigmoid_calibrated(yhat, T)

        def pack(mask: np.ndarray) -> Dict[str, float]:
            idx = np.where(mask)[0]
            if idx.size == 0:
                return {"n": 0, "MSE": 0.0, "ECE": 0.0, "risk_mse@80": 0.0}
            y_s = y01[idx]
            yhat_s = yhat[idx]
            p_s = p_cal[idx]
            mse = self._mse(y_s, yhat_s)
            ece = self._ece(y_s, p_s, self.ece_bins)
            rc = self._risk_coverage(y_s, yhat_s, p_for_conf=p_s)
            return {"n": int(idx.size), "MSE": float(mse), "ECE": float(ece), "risk_mse@80": float(rc["risk_mse@80"])}

        out: Dict[str, Dict[str, float]] = {}

        if "report_length_bucket" in df_test.columns:
            for v in ["short", "medium", "long"]:
                mask = (df_test["report_length_bucket"].astype(str).values == v)
                out[f"test_slice.report_length_bucket.{v}"] = pack(mask)

        if "device_present" in df_test.columns:
            vals = df_test["device_present"].astype(int).values
            out["test_slice.device_present.1"] = pack(vals == 1)
            out["test_slice.device_present.0"] = pack(vals == 0)

        if "uncertainty_present" in df_test.columns:
            vals = df_test["uncertainty_present"].astype(int).values
            out["test_slice.uncertainty_present.1"] = pack(vals == 1)
            out["test_slice.uncertainty_present.0"] = pack(vals == 0)

        if "label_vs_extractive" in df_test.columns:
            vals = df_test["label_vs_extractive"].astype(str).values
            out["test_slice.label_vs_extractive.label"] = pack(vals == "label")
            out["test_slice.label_vs_extractive.extractive"] = pack(vals == "extractive")

        return out


    # Run
    def run(self) -> str:
        if not os.path.exists(self.valid_parquet):
            raise FileNotFoundError(self.valid_parquet)
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)

        df_valid = pd.read_parquet(self.valid_parquet)
        df_test = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        # df_test = df_test.iloc[: int(len(df_test) * 0.005)].reset_index(drop=True)
        # df_valid = df_valid.iloc[: int(len(df_valid) * 0.005)].reset_index(drop=True)


        if "split" in df_valid.columns:
            df_valid = df_valid[df_valid["split"].astype(str) == "valid"].copy()
        if "split" in df_test.columns:
            df_test = df_test[df_test["split"].astype(str) == "test"].copy()

        df_test_eval = df_test
        if self.faithfulness_max_eval is not None and self.faithfulness_max_eval > 0:
            df_test_eval = df_test.sample(n=min(len(df_test), self.faithfulness_max_eval), random_state=self.seed).reset_index(drop=True)

        # load artifacts/models
        self.chunk_map = self._load_chunk_map()
        self._build_or_load_bm25()
        self._load_verifier()


        # VALID: run end-to-end, fit temperature, save valid report
        claim_valid = df_valid["claim_text"].astype(str).tolist()
        topk_valid = self._retrieve_topk_chunk_ids(claim_valid, desc="S3 Retrieval (BM25) [valid]")

        inputs_valid: List[str] = []
        for i in range(len(df_valid)):
            q = df_valid.iloc[i]["question"]
            c = df_valid.iloc[i]["case_context"]
            ev = self._materialize_evidence(topk_valid[i])
            inputs_valid.append(self._build_verifier_inputs(q, c, ev))

        yhat_valid = self._predict_support(inputs_valid, desc="S3 Verification (V*) [valid]")
        y_valid = pd.to_numeric(df_valid["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01_valid = np.clip(y_valid, 0.0, 1.0)

        self.calib_T = float(self._fit_temperature(yhat_valid, y01_valid))

        valid_unc, valid_cal = self._eval_metrics_bundle(y01_valid, yhat_valid, T=self.calib_T)
        p_valid_unc = self._sigmoid(yhat_valid)
        p_valid_cal = self._sigmoid_calibrated(yhat_valid, self.calib_T)
        valid_bins_unc = self._reliability_bins(y01_valid, p_valid_unc, self.ece_bins)
        valid_bins_cal = self._reliability_bins(y01_valid, p_valid_cal, self.ece_bins)

        valid_out_df = df_valid.copy()
        valid_out_df["s3_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_valid]
        valid_out_df["s3_pred_support_raw"] = yhat_valid.astype(np.float32)
        valid_out_df["s3_pred_prob_sigmoid_uncal"] = p_valid_unc.astype(np.float32)
        valid_out_df["s3_pred_prob_sigmoid_cal"] = p_valid_cal.astype(np.float32)

        valid_pred_path = os.path.join(self.valid_out_dir, "s3_valid_predictions.parquet")
        valid_out_df.to_parquet(valid_pred_path, index=False)

        valid_report = {
            "system": "S3 (BM25 retrieval -> V*)",
            "valid_parquet": self.valid_parquet,
            "n_valid": int(len(df_valid)),
            "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
            "valid_metrics_uncalibrated": valid_unc,
            "valid_metrics_calibrated": valid_cal,
            "valid_reliability_bins_uncalibrated": valid_bins_unc,
            "valid_reliability_bins_calibrated": valid_bins_cal,
            "artifacts": {"valid_predictions_parquet": valid_pred_path},
        }
        valid_report_path = os.path.join(self.valid_out_dir, "s3_valid_report.json")
        with open(valid_report_path, "w", encoding="utf-8") as f:
            json.dump(valid_report, f, indent=2)


        # TEST: Tier 1 (uncal + cal)
        claim_test = df_test_eval["claim_text"].astype(str).tolist()
        topk_test = self._retrieve_topk_chunk_ids(claim_test, desc="S3 Retrieval (BM25)")

        inputs_test: List[str] = []
        for i in range(len(df_test_eval)):
            q = df_test_eval.iloc[i]["question"]
            c = df_test_eval.iloc[i]["case_context"]
            ev = self._materialize_evidence(topk_test[i])
            inputs_test.append(self._build_verifier_inputs(q, c, ev))

        yhat_test = self._predict_support(inputs_test, desc="S3 Verification (V*)")
        y_test = pd.to_numeric(df_test_eval["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01_test = np.clip(y_test, 0.0, 1.0)

        test_unc, test_cal = self._eval_metrics_bundle(y01_test, yhat_test, T=self.calib_T)

        p_test_unc = self._sigmoid(yhat_test)
        p_test_cal = self._sigmoid_calibrated(yhat_test, self.calib_T)

        rc_unc = self._risk_coverage(y01_test, yhat_test, p_for_conf=p_test_unc)
        rc_cal = self._risk_coverage(y01_test, yhat_test, p_for_conf=p_test_cal)

        bins_unc = self._reliability_bins(y01_test, p_test_unc, self.ece_bins)
        bins_cal = self._reliability_bins(y01_test, p_test_cal, self.ece_bins)

        test_out_df = df_test_eval.copy()
        test_out_df["s3_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_test]
        test_out_df["s3_pred_support_raw"] = yhat_test.astype(np.float32)
        test_out_df["s3_pred_prob_sigmoid_uncal"] = p_test_unc.astype(np.float32)
        test_out_df["s3_pred_prob_sigmoid_cal"] = p_test_cal.astype(np.float32)

        test_pred_path = os.path.join(self.out_dir, "s3_test_predictions.parquet")
        test_out_df.to_parquet(test_pred_path, index=False)

        tier1_report = {
            "system": "S3 (BM25 retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "valid_parquet_for_calibration": self.valid_parquet,
            "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
            "retriever": {
                "bm25_index_pkl": self.bm25_index_path,
                "top_k": self.top_k,
                "k1": self.bm25_k1,
                "b": self.bm25_b,
            },
            "verifier": {"weights": self.verifier_weights_path, "max_tokens": self.verifier_max_tokens},
            "metrics_uncalibrated": {**test_unc, "SelectiveRisk": rc_unc},
            "metrics_calibrated": {**test_cal, "SelectiveRisk": rc_cal},
            "reliability_bins_uncalibrated": bins_unc,
            "reliability_bins_calibrated": bins_cal,
            "artifacts": {"predictions_parquet": test_pred_path},
            "n_test": int(len(df_test_eval)),
        }
        tier1_path = os.path.join(self.out_dir, "s3_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)


        # TEST: Tier 2A retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics(df_test_eval, topk_test)
        retrieval_path = os.path.join(self.out_dir, "s3_tier2_retrieval_metrics.json")
        with open(retrieval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S3 (BM25 retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "n_test": int(len(df_test_eval)),
                    "recall_ks": list(self.recall_ks),
                    "mrr_k": self.mrr_k,
                    "metrics": retrieval_metrics,
                },
                f,
                indent=2,
            )


        # TEST: Tier 2B faithfulness (uncal + cal)
        base_unc, base_cal = self._eval_metrics_bundle(y01_test, yhat_test, T=self.calib_T)

        ids_top1 = self._stress_truncate(topk_test, 1)
        ids_top3 = self._stress_truncate(topk_test, 3)
        ids_shuffle = self._stress_shuffle_order(topk_test)
        ids_swapped = self._stress_swap_between_samples(topk_test)

        rand_unc_ms: List[Dict[str, float]] = []
        rand_cal_ms: List[Dict[str, float]] = []
        rand_preds: List[np.ndarray] = []

        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(df_test_eval))
            yhat_rand = self._predict_with_ids(df_test_eval, ids_rand, desc=f"S3 Faithfulness random (trial {t+1})")
            rand_preds.append(yhat_rand)
            u, c = self._eval_metrics_bundle(y01_test, yhat_rand, T=self.calib_T)
            rand_unc_ms.append(u)
            rand_cal_ms.append(c)

        def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
            if not ms:
                return {"MSE": 0.0, "MAE": 0.0, "Pearson": 0.0, "Spearman": 0.0, "ECE": 0.0, "Brier": 0.0}
            keys = ms[0].keys()
            return {k: float(np.mean([m[k] for m in ms])) for k in keys}

        yhat_top1 = self._predict_with_ids(df_test_eval, ids_top1, desc="S3 Faithfulness top-1")
        yhat_top3 = self._predict_with_ids(df_test_eval, ids_top3, desc="S3 Faithfulness top-3")
        yhat_shuffle = self._predict_with_ids(df_test_eval, ids_shuffle, desc="S3 Faithfulness shuffle-order")
        yhat_swapped = self._predict_with_ids(df_test_eval, ids_swapped, desc="S3 Faithfulness swapped-evidence")

        top1_unc, top1_cal = self._eval_metrics_bundle(y01_test, yhat_top1, T=self.calib_T)
        top3_unc, top3_cal = self._eval_metrics_bundle(y01_test, yhat_top3, T=self.calib_T)
        shuf_unc, shuf_cal = self._eval_metrics_bundle(y01_test, yhat_shuffle, T=self.calib_T)
        swap_unc, swap_cal = self._eval_metrics_bundle(y01_test, yhat_swapped, T=self.calib_T)

        rand_unc = avg_metrics(rand_unc_ms)
        rand_cal = avg_metrics(rand_cal_ms)

        def delta(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
            return {f"Δ{k}": float(m[k] - base[k]) for k in base.keys()}

        faith_report = {
            "system": "S3 (BM25 retrieval -> V*)",
            "test_parquet": self.test_parquet,
            "n_test": int(len(df_test_eval)),
            "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
            "baseline_metrics_uncalibrated": base_unc,
            "baseline_metrics_calibrated": base_cal,
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "metrics_uncalibrated": rand_unc,
                    "metrics_calibrated": rand_cal,
                    "delta_vs_base_uncalibrated": delta(rand_unc, base_unc),
                    "delta_vs_base_calibrated": delta(rand_cal, base_cal),
                },
                "swapped_evidence": {
                    "metrics_uncalibrated": swap_unc,
                    "metrics_calibrated": swap_cal,
                    "delta_vs_base_uncalibrated": delta(swap_unc, base_unc),
                    "delta_vs_base_calibrated": delta(swap_cal, base_cal),
                },
                "truncate_top1": {
                    "metrics_uncalibrated": top1_unc,
                    "metrics_calibrated": top1_cal,
                    "delta_vs_base_uncalibrated": delta(top1_unc, base_unc),
                    "delta_vs_base_calibrated": delta(top1_cal, base_cal),
                },
                "truncate_top3": {
                    "metrics_uncalibrated": top3_unc,
                    "metrics_calibrated": top3_cal,
                    "delta_vs_base_uncalibrated": delta(top3_unc, base_unc),
                    "delta_vs_base_calibrated": delta(top3_cal, base_cal),
                },
                "shuffle_order": {
                    "metrics_uncalibrated": shuf_unc,
                    "metrics_calibrated": shuf_cal,
                    "delta_vs_base_uncalibrated": delta(shuf_unc, base_unc),
                    "delta_vs_base_calibrated": delta(shuf_cal, base_cal),
                },
            },
        }

        faith_df = df_test_eval.copy()
        faith_df["s3_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_test]
        faith_df["s3_pred_base_raw"] = yhat_test.astype(np.float32)
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
            json.dump({**faith_report, "artifacts": {"faithfulness_predictions_parquet": faith_pred_path}}, f, indent=2)


        # TEST: Tier 3 slices
        tier3_metrics = self._slice_report(df_test_eval, yhat_test, y01_test, T=float(self.calib_T))
        tier3_path = os.path.join(self.out_dir, "s3_tier3_slices_report.json")
        with open(tier3_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S3 (BM25 retrieval -> V*)",
                    "test_parquet": self.test_parquet,
                    "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
                    "n_test": int(len(df_test_eval)),
                    "metrics": tier3_metrics,
                },
                f,
                indent=2,
            )


        # Summary
        summary = {
            "valid_report": valid_report_path,
            "tier1_report": tier1_path,
            "tier2_retrieval_metrics": retrieval_path,
            "tier2_faithfulness_report": faith_path,
            "tier2_faithfulness_predictions": faith_pred_path,
            "tier3_slices_report": tier3_path,
        }
        summary_path = os.path.join(self.out_dir, "s3_tier123_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[S3] calib_T={self.calib_T:.6f}")
        print(json.dumps(tier1_report["metrics_uncalibrated"], indent=2))
        print(json.dumps(tier1_report["metrics_calibrated"], indent=2))
        print(json.dumps({"Retrieval": retrieval_metrics}, indent=2))
        print(f"Wrote: {valid_pred_path}")
        print(f"Wrote: {valid_report_path}")
        print(f"Wrote: {test_pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {retrieval_path}")
        print(f"Wrote: {faith_pred_path}")
        print(f"Wrote: {faith_path}")
        print(f"Wrote: {tier3_path}")
        print(f"Wrote: {summary_path}")
        return summary_path





class S1Tier123CaseOnlyEvaluator:
    """
    S1 — Case-only verifier V* (DeBERTa), no evidence segment.

    Adds everything that makes sense for S1:
      - Valid: run predictions, fit calibration (temperature) on valid, save valid report + predictions
      - Test Tier1: regression + calibration + selective prediction (uncal + calibrated), reliability bins
      - Test Tier3: distribution-shift slices on test (calibrated ECE + risk@80 + MSE)
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
        # Calibration knobs
        calib_max_steps: int = 500,
        calib_lr: float = 0.05,
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

        self.calib_max_steps = int(calib_max_steps)
        self.calib_lr = float(calib_lr)

        self.valid_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_valid_eval", "mimic_valid_eval.parquet"
        )
        self.test_parquet = os.path.join(
            self.base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet"
        )

        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "tier123_results_s1_end2end")
        os.makedirs(self.out_dir, exist_ok=True)

        self.valid_out_dir = os.path.join(self.out_dir, "valid_results")
        os.makedirs(self.valid_out_dir, exist_ok=True)

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        self.verifier_tok = None
        self.verifier_model = None
        self.calib_T = None

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


    # Text building
    def _build_verifier_inputs(self, question: str, case_context: str) -> str:
        q = " ".join(str(question).split()).strip()
        c = " ".join(str(case_context).split()).strip()
        return f"Question: {q}\nCase: {c}"


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
                y = self.verifier_model(enc["input_ids"], enc["attention_mask"])
                preds.append(y.detach().float().cpu().numpy())

        out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out.astype(np.float32)


    # Metrics + calibration
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _sigmoid_calibrated(self, x: np.ndarray, T: float) -> np.ndarray:
        T = float(max(T, 1e-6))
        return 1.0 / (1.0 + np.exp(-np.clip(x / T, -20, 20)))

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

    def _reliability_bins(self, y01: np.ndarray, p: np.ndarray, bins: int) -> Dict[str, object]:
        edges = np.linspace(0.0, 1.0, bins + 1)
        bin_lo, bin_hi, counts, accs, confs = [], [], [], [], []
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            bin_lo.append(lo)
            bin_hi.append(hi)
            counts.append(m)
            if m == 0:
                accs.append(None)
                confs.append(None)
            else:
                accs.append(float(y01[mask].mean()))
                confs.append(float(p[mask].mean()))
        return {"bin_lo": bin_lo, "bin_hi": bin_hi, "count": counts, "acc": accs, "conf": confs}

    def _risk_coverage(self, y01: np.ndarray, yhat: np.ndarray, p_for_conf: Optional[np.ndarray] = None) -> Dict[str, object]:
        if p_for_conf is None:
            p_for_conf = self._sigmoid(yhat)
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

    def _eval_metrics_bundle(self, y01: np.ndarray, yhat: np.ndarray, T: Optional[float]) -> (Dict[str, float], Dict[str, float]):
        mse = self._mse(y01, yhat)
        mae = self._mae(y01, yhat)
        pr = self._pearson(y01, yhat)
        sp = self._spearman(y01, yhat)

        p_unc = self._sigmoid(yhat)
        ece_unc = self._ece(y01, p_unc, self.ece_bins)
        brier_unc = self._brier(y01, p_unc)
        uncal = {"MSE": mse, "MAE": mae, "Pearson": pr, "Spearman": sp, "ECE": ece_unc, "Brier": brier_unc}

        if T is None:
            return uncal, dict(uncal)

        p_cal = self._sigmoid_calibrated(yhat, float(T))
        ece_cal = self._ece(y01, p_cal, self.ece_bins)
        brier_cal = self._brier(y01, p_cal)
        cal = {"MSE": mse, "MAE": mae, "Pearson": pr, "Spearman": sp, "ECE": ece_cal, "Brier": brier_cal}
        return uncal, cal

    def _fit_temperature(self, yhat_valid: np.ndarray, y01_valid: np.ndarray) -> float:
        z = torch.tensor(yhat_valid.astype(np.float32), device=self.device)
        y = torch.tensor(y01_valid.astype(np.float32), device=self.device)

        t = torch.zeros((), device=self.device, requires_grad=True)
        opt = torch.optim.Adam([t], lr=float(self.calib_lr))

        best_T = 1.0
        best_loss = float("inf")

        for _ in range(int(self.calib_max_steps)):
            T = torch.nn.functional.softplus(t) + 1e-6
            p = torch.sigmoid(torch.clamp(z / T, -20, 20))
            loss = -(y * torch.log(p + 1e-12) + (1.0 - y) * torch.log(1.0 - p + 1e-12)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            lv = float(loss.detach().cpu().item())
            Tv = float(T.detach().cpu().item())
            if lv < best_loss:
                best_loss = lv
                best_T = Tv

        return float(best_T)


    # Tier3: test slices
    def _slice_report(self, df_test: pd.DataFrame, yhat: np.ndarray, y01: np.ndarray, T: float) -> Dict[str, Dict[str, float]]:
        p_cal = self._sigmoid_calibrated(yhat, T)

        def pack(mask: np.ndarray) -> Dict[str, float]:
            idx = np.where(mask)[0]
            if idx.size == 0:
                return {"n": 0, "MSE": 0.0, "ECE": 0.0, "risk_mse@80": 0.0}
            y_s = y01[idx]
            yhat_s = yhat[idx]
            p_s = p_cal[idx]
            mse = self._mse(y_s, yhat_s)
            ece = self._ece(y_s, p_s, self.ece_bins)
            rc = self._risk_coverage(y_s, yhat_s, p_for_conf=p_s)
            return {"n": int(idx.size), "MSE": float(mse), "ECE": float(ece), "risk_mse@80": float(rc["risk_mse@80"])}

        out: Dict[str, Dict[str, float]] = {}

        if "report_length_bucket" in df_test.columns:
            for v in ["short", "medium", "long"]:
                mask = (df_test["report_length_bucket"].astype(str).values == v)
                out[f"test_slice.report_length_bucket.{v}"] = pack(mask)

        if "device_present" in df_test.columns:
            vals = df_test["device_present"].astype(int).values
            out["test_slice.device_present.1"] = pack(vals == 1)
            out["test_slice.device_present.0"] = pack(vals == 0)

        if "uncertainty_present" in df_test.columns:
            vals = df_test["uncertainty_present"].astype(int).values
            out["test_slice.uncertainty_present.1"] = pack(vals == 1)
            out["test_slice.uncertainty_present.0"] = pack(vals == 0)

        if "label_vs_extractive" in df_test.columns:
            vals = df_test["label_vs_extractive"].astype(str).values
            out["test_slice.label_vs_extractive.label"] = pack(vals == "label")
            out["test_slice.label_vs_extractive.extractive"] = pack(vals == "extractive")

        return out


    # Run
    def run(self) -> str:
        if not os.path.exists(self.valid_parquet):
            raise FileNotFoundError(self.valid_parquet)
        if not os.path.exists(self.test_parquet):
            raise FileNotFoundError(self.test_parquet)

        df_valid = pd.read_parquet(self.valid_parquet)
        df_test = pd.read_parquet(self.test_parquet)

        # TODO; remove this
        # df_test = df_test.iloc[: int(len(df_test) * 0.005)].reset_index(drop=True)
        # df_valid = df_valid.iloc[: int(len(df_valid) * 0.005)].reset_index(drop=True)


        if "split" in df_valid.columns:
            df_valid = df_valid[df_valid["split"].astype(str) == "valid"].copy()
        if "split" in df_test.columns:
            df_test = df_test[df_test["split"].astype(str) == "test"].copy()

        self._load_verifier()


        # VALID: predict + fit calibration
        inputs_valid: List[str] = []
        for i in range(len(df_valid)):
            q = df_valid.iloc[i]["question"]
            c = df_valid.iloc[i]["case_context"]
            inputs_valid.append(self._build_verifier_inputs(q, c))

        yhat_valid = self._predict_support(inputs_valid, desc="S1 Verification (case-only) [valid]")
        y_valid = pd.to_numeric(df_valid["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01_valid = np.clip(y_valid, 0.0, 1.0)

        self.calib_T = float(self._fit_temperature(yhat_valid, y01_valid))

        valid_unc, valid_cal = self._eval_metrics_bundle(y01_valid, yhat_valid, T=self.calib_T)
        p_valid_unc = self._sigmoid(yhat_valid)
        p_valid_cal = self._sigmoid_calibrated(yhat_valid, self.calib_T)

        valid_bins_unc = self._reliability_bins(y01_valid, p_valid_unc, self.ece_bins)
        valid_bins_cal = self._reliability_bins(y01_valid, p_valid_cal, self.ece_bins)

        valid_out_df = df_valid.copy()
        valid_out_df["s1_pred_support_raw"] = yhat_valid.astype(np.float32)
        valid_out_df["s1_pred_prob_sigmoid_uncal"] = p_valid_unc.astype(np.float32)
        valid_out_df["s1_pred_prob_sigmoid_cal"] = p_valid_cal.astype(np.float32)

        valid_pred_path = os.path.join(self.valid_out_dir, "s1_valid_predictions.parquet")
        valid_out_df.to_parquet(valid_pred_path, index=False)

        valid_report = {
            "system": "S1 (case-only verifier)",
            "valid_parquet": self.valid_parquet,
            "n_valid": int(len(df_valid)),
            "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
            "valid_metrics_uncalibrated": valid_unc,
            "valid_metrics_calibrated": valid_cal,
            "valid_reliability_bins_uncalibrated": valid_bins_unc,
            "valid_reliability_bins_calibrated": valid_bins_cal,
            "artifacts": {"valid_predictions_parquet": valid_pred_path},
        }
        valid_report_path = os.path.join(self.valid_out_dir, "s1_valid_report.json")
        with open(valid_report_path, "w", encoding="utf-8") as f:
            json.dump(valid_report, f, indent=2)


        # TEST: Tier 1 (uncal + calibrated)
        inputs_test: List[str] = []
        for i in range(len(df_test)):
            q = df_test.iloc[i]["question"]
            c = df_test.iloc[i]["case_context"]
            inputs_test.append(self._build_verifier_inputs(q, c))

        yhat_test = self._predict_support(inputs_test, desc="S1 Verification (case-only) [test]")
        y_test = pd.to_numeric(df_test["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01_test = np.clip(y_test, 0.0, 1.0)

        test_unc, test_cal = self._eval_metrics_bundle(y01_test, yhat_test, T=self.calib_T)

        p_test_unc = self._sigmoid(yhat_test)
        p_test_cal = self._sigmoid_calibrated(yhat_test, self.calib_T)

        rc_unc = self._risk_coverage(y01_test, yhat_test, p_for_conf=p_test_unc)
        rc_cal = self._risk_coverage(y01_test, yhat_test, p_for_conf=p_test_cal)

        bins_unc = self._reliability_bins(y01_test, p_test_unc, self.ece_bins)
        bins_cal = self._reliability_bins(y01_test, p_test_cal, self.ece_bins)

        test_out_df = df_test.copy()
        test_out_df["s1_pred_support_raw"] = yhat_test.astype(np.float32)
        test_out_df["s1_pred_prob_sigmoid_uncal"] = p_test_unc.astype(np.float32)
        test_out_df["s1_pred_prob_sigmoid_cal"] = p_test_cal.astype(np.float32)

        test_pred_path = os.path.join(self.out_dir, "s1_test_predictions.parquet")
        test_out_df.to_parquet(test_pred_path, index=False)

        tier1_report = {
            "system": "S1 (case-only verifier)",
            "test_parquet": self.test_parquet,
            "valid_parquet_for_calibration": self.valid_parquet,
            "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
            "verifier": {"weights": self.verifier_weights_path, "max_tokens": self.verifier_max_tokens},
            "metrics_uncalibrated": {**test_unc, "SelectiveRisk": rc_unc},
            "metrics_calibrated": {**test_cal, "SelectiveRisk": rc_cal},
            "reliability_bins_uncalibrated": bins_unc,
            "reliability_bins_calibrated": bins_cal,
            "artifacts": {"predictions_parquet": test_pred_path},
            "n_test": int(len(df_test)),
        }
        tier1_path = os.path.join(self.out_dir, "s1_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)


        # TEST: Tier 3 slices (calibrated)
        tier3_metrics = self._slice_report(df_test, yhat_test, y01_test, T=float(self.calib_T))
        tier3_path = os.path.join(self.out_dir, "s1_tier3_slices_report.json")
        with open(tier3_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system": "S1 (case-only verifier)",
                    "test_parquet": self.test_parquet,
                    "calibration_fit": {"method": "temperature", "T": float(self.calib_T)},
                    "n_test": int(len(df_test)),
                    "metrics": tier3_metrics,
                },
                f,
                indent=2,
            )

        summary = {
            "valid_report": valid_report_path,
            "tier1_report": tier1_path,
            "tier3_slices_report": tier3_path,
        }
        summary_path = os.path.join(self.out_dir, "s1_tier123_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[S1] calib_T={self.calib_T:.6f}")
        print(json.dumps(tier1_report["metrics_uncalibrated"], indent=2))
        print(json.dumps(tier1_report["metrics_calibrated"], indent=2))
        print(f"Wrote: {valid_pred_path}")
        print(f"Wrote: {valid_report_path}")
        print(f"Wrote: {test_pred_path}")
        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {tier3_path}")
        print(f"Wrote: {summary_path}")
        return summary_path







class CrossSystemComparatorPredOnly:
    """
    Evaluation set = intersection of qa_row_id across all provided systems.

    What it computes (on the intersection):
      - Per-system: MSE/MAE/Pearson/Spearman/ECE/Brier/NLL/AURC
      - Pairwise (all pairs, not just vs S4):
          * win rates (abs error, calibration error)
          * paired bootstrap CI for deltas (Δ metric = A - B)
      - faithfulness severity (if faithfulness prediction parquets are provided)

    Requirements inside each system test_predictions.parquet:
      - qa_row_id
      - soft_support_score  (gold target; must match across systems on eval ids)
      - a pred raw column containing "pred_support_raw" (or ending with "_pred_support_raw")
    """

    def __init__(
        self,
        system_test_predictions: Dict[str, str],
        out_path: str,
        *,
        ece_bins: int = 15,
        n_boot: int = 2000,
        seed: int = 13,
        system_faithfulness_predictions: Optional[Dict[str, str]] = None,
    ) -> None:
        self.system_test_predictions = dict(system_test_predictions)
        self.system_faithfulness_predictions = dict(system_faithfulness_predictions or {})
        self.out_path = out_path

        self.ece_bins = int(ece_bins)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)


    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _rankdata(self, x: np.ndarray) -> np.ndarray:
        return pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)

    def _pearson(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
        if denom <= 0:
            return 0.0
        return float((a * b).sum() / denom)

    def _spearman(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._pearson(self._rankdata(a), self._rankdata(b))

    def _mse(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean((y - yhat) ** 2))

    def _mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return float(np.mean(np.abs(y - yhat)))

    def _brier(self, y01: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean((p - y01) ** 2))

    def _nll_soft_bernoulli(self, y01: np.ndarray, p: np.ndarray) -> float:
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        return float(np.mean(-(y01 * np.log(p) + (1.0 - y01) * np.log(1.0 - p))))

    def _ece(self, y01: np.ndarray, p: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(y01)
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(y01[mask].mean())
            conf = float(p[mask].mean())
            ece += (m / n) * abs(acc - conf)
        return float(ece)

    def _aurc(self, y01: np.ndarray, yhat: np.ndarray) -> float:
        p = self._sigmoid(yhat)
        conf = np.maximum(p, 1.0 - p)
        order = np.argsort(-conf)
        y = y01[order]
        yh = yhat[order]
        csum = 0.0
        for k in range(1, len(y) + 1):
            csum += float(np.mean((y[:k] - yh[:k]) ** 2))
        return float(csum / max(1, len(y)))

    def _paired_bootstrap_delta(self, y01: np.ndarray, a: np.ndarray, b: np.ndarray, fn) -> Dict[str, float]:
        n = len(y01)
        deltas = np.zeros((self.n_boot,), dtype=np.float64)
        for t in range(self.n_boot):
            idx = self.rng.randint(0, n, size=n)
            deltas[t] = fn(y01[idx], a[idx]) - fn(y01[idx], b[idx])
        return {
            "mean_delta": float(deltas.mean()),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
        }

    # Column inference
    def _infer_cols(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        cols = list(df.columns)

        pred_raw = None
        for c in cols:
            if c.endswith("_pred_support_raw") or ("pred_support_raw" in c):
                pred_raw = c
                break

        pred_prob = None
        for c in cols:
            if ("pred_prob" in c) and ("sigmoid" in c):
                pred_prob = c
                break

        return {"pred_raw": pred_raw, "pred_prob": pred_prob}


    def _load_and_align_systems(self):
        systems_raw: Dict[str, pd.DataFrame] = {}

        for sys_name, path in self.system_test_predictions.items():
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

        # Intersection ids
        sys_ids = {k: set(v["qa_row_id"].tolist()) for k, v in systems_raw.items()}
        eval_ids = None
        for ids in sys_ids.values():
            eval_ids = set(ids) if eval_ids is None else (eval_ids & ids)
        eval_ids = np.array(sorted(list(eval_ids or [])), dtype=np.int64)

        if len(eval_ids) == 0:
            raise RuntimeError("No overlapping qa_row_id across provided systems.")

        # Filter + align
        systems: Dict[str, pd.DataFrame] = {}
        dropped = {}
        for sys_name, df in systems_raw.items():
            df_f = df[df["qa_row_id"].isin(eval_ids)].copy()
            df_f = df_f.sort_values("qa_row_id").reset_index(drop=True)
            systems[sys_name] = df_f
            dropped[f"{sys_name}_dropped"] = int(len(df) - len(df_f))

        # Sanity: exact id order match
        ref = next(iter(systems.values()))["qa_row_id"].to_numpy()
        for sys_name, df in systems.items():
            if not np.array_equal(df["qa_row_id"].to_numpy(), ref):
                raise RuntimeError(f"Internal alignment failure for {sys_name}")

        # Gold y: take from S4 if present else first system, and verify consistency
        prefer = "S4" if "S4" in systems else list(systems.keys())[0]
        y = pd.to_numeric(systems[prefer]["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y01 = np.clip(y, 0.0, 1.0)

        # Verify soft_support_score matches across systems on eval ids (small float tolerance)
        for sys_name, df in systems.items():
            y_other = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if not np.allclose(y_other, y, rtol=0.0, atol=1e-6):
                # Not ethical to compare if "gold" differs across files for same ids
                raise RuntimeError(
                    f"soft_support_score mismatch between {prefer} and {sys_name} on aligned eval ids. "
                    f"Make sure all test_predictions were produced from the same fixed test set rows."
                )

        meta = {
            "n_eval": int(len(eval_ids)),
            "eval_id_source": "intersection_of_system_outputs_only",
            "dropped_counts": dropped,
            "systems": list(systems.keys()),
        }
        return systems, y01, meta


    # Faithfulness severity
    def _faithfulness_severity(self, y01: np.ndarray, faith_df: pd.DataFrame, sys_prefix: str) -> Dict[str, object]:
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


    # Main run
    def run(self) -> str:
        systems, y01, meta = self._load_and_align_systems()

        per_system = {}
        series = {}

        for sys_name, df in systems.items():
            cols = self._infer_cols(df)
            if cols["pred_raw"] is None:
                raise RuntimeError(f"Could not infer pred_support_raw column for {sys_name}")

            yhat = pd.to_numeric(df[cols["pred_raw"]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

            if cols["pred_prob"] is not None:
                p = pd.to_numeric(df[cols["pred_prob"]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                p = np.clip(p, 0.0, 1.0)
            else:
                p = self._sigmoid(yhat).astype(np.float32)

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

        # Pairwise comparisons for ALL ordered pairs A vs B
        def ece_p_fn(yy, pp): return self._ece(yy, pp, self.ece_bins)
        def brier_p_fn(yy, pp): return self._brier(yy, pp)
        def nll_p_fn(yy, pp): return self._nll_soft_bernoulli(yy, pp)

        pairwise = {}
        names = list(systems.keys())
        for a in names:
            for b in names:
                if a == b:
                    continue

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
                        "ΔMSE": self._paired_bootstrap_delta(y01, yhat_a, yhat_b, self._mse),
                        "ΔMAE": self._paired_bootstrap_delta(y01, yhat_a, yhat_b, self._mae),
                        "ΔPearson": self._paired_bootstrap_delta(y01, yhat_a, yhat_b, lambda yy, aa: self._pearson(yy, aa)),
                        "ΔSpearman": self._paired_bootstrap_delta(y01, yhat_a, yhat_b, lambda yy, aa: self._spearman(yy, aa)),
                        "ΔAURC": self._paired_bootstrap_delta(y01, yhat_a, yhat_b, self._aurc),
                        "ΔECE": self._paired_bootstrap_delta(y01, p_a, p_b, ece_p_fn),
                        "ΔBrier": self._paired_bootstrap_delta(y01, p_a, p_b, brier_p_fn),
                        "ΔNLL": self._paired_bootstrap_delta(y01, p_a, p_b, nll_p_fn),
                    },
                }

        # Optional faithfulness severity
        faithfulness = {}
        if self.system_faithfulness_predictions:
            # Align by qa_row_id intersection ids (from systems) via the reference ids
            ref_ids = systems[next(iter(systems))]["qa_row_id"].to_numpy(dtype=np.int64)
            ref_ids_df = pd.DataFrame({"qa_row_id": ref_ids})

            for sys_name, path in self.system_faithfulness_predictions.items():
                if not os.path.exists(path):
                    continue
                fdf = pd.read_parquet(path)
                if "qa_row_id" not in fdf.columns:
                    faithfulness[sys_name] = {"available": False}
                    continue
                fdf = fdf.copy()
                fdf["qa_row_id"] = pd.to_numeric(fdf["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
                fdf = fdf.merge(ref_ids_df, on="qa_row_id", how="inner").sort_values("qa_row_id").reset_index(drop=True)

                base_cols = [c for c in fdf.columns if c.endswith("_pred_base_raw")]
                if not base_cols:
                    faithfulness[sys_name] = {"available": False}
                    continue

                prefix = base_cols[0].replace("_pred_base_raw", "")
                # y01 must align with fdf rows now
                y01_f = y01[: len(fdf)]
                faithfulness[sys_name] = self._faithfulness_severity(y01_f, fdf, prefix)

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
            },
        }

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)

        print(f"Wrote: {self.out_path}")
        print(f"n_eval={final['n_eval']} dropped={final['dropped_counts']}")
        return self.out_path







if __name__ == "__main__":

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]



    ''' S1 '''
    s1 = S1Tier123CaseOnlyEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier123_results_s1_end2end"),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        # optional:
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s1.run()



    ''' S3 '''
    s3 = S3Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier123_results_s3_end2end"),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        bm25_k1=1.2,
        bm25_b=0.75,
        # optional speed knobs:
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s3.run()




    ''' S2 '''
    s2 = S2Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier123_results_s2_end2end"),
        batch_size_embed_chunks=128,
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        # optional speed knobs
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s2.run()



    ''' S4 '''
    runner = S4Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        model_weights_path=model_weights_path,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier123_results_s4_end2end"),
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        # Tier2 knobs
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # Tier3 knobs
        calibration_method="temperature",
        # calibration_max_eval=10000,
        calibration_opt_steps=200,
        calibration_lr=0.05,
    )
    runner.run()



    ''' Cross comparison '''
    base = params["Preprocessing"]["base_path_evidenceData"]

    comparator = CrossSystemComparatorPredOnly(
        system_test_predictions={
            "S1": os.path.join(base, "tier123_results_s1_end2end", "s1_test_predictions.parquet"),
            "S2": os.path.join(base, "tier123_results_s2_end2end", "s2_test_predictions.parquet"),
            "S3": os.path.join(base, "tier123_results_s3_end2end", "s3_test_predictions.parquet"),
            "S4": os.path.join(base, "tier123_results_s4_end2end", "s4_test_predictions.parquet"),
        },
        system_faithfulness_predictions={
            "S2": os.path.join(base, "tier123_results_s2_end2end", "s2_tier2_faithfulness_predictions.parquet"),
            "S3": os.path.join(base, "tier123_results_s3_end2end", "s3_tier2_faithfulness_predictions.parquet"),
            "S4": os.path.join(base, "tier123_results_s4_end2end", "s4_tier2_faithfulness_predictions.parquet"),
        },
        out_path=os.path.join(base, "cross_system_results", "s1_s4_cross_system_comparison_predonly.json"),
        ece_bins=15,
        n_boot=2000,
        seed=13,
    )
    comparator.run()
