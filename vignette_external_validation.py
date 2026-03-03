"""
Created on March 2, 2026.
vignette_external_validation.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/


Tier1:
    End-to-end retrieval (R*) → verifier (V*) option scoring on the Vignette test set (169 MCQs).
    Classification metrics:
        Accuracy (top-1)
        Negative log-likelihood (NLL) of the correct option
        Multiclass Brier score
    Calibration metrics:
        Top-1 Expected Calibration Error (ECE) using softmax probabilities
    Selective risk:
        Risk–coverage curve (risk = 1 − accuracy)
            AURC (area under the risk–coverage curve)
Tier2:
    Faithfulness / perturbation stress tests on the fixed Vignette test set:
        (a) random evidence (for S2–S4, if available)
        (b) swapped evidence across samples (S2–S4)
        (c) truncate to top-1 / top-3 retrieved chunks (S2–S4)
        (d) shuffle chunk order (S2–S4)
        (e) input perturbations (S1):
            question-only
            option-only
            shuffled option text
    For each perturbation:
        Report baseline accuracy
        Report perturbed accuracy
        Report accuracy drop relative to the system’s own baseline
        Report fraction of questions with degraded correctness under corruption
Tier3: Distribution slices on the fixed Vignette test set:
        (a) option_count (4 / 5)
        (b) question_length_bucket (short / medium / long)
        (c) contains_modality_terms (0 / 1)
        (d) contains_uncertainty_terms (0 / 1)
    For each slice:
        Report n
        Report Accuracy
        Report NLL
        Report Brier
        Report ECE
        Report AURC
Cross-system:
    Align S1–S4 by qa_row_id intersection (Vignette test).
    Per-system metrics:
        Accuracy
        NLL
        Multiclass Brier
        ECE
        AURC
        Mean confidence
        Mean margin
    Pairwise comparisons (all ordered pairs):
        (a) paired bootstrap 95% CI for metric deltas (A minus B):
            ΔAccuracy
            ΔNLL
            ΔBrier
            ΔECE
            ΔAURC
        (b) example-level win rates:
            NLL win rate
            Brier win rate
            Correctness win rate
            Calibration error proxy win rate
    Faithfulness severity (if tier2 question-level perturbation parquets exist):
        Accuracy drop per perturbation
        Fraction of questions with degraded correctness under corruption
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




@dataclass
class VignetteEvalPaths:
    csv_4option: str
    csv_5option: str

class VignetteS4Tier123End2EndEvaluator:
    """
    External validation on the 169 radiology vignette MCQs (ALL questions are TEST).

    CSV formats:
      4-option CSV columns:
        - "q number", "question", "option 1", "option 2", "option 3", "option 4", "answer index"
        - answer index values: 1/2/3/4

      5-option CSV columns:
        - "question_number", "question", "option_A", "option_B", "option_C", "option_D", "option_E", "solution_index"
        - solution_index values: A/B/C/D/E

    Core:
      - Expand each question into option-rows.
      - For each (question, option): retrieve evidence with R* using query = question + option text.
      - Score each option with V* (scalar).
      - Softmax across options within the same question -> probabilities.
      - Argmax -> predicted option. Compute MCQ metrics.

    Tier1:
      - Accuracy
      - NLL (true option)
      - Multiclass Brier
      - Top-1 ECE (binning on max prob)
      - Selective risk: risk_01 = 1-acc vs coverage by confidence; AURC01

    Tier2 (faithfulness):
      - random evidence (avg over trials)
      - swapped evidence across questions
      - truncate top-1 / top-3
      - shuffle chunk order
      Report deltas vs baseline in Accuracy/NLL/ECE/Brier.

    Tier3 slices (dataset-only heuristics):
      - option_count: 4 vs 5
      - question_length_bucket: short/medium/long (tertiles)
      - contains_modality_terms: 0/1
      - contains_uncertainty_terms: 0/1
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        model_weights_path: str,        # retriever R* trained_model.pth (trained on MIMIC)
        verifier_weights_path: str,     # verifier V* trained_model.pth (trained on MIMIC)
        vignette_paths: VignetteEvalPaths,
        device: str = "cuda",
        top_k: int = 20,
        verifier_max_tokens: int = 512,
        evidence_sep: str = "\n\n---\n\n",
        batch_size_retrieve: int = 256,
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        seed: int = 13,
        # Optional: external multiclass temperature for MCQ softmax. If provided we compute calibrated metrics too.
        mcq_temperature_T: Optional[float] = None,
        # Tier2
        faithfulness_random_trials: int = 1,
        # Output
        out_dir: Optional[str] = None,
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.model_weights_path = model_weights_path
        self.verifier_weights_path = verifier_weights_path
        self.vignette_paths = vignette_paths

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.evidence_sep = str(evidence_sep)
        self.batch_size_retrieve = int(batch_size_retrieve)
        self.batch_size_verify = int(batch_size_verify)
        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.mcq_temperature_T = float(mcq_temperature_T) if mcq_temperature_T is not None else None
        self.faithfulness_random_trials = int(faithfulness_random_trials)

        # Evidence assets (same as MIMIC setup)
        self.step6_dir = os.path.join(self.base_path_evidenceData, "step6_faiss_rstar")
        self.faiss_index_path = os.path.join(self.step6_dir, "faiss_Rstar.index")
        self.chunk_ids_path = os.path.join(self.step6_dir, "chunk_ids.npy")
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        # Output
        self.out_dir = out_dir or os.path.join(self.base_path_evidenceData, "external_validation", "Vignette")
        os.makedirs(self.out_dir, exist_ok=True)

        self._set_determinism()

        # Loaded later
        self.params = None
        self.retriever_tok = None
        self.retriever_model = None
        self.max_query_tokens = 128

        self.verifier_tok = None
        self.verifier_model = None

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
        self.params = open_experiment(self.experiment_name, self.cfg_path)

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

    def _clean(self, s: str) -> str:
        return " ".join(str(s).split()).strip()

    def _load_vignette_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.dropna(how="all").reset_index(drop=True)
        # normalize column names (keep original too)
        df.columns = [str(c) for c in df.columns]
        return df

    def _build_mcq_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds:
          - questions_df: one row per question
          - options_df: one row per option
        Uses the fixed schemas you provided.
        """
        q_rows: List[Dict] = []
        opt_rows: List[Dict] = []

        qa_row_id = 0

        if self.vignette_paths.csv_4option and os.path.exists(self.vignette_paths.csv_4option):
            df4 = self._load_vignette_csv(self.vignette_paths.csv_4option)

            required4 = ["q number", "question", "option 1", "option 2", "option 3", "option 4", "answer index"]
            missing4 = [c for c in required4 if c not in df4.columns]
            if missing4:
                raise ValueError(f"4-option CSV missing columns: {missing4}. Found: {list(df4.columns)}")

            for _, r in df4.iterrows():
                qtext = self._clean(r["question"])
                # answer index is 1..4
                ans_idx_raw = r["answer index"]
                try:
                    ans_idx = int(pd.to_numeric(ans_idx_raw, errors="raise"))
                except Exception:
                    continue
                if ans_idx < 1 or ans_idx > 4:
                    continue
                correct_index = ans_idx - 1  # 0-based

                options = [
                    ("A", self._clean(r["option 1"])),
                    ("B", self._clean(r["option 2"])),
                    ("C", self._clean(r["option 3"])),
                    ("D", self._clean(r["option 4"])),
                ]

                options = [(L, t) for (L, t) in options if t != ""]
                if len(options) != 4:
                    continue

                qid = qa_row_id
                qa_row_id += 1

                q_rows.append(
                    {
                        "qa_row_id": qid,
                        "question": qtext,
                        "num_options": 4,
                        "options_json": json.dumps([{"letter": L, "text": t} for L, t in options]),
                        "correct_letter": options[correct_index][0],
                        "correct_index": int(correct_index),
                        "source_split": "4option",
                        "source_question_id": self._clean(r["q number"]),
                    }
                )

                for opt_idx, (L, opt_text) in enumerate(options):
                    retrieval_query = f"{qtext}\n\nOption ({L}): {opt_text}"
                    opt_rows.append(
                        {
                            "qa_row_id": qid,
                            "option_index": int(opt_idx),
                            "option_letter": L,
                            "option_text": opt_text,
                            "question": qtext,
                            "retrieval_query": retrieval_query,
                            "is_correct": int(opt_idx == correct_index),
                            "correct_index": int(correct_index),
                            "num_options": 4,
                            "source_split": "4option",
                        }
                    )

        if self.vignette_paths.csv_5option and os.path.exists(self.vignette_paths.csv_5option):
            df5 = self._load_vignette_csv(self.vignette_paths.csv_5option)

            required5 = ["question_number", "question", "option_A", "option_B", "option_C", "option_D", "option_E", "solution_index"]
            missing5 = [c for c in required5 if c not in df5.columns]
            if missing5:
                raise ValueError(f"5-option CSV missing columns: {missing5}. Found: {list(df5.columns)}")

            for _, r in df5.iterrows():
                qtext = self._clean(r["question"])
                ans_letter = self._clean(r["solution_index"]).upper()
                if ans_letter not in ["A", "B", "C", "D", "E"]:
                    continue

                options = [
                    ("A", self._clean(r["option_A"])),
                    ("B", self._clean(r["option_B"])),
                    ("C", self._clean(r["option_C"])),
                    ("D", self._clean(r["option_D"])),
                    ("E", self._clean(r["option_E"])),
                ]
                options = [(L, t) for (L, t) in options if t != ""]
                if len(options) != 5:
                    continue

                letter_to_index = {L: i for i, (L, _) in enumerate(options)}
                correct_index = letter_to_index.get(ans_letter, None)
                if correct_index is None:
                    continue

                qid = qa_row_id
                qa_row_id += 1

                q_rows.append(
                    {
                        "qa_row_id": qid,
                        "question": qtext,
                        "num_options": 5,
                        "options_json": json.dumps([{"letter": L, "text": t} for L, t in options]),
                        "correct_letter": ans_letter,
                        "correct_index": int(correct_index),
                        "source_split": "5option",
                        "source_question_id": self._clean(r["question_number"]),
                    }
                )

                for opt_idx, (L, opt_text) in enumerate(options):
                    retrieval_query = f"{qtext}\n\nOption ({L}): {opt_text}"
                    opt_rows.append(
                        {
                            "qa_row_id": qid,
                            "option_index": int(opt_idx),
                            "option_letter": L,
                            "option_text": opt_text,
                            "question": qtext,
                            "retrieval_query": retrieval_query,
                            "is_correct": int(opt_idx == correct_index),
                            "correct_index": int(correct_index),
                            "num_options": 5,
                            "source_split": "5option",
                        }
                    )

        questions_df = pd.DataFrame(q_rows)
        options_df = pd.DataFrame(opt_rows)

        if len(questions_df) == 0 or len(options_df) == 0:
            raise RuntimeError("No questions/options were parsed. Check CSV paths and schemas.")

        # Integrity check
        cnt = options_df.groupby("qa_row_id").size().reset_index(name="n")
        merged = questions_df.merge(cnt, on="qa_row_id", how="left")
        bad = merged[(merged["n"].fillna(-1).astype(int) != merged["num_options"].astype(int))]
        if len(bad) > 0:
            raise RuntimeError(f"Expanded rows mismatch for {len(bad)} questions. Example:\n{bad.head(3)}")

        return questions_df.reset_index(drop=True), options_df.reset_index(drop=True)


    def _normalize_query(self, x: str) -> str:
        return " ".join(str(x).split()).strip().lower()

    def _tokenize_queries(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.retriever_tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_query_tokens,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _retrieve_topk_chunk_ids(self, query_texts: List[str], desc: str) -> List[List[int]]:
        assert self.retriever_model is not None
        assert self.faiss_index is not None
        assert self.faiss_chunk_ids is not None

        all_ids: List[List[int]] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(query_texts), self.batch_size_retrieve),
                total=math.ceil(len(query_texts) / self.batch_size_retrieve),
                desc=desc,
                unit="batch",
            ):
                batch = [self._normalize_query(x) for x in query_texts[i : i + self.batch_size_retrieve]]
                tok = self._tokenize_queries(batch)
                q_emb = self.retriever_model.encode_queries(tok["input_ids"], tok["attention_mask"])
                q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).detach().cpu().numpy().astype(np.float32)

                D, I = self.faiss_index.search(q_emb, self.top_k)
                for j in range(I.shape[0]):
                    idxs = I[j].tolist()
                    cids = [int(self.faiss_chunk_ids[ii]) for ii in idxs if ii >= 0]
                    all_ids.append(cids)
        return all_ids

    def _materialize_evidence(self, chunk_ids: List[int]) -> str:
        texts = []
        for cid in chunk_ids:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)

    def _build_verifier_input(self, question: str, option_letter: str, option_text: str, evidence_text: str) -> str:
        q = self._clean(question)
        a = self._clean(f"({option_letter}) {option_text}")
        e = self._clean(evidence_text)
        return f"Question: {q}\nProposed Answer: {a}\nEvidence: {e}"

    def _predict_scores(self, inputs: List[str], desc: str) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), self.batch_size_verify),
                total=math.ceil(len(inputs) / self.batch_size_verify),
                desc=desc,
                unit="batch",
            ):
                batch = inputs[i : i + self.batch_size_verify]
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

    def _softmax_probs(self, scores: np.ndarray, T: float) -> np.ndarray:
        z = scores.astype(np.float64) / float(max(T, 1e-6))
        m = float(z.max())
        ex = np.exp(np.clip(z - m, -50, 50))
        return ex / max(float(ex.sum()), 1e-12)

    def _question_level_from_option_scores(self, opt_df: pd.DataFrame, score_col: str, T: float) -> pd.DataFrame:
        rows = []
        for qid, g in opt_df.groupby("qa_row_id", sort=True):
            scores = g[score_col].to_numpy(dtype=np.float64)
            p = self._softmax_probs(scores, T=T)

            correct_index = int(g["correct_index"].iloc[0])
            pred_index = int(np.argmax(p))
            conf = float(np.max(p))
            p_sorted = np.sort(p)
            margin = float(p_sorted[-1] - p_sorted[-2]) if len(p_sorted) >= 2 else float(p_sorted[-1])

            nll = float(-math.log(max(float(p[correct_index]), 1e-12)))
            y = np.zeros_like(p)
            y[correct_index] = 1.0
            brier = float(np.sum((p - y) ** 2))

            rows.append(
                {
                    "qa_row_id": int(qid),
                    "num_options": int(g["num_options"].iloc[0]),
                    "correct_index": correct_index,
                    "pred_index": pred_index,
                    "is_correct": int(pred_index == correct_index),
                    "conf": conf,
                    "margin": margin,
                    "p_true": float(p[correct_index]),
                    "nll": nll,
                    "brier_mcq": brier,
                    "source_split": str(g["source_split"].iloc[0]) if "source_split" in g.columns else None,
                }
            )
        return pd.DataFrame(rows)

    def _ece_top1(self, correct01: np.ndarray, conf: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(correct01)
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (conf >= lo) & (conf < hi) if b < bins - 1 else (conf >= lo) & (conf <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(correct01[mask].mean())
            cbar = float(conf[mask].mean())
            ece += (m / n) * abs(acc - cbar)
        return float(ece)

    def _selective_risk(self, is_correct: np.ndarray, conf: np.ndarray) -> Dict[str, object]:
        order = np.argsort(-conf)
        coverages = np.linspace(0.1, 1.0, 10)

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(order))))
            return float(1.0 - is_correct[order[:k]].mean())

        curve = [{"coverage": float(c), "risk_01": risk_at(float(c))} for c in coverages]
        fixed = [0.6, 0.8, 0.9]
        fixed_out = {f"risk_01@{int(c*100)}": risk_at(c) for c in fixed}
        aurc01 = float(np.mean([x["risk_01"] for x in curve]))
        return {"curve": curve, "AURC01": aurc01, **fixed_out}

    def _tier1_metrics(self, qdf: pd.DataFrame) -> Dict[str, object]:
        acc = float(qdf["is_correct"].mean()) if len(qdf) else 0.0
        nll = float(qdf["nll"].mean()) if len(qdf) else 0.0
        brier = float(qdf["brier_mcq"].mean()) if len(qdf) else 0.0
        ece = self._ece_top1(
            qdf["is_correct"].to_numpy(np.float32),
            qdf["conf"].to_numpy(np.float32),
            self.ece_bins,
        ) if len(qdf) else 0.0
        sel = self._selective_risk(
            qdf["is_correct"].to_numpy(np.float32),
            qdf["conf"].to_numpy(np.float32),
        ) if len(qdf) else {}
        return {"Accuracy": acc, "NLL": nll, "Brier": brier, "ECE": float(ece), "SelectiveRisk": sel}


    def _token_count(self, s: str) -> int:
        return len(self._clean(s).split())

    def _slice_tables(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        df = questions_df.copy()
        qlens = df["question"].astype(str).map(self._token_count).to_numpy()

        q1 = float(np.quantile(qlens, 1 / 3))
        q2 = float(np.quantile(qlens, 2 / 3))

        def bucket(x: int) -> str:
            if x <= q1:
                return "short"
            if x <= q2:
                return "medium"
            return "long"

        df["question_len_tokens"] = qlens.astype(int)
        df["question_length_bucket"] = [bucket(int(x)) for x in qlens.tolist()]

        modality_terms = ["ct", "mri", "ultrasound", "us", "x-ray", "radiograph", "mammogram", "pet", "angiogram"]
        uncertain_terms = ["likely", "possibly", "may represent", "could represent", "suggest", "suspicious", "indeterminate"]

        qlow = df["question"].astype(str).str.lower()
        df["contains_modality_terms"] = qlow.apply(lambda s: int(any(t in s for t in modality_terms)))
        df["contains_uncertainty_terms"] = qlow.apply(lambda s: int(any(t in s for t in uncertain_terms)))

        df["option_count"] = pd.to_numeric(df["num_options"], errors="coerce").astype("Int64")
        return df

    def _compute_slice_metrics(self, qpred: pd.DataFrame, qtags: pd.DataFrame, prefix: str) -> Dict[str, object]:
        df = qpred.merge(
            qtags[["qa_row_id", "question_length_bucket", "contains_modality_terms", "contains_uncertainty_terms", "option_count"]],
            on="qa_row_id",
            how="left",
        )
        out: Dict[str, object] = {}

        def pack(mask: np.ndarray) -> Dict[str, object]:
            if int(mask.sum()) == 0:
                return {"n": 0, "Accuracy": None, "NLL": None, "ECE": None, "Brier": None, "AURC01": None}
            sub = df.loc[mask].reset_index(drop=True)
            m = self._tier1_metrics(sub)
            return {
                "n": int(len(sub)),
                "Accuracy": m["Accuracy"],
                "NLL": m["NLL"],
                "ECE": m["ECE"],
                "Brier": m["Brier"],
                "AURC01": float(m["SelectiveRisk"]["AURC01"]) if "SelectiveRisk" in m and "AURC01" in m["SelectiveRisk"] else None,
            }

        # option_count 4 vs 5
        oc = pd.to_numeric(df["option_count"], errors="coerce").fillna(-1).astype(int).to_numpy()
        for v in [4, 5]:
            out[f"{prefix}.option_count.{v}"] = pack(oc == v)

        # length bucket
        for v in ["short", "medium", "long"]:
            out[f"{prefix}.question_length_bucket.{v}"] = pack(df["question_length_bucket"].astype(str).to_numpy() == v)

        # modality terms
        m1 = pd.to_numeric(df["contains_modality_terms"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        out[f"{prefix}.contains_modality_terms.1"] = pack(m1)
        out[f"{prefix}.contains_modality_terms.0"] = pack(~m1)

        # uncertainty terms
        u1 = pd.to_numeric(df["contains_uncertainty_terms"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        out[f"{prefix}.contains_uncertainty_terms.1"] = pack(u1)
        out[f"{prefix}.contains_uncertainty_terms.0"] = pack(~u1)

        return out


    def _stress_random_ids(self, n: int) -> List[List[int]]:
        U = self.all_chunk_ids_universe
        out = []
        for _ in range(n):
            idx = np.random.choice(len(U), size=self.top_k, replace=False)
            out.append([int(x) for x in U[idx].tolist()])
        return out

    def _stress_swap_between_questions(self, ids_per_row: List[List[int]], opt_df: pd.DataFrame) -> List[List[int]]:
        qids = opt_df["qa_row_id"].astype(int).to_numpy()
        uniq = sorted(pd.unique(qids).tolist())
        if len(uniq) <= 1:
            return ids_per_row

        q_to_rows = {qid: opt_df.index[opt_df["qa_row_id"].astype(int) == qid].tolist() for qid in uniq}
        q_to_firstrow = {qid: q_to_rows[qid][0] for qid in uniq}
        q_to_evidence = {qid: ids_per_row[q_to_firstrow[qid]] for qid in uniq}

        shifted = {}
        for i, qid in enumerate(uniq):
            q_next = uniq[(i + 1) % len(uniq)]
            shifted[qid] = q_to_evidence[q_next]

        out = []
        for qid in qids.tolist():
            out.append(list(shifted[int(qid)][: self.top_k]))
        return out

    def _stress_truncate(self, ids_per_row: List[List[int]], k: int) -> List[List[int]]:
        kk = max(1, int(k))
        return [list(x[:kk]) for x in ids_per_row]

    def _stress_shuffle_order(self, ids_per_row: List[List[int]]) -> List[List[int]]:
        out = []
        for x in ids_per_row:
            y = list(x[: self.top_k])
            random.shuffle(y)
            out.append(y)
        return out


    def run(self) -> str:
        self.chunk_map = self._load_chunk_map()
        self._load_faiss()
        self._load_retriever()
        self._load_verifier()

        questions_df, options_df = self._build_mcq_tables()
        qtags = self._slice_tables(questions_df)

        # Retrieve for each option-row
        queries = options_df["retrieval_query"].astype(str).tolist()
        topk_ids = self._retrieve_topk_chunk_ids(queries, desc="S4 Retrieval (R*) [Vignette options]")

        # Verifier score for each option-row
        verifier_inputs = []
        for i in range(len(options_df)):
            row = options_df.iloc[i]
            ev = self._materialize_evidence(topk_ids[i])
            verifier_inputs.append(self._build_verifier_input(row["question"], row["option_letter"], row["option_text"], ev))

        scores = self._predict_scores(verifier_inputs, desc="S4 Verification (V*) [Vignette options]")

        opt_pred = options_df.copy()
        opt_pred["s4_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        opt_pred["s4_score_raw"] = scores.astype(np.float32)

        # Tier1 baseline (uncalibrated: T=1)
        qpred_uncal = self._question_level_from_option_scores(opt_pred, score_col="s4_score_raw", T=1.0)
        tier1_uncal = self._tier1_metrics(qpred_uncal)

        # Optional calibrated (external temperature)
        tier1_cal = None
        qpred_cal = None
        if self.mcq_temperature_T is not None:
            T_ext = float(max(self.mcq_temperature_T, 1e-6))
            qpred_cal = self._question_level_from_option_scores(opt_pred, score_col="s4_score_raw", T=T_ext)
            tier1_cal = self._tier1_metrics(qpred_cal)

        # Save predictions
        opt_path = os.path.join(self.out_dir, "option_level_predictions.parquet")
        q_uncal_path = os.path.join(self.out_dir, "question_level_predictions_uncalibrated.parquet")
        opt_pred.to_parquet(opt_path, index=False)
        qpred_uncal.to_parquet(q_uncal_path, index=False)

        q_cal_path = None
        if qpred_cal is not None:
            q_cal_path = os.path.join(self.out_dir, "question_level_predictions_calibrated.parquet")
            qpred_cal.to_parquet(q_cal_path, index=False)

        # Tier3 slices
        slice_uncal = self._compute_slice_metrics(qpred_uncal, qtags, prefix="slice_uncal")
        slice_cal = None
        if qpred_cal is not None:
            slice_cal = self._compute_slice_metrics(qpred_cal, qtags, prefix="slice_cal")

        # Tier2 faithfulness
        base_uncal = tier1_uncal
        base_cal = tier1_cal

        base_ids = [json.loads(s) for s in opt_pred["s4_retrieved_chunk_ids_json"].astype(str).tolist()]

        def score_with_ids(ids_per_row: List[List[int]], desc: str) -> pd.DataFrame:
            inputs = []
            for i in range(len(options_df)):
                r = options_df.iloc[i]
                ev = self._materialize_evidence(ids_per_row[i])
                inputs.append(self._build_verifier_input(r["question"], r["option_letter"], r["option_text"], ev))
            sc = self._predict_scores(inputs, desc=desc)
            out = options_df.copy()
            out["score_raw"] = sc.astype(np.float32)
            return out

        def deltas(m: Dict[str, object], base: Dict[str, object]) -> Dict[str, float]:
            keys = ["Accuracy", "NLL", "Brier", "ECE"]
            return {f"Δ{k}": float(m[k] - base[k]) for k in keys}

        # Random evidence (avg)
        rand_metrics_uncal = []
        rand_metrics_cal = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(options_df))
            opt_rand = score_with_ids(ids_rand, desc=f"Faithfulness: random evidence (trial {t+1})")
            q_rand_uncal = self._question_level_from_option_scores(opt_rand.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
            rand_metrics_uncal.append(self._tier1_metrics(q_rand_uncal))
            if self.mcq_temperature_T is not None:
                q_rand_cal = self._question_level_from_option_scores(opt_rand.rename(columns={"score_raw": "tmp"}), "tmp", T=float(self.mcq_temperature_T))
                rand_metrics_cal.append(self._tier1_metrics(q_rand_cal))

        def avg_metrics(ms: List[Dict[str, object]]) -> Dict[str, object]:
            if not ms:
                return {"Accuracy": 0.0, "NLL": 0.0, "Brier": 0.0, "ECE": 0.0}
            return {
                "Accuracy": float(np.mean([m["Accuracy"] for m in ms])),
                "NLL": float(np.mean([m["NLL"] for m in ms])),
                "Brier": float(np.mean([m["Brier"] for m in ms])),
                "ECE": float(np.mean([m["ECE"] for m in ms])),
            }

        m_rand_uncal = avg_metrics(rand_metrics_uncal)
        m_rand_cal = avg_metrics(rand_metrics_cal) if self.mcq_temperature_T is not None else None

        # Swap, truncate, shuffle
        ids_swap = self._stress_swap_between_questions(base_ids, options_df)
        ids_top1 = self._stress_truncate(base_ids, 1)
        ids_top3 = self._stress_truncate(base_ids, 3)
        ids_shuffle = self._stress_shuffle_order(base_ids)

        opt_swap = score_with_ids(ids_swap, desc="Faithfulness: swapped evidence (question-level)")
        opt_top1 = score_with_ids(ids_top1, desc="Faithfulness: truncate top-1")
        opt_top3 = score_with_ids(ids_top3, desc="Faithfulness: truncate top-3")
        opt_shuf = score_with_ids(ids_shuffle, desc="Faithfulness: shuffle order")

        q_swap_uncal = self._question_level_from_option_scores(opt_swap.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_top1_uncal = self._question_level_from_option_scores(opt_top1.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_top3_uncal = self._question_level_from_option_scores(opt_top3.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_shuf_uncal = self._question_level_from_option_scores(opt_shuf.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)

        m_swap_uncal = self._tier1_metrics(q_swap_uncal)
        m_top1_uncal = self._tier1_metrics(q_top1_uncal)
        m_top3_uncal = self._tier1_metrics(q_top3_uncal)
        m_shuf_uncal = self._tier1_metrics(q_shuf_uncal)

        m_swap_cal = m_top1_cal = m_top3_cal = m_shuf_cal = None
        if self.mcq_temperature_T is not None:
            T_ext = float(self.mcq_temperature_T)
            q_swap_cal = self._question_level_from_option_scores(opt_swap.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_top1_cal = self._question_level_from_option_scores(opt_top1.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_top3_cal = self._question_level_from_option_scores(opt_top3.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_shuf_cal = self._question_level_from_option_scores(opt_shuf.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            m_swap_cal = self._tier1_metrics(q_swap_cal)
            m_top1_cal = self._tier1_metrics(q_top1_cal)
            m_top3_cal = self._tier1_metrics(q_top3_cal)
            m_shuf_cal = self._tier1_metrics(q_shuf_cal)

        tier2_report = {
            "system": "S4 (R* trained on MIMIC, V* trained on MIMIC)",
            "n_questions": int(len(qpred_uncal)),
            "baseline_uncalibrated": {k: base_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
            "baseline_calibrated": ({k: base_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if base_cal is not None else None),
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "uncalibrated": m_rand_uncal,
                    "delta_uncalibrated": deltas(m_rand_uncal, base_uncal),
                    "calibrated": m_rand_cal,
                    "delta_calibrated": (deltas(m_rand_cal, base_cal) if (m_rand_cal is not None and base_cal is not None) else None),
                },
                "swapped_evidence": {
                    "uncalibrated": {k: m_swap_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_swap_uncal, base_uncal),
                    "calibrated": ({k: m_swap_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_swap_cal is not None else None),
                    "delta_calibrated": (deltas(m_swap_cal, base_cal) if (m_swap_cal is not None and base_cal is not None) else None),
                },
                "truncate_top1": {
                    "uncalibrated": {k: m_top1_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_top1_uncal, base_uncal),
                    "calibrated": ({k: m_top1_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_top1_cal is not None else None),
                    "delta_calibrated": (deltas(m_top1_cal, base_cal) if (m_top1_cal is not None and base_cal is not None) else None),
                },
                "truncate_top3": {
                    "uncalibrated": {k: m_top3_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_top3_uncal, base_uncal),
                    "calibrated": ({k: m_top3_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_top3_cal is not None else None),
                    "delta_calibrated": (deltas(m_top3_cal, base_cal) if (m_top3_cal is not None and base_cal is not None) else None),
                },
                "shuffle_order": {
                    "uncalibrated": {k: m_shuf_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_shuf_uncal, base_uncal),
                    "calibrated": ({k: m_shuf_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_shuf_cal is not None else None),
                    "delta_calibrated": (deltas(m_shuf_cal, base_cal) if (m_shuf_cal is not None and base_cal is not None) else None),
                },
            },
        }

        # Save question-level faithfulness parquet (uncal only)
        tier2_q = qpred_uncal.copy()
        tier2_q = tier2_q.rename(columns={"is_correct": "base_is_correct", "conf": "base_conf", "nll": "base_nll"})
        tier2_q["swap_is_correct"] = q_swap_uncal["is_correct"].astype(int).values
        tier2_q["top1_is_correct"] = q_top1_uncal["is_correct"].astype(int).values
        tier2_q["top3_is_correct"] = q_top3_uncal["is_correct"].astype(int).values
        tier2_q["shuffle_is_correct"] = q_shuf_uncal["is_correct"].astype(int).values
        tier2_pred_path = os.path.join(self.out_dir, "tier2_faithfulness_predictions_question_level.parquet")
        tier2_q.to_parquet(tier2_pred_path, index=False)

        tier2_path = os.path.join(self.out_dir, "tier2_faithfulness_report.json")
        with open(tier2_path, "w", encoding="utf-8") as f:
            json.dump({**tier2_report, "artifacts": {"question_level_parquet": tier2_pred_path}}, f, indent=2)

        # Tier1 report
        tier1_report = {
            "system": "S4 (R* trained on MIMIC, V* trained on MIMIC)",
            "dataset": "Vignette MCQ (169 questions; 4/5 options mixed)",
            "paths": {"csv_4option": self.vignette_paths.csv_4option, "csv_5option": self.vignette_paths.csv_5option},
            "n_questions": int(len(qpred_uncal)),
            "n_options_rows": int(len(opt_pred)),
            "temperature_for_mcq_softmax": {
                "uncalibrated_T": 1.0,
                "external_calibrated_T": float(self.mcq_temperature_T) if self.mcq_temperature_T is not None else None,
            },
            "tier1_uncalibrated": tier1_uncal,
            "tier1_calibrated": tier1_cal,
            "tier3_slices_uncalibrated": slice_uncal,
            "tier3_slices_calibrated": slice_cal,
            "artifacts": {
                "option_level_predictions_parquet": opt_path,
                "question_level_predictions_uncalibrated_parquet": q_uncal_path,
                "question_level_predictions_calibrated_parquet": q_cal_path,
            },
        }
        tier1_path = os.path.join(self.out_dir, "tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        summary = {
            "tier1_report": tier1_path,
            "tier2_faithfulness_report": tier2_path,
            "option_level_predictions": opt_path,
            "question_level_predictions_uncalibrated": q_uncal_path,
            "question_level_predictions_calibrated": q_cal_path,
            "tier2_faithfulness_predictions_question_level": tier2_pred_path,
        }
        summary_path = os.path.join(self.out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {tier2_path}")
        print(f"Wrote: {summary_path}")




        # summarizing all to a single csv ...
        # summarizing all to a single csv ...
        # summarizing all to a single csv ...

        tier1_path = os.path.join(self.out_dir, "tier1_report.json")
        tier2_faith_path = os.path.join(self.out_dir, "tier2_faithfulness_report.json")

        out_csv = os.path.join(self.out_dir, "S4_vignette_full_results.csv")

        def f3(x):
            if x is None:
                return None
            try:
                return float(f"{float(x):.3f}")
            except Exception:
                return None

        def add_metrics(rows, tier, subgroup, metrics_dict, keys):
            for k in keys:
                if k in metrics_dict:
                    rows.append([tier, subgroup, k, f3(metrics_dict.get(k))])

        with open(tier1_path, "r") as f:
            t1 = json.load(f)

        with open(tier2_faith_path, "r") as f:
            t2f = json.load(f)

        rows = []

        t1u = t1.get("tier1_uncalibrated", {}) or {}
        add_metrics(rows, "Tier1", "overall", t1u, ["Accuracy", "NLL", "Brier", "ECE"])

        sr = (t1u.get("SelectiveRisk") or {})
        if sr:
            rows.append(["Tier1", "overall", "AURC01", f3(sr.get("AURC01"))])
            rows.append(["Tier1", "overall", "risk_01@60", f3(sr.get("risk_01@60"))])
            rows.append(["Tier1", "overall", "risk_01@80", f3(sr.get("risk_01@80"))])
            rows.append(["Tier1", "overall", "risk_01@90", f3(sr.get("risk_01@90"))])

        base = (t2f.get("baseline_uncalibrated") or {})
        if base:
            add_metrics(rows, "Tier2_Faithfulness", "baseline", base, ["Accuracy", "NLL", "Brier", "ECE"])

        pert = (t2f.get("perturbations") or {})
        for name, obj in pert.items():
            du = (obj.get("delta_uncalibrated") or {})
            # keep deltas (main ones)
            for k in ["ΔAccuracy", "ΔNLL", "ΔBrier", "ΔECE"]:
                if k in du:
                    rows.append(["Tier2_Faithfulness", name, k, f3(du.get(k))])

        t3u = (t1.get("tier3_slices_uncalibrated") or {})
        for slice_name, m in t3u.items():
            add_metrics(rows, "Tier3", slice_name, m, ["n", "Accuracy", "NLL", "Brier", "ECE", "AURC01"])

        df = pd.DataFrame(rows, columns=["Tier", "Subgroup", "Metric", "Value"])
        df.to_csv(out_csv, index=False)

        print(f"Saved: {out_csv}")



        return summary_path







@dataclass
class VignetteEvalPaths:
    csv_4option: str
    csv_5option: str

class VignetteS3Tier123End2EndEvaluator:
    """
    S3 external validation on Vignette MCQs (169 Qs, TEST ONLY):
      - BM25 lexical retrieval over chunks.parquet (no FAISS)
      - Verifier V* (trained on MIMIC) scores each option
      - Softmax across options per question -> MCQ probabilities

    Metrics (adapted for MCQ):
      Tier1:
        - Accuracy
        - NLL (true option)
        - Multiclass Brier
        - Top-1 ECE (confidence calibration of max prob)
        - Selective risk (risk_01 = 1-acc vs coverage by confidence), AURC01
        - Confidence / margin distributions in prediction parquet

      Tier2 (faithfulness stress tests on evidence):
        - random evidence (avg over trials)
        - swapped evidence across questions
        - truncate to top-1 / top-3
        - shuffle chunk order
        Report deltas vs baseline in Accuracy/NLL/ECE/Brier.

      Tier3 slices (dataset-only heuristics):
        - option_count: 4 vs 5
        - question_length_bucket: short/medium/long (tertiles)
        - contains_modality_terms: 0/1
        - contains_uncertainty_terms: 0/1
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,      # verifier trained_model.pth (trained on MIMIC)
        vignette_paths: VignetteEvalPaths,
        device: str = "cuda",
        top_k: int = 20,
        verifier_max_tokens: int = 512,
        evidence_sep: str = "\n\n---\n\n",
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        seed: int = 13,
        mcq_temperature_T: Optional[float] = None,
        # BM25 knobs
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        # Tier2 knobs
        faithfulness_random_trials: int = 1,
        faithfulness_max_eval_questions: Optional[int] = None,  # subsample QUESTIONS for tier2 speed
        # Output
        out_dir: Optional[str] = None,
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path
        self.vignette_paths = vignette_paths

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.evidence_sep = str(evidence_sep)
        self.batch_size_verify = int(batch_size_verify)
        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.mcq_temperature_T = float(mcq_temperature_T) if mcq_temperature_T is not None else None

        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        self.faithfulness_random_trials = int(faithfulness_random_trials)
        self.faithfulness_max_eval_questions = int(faithfulness_max_eval_questions) if faithfulness_max_eval_questions is not None else None

        # Evidence
        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        # Output
        default_out = os.path.join(
            self.base_path_evidenceData,
            "external_validation",
            "Vignette",
            "tier123_results_s3_end2end",
        )
        self.out_dir = out_dir or default_out
        os.makedirs(self.out_dir, exist_ok=True)

        self.bm25_index_path = os.path.join(self.out_dir, "bm25_index.pkl")

        self._set_determinism()

        # Loaded later
        self.params = open_experiment(self.experiment_name, self.cfg_path)

        self.verifier_tok = None
        self.verifier_model = None
        self.chunk_map = None

        self._bm25 = None
        self.all_chunk_ids_universe = None

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

    def _tokenize_bm25(self, text: str) -> List[str]:
        t = " ".join(str(text).lower().split()).strip()
        out: List[str] = []
        cur: List[str] = []
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

    def _retrieve_topk_chunk_ids(self, query_texts: List[str], desc: str) -> List[List[int]]:
        assert self._bm25 is not None
        chunk_ids = self._bm25["chunk_ids"]
        postings = self._bm25["postings"]
        idf = self._bm25["idf"]
        doc_len = self._bm25["doc_len"]
        avgdl = float(self._bm25["avgdl"])
        k1 = float(self._bm25["k1"])
        b = float(self._bm25["b"])

        out: List[List[int]] = []

        for q in tqdm(query_texts, desc=desc, unit="query"):
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
        for cid in chunk_ids:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)

    def _clean(self, s: str) -> str:
        return " ".join(str(s).split()).strip()

    def _load_vignette_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.dropna(how="all").reset_index(drop=True)
        df.columns = [str(c) for c in df.columns]
        return df

    def _build_mcq_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q_rows: List[Dict] = []
        opt_rows: List[Dict] = []
        qa_row_id = 0

        # 4-option
        if self.vignette_paths.csv_4option and os.path.exists(self.vignette_paths.csv_4option):
            df4 = self._load_vignette_csv(self.vignette_paths.csv_4option)
            required4 = ["q number", "question", "option 1", "option 2", "option 3", "option 4", "answer index"]
            missing4 = [c for c in required4 if c not in df4.columns]
            if missing4:
                raise ValueError(f"4-option CSV missing columns: {missing4}. Found: {list(df4.columns)}")

            for _, r in df4.iterrows():
                qtext = self._clean(r["question"])
                try:
                    ans_idx = int(pd.to_numeric(r["answer index"], errors="raise"))
                except Exception:
                    continue
                if ans_idx < 1 or ans_idx > 4:
                    continue
                correct_index = ans_idx - 1

                options = [
                    ("A", self._clean(r["option 1"])),
                    ("B", self._clean(r["option 2"])),
                    ("C", self._clean(r["option 3"])),
                    ("D", self._clean(r["option 4"])),
                ]
                options = [(L, t) for (L, t) in options if t != ""]
                if len(options) != 4:
                    continue

                qid = qa_row_id
                qa_row_id += 1

                q_rows.append(
                    {
                        "qa_row_id": qid,
                        "question": qtext,
                        "num_options": 4,
                        "options_json": json.dumps([{"letter": L, "text": t} for L, t in options]),
                        "correct_letter": options[correct_index][0],
                        "correct_index": int(correct_index),
                        "source_split": "4option",
                        "source_question_id": self._clean(r["q number"]),
                    }
                )

                for opt_idx, (L, opt_text) in enumerate(options):
                    retrieval_query = f"{qtext}\n\nOption ({L}): {opt_text}"
                    opt_rows.append(
                        {
                            "qa_row_id": qid,
                            "option_index": int(opt_idx),
                            "option_letter": L,
                            "option_text": opt_text,
                            "question": qtext,
                            "retrieval_query": retrieval_query,
                            "is_correct": int(opt_idx == correct_index),
                            "correct_index": int(correct_index),
                            "num_options": 4,
                            "source_split": "4option",
                        }
                    )

        # 5-option
        if self.vignette_paths.csv_5option and os.path.exists(self.vignette_paths.csv_5option):
            df5 = self._load_vignette_csv(self.vignette_paths.csv_5option)
            required5 = ["question_number", "question", "option_A", "option_B", "option_C", "option_D", "option_E", "solution_index"]
            missing5 = [c for c in required5 if c not in df5.columns]
            if missing5:
                raise ValueError(f"5-option CSV missing columns: {missing5}. Found: {list(df5.columns)}")

            for _, r in df5.iterrows():
                qtext = self._clean(r["question"])
                ans_letter = self._clean(r["solution_index"]).upper()
                if ans_letter not in ["A", "B", "C", "D", "E"]:
                    continue

                options = [
                    ("A", self._clean(r["option_A"])),
                    ("B", self._clean(r["option_B"])),
                    ("C", self._clean(r["option_C"])),
                    ("D", self._clean(r["option_D"])),
                    ("E", self._clean(r["option_E"])),
                ]
                options = [(L, t) for (L, t) in options if t != ""]
                if len(options) != 5:
                    continue

                letter_to_index = {L: i for i, (L, _) in enumerate(options)}
                correct_index = letter_to_index.get(ans_letter, None)
                if correct_index is None:
                    continue

                qid = qa_row_id
                qa_row_id += 1

                q_rows.append(
                    {
                        "qa_row_id": qid,
                        "question": qtext,
                        "num_options": 5,
                        "options_json": json.dumps([{"letter": L, "text": t} for L, t in options]),
                        "correct_letter": ans_letter,
                        "correct_index": int(correct_index),
                        "source_split": "5option",
                        "source_question_id": self._clean(r["question_number"]),
                    }
                )

                for opt_idx, (L, opt_text) in enumerate(options):
                    retrieval_query = f"{qtext}\n\nOption ({L}): {opt_text}"
                    opt_rows.append(
                        {
                            "qa_row_id": qid,
                            "option_index": int(opt_idx),
                            "option_letter": L,
                            "option_text": opt_text,
                            "question": qtext,
                            "retrieval_query": retrieval_query,
                            "is_correct": int(opt_idx == correct_index),
                            "correct_index": int(correct_index),
                            "num_options": 5,
                            "source_split": "5option",
                        }
                    )

        questions_df = pd.DataFrame(q_rows)
        options_df = pd.DataFrame(opt_rows)

        if len(questions_df) == 0 or len(options_df) == 0:
            raise RuntimeError("No questions/options parsed. Check CSV paths and schemas.")

        cnt = options_df.groupby("qa_row_id").size().reset_index(name="n")
        merged = questions_df.merge(cnt, on="qa_row_id", how="left")
        bad = merged[(merged["n"].fillna(-1).astype(int) != merged["num_options"].astype(int))]
        if len(bad) > 0:
            raise RuntimeError(f"Expanded rows mismatch for {len(bad)} questions. Example:\n{bad.head(3)}")

        return questions_df.reset_index(drop=True), options_df.reset_index(drop=True)

    def _build_verifier_input(self, question: str, option_letter: str, option_text: str, evidence_text: str) -> str:
        q = self._clean(question)
        a = self._clean(f"({option_letter}) {option_text}")
        e = self._clean(evidence_text)
        return f"Question: {q}\nProposed Answer: {a}\nEvidence: {e}"

    def _predict_scores(self, inputs: List[str], desc: str) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), self.batch_size_verify),
                total=math.ceil(len(inputs) / self.batch_size_verify),
                desc=desc,
                unit="batch",
            ):
                batch = inputs[i : i + self.batch_size_verify]
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

    def _softmax_probs(self, scores: np.ndarray, T: float) -> np.ndarray:
        z = scores.astype(np.float64) / float(max(T, 1e-6))
        m = float(z.max())
        ex = np.exp(np.clip(z - m, -50, 50))
        return ex / max(float(ex.sum()), 1e-12)

    def _question_level_from_option_scores(self, opt_df: pd.DataFrame, score_col: str, T: float) -> pd.DataFrame:
        rows = []
        for qid, g in opt_df.groupby("qa_row_id", sort=True):
            scores = g[score_col].to_numpy(dtype=np.float64)
            p = self._softmax_probs(scores, T=T)

            correct_index = int(g["correct_index"].iloc[0])
            pred_index = int(np.argmax(p))
            conf = float(np.max(p))
            p_sorted = np.sort(p)
            margin = float(p_sorted[-1] - p_sorted[-2]) if len(p_sorted) >= 2 else float(p_sorted[-1])

            nll = float(-math.log(max(float(p[correct_index]), 1e-12)))
            y = np.zeros_like(p)
            y[correct_index] = 1.0
            brier = float(np.sum((p - y) ** 2))

            rows.append(
                {
                    "qa_row_id": int(qid),
                    "num_options": int(g["num_options"].iloc[0]),
                    "correct_index": correct_index,
                    "pred_index": pred_index,
                    "is_correct": int(pred_index == correct_index),
                    "conf": conf,
                    "margin": margin,
                    "p_true": float(p[correct_index]),
                    "nll": nll,
                    "brier_mcq": brier,
                    "source_split": str(g["source_split"].iloc[0]) if "source_split" in g.columns else None,
                }
            )
        return pd.DataFrame(rows)

    def _ece_top1(self, correct01: np.ndarray, conf: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(correct01)
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (conf >= lo) & (conf < hi) if b < bins - 1 else (conf >= lo) & (conf <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(correct01[mask].mean())
            cbar = float(conf[mask].mean())
            ece += (m / n) * abs(acc - cbar)
        return float(ece)

    def _selective_risk(self, is_correct: np.ndarray, conf: np.ndarray) -> Dict[str, object]:
        order = np.argsort(-conf)
        coverages = np.linspace(0.1, 1.0, 10)

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(order))))
            return float(1.0 - is_correct[order[:k]].mean())

        curve = [{"coverage": float(c), "risk_01": risk_at(float(c))} for c in coverages]
        fixed = [0.6, 0.8, 0.9]
        fixed_out = {f"risk_01@{int(c*100)}": risk_at(c) for c in fixed}
        aurc01 = float(np.mean([x["risk_01"] for x in curve]))
        return {"curve": curve, "AURC01": aurc01, **fixed_out}

    def _tier1_metrics(self, qdf: pd.DataFrame) -> Dict[str, object]:
        acc = float(qdf["is_correct"].mean()) if len(qdf) else 0.0
        nll = float(qdf["nll"].mean()) if len(qdf) else 0.0
        brier = float(qdf["brier_mcq"].mean()) if len(qdf) else 0.0
        ece = self._ece_top1(
            qdf["is_correct"].to_numpy(np.float32),
            qdf["conf"].to_numpy(np.float32),
            self.ece_bins,
        ) if len(qdf) else 0.0
        sel = self._selective_risk(
            qdf["is_correct"].to_numpy(np.float32),
            qdf["conf"].to_numpy(np.float32),
        ) if len(qdf) else {}
        return {"Accuracy": acc, "NLL": nll, "Brier": brier, "ECE": float(ece), "SelectiveRisk": sel}


    def _token_count(self, s: str) -> int:
        return len(self._clean(s).split())

    def _slice_tables(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        df = questions_df.copy()
        qlens = df["question"].astype(str).map(self._token_count).to_numpy()

        q1 = float(np.quantile(qlens, 1 / 3))
        q2 = float(np.quantile(qlens, 2 / 3))

        def bucket(x: int) -> str:
            if x <= q1:
                return "short"
            if x <= q2:
                return "medium"
            return "long"

        df["question_len_tokens"] = qlens.astype(int)
        df["question_length_bucket"] = [bucket(int(x)) for x in qlens.tolist()]

        modality_terms = ["ct", "mri", "ultrasound", "us", "x-ray", "radiograph", "mammogram", "pet", "angiogram"]
        uncertain_terms = ["likely", "possibly", "may represent", "could represent", "suggest", "suspicious", "indeterminate"]

        qlow = df["question"].astype(str).str.lower()
        df["contains_modality_terms"] = qlow.apply(lambda s: int(any(t in s for t in modality_terms)))
        df["contains_uncertainty_terms"] = qlow.apply(lambda s: int(any(t in s for t in uncertain_terms)))

        df["option_count"] = pd.to_numeric(df["num_options"], errors="coerce").astype("Int64")
        return df

    def _compute_slice_metrics(self, qpred: pd.DataFrame, qtags: pd.DataFrame, prefix: str) -> Dict[str, object]:
        df = qpred.merge(
            qtags[["qa_row_id", "question_length_bucket", "contains_modality_terms", "contains_uncertainty_terms", "option_count"]],
            on="qa_row_id",
            how="left",
        )
        out: Dict[str, object] = {}

        def pack(mask: np.ndarray) -> Dict[str, object]:
            if int(mask.sum()) == 0:
                return {"n": 0, "Accuracy": None, "NLL": None, "ECE": None, "Brier": None, "AURC01": None}
            sub = df.loc[mask].reset_index(drop=True)
            m = self._tier1_metrics(sub)
            return {
                "n": int(len(sub)),
                "Accuracy": m["Accuracy"],
                "NLL": m["NLL"],
                "ECE": m["ECE"],
                "Brier": m["Brier"],
                "AURC01": float(m["SelectiveRisk"]["AURC01"]) if "SelectiveRisk" in m and "AURC01" in m["SelectiveRisk"] else None,
            }

        oc = pd.to_numeric(df["option_count"], errors="coerce").fillna(-1).astype(int).to_numpy()
        for v in [4, 5]:
            out[f"{prefix}.option_count.{v}"] = pack(oc == v)

        for v in ["short", "medium", "long"]:
            out[f"{prefix}.question_length_bucket.{v}"] = pack(df["question_length_bucket"].astype(str).to_numpy() == v)

        m1 = pd.to_numeric(df["contains_modality_terms"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        out[f"{prefix}.contains_modality_terms.1"] = pack(m1)
        out[f"{prefix}.contains_modality_terms.0"] = pack(~m1)

        u1 = pd.to_numeric(df["contains_uncertainty_terms"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        out[f"{prefix}.contains_uncertainty_terms.1"] = pack(u1)
        out[f"{prefix}.contains_uncertainty_terms.0"] = pack(~u1)

        return out


    def _stress_random_ids(self, n_rows: int) -> List[List[int]]:
        U = self.all_chunk_ids_universe
        out = []
        for _ in range(n_rows):
            size = min(self.top_k, len(U))
            idx = np.random.choice(len(U), size=size, replace=False)
            out.append([int(x) for x in U[idx].tolist()])
        return out

    def _stress_swap_between_questions(self, ids_per_row: List[List[int]], opt_df: pd.DataFrame) -> List[List[int]]:
        qids = opt_df["qa_row_id"].astype(int).to_numpy()
        uniq = sorted(pd.unique(qids).tolist())
        if len(uniq) <= 1:
            return ids_per_row

        q_to_rows = {qid: opt_df.index[opt_df["qa_row_id"].astype(int) == qid].tolist() for qid in uniq}
        q_to_firstrow = {qid: q_to_rows[qid][0] for qid in uniq}
        q_to_evidence = {qid: ids_per_row[q_to_firstrow[qid]] for qid in uniq}

        shifted = {}
        for i, qid in enumerate(uniq):
            q_next = uniq[(i + 1) % len(uniq)]
            shifted[qid] = q_to_evidence[q_next]

        out = []
        for qid in qids.tolist():
            out.append(list(shifted[int(qid)][: self.top_k]))
        return out

    def _stress_truncate(self, ids_per_row: List[List[int]], k: int) -> List[List[int]]:
        kk = max(1, int(k))
        return [list(x[:kk]) for x in ids_per_row]

    def _stress_shuffle_order(self, ids_per_row: List[List[int]]) -> List[List[int]]:
        out = []
        for x in ids_per_row:
            y = list(x[: self.top_k])
            random.shuffle(y)
            out.append(y)
        return out


    def run(self) -> str:
        self.chunk_map = self._load_chunk_map()
        self._build_or_load_bm25()
        self._load_verifier()

        # Build dataset (all TEST)
        questions_df, options_df = self._build_mcq_tables()
        qtags = self._slice_tables(questions_df)

        # End2End baseline: retrieve -> score each option
        queries = options_df["retrieval_query"].astype(str).tolist()
        topk_ids = self._retrieve_topk_chunk_ids(queries, desc="S3 Retrieval (BM25) [Vignette options]")

        verifier_inputs = []
        for i in range(len(options_df)):
            row = options_df.iloc[i]
            ev = self._materialize_evidence(topk_ids[i])
            verifier_inputs.append(self._build_verifier_input(row["question"], row["option_letter"], row["option_text"], ev))

        scores = self._predict_scores(verifier_inputs, desc="S3 Verification (V*) [Vignette options]")

        opt_pred = options_df.copy()
        opt_pred["s3_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        opt_pred["s3_score_raw"] = scores.astype(np.float32)

        # Tier1 baseline (T=1)
        qpred_uncal = self._question_level_from_option_scores(opt_pred, score_col="s3_score_raw", T=1.0)
        tier1_uncal = self._tier1_metrics(qpred_uncal)

        # Optional calibrated MCQ temperature (external)
        tier1_cal = None
        qpred_cal = None
        if self.mcq_temperature_T is not None:
            T_ext = float(max(self.mcq_temperature_T, 1e-6))
            qpred_cal = self._question_level_from_option_scores(opt_pred, score_col="s3_score_raw", T=T_ext)
            tier1_cal = self._tier1_metrics(qpred_cal)

        opt_path = os.path.join(self.out_dir, "s3_option_level_predictions.parquet")
        q_uncal_path = os.path.join(self.out_dir, "s3_question_level_predictions_uncalibrated.parquet")
        opt_pred.to_parquet(opt_path, index=False)
        qpred_uncal.to_parquet(q_uncal_path, index=False)

        q_cal_path = None
        if qpred_cal is not None:
            q_cal_path = os.path.join(self.out_dir, "s3_question_level_predictions_calibrated.parquet")
            qpred_cal.to_parquet(q_cal_path, index=False)

        # Tier3 slices
        slice_uncal = self._compute_slice_metrics(qpred_uncal, qtags, prefix="slice_uncal")
        slice_cal = None
        if qpred_cal is not None:
            slice_cal = self._compute_slice_metrics(qpred_cal, qtags, prefix="slice_cal")

        # Tier2 faithfulness (optional subsample QUESTIONS for speed)
        opt_eval = opt_pred
        if self.faithfulness_max_eval_questions is not None and self.faithfulness_max_eval_questions > 0:
            rng = np.random.RandomState(self.seed)
            qids = qpred_uncal["qa_row_id"].astype(int).to_numpy()
            n = min(len(qids), self.faithfulness_max_eval_questions)
            keep_qids = set(rng.choice(qids, size=n, replace=False).tolist())
            opt_eval = opt_pred[opt_pred["qa_row_id"].astype(int).isin(keep_qids)].reset_index(drop=True)

        base_uncal = tier1_uncal
        base_cal = tier1_cal

        base_ids = [json.loads(s) for s in opt_eval["s3_retrieved_chunk_ids_json"].astype(str).tolist()]

        def score_with_ids(opt_subset: pd.DataFrame, ids_per_row: List[List[int]], desc: str) -> pd.DataFrame:
            inputs = []
            for i in range(len(opt_subset)):
                r = opt_subset.iloc[i]
                ev = self._materialize_evidence(ids_per_row[i])
                inputs.append(self._build_verifier_input(r["question"], r["option_letter"], r["option_text"], ev))
            sc = self._predict_scores(inputs, desc=desc)
            out = opt_subset.copy()
            out["score_raw"] = sc.astype(np.float32)
            return out

        def deltas(m: Dict[str, object], base: Dict[str, object]) -> Dict[str, float]:
            keys = ["Accuracy", "NLL", "Brier", "ECE"]
            return {f"Δ{k}": float(m[k] - base[k]) for k in keys}

        # Random evidence (avg)
        rand_metrics_uncal = []
        rand_metrics_cal = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(opt_eval))
            opt_rand = score_with_ids(opt_eval, ids_rand, desc=f"Faithfulness: random evidence (trial {t+1})")
            q_rand_uncal = self._question_level_from_option_scores(opt_rand.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
            rand_metrics_uncal.append(self._tier1_metrics(q_rand_uncal))
            if self.mcq_temperature_T is not None:
                q_rand_cal = self._question_level_from_option_scores(opt_rand.rename(columns={"score_raw": "tmp"}), "tmp", T=float(self.mcq_temperature_T))
                rand_metrics_cal.append(self._tier1_metrics(q_rand_cal))

        def avg_metrics(ms: List[Dict[str, object]]) -> Dict[str, object]:
            if not ms:
                return {"Accuracy": 0.0, "NLL": 0.0, "Brier": 0.0, "ECE": 0.0}
            return {
                "Accuracy": float(np.mean([m["Accuracy"] for m in ms])),
                "NLL": float(np.mean([m["NLL"] for m in ms])),
                "Brier": float(np.mean([m["Brier"] for m in ms])),
                "ECE": float(np.mean([m["ECE"] for m in ms])),
            }

        m_rand_uncal = avg_metrics(rand_metrics_uncal)
        m_rand_cal = avg_metrics(rand_metrics_cal) if self.mcq_temperature_T is not None else None

        # Swap, truncate, shuffle
        ids_swap = self._stress_swap_between_questions(base_ids, opt_eval)
        ids_top1 = self._stress_truncate(base_ids, 1)
        ids_top3 = self._stress_truncate(base_ids, 3)
        ids_shuffle = self._stress_shuffle_order(base_ids)

        opt_swap = score_with_ids(opt_eval, ids_swap, desc="Faithfulness: swapped evidence (question-level)")
        opt_top1 = score_with_ids(opt_eval, ids_top1, desc="Faithfulness: truncate top-1")
        opt_top3 = score_with_ids(opt_eval, ids_top3, desc="Faithfulness: truncate top-3")
        opt_shuf = score_with_ids(opt_eval, ids_shuffle, desc="Faithfulness: shuffle order")

        q_swap_uncal = self._question_level_from_option_scores(opt_swap.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_top1_uncal = self._question_level_from_option_scores(opt_top1.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_top3_uncal = self._question_level_from_option_scores(opt_top3.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_shuf_uncal = self._question_level_from_option_scores(opt_shuf.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)

        m_swap_uncal = self._tier1_metrics(q_swap_uncal)
        m_top1_uncal = self._tier1_metrics(q_top1_uncal)
        m_top3_uncal = self._tier1_metrics(q_top3_uncal)
        m_shuf_uncal = self._tier1_metrics(q_shuf_uncal)

        m_swap_cal = m_top1_cal = m_top3_cal = m_shuf_cal = None
        if self.mcq_temperature_T is not None:
            T_ext = float(self.mcq_temperature_T)
            q_swap_cal = self._question_level_from_option_scores(opt_swap.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_top1_cal = self._question_level_from_option_scores(opt_top1.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_top3_cal = self._question_level_from_option_scores(opt_top3.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_shuf_cal = self._question_level_from_option_scores(opt_shuf.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            m_swap_cal = self._tier1_metrics(q_swap_cal)
            m_top1_cal = self._tier1_metrics(q_top1_cal)
            m_top3_cal = self._tier1_metrics(q_top3_cal)
            m_shuf_cal = self._tier1_metrics(q_shuf_cal)

        tier2_report = {
            "system": "S3 (BM25 retrieval -> V* trained on MIMIC)",
            "n_questions_total": int(len(qpred_uncal)),
            "n_questions_used_for_tier2": int(len(pd.unique(opt_eval["qa_row_id"].astype(int)))),
            "baseline_uncalibrated": {k: tier1_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
            "baseline_calibrated": ({k: tier1_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if tier1_cal is not None else None),
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "uncalibrated": m_rand_uncal,
                    "delta_uncalibrated": deltas(m_rand_uncal, tier1_uncal),
                    "calibrated": m_rand_cal,
                    "delta_calibrated": (deltas(m_rand_cal, tier1_cal) if (m_rand_cal is not None and tier1_cal is not None) else None),
                },
                "swapped_evidence": {
                    "uncalibrated": {k: m_swap_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_swap_uncal, tier1_uncal),
                    "calibrated": ({k: m_swap_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_swap_cal is not None else None),
                    "delta_calibrated": (deltas(m_swap_cal, tier1_cal) if (m_swap_cal is not None and tier1_cal is not None) else None),
                },
                "truncate_top1": {
                    "uncalibrated": {k: m_top1_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_top1_uncal, tier1_uncal),
                    "calibrated": ({k: m_top1_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_top1_cal is not None else None),
                    "delta_calibrated": (deltas(m_top1_cal, tier1_cal) if (m_top1_cal is not None and tier1_cal is not None) else None),
                },
                "truncate_top3": {
                    "uncalibrated": {k: m_top3_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_top3_uncal, tier1_uncal),
                    "calibrated": ({k: m_top3_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_top3_cal is not None else None),
                    "delta_calibrated": (deltas(m_top3_cal, tier1_cal) if (m_top3_cal is not None and tier1_cal is not None) else None),
                },
                "shuffle_order": {
                    "uncalibrated": {k: m_shuf_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_shuf_uncal, tier1_uncal),
                    "calibrated": ({k: m_shuf_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_shuf_cal is not None else None),
                    "delta_calibrated": (deltas(m_shuf_cal, tier1_cal) if (m_shuf_cal is not None and tier1_cal is not None) else None),
                },
            },
        }

        # Save question-level faithfulness parquet (uncal only) for the subset used
        tier2_q = qpred_uncal[qpred_uncal["qa_row_id"].astype(int).isin(pd.unique(opt_eval["qa_row_id"].astype(int)))].copy()
        tier2_q = tier2_q.rename(columns={"is_correct": "base_is_correct", "conf": "base_conf", "nll": "base_nll"})
        tier2_q["swap_is_correct"] = q_swap_uncal["is_correct"].astype(int).values
        tier2_q["top1_is_correct"] = q_top1_uncal["is_correct"].astype(int).values
        tier2_q["top3_is_correct"] = q_top3_uncal["is_correct"].astype(int).values
        tier2_q["shuffle_is_correct"] = q_shuf_uncal["is_correct"].astype(int).values
        tier2_pred_path = os.path.join(self.out_dir, "s3_tier2_faithfulness_predictions_question_level.parquet")
        tier2_q.to_parquet(tier2_pred_path, index=False)

        tier2_path = os.path.join(self.out_dir, "s3_tier2_faithfulness_report.json")
        with open(tier2_path, "w", encoding="utf-8") as f:
            json.dump({**tier2_report, "artifacts": {"question_level_parquet": tier2_pred_path}}, f, indent=2)

        # Tier1 report
        tier1_report = {
            "system": "S3 (BM25 retrieval -> V* trained on MIMIC)",
            "dataset": "Vignette MCQ (169 questions; 4/5 options mixed)",
            "paths": {"csv_4option": self.vignette_paths.csv_4option, "csv_5option": self.vignette_paths.csv_5option},
            "n_questions": int(len(qpred_uncal)),
            "n_options_rows": int(len(opt_pred)),
            "retriever": {
                "type": "BM25",
                "bm25_index_pkl": self.bm25_index_path,
                "top_k": self.top_k,
                "k1": self.bm25_k1,
                "b": self.bm25_b,
            },
            "temperature_for_mcq_softmax": {
                "uncalibrated_T": 1.0,
                "external_calibrated_T": float(self.mcq_temperature_T) if self.mcq_temperature_T is not None else None,
            },
            "tier1_uncalibrated": tier1_uncal,
            "tier1_calibrated": tier1_cal,
            "tier3_slices_uncalibrated": slice_uncal,
            "tier3_slices_calibrated": slice_cal,
            "artifacts": {
                "option_level_predictions_parquet": opt_path,
                "question_level_predictions_uncalibrated_parquet": q_uncal_path,
                "question_level_predictions_calibrated_parquet": q_cal_path,
            },
        }
        tier1_path = os.path.join(self.out_dir, "s3_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        summary = {
            "tier1_report": tier1_path,
            "tier2_faithfulness_report": tier2_path,
            "option_level_predictions": opt_path,
            "question_level_predictions_uncalibrated": q_uncal_path,
            "question_level_predictions_calibrated": q_cal_path,
            "tier2_faithfulness_predictions_question_level": tier2_pred_path,
        }
        summary_path = os.path.join(self.out_dir, "s3_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {tier2_path}")
        print(f"Wrote: {summary_path}")




        # summarizing all into a single csv ....
        # summarizing all into a single csv ....
        # summarizing all into a single csv ....
        tier1_path = os.path.join(self.out_dir, "s3_tier1_report.json")
        tier2_faith_path = os.path.join(self.out_dir, "s3_tier2_faithfulness_report.json")
        out_csv = os.path.join(self.out_dir, "S3_vignette_full_results.csv")

        with open(tier1_path, "r") as f:
            t1 = json.load(f)

        with open(tier2_faith_path, "r") as f:
            t2f = json.load(f)

        rows = []

        def add_row(tier, subgroup, metric, value):
            rows.append({"Tier": tier, "Subgroup": subgroup, "Metric": metric, "Value": value})

        def add_metrics_block(tier, subgroup, block, keys):
            if not block:
                return
            for k in keys:
                if k in block and block[k] is not None:
                    add_row(tier, subgroup, k, block[k])

        t1_uncal = t1.get("tier1_uncalibrated")
        add_metrics_block("Tier1", "overall_uncal", t1_uncal, ["Accuracy", "NLL", "Brier", "ECE"])
        if t1_uncal and t1_uncal.get("SelectiveRisk"):
            sr = t1_uncal["SelectiveRisk"]
            add_metrics_block("Tier1", "overall_uncal", sr, ["AURC01", "risk_01@60", "risk_01@80", "risk_01@90"])

        t1_cal = t1.get("tier1_calibrated")
        add_metrics_block("Tier1", "overall_cal", t1_cal, ["Accuracy", "NLL", "Brier", "ECE"])
        if t1_cal and t1_cal.get("SelectiveRisk"):
            sr = t1_cal["SelectiveRisk"]
            add_metrics_block("Tier1", "overall_cal", sr, ["AURC01", "risk_01@60", "risk_01@80", "risk_01@90"])

        base_u = t2f.get("baseline_uncalibrated")
        add_metrics_block("Tier2_Faithfulness", "baseline_uncal", base_u, ["Accuracy", "NLL", "Brier", "ECE"])

        base_c = t2f.get("baseline_calibrated")
        add_metrics_block("Tier2_Faithfulness", "baseline_cal", base_c, ["Accuracy", "NLL", "Brier", "ECE"])

        for pert_name, pert in (t2f.get("perturbations") or {}).items():
            # keep both absolute metrics and deltas (when present)
            add_metrics_block("Tier2_Faithfulness", pert_name, pert.get("uncalibrated"),
                              ["Accuracy", "NLL", "Brier", "ECE"])
            add_metrics_block("Tier2_Faithfulness", pert_name, pert.get("delta_uncalibrated"),
                              ["ΔAccuracy", "ΔNLL", "ΔBrier", "ΔECE"])
            add_metrics_block("Tier2_Faithfulness", pert_name, pert.get("calibrated"),
                              ["Accuracy", "NLL", "Brier", "ECE"])
            add_metrics_block("Tier2_Faithfulness", pert_name, pert.get("delta_calibrated"),
                              ["ΔAccuracy", "ΔNLL", "ΔBrier", "ΔECE"])

        slices_u = t1.get("tier3_slices_uncalibrated") or {}
        for slice_name, vals in slices_u.items():
            if "n" in vals and vals["n"] is not None:
                add_row("Tier3", slice_name, "n", vals["n"])
            add_metrics_block("Tier3", slice_name, vals, ["Accuracy", "NLL", "Brier", "ECE", "AURC01"])

        slices_c = t1.get("tier3_slices_calibrated") or {}
        for slice_name, vals in slices_c.items():
            if "n" in vals and vals["n"] is not None:
                add_row("Tier3_cal", slice_name, "n", vals["n"])
            add_metrics_block("Tier3_cal", slice_name, vals, ["Accuracy", "NLL", "Brier", "ECE", "AURC01"])

        df = pd.DataFrame(rows, columns=["Tier", "Subgroup", "Metric", "Value"])

        def fmt3(x):
            if pd.isna(x):
                return x
            # keep ints as ints (like n), format everything else to 3 decimals
            if isinstance(x, (int,)) or (
                    isinstance(x, float) and float(x).is_integer() and df.loc[df["Value"] == x, "Metric"].eq(
                    "n").any()):
                return int(x)
            try:
                return f"{float(x):.3f}"
            except Exception:
                return x

        df["Value"] = df["Value"].apply(fmt3)

        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")


        return summary_path








@dataclass
class VignetteEvalPaths:
    csv_4option: str
    csv_5option: str


class VignetteS2Tier123End2EndEvaluator:
    """
    S2 external validation on Vignette MCQs (169 Qs, TEST ONLY):
      - Off-the-shelf MPNet retrieval (no finetune)
      - Builds its own FAISS + chunk embeddings inside out_dir
      - Verifier V* (trained on MIMIC) scores each option
      - Softmax across options per question -> MCQ probabilities

    Metrics (MCQ):
      Tier1:
        - Accuracy
        - NLL (true option)
        - Multiclass Brier
        - Top-1 ECE (max-prob calibration)
        - Selective risk (risk_01 = 1-acc vs coverage), AURC01
        - Conf and margin in question parquet

      Tier2 faithfulness (evidence corruption, deltas vs baseline):
        - random evidence (avg over trials)
        - swapped evidence across questions
        - truncate to top-1 / top-3
        - shuffle chunk order

      Tier3 slices (dataset heuristics):
        - option_count: 4 vs 5
        - question_length_bucket: short/medium/long (tertiles)
        - contains_modality_terms: 0/1
        - contains_uncertainty_terms: 0/1
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,
        vignette_paths: VignetteEvalPaths,
        device: str = "cuda",
        top_k: int = 20,
        verifier_max_tokens: int = 512,
        evidence_sep: str = "\n\n---\n\n",
        batch_size_embed_chunks: int = 128,
        batch_size_retrieve: int = 256,
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        seed: int = 13,
        mcq_temperature_T: Optional[float] = None,
        # Tier2 knobs
        faithfulness_random_trials: int = 1,
        faithfulness_max_eval_questions: Optional[int] = None,
        # Output
        out_dir: Optional[str] = None,
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path
        self.vignette_paths = vignette_paths

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.top_k = int(top_k)
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.evidence_sep = str(evidence_sep)

        self.batch_size_embed_chunks = int(batch_size_embed_chunks)
        self.batch_size_retrieve = int(batch_size_retrieve)
        self.batch_size_verify = int(batch_size_verify)

        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.mcq_temperature_T = float(mcq_temperature_T) if mcq_temperature_T is not None else None
        self.faithfulness_random_trials = int(faithfulness_random_trials)
        self.faithfulness_max_eval_questions = int(faithfulness_max_eval_questions) if faithfulness_max_eval_questions is not None else None

        self.chunks_parquet = os.path.join(self.base_path_evidenceData, "chunks.parquet")

        default_out = os.path.join(
            self.base_path_evidenceData,
            "external_validation",
            "Vignette",
            "tier123_results_s2_end2end",
        )
        self.out_dir = out_dir or default_out
        os.makedirs(self.out_dir, exist_ok=True)

        # S2 artifacts stored inside out_dir
        self.faiss_index_path = os.path.join(self.out_dir, "faiss_mpnet_offtheshelf.index")
        self.chunk_ids_path = os.path.join(self.out_dir, "chunk_ids.npy")
        self.chunk_embeds_path = os.path.join(self.out_dir, "chunk_embeds_mpnet_offtheshelf.npy")

        self._set_determinism()

        self.params = open_experiment(self.experiment_name, self.cfg_path)

        # retriever (off-the-shelf mpnet)
        self.retriever_tok = None
        self.retriever_model = None
        self.max_query_tokens = None
        self.max_doc_tokens = None

        # verifier
        self.verifier_tok = None
        self.verifier_model = None

        # evidence + index
        self.chunk_map = None
        self.faiss_index = None
        self.faiss_chunk_ids = None
        self.all_chunk_ids_universe = None

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

    def _materialize_evidence(self, chunk_ids: List[int]) -> str:
        assert self.chunk_map is not None
        texts = []
        for cid in chunk_ids:
            t = self.chunk_map.get(int(cid), "")
            if t:
                texts.append(t)
        return self.evidence_sep.join(texts)


    def _load_offtheshelf_mpnet(self) -> None:
        model_name = self.params["retriever"]["model_name"]  # sentence-transformers/all-mpnet-base-v2
        self.max_query_tokens = int(self.params["retriever"]["max_query_tokens"])
        self.max_doc_tokens = int(self.params["retriever"]["max_doc_tokens"])

        self.retriever_tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.retriever_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.retriever_model.eval()

    def _mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _normalize_text(self, x: str) -> str:
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
                batch = [self._normalize_text(x) for x in texts[i : i + batch_size]]
                enc = self.retriever_tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=int(max_len),
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
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.chunk_ids_path) and os.path.exists(self.chunk_embeds_path):
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
            max_len=int(self.max_doc_tokens),
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

    def _retrieve_topk_chunk_ids(self, query_texts: List[str], desc: str) -> List[List[int]]:
        assert self.faiss_index is not None
        assert self.faiss_chunk_ids is not None
        q_emb = self._encode_texts(
            texts=query_texts,
            max_len=int(self.max_query_tokens),
            batch_size=self.batch_size_retrieve,
            desc=desc,
        )
        _, I = self.faiss_index.search(q_emb.astype(np.float32), self.top_k)

        all_ids: List[List[int]] = []
        for j in range(I.shape[0]):
            idxs = I[j].tolist()
            cids = [int(self.faiss_chunk_ids[ii]) for ii in idxs if ii >= 0]
            all_ids.append(cids)
        return all_ids


    def _clean(self, s: str) -> str:
        return " ".join(str(s).split()).strip()

    def _load_vignette_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.dropna(how="all").reset_index(drop=True)
        df.columns = [str(c) for c in df.columns]
        return df

    def _build_mcq_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q_rows: List[Dict] = []
        opt_rows: List[Dict] = []
        qa_row_id = 0

        # 4-option
        if self.vignette_paths.csv_4option and os.path.exists(self.vignette_paths.csv_4option):
            df4 = self._load_vignette_csv(self.vignette_paths.csv_4option)
            required4 = ["q number", "question", "option 1", "option 2", "option 3", "option 4", "answer index"]
            missing4 = [c for c in required4 if c not in df4.columns]
            if missing4:
                raise ValueError(f"4-option CSV missing columns: {missing4}. Found: {list(df4.columns)}")

            for _, r in df4.iterrows():
                qtext = self._clean(r["question"])
                try:
                    ans_idx = int(pd.to_numeric(r["answer index"], errors="raise"))
                except Exception:
                    continue
                if ans_idx < 1 or ans_idx > 4:
                    continue
                correct_index = ans_idx - 1

                options = [
                    ("A", self._clean(r["option 1"])),
                    ("B", self._clean(r["option 2"])),
                    ("C", self._clean(r["option 3"])),
                    ("D", self._clean(r["option 4"])),
                ]
                options = [(L, t) for (L, t) in options if t != ""]
                if len(options) != 4:
                    continue

                qid = qa_row_id
                qa_row_id += 1

                q_rows.append(
                    {
                        "qa_row_id": qid,
                        "question": qtext,
                        "num_options": 4,
                        "options_json": json.dumps([{"letter": L, "text": t} for L, t in options]),
                        "correct_letter": options[correct_index][0],
                        "correct_index": int(correct_index),
                        "source_split": "4option",
                        "source_question_id": self._clean(r["q number"]),
                    }
                )

                for opt_idx, (L, opt_text) in enumerate(options):
                    retrieval_query = f"{qtext}\n\nOption ({L}): {opt_text}"
                    opt_rows.append(
                        {
                            "qa_row_id": qid,
                            "option_index": int(opt_idx),
                            "option_letter": L,
                            "option_text": opt_text,
                            "question": qtext,
                            "retrieval_query": retrieval_query,
                            "is_correct": int(opt_idx == correct_index),
                            "correct_index": int(correct_index),
                            "num_options": 4,
                            "source_split": "4option",
                        }
                    )

        # 5-option
        if self.vignette_paths.csv_5option and os.path.exists(self.vignette_paths.csv_5option):
            df5 = self._load_vignette_csv(self.vignette_paths.csv_5option)
            required5 = ["question_number", "question", "option_A", "option_B", "option_C", "option_D", "option_E", "solution_index"]
            missing5 = [c for c in required5 if c not in df5.columns]
            if missing5:
                raise ValueError(f"5-option CSV missing columns: {missing5}. Found: {list(df5.columns)}")

            for _, r in df5.iterrows():
                qtext = self._clean(r["question"])
                ans_letter = self._clean(r["solution_index"]).upper()
                if ans_letter not in ["A", "B", "C", "D", "E"]:
                    continue

                options = [
                    ("A", self._clean(r["option_A"])),
                    ("B", self._clean(r["option_B"])),
                    ("C", self._clean(r["option_C"])),
                    ("D", self._clean(r["option_D"])),
                    ("E", self._clean(r["option_E"])),
                ]
                options = [(L, t) for (L, t) in options if t != ""]
                if len(options) != 5:
                    continue

                letter_to_index = {L: i for i, (L, _) in enumerate(options)}
                correct_index = letter_to_index.get(ans_letter, None)
                if correct_index is None:
                    continue

                qid = qa_row_id
                qa_row_id += 1

                q_rows.append(
                    {
                        "qa_row_id": qid,
                        "question": qtext,
                        "num_options": 5,
                        "options_json": json.dumps([{"letter": L, "text": t} for L, t in options]),
                        "correct_letter": ans_letter,
                        "correct_index": int(correct_index),
                        "source_split": "5option",
                        "source_question_id": self._clean(r["question_number"]),
                    }
                )

                for opt_idx, (L, opt_text) in enumerate(options):
                    retrieval_query = f"{qtext}\n\nOption ({L}): {opt_text}"
                    opt_rows.append(
                        {
                            "qa_row_id": qid,
                            "option_index": int(opt_idx),
                            "option_letter": L,
                            "option_text": opt_text,
                            "question": qtext,
                            "retrieval_query": retrieval_query,
                            "is_correct": int(opt_idx == correct_index),
                            "correct_index": int(correct_index),
                            "num_options": 5,
                            "source_split": "5option",
                        }
                    )

        questions_df = pd.DataFrame(q_rows)
        options_df = pd.DataFrame(opt_rows)

        if len(questions_df) == 0 or len(options_df) == 0:
            raise RuntimeError("No questions/options parsed. Check CSV paths and schemas.")

        cnt = options_df.groupby("qa_row_id").size().reset_index(name="n")
        merged = questions_df.merge(cnt, on="qa_row_id", how="left")
        bad = merged[(merged["n"].fillna(-1).astype(int) != merged["num_options"].astype(int))]
        if len(bad) > 0:
            raise RuntimeError(f"Expanded rows mismatch for {len(bad)} questions. Example:\n{bad.head(3)}")

        return questions_df.reset_index(drop=True), options_df.reset_index(drop=True)

    def _build_verifier_input(self, question: str, option_letter: str, option_text: str, evidence_text: str) -> str:
        q = self._clean(question)
        a = self._clean(f"({option_letter}) {option_text}")
        e = self._clean(evidence_text)
        return f"Question: {q}\nProposed Answer: {a}\nEvidence: {e}"

    def _predict_scores(self, inputs: List[str], desc: str) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), self.batch_size_verify),
                total=math.ceil(len(inputs) / self.batch_size_verify),
                desc=desc,
                unit="batch",
            ):
                batch = inputs[i : i + self.batch_size_verify]
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

    def _softmax_probs(self, scores: np.ndarray, T: float) -> np.ndarray:
        z = scores.astype(np.float64) / float(max(T, 1e-6))
        m = float(z.max())
        ex = np.exp(np.clip(z - m, -50, 50))
        return ex / max(float(ex.sum()), 1e-12)

    def _question_level_from_option_scores(self, opt_df: pd.DataFrame, score_col: str, T: float) -> pd.DataFrame:
        rows = []
        for qid, g in opt_df.groupby("qa_row_id", sort=True):
            scores = g[score_col].to_numpy(dtype=np.float64)
            p = self._softmax_probs(scores, T=T)

            correct_index = int(g["correct_index"].iloc[0])
            pred_index = int(np.argmax(p))
            conf = float(np.max(p))
            p_sorted = np.sort(p)
            margin = float(p_sorted[-1] - p_sorted[-2]) if len(p_sorted) >= 2 else float(p_sorted[-1])

            nll = float(-math.log(max(float(p[correct_index]), 1e-12)))
            y = np.zeros_like(p)
            y[correct_index] = 1.0
            brier = float(np.sum((p - y) ** 2))

            rows.append(
                {
                    "qa_row_id": int(qid),
                    "num_options": int(g["num_options"].iloc[0]),
                    "correct_index": correct_index,
                    "pred_index": pred_index,
                    "is_correct": int(pred_index == correct_index),
                    "conf": conf,
                    "margin": margin,
                    "p_true": float(p[correct_index]),
                    "nll": nll,
                    "brier_mcq": brier,
                    "source_split": str(g["source_split"].iloc[0]) if "source_split" in g.columns else None,
                }
            )
        return pd.DataFrame(rows)

    def _ece_top1(self, correct01: np.ndarray, conf: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(correct01)
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (conf >= lo) & (conf < hi) if b < bins - 1 else (conf >= lo) & (conf <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(correct01[mask].mean())
            cbar = float(conf[mask].mean())
            ece += (m / n) * abs(acc - cbar)
        return float(ece)

    def _selective_risk(self, is_correct: np.ndarray, conf: np.ndarray) -> Dict[str, object]:
        order = np.argsort(-conf)
        coverages = np.linspace(0.1, 1.0, 10)

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(order))))
            return float(1.0 - is_correct[order[:k]].mean())

        curve = [{"coverage": float(c), "risk_01": risk_at(float(c))} for c in coverages]
        fixed = [0.6, 0.8, 0.9]
        fixed_out = {f"risk_01@{int(c*100)}": risk_at(c) for c in fixed}
        aurc01 = float(np.mean([x["risk_01"] for x in curve]))
        return {"curve": curve, "AURC01": aurc01, **fixed_out}

    def _tier1_metrics(self, qdf: pd.DataFrame) -> Dict[str, object]:
        acc = float(qdf["is_correct"].mean()) if len(qdf) else 0.0
        nll = float(qdf["nll"].mean()) if len(qdf) else 0.0
        brier = float(qdf["brier_mcq"].mean()) if len(qdf) else 0.0
        ece = self._ece_top1(
            qdf["is_correct"].to_numpy(np.float32),
            qdf["conf"].to_numpy(np.float32),
            self.ece_bins,
        ) if len(qdf) else 0.0
        sel = self._selective_risk(
            qdf["is_correct"].to_numpy(np.float32),
            qdf["conf"].to_numpy(np.float32),
        ) if len(qdf) else {}
        return {"Accuracy": acc, "NLL": nll, "Brier": brier, "ECE": float(ece), "SelectiveRisk": sel}

    def _token_count(self, s: str) -> int:
        return len(self._clean(s).split())

    def _slice_tables(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        df = questions_df.copy()
        qlens = df["question"].astype(str).map(self._token_count).to_numpy()

        q1 = float(np.quantile(qlens, 1 / 3))
        q2 = float(np.quantile(qlens, 2 / 3))

        def bucket(x: int) -> str:
            if x <= q1:
                return "short"
            if x <= q2:
                return "medium"
            return "long"

        df["question_len_tokens"] = qlens.astype(int)
        df["question_length_bucket"] = [bucket(int(x)) for x in qlens.tolist()]

        modality_terms = ["ct", "mri", "ultrasound", "us", "x-ray", "radiograph", "mammogram", "pet", "angiogram"]
        uncertain_terms = ["likely", "possibly", "may represent", "could represent", "suggest", "suspicious", "indeterminate"]

        qlow = df["question"].astype(str).str.lower()
        df["contains_modality_terms"] = qlow.apply(lambda s: int(any(t in s for t in modality_terms)))
        df["contains_uncertainty_terms"] = qlow.apply(lambda s: int(any(t in s for t in uncertain_terms)))

        df["option_count"] = pd.to_numeric(df["num_options"], errors="coerce").astype("Int64")
        return df

    def _compute_slice_metrics(self, qpred: pd.DataFrame, qtags: pd.DataFrame, prefix: str) -> Dict[str, object]:
        df = qpred.merge(
            qtags[["qa_row_id", "question_length_bucket", "contains_modality_terms", "contains_uncertainty_terms", "option_count"]],
            on="qa_row_id",
            how="left",
        )
        out: Dict[str, object] = {}

        def pack(mask: np.ndarray) -> Dict[str, object]:
            if int(mask.sum()) == 0:
                return {"n": 0, "Accuracy": None, "NLL": None, "ECE": None, "Brier": None, "AURC01": None}
            sub = df.loc[mask].reset_index(drop=True)
            m = self._tier1_metrics(sub)
            return {
                "n": int(len(sub)),
                "Accuracy": m["Accuracy"],
                "NLL": m["NLL"],
                "ECE": m["ECE"],
                "Brier": m["Brier"],
                "AURC01": float(m["SelectiveRisk"]["AURC01"]) if "SelectiveRisk" in m and "AURC01" in m["SelectiveRisk"] else None,
            }

        oc = pd.to_numeric(df["option_count"], errors="coerce").fillna(-1).astype(int).to_numpy()
        for v in [4, 5]:
            out[f"{prefix}.option_count.{v}"] = pack(oc == v)

        for v in ["short", "medium", "long"]:
            out[f"{prefix}.question_length_bucket.{v}"] = pack(df["question_length_bucket"].astype(str).to_numpy() == v)

        m1 = pd.to_numeric(df["contains_modality_terms"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        out[f"{prefix}.contains_modality_terms.1"] = pack(m1)
        out[f"{prefix}.contains_modality_terms.0"] = pack(~m1)

        u1 = pd.to_numeric(df["contains_uncertainty_terms"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        out[f"{prefix}.contains_uncertainty_terms.1"] = pack(u1)
        out[f"{prefix}.contains_uncertainty_terms.0"] = pack(~u1)

        return out

    def _stress_random_ids(self, n_rows: int) -> List[List[int]]:
        U = self.all_chunk_ids_universe
        out = []
        for _ in range(n_rows):
            size = min(self.top_k, len(U))
            idx = np.random.choice(len(U), size=size, replace=False)
            out.append([int(x) for x in U[idx].tolist()])
        return out

    def _stress_swap_between_questions(self, ids_per_row: List[List[int]], opt_df: pd.DataFrame) -> List[List[int]]:
        qids = opt_df["qa_row_id"].astype(int).to_numpy()
        uniq = sorted(pd.unique(qids).tolist())
        if len(uniq) <= 1:
            return ids_per_row

        q_to_rows = {qid: opt_df.index[opt_df["qa_row_id"].astype(int) == qid].tolist() for qid in uniq}
        q_to_firstrow = {qid: q_to_rows[qid][0] for qid in uniq}
        q_to_evidence = {qid: ids_per_row[q_to_firstrow[qid]] for qid in uniq}

        shifted = {}
        for i, qid in enumerate(uniq):
            q_next = uniq[(i + 1) % len(uniq)]
            shifted[qid] = q_to_evidence[q_next]

        out = []
        for qid in qids.tolist():
            out.append(list(shifted[int(qid)][: self.top_k]))
        return out

    def _stress_truncate(self, ids_per_row: List[List[int]], k: int) -> List[List[int]]:
        kk = max(1, int(k))
        return [list(x[:kk]) for x in ids_per_row]

    def _stress_shuffle_order(self, ids_per_row: List[List[int]]) -> List[List[int]]:
        out = []
        for x in ids_per_row:
            y = list(x[: self.top_k])
            random.shuffle(y)
            out.append(y)
        return out


    def run(self) -> str:
        self.chunk_map = self._load_chunk_map()
        self._load_offtheshelf_mpnet()
        self._build_or_load_faiss_s2()
        self._load_verifier()

        questions_df, options_df = self._build_mcq_tables()
        qtags = self._slice_tables(questions_df)

        # Baseline: retrieve per option-row, then verify
        queries = options_df["retrieval_query"].astype(str).tolist()
        topk_ids = self._retrieve_topk_chunk_ids(queries, desc="S2 Retrieval (MPNet off-the-shelf) [Vignette options]")

        verifier_inputs = []
        for i in range(len(options_df)):
            row = options_df.iloc[i]
            ev = self._materialize_evidence(topk_ids[i])
            verifier_inputs.append(self._build_verifier_input(row["question"], row["option_letter"], row["option_text"], ev))

        scores = self._predict_scores(verifier_inputs, desc="S2 Verification (V*) [Vignette options]")

        opt_pred = options_df.copy()
        opt_pred["s2_retrieved_chunk_ids_json"] = [json.dumps(x) for x in topk_ids]
        opt_pred["s2_score_raw"] = scores.astype(np.float32)

        # Tier1 baseline
        qpred_uncal = self._question_level_from_option_scores(opt_pred, score_col="s2_score_raw", T=1.0)
        tier1_uncal = self._tier1_metrics(qpred_uncal)

        tier1_cal = None
        qpred_cal = None
        if self.mcq_temperature_T is not None:
            T_ext = float(max(self.mcq_temperature_T, 1e-6))
            qpred_cal = self._question_level_from_option_scores(opt_pred, score_col="s2_score_raw", T=T_ext)
            tier1_cal = self._tier1_metrics(qpred_cal)

        # Save predictions
        opt_path = os.path.join(self.out_dir, "s2_option_level_predictions.parquet")
        q_uncal_path = os.path.join(self.out_dir, "s2_question_level_predictions_uncalibrated.parquet")
        opt_pred.to_parquet(opt_path, index=False)
        qpred_uncal.to_parquet(q_uncal_path, index=False)

        q_cal_path = None
        if qpred_cal is not None:
            q_cal_path = os.path.join(self.out_dir, "s2_question_level_predictions_calibrated.parquet")
            qpred_cal.to_parquet(q_cal_path, index=False)

        # Tier3 slices
        slice_uncal = self._compute_slice_metrics(qpred_uncal, qtags, prefix="slice_uncal")
        slice_cal = None
        if qpred_cal is not None:
            slice_cal = self._compute_slice_metrics(qpred_cal, qtags, prefix="slice_cal")

        # Tier2 faithfulness (optional sub-sample questions)
        opt_eval = opt_pred
        if self.faithfulness_max_eval_questions is not None and self.faithfulness_max_eval_questions > 0:
            rng = np.random.RandomState(self.seed)
            qids = qpred_uncal["qa_row_id"].astype(int).to_numpy()
            n = min(len(qids), self.faithfulness_max_eval_questions)
            keep_qids = set(rng.choice(qids, size=n, replace=False).tolist())
            opt_eval = opt_pred[opt_pred["qa_row_id"].astype(int).isin(keep_qids)].reset_index(drop=True)

        base_ids = [json.loads(s) for s in opt_eval["s2_retrieved_chunk_ids_json"].astype(str).tolist()]

        def score_with_ids(opt_subset: pd.DataFrame, ids_per_row: List[List[int]], desc: str) -> pd.DataFrame:
            inputs = []
            for i in range(len(opt_subset)):
                r = opt_subset.iloc[i]
                ev = self._materialize_evidence(ids_per_row[i])
                inputs.append(self._build_verifier_input(r["question"], r["option_letter"], r["option_text"], ev))
            sc = self._predict_scores(inputs, desc=desc)
            out = opt_subset.copy()
            out["score_raw"] = sc.astype(np.float32)
            return out

        def deltas(m: Dict[str, object], base: Dict[str, object]) -> Dict[str, float]:
            keys = ["Accuracy", "NLL", "Brier", "ECE"]
            return {f"Δ{k}": float(m[k] - base[k]) for k in keys}

        # Random evidence (avg)
        rand_metrics_uncal = []
        rand_metrics_cal = []
        for t in range(self.faithfulness_random_trials):
            ids_rand = self._stress_random_ids(len(opt_eval))
            opt_rand = score_with_ids(opt_eval, ids_rand, desc=f"Faithfulness: random evidence (trial {t+1})")
            q_rand_uncal = self._question_level_from_option_scores(opt_rand.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
            rand_metrics_uncal.append(self._tier1_metrics(q_rand_uncal))
            if self.mcq_temperature_T is not None:
                q_rand_cal = self._question_level_from_option_scores(opt_rand.rename(columns={"score_raw": "tmp"}), "tmp", T=float(self.mcq_temperature_T))
                rand_metrics_cal.append(self._tier1_metrics(q_rand_cal))

        def avg_metrics(ms: List[Dict[str, object]]) -> Dict[str, object]:
            if not ms:
                return {"Accuracy": 0.0, "NLL": 0.0, "Brier": 0.0, "ECE": 0.0}
            return {
                "Accuracy": float(np.mean([m["Accuracy"] for m in ms])),
                "NLL": float(np.mean([m["NLL"] for m in ms])),
                "Brier": float(np.mean([m["Brier"] for m in ms])),
                "ECE": float(np.mean([m["ECE"] for m in ms])),
            }

        m_rand_uncal = avg_metrics(rand_metrics_uncal)
        m_rand_cal = avg_metrics(rand_metrics_cal) if self.mcq_temperature_T is not None else None

        # Swap, truncate, shuffle
        ids_swap = self._stress_swap_between_questions(base_ids, opt_eval)
        ids_top1 = self._stress_truncate(base_ids, 1)
        ids_top3 = self._stress_truncate(base_ids, 3)
        ids_shuffle = self._stress_shuffle_order(base_ids)

        opt_swap = score_with_ids(opt_eval, ids_swap, desc="Faithfulness: swapped evidence (question-level)")
        opt_top1 = score_with_ids(opt_eval, ids_top1, desc="Faithfulness: truncate top-1")
        opt_top3 = score_with_ids(opt_eval, ids_top3, desc="Faithfulness: truncate top-3")
        opt_shuf = score_with_ids(opt_eval, ids_shuffle, desc="Faithfulness: shuffle order")

        q_swap_uncal = self._question_level_from_option_scores(opt_swap.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_top1_uncal = self._question_level_from_option_scores(opt_top1.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_top3_uncal = self._question_level_from_option_scores(opt_top3.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_shuf_uncal = self._question_level_from_option_scores(opt_shuf.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)

        m_swap_uncal = self._tier1_metrics(q_swap_uncal)
        m_top1_uncal = self._tier1_metrics(q_top1_uncal)
        m_top3_uncal = self._tier1_metrics(q_top3_uncal)
        m_shuf_uncal = self._tier1_metrics(q_shuf_uncal)

        m_swap_cal = m_top1_cal = m_top3_cal = m_shuf_cal = None
        if self.mcq_temperature_T is not None:
            T_ext = float(self.mcq_temperature_T)
            q_swap_cal = self._question_level_from_option_scores(opt_swap.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_top1_cal = self._question_level_from_option_scores(opt_top1.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_top3_cal = self._question_level_from_option_scores(opt_top3.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_shuf_cal = self._question_level_from_option_scores(opt_shuf.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            m_swap_cal = self._tier1_metrics(q_swap_cal)
            m_top1_cal = self._tier1_metrics(q_top1_cal)
            m_top3_cal = self._tier1_metrics(q_top3_cal)
            m_shuf_cal = self._tier1_metrics(q_shuf_cal)

        tier2_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V* trained on MIMIC)",
            "n_questions_total": int(len(qpred_uncal)),
            "n_questions_used_for_tier2": int(len(pd.unique(opt_eval["qa_row_id"].astype(int)))),
            "baseline_uncalibrated": {k: tier1_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
            "baseline_calibrated": ({k: tier1_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if tier1_cal is not None else None),
            "perturbations": {
                "random_evidence": {
                    "trials": int(self.faithfulness_random_trials),
                    "uncalibrated": m_rand_uncal,
                    "delta_uncalibrated": deltas(m_rand_uncal, tier1_uncal),
                    "calibrated": m_rand_cal,
                    "delta_calibrated": (deltas(m_rand_cal, tier1_cal) if (m_rand_cal is not None and tier1_cal is not None) else None),
                },
                "swapped_evidence": {
                    "uncalibrated": {k: m_swap_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_swap_uncal, tier1_uncal),
                    "calibrated": ({k: m_swap_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_swap_cal is not None else None),
                    "delta_calibrated": (deltas(m_swap_cal, tier1_cal) if (m_swap_cal is not None and tier1_cal is not None) else None),
                },
                "truncate_top1": {
                    "uncalibrated": {k: m_top1_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_top1_uncal, tier1_uncal),
                    "calibrated": ({k: m_top1_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_top1_cal is not None else None),
                    "delta_calibrated": (deltas(m_top1_cal, tier1_cal) if (m_top1_cal is not None and tier1_cal is not None) else None),
                },
                "truncate_top3": {
                    "uncalibrated": {k: m_top3_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_top3_uncal, tier1_uncal),
                    "calibrated": ({k: m_top3_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_top3_cal is not None else None),
                    "delta_calibrated": (deltas(m_top3_cal, tier1_cal) if (m_top3_cal is not None and tier1_cal is not None) else None),
                },
                "shuffle_order": {
                    "uncalibrated": {k: m_shuf_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_shuf_uncal, tier1_uncal),
                    "calibrated": ({k: m_shuf_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_shuf_cal is not None else None),
                    "delta_calibrated": (deltas(m_shuf_cal, tier1_cal) if (m_shuf_cal is not None and tier1_cal is not None) else None),
                },
            },
        }

        tier2_q = qpred_uncal[qpred_uncal["qa_row_id"].astype(int).isin(pd.unique(opt_eval["qa_row_id"].astype(int)))].copy()
        tier2_q = tier2_q.rename(columns={"is_correct": "base_is_correct", "conf": "base_conf", "nll": "base_nll"})
        tier2_q["swap_is_correct"] = q_swap_uncal["is_correct"].astype(int).values
        tier2_q["top1_is_correct"] = q_top1_uncal["is_correct"].astype(int).values
        tier2_q["top3_is_correct"] = q_top3_uncal["is_correct"].astype(int).values
        tier2_q["shuffle_is_correct"] = q_shuf_uncal["is_correct"].astype(int).values

        tier2_pred_path = os.path.join(self.out_dir, "s2_tier2_faithfulness_predictions_question_level.parquet")
        tier2_q.to_parquet(tier2_pred_path, index=False)

        tier2_path = os.path.join(self.out_dir, "s2_tier2_faithfulness_report.json")
        with open(tier2_path, "w", encoding="utf-8") as f:
            json.dump({**tier2_report, "artifacts": {"question_level_parquet": tier2_pred_path}}, f, indent=2)

        # Tier1 report
        tier1_report = {
            "system": "S2 (MPNet off-the-shelf retrieval -> V* trained on MIMIC)",
            "dataset": "Vignette MCQ (169 questions; 4/5 options mixed)",
            "paths": {"csv_4option": self.vignette_paths.csv_4option, "csv_5option": self.vignette_paths.csv_5option},
            "n_questions": int(len(qpred_uncal)),
            "n_options_rows": int(len(opt_pred)),
            "retriever": {
                "type": "MPNet off-the-shelf + FAISS",
                "mpnet_model": self.params["retriever"]["model_name"],
                "faiss_index": self.faiss_index_path,
                "chunk_ids_npy": self.chunk_ids_path,
                "chunk_embeds_npy": self.chunk_embeds_path,
                "top_k": self.top_k,
            },
            "temperature_for_mcq_softmax": {
                "uncalibrated_T": 1.0,
                "external_calibrated_T": float(self.mcq_temperature_T) if self.mcq_temperature_T is not None else None,
            },
            "tier1_uncalibrated": tier1_uncal,
            "tier1_calibrated": tier1_cal,
            "tier3_slices_uncalibrated": slice_uncal,
            "tier3_slices_calibrated": slice_cal,
            "artifacts": {
                "option_level_predictions_parquet": opt_path,
                "question_level_predictions_uncalibrated_parquet": q_uncal_path,
                "question_level_predictions_calibrated_parquet": q_cal_path,
            },
        }
        tier1_path = os.path.join(self.out_dir, "s2_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        summary = {
            "tier1_report": tier1_path,
            "tier2_faithfulness_report": tier2_path,
            "option_level_predictions": opt_path,
            "question_level_predictions_uncalibrated": q_uncal_path,
            "question_level_predictions_calibrated": q_cal_path,
            "tier2_faithfulness_predictions_question_level": tier2_pred_path,
        }
        summary_path = os.path.join(self.out_dir, "s2_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {tier2_path}")
        print(f"Wrote: {summary_path}")



        # summarizing all to a single csv ...
        # summarizing all to a single csv ...
        # summarizing all to a single csv ...

        tier1_path = os.path.join(self.out_dir, "s2_tier1_report.json")
        tier2_faith_path = os.path.join(self.out_dir, "s2_tier2_faithfulness_report.json")
        out_csv = os.path.join(self.out_dir, "S2_vignette_full_results.csv")

        def load_json(p):
            with open(p, "r") as f:
                return json.load(f)

        def add_row(rows, tier, subgroup, metric, value):
            rows.append({"Tier": tier, "Subgroup": subgroup, "Metric": metric, "Value": value})

        t1 = load_json(tier1_path)
        t2f = load_json(tier2_faith_path)

        rows = []

        t1u = t1.get("tier1_uncalibrated", {})

        for k in ["Accuracy", "NLL", "Brier", "ECE"]:
            if k in t1u:
                add_row(rows, "Tier1", "overall", k, t1u[k])

        sr = (t1u.get("SelectiveRisk") or {})
        for k in ["AURC01", "risk_01@60", "risk_01@80", "risk_01@90"]:
            if k in sr:
                add_row(rows, "Tier1", "overall", k, sr[k])

        # Tier 3 slices (stored inside tier1 report for Vignette)
        slices = t1.get("tier3_slices_uncalibrated", {}) or {}
        for slice_name, vals in slices.items():
            for k in ["n", "Accuracy", "NLL", "Brier", "ECE", "AURC01"]:
                if k in vals:
                    add_row(rows, "Tier3", slice_name, k, vals[k])

        base = t2f.get("baseline_uncalibrated", {}) or {}
        for k in ["Accuracy", "NLL", "Brier", "ECE"]:
            if k in base:
                add_row(rows, "Tier2_Faithfulness", "baseline", k, base[k])

        perts = t2f.get("perturbations", {}) or {}
        for pname, p in perts.items():
            uncal = p.get("uncalibrated", {}) or {}
            delta = p.get("delta_uncalibrated", {}) or {}

            for k in ["Accuracy", "NLL", "Brier", "ECE"]:
                if k in uncal:
                    add_row(rows, "Tier2_Faithfulness", pname, k, uncal[k])

            for k in ["ΔAccuracy", "ΔNLL", "ΔBrier", "ΔECE"]:
                if k in delta:
                    add_row(rows, "Tier2_Faithfulness", pname, k, delta[k])

        df = pd.DataFrame(rows)

        # round ALL numeric values to 3 decimals
        df["Value"] = pd.to_numeric(df["Value"], errors="ignore")
        df["Value"] = df["Value"].apply(lambda x: round(float(x), 3) if isinstance(x, (int, float)) else x)

        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        print(df.head(20).to_string(index=False))



        return summary_path








@dataclass
class VignetteEvalPaths:
    csv_4option: str
    csv_5option: str


class VignetteS1Tier123End2EndEvaluator:
    """
    S1 external validation on Vignette MCQs (169 Qs, TEST ONLY):
      - No retrieval, no evidence segment
      - Verifier V* scores each (question, option) pair
      - Softmax across options per question -> MCQ probabilities

    Tier1 (MCQ):
      - Accuracy
      - NLL (true option)
      - Multiclass Brier
      - Top-1 ECE
      - Selective risk (risk_01 = 1-acc vs coverage), AURC01
      - Conf and margin in question parquet

    Tier2 (faithfulness-like input ablations, no evidence):
      - option_text_shuffled_within_question (permute option texts, fixed seed)
      - question_only (option removed)
      - option_only (question removed)
      Reports deltas vs baseline.

    Tier3 slices:
      - option_count: 4 vs 5
      - question_length_bucket: short/medium/long (tertiles)
      - contains_modality_terms: 0/1
      - contains_uncertainty_terms: 0/1
    """

    def __init__(
        self,
        cfg_path: str,
        experiment_name: str,
        base_path_evidenceData: str,
        verifier_weights_path: str,
        vignette_paths: VignetteEvalPaths,
        device: str = "cuda",
        verifier_max_tokens: int = 512,
        batch_size_verify: int = 16,
        ece_bins: int = 15,
        seed: int = 13,
        mcq_temperature_T: Optional[float] = None,
        # Output
        out_dir: Optional[str] = None,
    ) -> None:
        self.cfg_path = cfg_path
        self.experiment_name = experiment_name
        self.base_path_evidenceData = base_path_evidenceData
        self.verifier_weights_path = verifier_weights_path
        self.vignette_paths = vignette_paths

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.verifier_max_tokens = int(verifier_max_tokens)
        self.batch_size_verify = int(batch_size_verify)
        self.ece_bins = int(ece_bins)
        self.seed = int(seed)

        self.mcq_temperature_T = float(mcq_temperature_T) if mcq_temperature_T is not None else None

        default_out = os.path.join(
            self.base_path_evidenceData,
            "external_validation",
            "Vignette",
            "tier123_results_s1_end2end",
        )
        self.out_dir = out_dir or default_out
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

    def _clean(self, s: str) -> str:
        return " ".join(str(s).split()).strip()

    def _load_vignette_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.dropna(how="all").reset_index(drop=True)
        df.columns = [str(c) for c in df.columns]
        return df

    def _build_mcq_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q_rows: List[Dict] = []
        opt_rows: List[Dict] = []
        qa_row_id = 0

        # 4-option
        if self.vignette_paths.csv_4option and os.path.exists(self.vignette_paths.csv_4option):
            df4 = self._load_vignette_csv(self.vignette_paths.csv_4option)
            required4 = ["q number", "question", "option 1", "option 2", "option 3", "option 4", "answer index"]
            missing4 = [c for c in required4 if c not in df4.columns]
            if missing4:
                raise ValueError(f"4-option CSV missing columns: {missing4}. Found: {list(df4.columns)}")

            for _, r in df4.iterrows():
                qtext = self._clean(r["question"])
                try:
                    ans_idx = int(pd.to_numeric(r["answer index"], errors="raise"))
                except Exception:
                    continue
                if ans_idx < 1 or ans_idx > 4:
                    continue
                correct_index = ans_idx - 1

                options = [
                    ("A", self._clean(r["option 1"])),
                    ("B", self._clean(r["option 2"])),
                    ("C", self._clean(r["option 3"])),
                    ("D", self._clean(r["option 4"])),
                ]
                options = [(L, t) for (L, t) in options if t != ""]
                if len(options) != 4:
                    continue

                qid = qa_row_id
                qa_row_id += 1

                q_rows.append(
                    {
                        "qa_row_id": qid,
                        "question": qtext,
                        "num_options": 4,
                        "options_json": json.dumps([{"letter": L, "text": t} for L, t in options]),
                        "correct_letter": options[correct_index][0],
                        "correct_index": int(correct_index),
                        "source_split": "4option",
                        "source_question_id": self._clean(r["q number"]),
                    }
                )

                for opt_idx, (L, opt_text) in enumerate(options):
                    opt_rows.append(
                        {
                            "qa_row_id": qid,
                            "option_index": int(opt_idx),
                            "option_letter": L,
                            "option_text": opt_text,
                            "question": qtext,
                            "is_correct": int(opt_idx == correct_index),
                            "correct_index": int(correct_index),
                            "num_options": 4,
                            "source_split": "4option",
                        }
                    )

        # 5-option
        if self.vignette_paths.csv_5option and os.path.exists(self.vignette_paths.csv_5option):
            df5 = self._load_vignette_csv(self.vignette_paths.csv_5option)
            required5 = ["question_number", "question", "option_A", "option_B", "option_C", "option_D", "option_E", "solution_index"]
            missing5 = [c for c in required5 if c not in df5.columns]
            if missing5:
                raise ValueError(f"5-option CSV missing columns: {missing5}. Found: {list(df5.columns)}")

            for _, r in df5.iterrows():
                qtext = self._clean(r["question"])
                ans_letter = self._clean(r["solution_index"]).upper()
                if ans_letter not in ["A", "B", "C", "D", "E"]:
                    continue

                options = [
                    ("A", self._clean(r["option_A"])),
                    ("B", self._clean(r["option_B"])),
                    ("C", self._clean(r["option_C"])),
                    ("D", self._clean(r["option_D"])),
                    ("E", self._clean(r["option_E"])),
                ]
                options = [(L, t) for (L, t) in options if t != ""]
                if len(options) != 5:
                    continue

                letter_to_index = {L: i for i, (L, _) in enumerate(options)}
                correct_index = letter_to_index.get(ans_letter, None)
                if correct_index is None:
                    continue

                qid = qa_row_id
                qa_row_id += 1

                q_rows.append(
                    {
                        "qa_row_id": qid,
                        "question": qtext,
                        "num_options": 5,
                        "options_json": json.dumps([{"letter": L, "text": t} for L, t in options]),
                        "correct_letter": ans_letter,
                        "correct_index": int(correct_index),
                        "source_split": "5option",
                        "source_question_id": self._clean(r["question_number"]),
                    }
                )

                for opt_idx, (L, opt_text) in enumerate(options):
                    opt_rows.append(
                        {
                            "qa_row_id": qid,
                            "option_index": int(opt_idx),
                            "option_letter": L,
                            "option_text": opt_text,
                            "question": qtext,
                            "is_correct": int(opt_idx == correct_index),
                            "correct_index": int(correct_index),
                            "num_options": 5,
                            "source_split": "5option",
                        }
                    )

        questions_df = pd.DataFrame(q_rows)
        options_df = pd.DataFrame(opt_rows)

        if len(questions_df) == 0 or len(options_df) == 0:
            raise RuntimeError("No questions/options parsed. Check CSV paths and schemas.")

        cnt = options_df.groupby("qa_row_id").size().reset_index(name="n")
        merged = questions_df.merge(cnt, on="qa_row_id", how="left")
        bad = merged[(merged["n"].fillna(-1).astype(int) != merged["num_options"].astype(int))]
        if len(bad) > 0:
            raise RuntimeError(f"Expanded rows mismatch for {len(bad)} questions. Example:\n{bad.head(3)}")

        return questions_df.reset_index(drop=True), options_df.reset_index(drop=True)

    def _build_input_baseline(self, question: str, option_letter: str, option_text: str) -> str:
        q = self._clean(question)
        a = self._clean(f"({option_letter}) {option_text}")
        return f"Question: {q}\nProposed Answer: {a}"

    def _build_input_question_only(self, question: str) -> str:
        q = self._clean(question)
        return f"Question: {q}"

    def _build_input_option_only(self, option_letter: str, option_text: str) -> str:
        a = self._clean(f"({option_letter}) {option_text}")
        return f"Proposed Answer: {a}"

    def _predict_scores(self, inputs: List[str], desc: str) -> np.ndarray:
        assert self.verifier_tok is not None
        assert self.verifier_model is not None

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), self.batch_size_verify),
                total=math.ceil(len(inputs) / self.batch_size_verify),
                desc=desc,
                unit="batch",
            ):
                batch = inputs[i : i + self.batch_size_verify]
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

    def _softmax_probs(self, scores: np.ndarray, T: float) -> np.ndarray:
        z = scores.astype(np.float64) / float(max(T, 1e-6))
        m = float(z.max())
        ex = np.exp(np.clip(z - m, -50, 50))
        return ex / max(float(ex.sum()), 1e-12)

    def _question_level_from_option_scores(self, opt_df: pd.DataFrame, score_col: str, T: float) -> pd.DataFrame:
        rows = []
        for qid, g in opt_df.groupby("qa_row_id", sort=True):
            scores = g[score_col].to_numpy(dtype=np.float64)
            p = self._softmax_probs(scores, T=T)

            correct_index = int(g["correct_index"].iloc[0])
            pred_index = int(np.argmax(p))
            conf = float(np.max(p))
            p_sorted = np.sort(p)
            margin = float(p_sorted[-1] - p_sorted[-2]) if len(p_sorted) >= 2 else float(p_sorted[-1])

            nll = float(-math.log(max(float(p[correct_index]), 1e-12)))
            y = np.zeros_like(p)
            y[correct_index] = 1.0
            brier = float(np.sum((p - y) ** 2))

            rows.append(
                {
                    "qa_row_id": int(qid),
                    "num_options": int(g["num_options"].iloc[0]),
                    "correct_index": correct_index,
                    "pred_index": pred_index,
                    "is_correct": int(pred_index == correct_index),
                    "conf": conf,
                    "margin": margin,
                    "p_true": float(p[correct_index]),
                    "nll": nll,
                    "brier_mcq": brier,
                    "source_split": str(g["source_split"].iloc[0]) if "source_split" in g.columns else None,
                }
            )
        return pd.DataFrame(rows)

    def _ece_top1(self, correct01: np.ndarray, conf: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(correct01)
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (conf >= lo) & (conf < hi) if b < bins - 1 else (conf >= lo) & (conf <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(correct01[mask].mean())
            cbar = float(conf[mask].mean())
            ece += (m / n) * abs(acc - cbar)
        return float(ece)

    def _selective_risk(self, is_correct: np.ndarray, conf: np.ndarray) -> Dict[str, object]:
        order = np.argsort(-conf)
        coverages = np.linspace(0.1, 1.0, 10)

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(order))))
            return float(1.0 - is_correct[order[:k]].mean())

        curve = [{"coverage": float(c), "risk_01": risk_at(float(c))} for c in coverages]
        fixed = [0.6, 0.8, 0.9]
        fixed_out = {f"risk_01@{int(c*100)}": risk_at(c) for c in fixed}
        aurc01 = float(np.mean([x["risk_01"] for x in curve]))
        return {"curve": curve, "AURC01": aurc01, **fixed_out}

    def _tier1_metrics(self, qdf: pd.DataFrame) -> Dict[str, object]:
        acc = float(qdf["is_correct"].mean()) if len(qdf) else 0.0
        nll = float(qdf["nll"].mean()) if len(qdf) else 0.0
        brier = float(qdf["brier_mcq"].mean()) if len(qdf) else 0.0
        ece = self._ece_top1(
            qdf["is_correct"].to_numpy(np.float32),
            qdf["conf"].to_numpy(np.float32),
            self.ece_bins,
        ) if len(qdf) else 0.0
        sel = self._selective_risk(
            qdf["is_correct"].to_numpy(np.float32),
            qdf["conf"].to_numpy(np.float32),
        ) if len(qdf) else {}
        return {"Accuracy": acc, "NLL": nll, "Brier": brier, "ECE": float(ece), "SelectiveRisk": sel}

    def _token_count(self, s: str) -> int:
        return len(self._clean(s).split())

    def _slice_tables(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        df = questions_df.copy()
        qlens = df["question"].astype(str).map(self._token_count).to_numpy()

        q1 = float(np.quantile(qlens, 1 / 3))
        q2 = float(np.quantile(qlens, 2 / 3))

        def bucket(x: int) -> str:
            if x <= q1:
                return "short"
            if x <= q2:
                return "medium"
            return "long"

        df["question_len_tokens"] = qlens.astype(int)
        df["question_length_bucket"] = [bucket(int(x)) for x in qlens.tolist()]

        modality_terms = ["ct", "mri", "ultrasound", "us", "x-ray", "radiograph", "mammogram", "pet", "angiogram"]
        uncertain_terms = ["likely", "possibly", "may represent", "could represent", "suggest", "suspicious", "indeterminate"]

        qlow = df["question"].astype(str).str.lower()
        df["contains_modality_terms"] = qlow.apply(lambda s: int(any(t in s for t in modality_terms)))
        df["contains_uncertainty_terms"] = qlow.apply(lambda s: int(any(t in s for t in uncertain_terms)))

        df["option_count"] = pd.to_numeric(df["num_options"], errors="coerce").astype("Int64")
        return df

    def _compute_slice_metrics(self, qpred: pd.DataFrame, qtags: pd.DataFrame, prefix: str) -> Dict[str, object]:
        df = qpred.merge(
            qtags[["qa_row_id", "question_length_bucket", "contains_modality_terms", "contains_uncertainty_terms", "option_count"]],
            on="qa_row_id",
            how="left",
        )
        out: Dict[str, object] = {}

        def pack(mask: np.ndarray) -> Dict[str, object]:
            if int(mask.sum()) == 0:
                return {"n": 0, "Accuracy": None, "NLL": None, "ECE": None, "Brier": None, "AURC01": None}
            sub = df.loc[mask].reset_index(drop=True)
            m = self._tier1_metrics(sub)
            return {
                "n": int(len(sub)),
                "Accuracy": m["Accuracy"],
                "NLL": m["NLL"],
                "ECE": m["ECE"],
                "Brier": m["Brier"],
                "AURC01": float(m["SelectiveRisk"]["AURC01"]) if "SelectiveRisk" in m and "AURC01" in m["SelectiveRisk"] else None,
            }

        oc = pd.to_numeric(df["option_count"], errors="coerce").fillna(-1).astype(int).to_numpy()
        for v in [4, 5]:
            out[f"{prefix}.option_count.{v}"] = pack(oc == v)

        for v in ["short", "medium", "long"]:
            out[f"{prefix}.question_length_bucket.{v}"] = pack(df["question_length_bucket"].astype(str).to_numpy() == v)

        m1 = pd.to_numeric(df["contains_modality_terms"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        out[f"{prefix}.contains_modality_terms.1"] = pack(m1)
        out[f"{prefix}.contains_modality_terms.0"] = pack(~m1)

        u1 = pd.to_numeric(df["contains_uncertainty_terms"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        out[f"{prefix}.contains_uncertainty_terms.1"] = pack(u1)
        out[f"{prefix}.contains_uncertainty_terms.0"] = pack(~u1)

        return out

    def _perturb_shuffle_option_texts_within_question(self, opt_df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed)
        out = opt_df.copy()
        new_texts = out["option_text"].astype(str).tolist()

        for qid, idxs in out.groupby("qa_row_id", sort=True).indices.items():
            idx_list = list(idxs)
            texts = [new_texts[i] for i in idx_list]
            perm = rng.permutation(len(texts))
            for j, i in enumerate(idx_list):
                new_texts[i] = texts[int(perm[j])]

        out["option_text"] = new_texts
        return out


    def run(self) -> str:
        self._load_verifier()

        questions_df, options_df = self._build_mcq_tables()
        qtags = self._slice_tables(questions_df)

        # Baseline (question + option)
        inputs_base = [
            self._build_input_baseline(r["question"], r["option_letter"], r["option_text"])
            for _, r in options_df.iterrows()
        ]
        scores_base = self._predict_scores(inputs_base, desc="S1 Verification (no-evidence) [Vignette options]")

        opt_pred = options_df.copy()
        opt_pred["s1_score_raw"] = scores_base.astype(np.float32)

        qpred_uncal = self._question_level_from_option_scores(opt_pred, score_col="s1_score_raw", T=1.0)
        tier1_uncal = self._tier1_metrics(qpred_uncal)

        tier1_cal = None
        qpred_cal = None
        if self.mcq_temperature_T is not None:
            T_ext = float(max(self.mcq_temperature_T, 1e-6))
            qpred_cal = self._question_level_from_option_scores(opt_pred, score_col="s1_score_raw", T=T_ext)
            tier1_cal = self._tier1_metrics(qpred_cal)

        opt_path = os.path.join(self.out_dir, "s1_option_level_predictions.parquet")
        q_uncal_path = os.path.join(self.out_dir, "s1_question_level_predictions_uncalibrated.parquet")
        opt_pred.to_parquet(opt_path, index=False)
        qpred_uncal.to_parquet(q_uncal_path, index=False)

        q_cal_path = None
        if qpred_cal is not None:
            q_cal_path = os.path.join(self.out_dir, "s1_question_level_predictions_calibrated.parquet")
            qpred_cal.to_parquet(q_cal_path, index=False)

        # Tier3 slices
        slice_uncal = self._compute_slice_metrics(qpred_uncal, qtags, prefix="slice_uncal")
        slice_cal = None
        if qpred_cal is not None:
            slice_cal = self._compute_slice_metrics(qpred_cal, qtags, prefix="slice_cal")

        # Tier2: input perturbations
        base_uncal = tier1_uncal
        base_cal = tier1_cal

        def score_from_opt_df(opt_df_in: pd.DataFrame, desc: str) -> pd.DataFrame:
            inputs = [
                self._build_input_baseline(r["question"], r["option_letter"], r["option_text"])
                for _, r in opt_df_in.iterrows()
            ]
            sc = self._predict_scores(inputs, desc=desc)
            out = opt_df_in.copy()
            out["score_raw"] = sc.astype(np.float32)
            return out

        def score_question_only(opt_df_in: pd.DataFrame, desc: str) -> pd.DataFrame:
            inputs = [self._build_input_question_only(r["question"]) for _, r in opt_df_in.iterrows()]
            sc = self._predict_scores(inputs, desc=desc)
            out = opt_df_in.copy()
            out["score_raw"] = sc.astype(np.float32)
            return out

        def score_option_only(opt_df_in: pd.DataFrame, desc: str) -> pd.DataFrame:
            inputs = [self._build_input_option_only(r["option_letter"], r["option_text"]) for _, r in opt_df_in.iterrows()]
            sc = self._predict_scores(inputs, desc=desc)
            out = opt_df_in.copy()
            out["score_raw"] = sc.astype(np.float32)
            return out

        def deltas(m: Dict[str, object], base: Dict[str, object]) -> Dict[str, float]:
            keys = ["Accuracy", "NLL", "Brier", "ECE"]
            return {f"Δ{k}": float(m[k] - base[k]) for k in keys}

        opt_shuf_txt = self._perturb_shuffle_option_texts_within_question(options_df)
        opt_qonly = options_df.copy()
        opt_oonly = options_df.copy()

        scored_shuf = score_from_opt_df(opt_shuf_txt, desc="S1 Tier2: option_text_shuffled_within_question")
        scored_qonly = score_question_only(opt_qonly, desc="S1 Tier2: question_only")
        scored_oonly = score_option_only(opt_oonly, desc="S1 Tier2: option_only")

        q_shuf_uncal = self._question_level_from_option_scores(scored_shuf.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_qonly_uncal = self._question_level_from_option_scores(scored_qonly.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)
        q_oonly_uncal = self._question_level_from_option_scores(scored_oonly.rename(columns={"score_raw": "tmp"}), "tmp", T=1.0)

        m_shuf_uncal = self._tier1_metrics(q_shuf_uncal)
        m_qonly_uncal = self._tier1_metrics(q_qonly_uncal)
        m_oonly_uncal = self._tier1_metrics(q_oonly_uncal)

        m_shuf_cal = m_qonly_cal = m_oonly_cal = None
        if self.mcq_temperature_T is not None:
            T_ext = float(self.mcq_temperature_T)
            q_shuf_cal = self._question_level_from_option_scores(scored_shuf.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_qonly_cal = self._question_level_from_option_scores(scored_qonly.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            q_oonly_cal = self._question_level_from_option_scores(scored_oonly.rename(columns={"score_raw": "tmp"}), "tmp", T=T_ext)
            m_shuf_cal = self._tier1_metrics(q_shuf_cal)
            m_qonly_cal = self._tier1_metrics(q_qonly_cal)
            m_oonly_cal = self._tier1_metrics(q_oonly_cal)

        tier2_report = {
            "system": "S1 (no retrieval, no evidence, V* trained on MIMIC)",
            "n_questions": int(len(qpred_uncal)),
            "baseline_uncalibrated": {k: base_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
            "baseline_calibrated": ({k: base_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if base_cal is not None else None),
            "perturbations": {
                "option_text_shuffled_within_question": {
                    "uncalibrated": {k: m_shuf_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_shuf_uncal, base_uncal),
                    "calibrated": ({k: m_shuf_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_shuf_cal is not None else None),
                    "delta_calibrated": (deltas(m_shuf_cal, base_cal) if (m_shuf_cal is not None and base_cal is not None) else None),
                },
                "question_only": {
                    "uncalibrated": {k: m_qonly_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_qonly_uncal, base_uncal),
                    "calibrated": ({k: m_qonly_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_qonly_cal is not None else None),
                    "delta_calibrated": (deltas(m_qonly_cal, base_cal) if (m_qonly_cal is not None and base_cal is not None) else None),
                },
                "option_only": {
                    "uncalibrated": {k: m_oonly_uncal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]},
                    "delta_uncalibrated": deltas(m_oonly_uncal, base_uncal),
                    "calibrated": ({k: m_oonly_cal[k] for k in ["Accuracy", "NLL", "Brier", "ECE"]} if m_oonly_cal is not None else None),
                    "delta_calibrated": (deltas(m_oonly_cal, base_cal) if (m_oonly_cal is not None and base_cal is not None) else None),
                },
            },
        }

        tier2_q = qpred_uncal.copy().rename(columns={"is_correct": "base_is_correct", "conf": "base_conf", "nll": "base_nll"})
        tier2_q["shuf_text_is_correct"] = q_shuf_uncal["is_correct"].astype(int).values
        tier2_q["question_only_is_correct"] = q_qonly_uncal["is_correct"].astype(int).values
        tier2_q["option_only_is_correct"] = q_oonly_uncal["is_correct"].astype(int).values

        tier2_pred_path = os.path.join(self.out_dir, "s1_tier2_input_perturbations_predictions_question_level.parquet")
        tier2_q.to_parquet(tier2_pred_path, index=False)

        tier2_path = os.path.join(self.out_dir, "s1_tier2_input_perturbations_report.json")
        with open(tier2_path, "w", encoding="utf-8") as f:
            json.dump({**tier2_report, "artifacts": {"question_level_parquet": tier2_pred_path}}, f, indent=2)

        # Tier1 report
        tier1_report = {
            "system": "S1 (no retrieval, no evidence, V* trained on MIMIC)",
            "dataset": "Vignette MCQ (169 questions; 4/5 options mixed)",
            "paths": {"csv_4option": self.vignette_paths.csv_4option, "csv_5option": self.vignette_paths.csv_5option},
            "n_questions": int(len(qpred_uncal)),
            "n_options_rows": int(len(opt_pred)),
            "temperature_for_mcq_softmax": {
                "uncalibrated_T": 1.0,
                "external_calibrated_T": float(self.mcq_temperature_T) if self.mcq_temperature_T is not None else None,
            },
            "tier1_uncalibrated": tier1_uncal,
            "tier1_calibrated": tier1_cal,
            "tier2_input_perturbations": {"report_json": tier2_path, "question_level_parquet": tier2_pred_path},
            "tier3_slices_uncalibrated": slice_uncal,
            "tier3_slices_calibrated": slice_cal,
            "artifacts": {
                "option_level_predictions_parquet": opt_path,
                "question_level_predictions_uncalibrated_parquet": q_uncal_path,
                "question_level_predictions_calibrated_parquet": q_cal_path,
            },
        }
        tier1_path = os.path.join(self.out_dir, "s1_tier1_report.json")
        with open(tier1_path, "w", encoding="utf-8") as f:
            json.dump(tier1_report, f, indent=2)

        summary = {
            "tier1_report": tier1_path,
            "tier2_input_perturbations_report": tier2_path,
            "tier2_input_perturbations_predictions_question_level": tier2_pred_path,
            "option_level_predictions": opt_path,
            "question_level_predictions_uncalibrated": q_uncal_path,
            "question_level_predictions_calibrated": q_cal_path,
        }
        summary_path = os.path.join(self.out_dir, "s1_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {tier1_path}")
        print(f"Wrote: {tier2_path}")
        print(f"Wrote: {summary_path}")



        # summarizing all to a single csv ...
        # summarizing all to a single csv ...
        # summarizing all to a single csv ...

        tier1_path = os.path.join(self.out_dir, "s1_tier1_report.json")
        tier2_path = os.path.join(self.out_dir, "s1_tier2_input_perturbations_report.json")
        out_csv = os.path.join(self.out_dir, "S1_vignette_full_results.csv")

        def _fmt3(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ""
            if isinstance(x, (int, float)):
                return f"{x:.3f}"
            return str(x)

        def add_row(rows, tier, subgroup, metric, value, split="uncalibrated", n=None):
            rows.append({
                "System": "S1",
                "Dataset": "Vignette",
                "Tier": tier,
                "Split": split,
                "Subgroup": subgroup,
                "Metric": metric,
                "Value": value,
                "N": n if n is not None else ""
            })

        # -------- Load --------
        with open(tier1_path, "r") as f:
            t1 = json.load(f)

        with open(tier2_path, "r") as f:
            t2 = json.load(f)

        rows = []

        t1u = t1.get("tier1_uncalibrated", {}) or {}
        n_q = t1.get("n_questions", "")

        for k in ["Accuracy", "NLL", "Brier", "ECE"]:
            if k in t1u:
                add_row(rows, "Tier1", "overall", k, t1u[k], "uncalibrated", n_q)

        sel = (t1u.get("SelectiveRisk") or {})
        if "AURC01" in sel:
            add_row(rows, "Tier1", "overall", "AURC01", sel["AURC01"], "uncalibrated", n_q)
        for k in ["risk_01@60", "risk_01@80", "risk_01@90"]:
            if k in sel:
                add_row(rows, "Tier1", "overall", k, sel[k], "uncalibrated", n_q)

        base = (t2.get("baseline_uncalibrated") or {})
        for k in ["Accuracy", "NLL", "Brier", "ECE"]:
            if k in base:
                add_row(rows, "Tier2_InputPerturb", "baseline", k, base[k], "uncalibrated", n_q)

        perts = (t2.get("perturbations") or {})
        for pname, pobj in perts.items():
            d = (pobj.get("delta_uncalibrated") or {})
            for k in ["ΔAccuracy", "ΔNLL", "ΔBrier", "ΔECE"]:
                if k in d:
                    add_row(rows, "Tier2_InputPerturb", pname, k, d[k], "uncalibrated", n_q)

        t3u = t1.get("tier3_slices_uncalibrated", {}) or {}
        for slice_name, vals in t3u.items():
            n = vals.get("n", "")
            for k in ["Accuracy", "NLL", "Brier", "ECE", "AURC01"]:
                if k in vals:
                    add_row(rows, "Tier3", slice_name, k, vals[k], "uncalibrated", n)

        df = pd.DataFrame(rows, columns=["System", "Dataset", "Tier", "Split", "Subgroup", "Metric", "Value", "N"])
        df["Value"] = df["Value"].apply(_fmt3)
        df.to_csv(out_csv, index=False)

        print(f"Saved: {out_csv}")


        return summary_path







class CrossSystemComparatorPredOnlyVignette:
    """
    Cross-system comparison for Vignette MCQ external validation (ALL questions are TEST).

    Inputs:
      - system_question_predictions: { "S1": <parquet>, "S2": <parquet>, ... }
        Each parquet must contain:
          qa_row_id, correct_index, pred_index, is_correct, conf, margin, p_true, nll, brier_mcq

      - system_tier2_question_predictions (optional):
          Used to compute "faithfulness / perturbation severity" at question-level per system.
          This supports two formats:
            (A) Retrieval faithfulness (S2/S3/S4 style):
                columns: qa_row_id, base_is_correct, swap_is_correct, top1_is_correct, top3_is_correct, shuffle_is_correct
                (optional) random_is_correct
            (B) Input perturbations (S1 style):
                columns: qa_row_id, base_is_correct, <any number of other *_is_correct columns>
                We will summarize each perturbation column automatically.

    What it computes on the intersection of qa_row_id across provided systems:
      - Per-system Tier1 (MCQ):
          Accuracy, NLL, Multiclass-Brier, Top1-ECE, AURC01, MeanConf, MeanMargin
      - Pairwise (ordered pairs A vs B):
          win rates per-question: NLL, Brier, correctness, ECE-proxy
          paired bootstrap CI for deltas: Accuracy, NLL, Brier, ECE, AURC01
    """

    def __init__(
        self,
        system_question_predictions: Dict[str, str],
        out_path: str,
        *,
        ece_bins: int = 15,
        n_boot: int = 2000,
        seed: int = 13,
        system_tier2_question_predictions: Optional[Dict[str, str]] = None,
    ) -> None:
        self.system_question_predictions = dict(system_question_predictions)
        self.system_tier2_question_predictions = dict(system_tier2_question_predictions or {})
        self.out_path = out_path

        self.ece_bins = int(ece_bins)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)


    def _ece_top1(self, correct01: np.ndarray, conf: np.ndarray, bins: int) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(correct01)
        for b in range(bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (conf >= lo) & (conf < hi) if b < bins - 1 else (conf >= lo) & (conf <= hi)
            m = int(mask.sum())
            if m == 0:
                continue
            acc = float(correct01[mask].mean())
            cbar = float(conf[mask].mean())
            ece += (m / n) * abs(acc - cbar)
        return float(ece)

    def _aurc01(self, is_correct: np.ndarray, conf: np.ndarray) -> float:
        order = np.argsort(-conf)
        coverages = np.linspace(0.1, 1.0, 10)

        def risk_at(cov: float) -> float:
            k = max(1, int(round(cov * len(order))))
            return float(1.0 - is_correct[order[:k]].mean())

        return float(np.mean([risk_at(float(c)) for c in coverages]))

    def _tier1_bundle(self, df: pd.DataFrame) -> Dict[str, float]:
        is_correct = df["is_correct"].to_numpy(dtype=np.float32)
        conf = df["conf"].to_numpy(dtype=np.float32)

        return {
            "Accuracy": float(is_correct.mean()) if len(df) else 0.0,
            "NLL": float(df["nll"].to_numpy(dtype=np.float32).mean()) if len(df) else 0.0,
            "Brier": float(df["brier_mcq"].to_numpy(dtype=np.float32).mean()) if len(df) else 0.0,
            "ECE": self._ece_top1(is_correct, conf, self.ece_bins) if len(df) else 0.0,
            "AURC01": self._aurc01(is_correct, conf) if len(df) else 0.0,
            "MeanConf": float(conf.mean()) if len(df) else 0.0,
            "MeanMargin": float(df["margin"].to_numpy(dtype=np.float32).mean()) if len(df) else 0.0,
        }

    def _paired_bootstrap_delta(self, a: np.ndarray, b: np.ndarray, fn_mean) -> Dict[str, float]:
        n = len(a)
        deltas = np.zeros((self.n_boot,), dtype=np.float64)
        for t in range(self.n_boot):
            idx = self.rng.randint(0, n, size=n)
            deltas[t] = float(fn_mean(a[idx]) - fn_mean(b[idx]))
        return {
            "mean_delta": float(deltas.mean()),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
        }

    def _load_system_qpred(self, sys_name: str, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)

        req = ["qa_row_id", "correct_index", "pred_index", "is_correct", "conf", "margin", "p_true", "nll", "brier_mcq"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise RuntimeError(f"{sys_name} question-level parquet missing: {missing}. Found={list(df.columns)}")

        df = df.copy()
        df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        df = df[df["qa_row_id"] >= 0].copy()
        df = df.sort_values("qa_row_id").reset_index(drop=True)

        for c in ["correct_index", "pred_index", "is_correct"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)
        for c in ["conf", "margin", "p_true", "nll", "brier_mcq"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

        return df

    def _load_and_align(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
        raw = {k: self._load_system_qpred(k, p) for k, p in self.system_question_predictions.items()}

        sys_ids = {k: set(v["qa_row_id"].tolist()) for k, v in raw.items()}
        eval_ids = None
        for ids in sys_ids.values():
            eval_ids = set(ids) if eval_ids is None else (eval_ids & ids)
        eval_ids = np.array(sorted(list(eval_ids or [])), dtype=np.int64)

        if len(eval_ids) == 0:
            raise RuntimeError("No overlapping qa_row_id across provided systems.")

        systems: Dict[str, pd.DataFrame] = {}
        dropped: Dict[str, int] = {}
        for sys_name, df in raw.items():
            df_f = df[df["qa_row_id"].isin(eval_ids)].copy()
            df_f = df_f.sort_values("qa_row_id").reset_index(drop=True)
            systems[sys_name] = df_f
            dropped[f"{sys_name}_dropped"] = int(len(df) - len(df_f))

        ref_ids = next(iter(systems.values()))["qa_row_id"].to_numpy(dtype=np.int64)
        ref_correct = next(iter(systems.values()))["correct_index"].to_numpy(dtype=np.int64)

        for sys_name, df in systems.items():
            if not np.array_equal(df["qa_row_id"].to_numpy(dtype=np.int64), ref_ids):
                raise RuntimeError(f"Alignment failure for {sys_name} (qa_row_id mismatch).")
            if not np.array_equal(df["correct_index"].to_numpy(dtype=np.int64), ref_correct):
                raise RuntimeError(f"correct_index mismatch for {sys_name} vs reference on aligned ids.")

        meta = {"n_eval": int(len(eval_ids)), "dropped_counts": dropped, "systems": list(systems.keys())}
        return systems, meta

    def _load_tier2_question_df(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        if "qa_row_id" not in df.columns:
            raise RuntimeError(f"Tier2 parquet missing qa_row_id: {path}")

        df = df.copy()
        df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        df = df[df["qa_row_id"] >= 0].copy()
        df = df.sort_values("qa_row_id").reset_index(drop=True)
        return df

    def _summarize_tier2_is_correct(self, df: pd.DataFrame) -> Dict[str, object]:
        if "base_is_correct" not in df.columns:
            return {"available": False, "reason": "missing base_is_correct"}

        base = pd.to_numeric(df["base_is_correct"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        base_acc = float(base.mean())

        # any other columns ending with _is_correct are treated as perturbations
        pert_cols = [c for c in df.columns if c.endswith("_is_correct") and c != "base_is_correct"]
        perts: Dict[str, object] = {}
        for c in pert_cols:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            pert_acc = float(v.mean())
            perts[c] = {
                "base_acc": base_acc,
                "pert_acc": pert_acc,
                "acc_drop": float(pert_acc - base_acc),
                "frac_questions_worse": float(np.mean(v < base)),
            }

        return {"available": True, "base_acc": base_acc, "perturbations": perts}


    def run(self) -> str:
        systems, meta = self._load_and_align()
        names = list(systems.keys())

        per_system = {sys: self._tier1_bundle(df) for sys, df in systems.items()}

        aligned = pd.DataFrame({"qa_row_id": systems[names[0]]["qa_row_id"].to_numpy(dtype=np.int64)})
        aligned["correct_index"] = systems[names[0]]["correct_index"].to_numpy(dtype=np.int64)
        for sys, df in systems.items():
            aligned[f"{sys}.pred_index"] = df["pred_index"].to_numpy(dtype=np.int64)
            aligned[f"{sys}.is_correct"] = df["is_correct"].to_numpy(dtype=np.int64)
            aligned[f"{sys}.conf"] = df["conf"].to_numpy(dtype=np.float32)
            aligned[f"{sys}.margin"] = df["margin"].to_numpy(dtype=np.float32)
            aligned[f"{sys}.nll"] = df["nll"].to_numpy(dtype=np.float32)
            aligned[f"{sys}.brier"] = df["brier_mcq"].to_numpy(dtype=np.float32)

        out_dir = os.path.dirname(self.out_path)
        os.makedirs(out_dir, exist_ok=True)
        aligned_path = os.path.join(out_dir, "aligned_question_level_predictions.parquet")
        aligned.to_parquet(aligned_path, index=False)

        # pairwise
        pairwise: Dict[str, object] = {}
        for a in names:
            for b in names:
                if a == b:
                    continue

                df_a = systems[a]
                df_b = systems[b]

                acc_a = df_a["is_correct"].to_numpy(dtype=np.float32)
                acc_b = df_b["is_correct"].to_numpy(dtype=np.float32)
                nll_a = df_a["nll"].to_numpy(dtype=np.float32)
                nll_b = df_b["nll"].to_numpy(dtype=np.float32)
                br_a = df_a["brier_mcq"].to_numpy(dtype=np.float32)
                br_b = df_b["brier_mcq"].to_numpy(dtype=np.float32)

                # ECE proxy: per-question |conf - correctness|
                ece_proxy_a = np.abs(df_a["conf"].to_numpy(dtype=np.float32) - acc_a)
                ece_proxy_b = np.abs(df_b["conf"].to_numpy(dtype=np.float32) - acc_b)

                win = {
                    "frac_questions_NLL_A_better": float(np.mean(nll_a < nll_b)),
                    "frac_questions_Brier_A_better": float(np.mean(br_a < br_b)),
                    "frac_questions_ECEproxy_A_better": float(np.mean(ece_proxy_a < ece_proxy_b)),
                    "frac_questions_correct_A_better": float(np.mean(acc_a > acc_b)),
                    "frac_questions_tie_correct": float(np.mean(acc_a == acc_b)),
                }

                # bootstrap deltas for system-level metrics
                pairwise[f"{a}_vs_{b}"] = {
                    "win_rates": win,
                    "bootstrap_deltas_A_minus_B": {
                        "ΔAccuracy": self._paired_bootstrap_delta(acc_a, acc_b, np.mean),
                        "ΔNLL": self._paired_bootstrap_delta(nll_a, nll_b, np.mean),
                        "ΔBrier": self._paired_bootstrap_delta(br_a, br_b, np.mean),
                        "ΔECE": self._paired_bootstrap_delta(
                            df_a["conf"].to_numpy(dtype=np.float32),
                            df_b["conf"].to_numpy(dtype=np.float32),
                            lambda conf_vec: self._ece_top1(acc_a, conf_vec.astype(np.float32), self.ece_bins),
                        ),
                        "ΔAURC01": self._paired_bootstrap_delta(
                            df_a["conf"].to_numpy(dtype=np.float32),
                            df_b["conf"].to_numpy(dtype=np.float32),
                            lambda conf_vec: self._aurc01(acc_a, conf_vec.astype(np.float32)),
                        ),
                    },
                }

        tier2_severity: Dict[str, object] = {}
        if self.system_tier2_question_predictions:
            ref_ids = aligned["qa_row_id"].to_numpy(dtype=np.int64)
            ref = pd.DataFrame({"qa_row_id": ref_ids})

            for sys, p in self.system_tier2_question_predictions.items():
                try:
                    df2 = self._load_tier2_question_df(p)
                    df2 = df2.merge(ref, on="qa_row_id", how="inner").sort_values("qa_row_id").reset_index(drop=True)
                    tier2_severity[sys] = self._summarize_tier2_is_correct(df2)
                    tier2_severity[sys]["path"] = p
                    tier2_severity[sys]["n_aligned"] = int(len(df2))
                except Exception as e:
                    tier2_severity[sys] = {"available": False, "path": p, "error": str(e)}

        final = {
            "dataset": "Vignette MCQ (169 questions; all TEST)",
            "n_eval": int(meta["n_eval"]),
            "alignment_mode": "intersection_of_system_question_outputs",
            "dropped_counts": meta["dropped_counts"],
            "per_system_metrics": per_system,
            "pairwise_all_pairs": pairwise,
            "tier2_severity_optional": tier2_severity,
            "artifacts": {"aligned_question_level_predictions_parquet": aligned_path},
            "params": {"ece_bins": int(self.ece_bins), "n_boot": int(self.n_boot), "seed": int(self.seed)},
        }

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)

        print(f"Wrote: {self.out_path}")
        print(f"Wrote: {aligned_path}")
        print(f"n_eval={final['n_eval']} dropped={final['dropped_counts']}")



        # summarizing all to a single csv ...
        # summarizing all to a single csv ...
        # summarizing all to a single csv ...
        ppath = os.path.dirname(self.out_path)
        base_path = os.path.dirname(ppath)

        s1_csv = os.path.join(base_path, "tier123_results_s1_end2end", "S1_vignette_full_results.csv")
        s2_csv = os.path.join(base_path, "tier123_results_s2_end2end", "S2_vignette_full_results.csv")
        s3_csv = os.path.join(base_path, "tier123_results_s3_end2end", "S3_vignette_full_results.csv")
        s4_csv = os.path.join(base_path, "tier123_results_s4_end2end", "S4_vignette_full_results.csv")

        cross_json = self.out_path

        out_csv = os.path.join(ppath, "Vignette_ALL_systems_plus_cross.csv")

        def _to_float(x):
            try:
                if x is None:
                    return np.nan
                return float(x)
            except Exception:
                return np.nan

        def _round3(x):
            x = _to_float(x)
            return np.nan if pd.isna(x) else round(x, 3)

        def load_summary_csv(path, system_name):
            df = pd.read_csv(path)

            # Normalize columns (keep it robust to slightly different headers)
            col_map = {}
            for c in df.columns:
                lc = c.strip().lower()
                if lc in {"tier"}:
                    col_map[c] = "Tier"
                elif lc in {"subgroup", "slice", "group"}:
                    col_map[c] = "Subgroup"
                elif lc in {"metric"}:
                    col_map[c] = "Metric"
                elif lc in {"value", "val"}:
                    col_map[c] = "Value"
            df = df.rename(columns=col_map)

            # Minimal required columns
            for req in ["Tier", "Subgroup", "Metric", "Value"]:
                if req not in df.columns:
                    raise ValueError(f"{path} is missing required column: {req}. Found: {list(df.columns)}")

            df["System"] = system_name

            # Force numeric where possible and round to 3 decimals
            df["Value"] = df["Value"].apply(_to_float).apply(_round3)

            # Order columns
            df = df[["Dataset", "System", "Tier", "Subgroup", "Metric", "Value"]] if "Dataset" in df.columns else df
            if "Dataset" not in df.columns:
                df.insert(0, "Dataset", "Vignette")

            return df[["Dataset", "System", "Tier", "Subgroup", "Metric", "Value"]]

        def add_row(rows, system, tier, subgroup, metric, value, dataset="Vignette"):
            rows.append({
                "Dataset": dataset,
                "System": system,
                "Tier": tier,
                "Subgroup": subgroup,
                "Metric": metric,
                "Value": _round3(value),
            })

        df_all = pd.concat(
            [
                load_summary_csv(s1_csv, "S1"),
                load_summary_csv(s2_csv, "S2"),
                load_summary_csv(s3_csv, "S3"),
                load_summary_csv(s4_csv, "S4"),
            ],
            ignore_index=True,
        )

        with open(cross_json, "r") as f:
            cx = json.load(f)

        rows = []

        # Per-system metrics (from cross JSON)
        per_sys = cx.get("per_system_metrics", {})
        for sys_name, metrics in per_sys.items():
            for m, v in metrics.items():
                add_row(rows, sys_name, "CrossSystem", "per_system_metrics", m, v)

        # Pairwise metrics: win_rates and bootstrap deltas
        pairs = cx.get("pairwise_all_pairs", {})
        for pair_name, block in pairs.items():
            # win rates
            win = block.get("win_rates", {})
            for m, v in win.items():
                add_row(rows, "ALL", "CrossSystem", f"{pair_name}::win_rates", m, v)

            # bootstrap deltas
            deltas = block.get("bootstrap_deltas_A_minus_B", {})
            for m, stats in deltas.items():
                add_row(rows, "ALL", "CrossSystem", f"{pair_name}::bootstrap", f"{m}::mean", stats.get("mean_delta"))
                add_row(rows, "ALL", "CrossSystem", f"{pair_name}::bootstrap", f"{m}::ci_low", stats.get("ci_low"))
                add_row(rows, "ALL", "CrossSystem", f"{pair_name}::bootstrap", f"{m}::ci_high", stats.get("ci_high"))

        # Tier2 severity optional (accuracy drops etc)
        sev = cx.get("tier2_severity_optional", {})
        for sys_name, block in sev.items():
            if not isinstance(block, dict):
                continue
            add_row(rows, sys_name, "CrossSystem", "tier2_severity_optional", "available",
                    int(bool(block.get("available", False))))
            if block.get("available", False):
                add_row(rows, sys_name, "CrossSystem", "tier2_severity_optional", "base_acc", block.get("base_acc"))
                perts = block.get("perturbations", {})
                for pert_name, pvals in perts.items():
                    add_row(rows, sys_name, "CrossSystem", f"tier2_severity::{pert_name}", "pert_acc",
                            pvals.get("pert_acc"))
                    add_row(rows, sys_name, "CrossSystem", f"tier2_severity::{pert_name}", "acc_drop",
                            pvals.get("acc_drop"))
                    add_row(rows, sys_name, "CrossSystem", f"tier2_severity::{pert_name}", "frac_questions_worse",
                            pvals.get("frac_questions_worse"))

        df_cross = pd.DataFrame(rows, columns=["Dataset", "System", "Tier", "Subgroup", "Metric", "Value"])

        df_final = pd.concat([df_all, df_cross], ignore_index=True)

        # Keep it clean
        df_final["Dataset"] = df_final["Dataset"].fillna("Vignette")
        df_final["System"] = df_final["System"].fillna("")

        df_final.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")



        return self.out_path








if __name__ == "__main__":

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_chexpertData = params["Preprocessing"]["base_path_mimicData"]
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]


    ''' S4 Vignette '''
    runner = VignetteS4Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        model_weights_path=model_weights_path,
        verifier_weights_path=verifier_weights_path,
        vignette_paths=VignetteEvalPaths(
            csv_4option=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
            csv_5option=os.path.join(base_path_chexpertData, "Vignette_5option.csv"),
        ),
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        faithfulness_random_trials=1,
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "Vignette", "tier123_results_s4_end2end"),
    )
    runner.run()



    ''' S3 Vignette '''
    s3 = VignetteS3Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        vignette_paths=VignetteEvalPaths(
            csv_4option=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
            csv_5option=os.path.join(base_path_chexpertData, "Vignette_5option.csv")),
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        bm25_k1=1.2,
        bm25_b=0.75,
        # tier2 speed knobs
        # faithfulness_max_eval_questions=100,
        # faithfulness_random_trials=1,
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "Vignette", "tier123_results_s3_end2end"),
    )
    s3.run()



    ''' S2 Vignette '''
    s2 = VignetteS2Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        vignette_paths=VignetteEvalPaths(
            csv_4option=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
            csv_5option=os.path.join(base_path_chexpertData, "Vignette_5option.csv")),
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        batch_size_embed_chunks=128,
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        # tier2 speed knobs
        # faithfulness_max_eval_questions=100,
        # faithfulness_random_trials=1,
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "Vignette", "tier123_results_s2_end2end"),
    )
    s2.run()



    ''' S1 Vignette '''
    s1 = VignetteS1Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        vignette_paths=VignetteEvalPaths(
            csv_4option=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
            csv_5option=os.path.join(base_path_chexpertData, "Vignette_5option.csv")),
        device="cuda",
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "Vignette", "tier123_results_s1_end2end"),
    )
    s1.run()




    ''' Cross comparison Vignette'''
    base = params["Preprocessing"]["base_path_evidenceData"]
    base_out = os.path.join(base, "external_validation", "Vignette")

    comparator = CrossSystemComparatorPredOnlyVignette(
        system_question_predictions={
            "S1": os.path.join(base_out, "tier123_results_s1_end2end", "s1_question_level_predictions_uncalibrated.parquet"),
            "S2": os.path.join(base_out, "tier123_results_s2_end2end", "s2_question_level_predictions_uncalibrated.parquet"),
            "S3": os.path.join(base_out, "tier123_results_s3_end2end", "s3_question_level_predictions_uncalibrated.parquet"),
            "S4": os.path.join(base_out, "tier123_results_s4_end2end", "question_level_predictions_uncalibrated.parquet"),
        },
        system_tier2_question_predictions={
            "S1": os.path.join(base_out, "tier123_results_s1_end2end", "s1_tier2_input_perturbations_predictions_question_level.parquet"),
            "S2": os.path.join(base_out, "tier123_results_s2_end2end", "s2_tier2_faithfulness_predictions_question_level.parquet"),
            "S3": os.path.join(base_out, "tier123_results_s3_end2end", "s3_tier2_faithfulness_predictions_question_level.parquet"),
            "S4": os.path.join(base_out, "tier123_results_s3_end2end", "tier2_faithfulness_predictions_question_level.parquet"),
        },
        out_path=os.path.join(
            base_out,
            "cross_system_results",
            "s1_s4_cross_system_comparison_vignette.json",
        ),
        ece_bins=15,
        n_boot=2000,
        seed=13,
    )
    comparator.run()
