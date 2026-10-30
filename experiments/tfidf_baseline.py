"""
experiments/tfidf_baseline.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
import pickle
from typing import Dict, List

import pandas as pd

from experiments._common import MissingInput, atomic_write, digest_file, get_cfg, require, run_units
from experiments.score import run_scoring, score_dir, verifiers
from experiments.study import out, read_evidence, train_set_path

_REGIME = "tfidf_baseline-v1"


def model_path(cfg: Dict, vid: str) -> str:
    return os.path.join(score_dir(cfg, vid), "model.pkl")


def case_blind_texts(df: pd.DataFrame, k: int) -> List[str]:
    return [f"claim: {c} evidence: " + "\n\n".join(read_evidence(e, k))
            for c, e in zip(df["claim_text"].astype(str), df["evidence_sentences_json"])]


def fit_case_blind(k: int, grid: Dict, trainvaltest_path: str, tag: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    tvt = pd.read_parquet(trainvaltest_path)
    tr, va = tvt[tvt["split"] == "train"], tvt[tvt["split"] == "val"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=int(grid["min_df"]), sublinear_tf=True, lowercase=True)
    xtr = vec.fit_transform(case_blind_texts(tr, k))
    xva = vec.transform(case_blind_texts(va, k))
    best = None
    for c in grid["c_grid"]:
        m = LogisticRegression(C=float(c), class_weight="balanced", max_iter=5000, solver="liblinear",
                               random_state=0).fit(xtr, tr["label_id"].to_numpy())
        auc = roc_auc_score(va["label_id"], m.predict_proba(xva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, float(c), m)
    return vec, best, len(tr), len(va)


class TfidfScorer:
    def __init__(self, cfg: Dict, vid: str):
        path = model_path(cfg, vid)
        if not os.path.exists(path):
            raise MissingInput(f"{path} is missing; the fit of main_tfidf did not finish")
        with open(path, "rb") as f:
            self.vec, self.model = pickle.load(f)
        self.k = int(cfg["max_evidence_sentences"])

    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        p = self.model.predict_proba(self.vec.transform(case_blind_texts(df, self.k)))[:, 1]
        return pd.DataFrame({"verifier_row_id": df["verifier_row_id"].to_numpy(), "prob_supported": p})


def tfidf_identity(cfg: Dict, vid: str) -> Dict:
    return {"regime": _REGIME, "model": digest_file(require(model_path(cfg, vid), "the fit of main_tfidf"))}


def _fit(cfg: Dict, vx: Dict) -> list:
    parquet = require(train_set_path(cfg, vx["train_set"]), "main_build_examples")
    vec, best, n_train, n_val = fit_case_blind(int(cfg["max_evidence_sentences"]), cfg["tfidf"], parquet, f"tfidf_baseline {vx['id']}")
    def _write(tmp):
        with open(tmp, "wb") as f:
            pickle.dump((vec, best[2]), f)
            f.flush()
            os.fsync(f.fileno())
    atomic_write(model_path(cfg, vx["id"]), _write)
    return [{"verifier": vx["id"], "selected_c": best[1], "validation_auroc": best[0], "n_train": n_train,
             "n_val": n_val, "n_features": len(vec.vocabulary_)}]


def main_tfidf(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    refused = 0
    for vx in [x for x in verifiers(cfg) if x["family"] == "tfidf"]:
        parquet = train_set_path(cfg, vx["train_set"])
        params = {"regime": _REGIME, "train": digest_file(parquet) if os.path.exists(parquet) else None,
                  "baseline": cfg["tfidf"], "max_evidence_sentences": int(cfg["max_evidence_sentences"])}

        def valid(unit, done, vid=vx["id"]):
            return os.path.exists(model_path(cfg, vid))

        _, n = run_units(f"main_tfidf fit {vx['id']}", [{"key": "fit", "params": params}], lambda u, vx=vx: _fit(cfg, vx),
                         score_dir(cfg, vx["id"]), os.path.join(score_dir(cfg, vx["id"]), "status.txt"),
                         validate=valid)
        if n:
            refused += n
            continue
        refused += run_scoring(cfg, "main_tfidf", vx, lambda vx=vx: TfidfScorer(cfg, vx["id"]), tfidf_identity(cfg, vx["id"]))
    if refused:
        raise MissingInput(f"{refused} TF-IDF unit(s) could not run yet")
    return out(cfg, "scores")
