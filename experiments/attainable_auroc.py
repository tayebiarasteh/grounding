"""
experiments/attainable_auroc.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments._common import MissingInput, digest_file, flush_status, get_cfg, run_units, write_table
from experiments.study import examples_path, out, read_evidence, train_set_path
from Inference.cluster_metrics import weighted_auroc

_REGIME = "attainable_auroc-v1"
CHANNELS = {"evidence_claim": ["ev_key", "claim_text"], "case_claim": ["case_context", "claim_text"],
            "claim": ["claim_text"]}


def _bound(y: np.ndarray, groups: np.ndarray) -> Dict:
    pos = np.bincount(groups, weights=y)
    tot = np.bincount(groups).astype(float)
    rate = pos / tot
    mixed = (pos > 0) & (pos < tot)
    return {"auroc_bound": float(weighted_auroc(y, rate[groups], np.ones((1, len(y))))[0]),
            "acc_bound": float(np.maximum(pos, tot - pos).sum() / tot.sum()),
            "n_groups": int(len(tot)), "n_mixed_groups": int(mixed.sum()),
            "n_rows_in_mixed_groups": int(tot[mixed].sum())}


def bound_sets(cfg: Dict) -> List[Dict]:
    sets = []
    for ds, conds in cfg["test_sets"].items():
        for c in conds:
            if c in ("unchanged", "heldout"):
                sets.append({"key": f"{ds}__{c}", "path": examples_path(cfg, ds, c), "split": "test"})
    nod = sorted({r["train_set"] for r in cfg["runs"] if r["train_set"].startswith("noD")})
    for name in ["main"] + nod:
        sets.append({"key": f"train__{name}", "path": train_set_path(cfg, name), "split": "train"})
    return sets


def bounds(df: pd.DataFrame, k: int, key: str) -> List[Dict]:
    df = df.sort_values("verifier_row_id").reset_index(drop=True)
    df["ev_key"] = df["evidence_sentences_json"].apply(lambda js: "\n\n".join(read_evidence(js, k)))
    y = df["label_id"].to_numpy(dtype=float)
    rows = []
    for finding in ["all"] + sorted(df["concept"].unique()):
        sel = np.ones(len(df), bool) if finding == "all" else (df["concept"] == finding).to_numpy()
        base = {"set": key, "finding": finding, "n_rows": int(sel.sum()), "n_pos": int(y[sel].sum())}
        if not 0 < y[sel].sum() < sel.sum():
            rows.append({**base, "channel": "none"})
            continue
        for name, keys in CHANNELS.items():
            codes = pd.factorize(df.loc[sel, keys].astype(str).agg("\x1f".join, axis=1))[0]
            rows.append({**base, "channel": name, **_bound(y[sel], codes)})
    return rows


def main_attainable_auroc(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    k = int(cfg["max_evidence_sentences"])
    out_dir = out(cfg, "attainable_auroc")
    status = os.path.join(out_dir, "status.txt")
    sets = bound_sets(cfg)
    units = [{"key": s["key"], "set": s, "params": {"regime": _REGIME, "max_evidence_sentences": k,
                                                  "input": digest_file(s["path"]) if os.path.exists(s["path"]) else None}}
             for s in sets]

    def compute(unit):
        s = unit["set"]
        if unit["params"]["input"] is None:
            raise MissingInput(f"{s['path']} is missing; run main_build_examples first")
        df = pd.read_parquet(s["path"])
        df = df[df["split"] == s["split"]]
        return bounds(df, k, s["key"])

    rows, refused = run_units("main_attainable_auroc", units, compute, out_dir, status)
    if refused:
        raise MissingInput(f"{refused} set(s) could not be read yet")
    write_table(pd.DataFrame(rows), os.path.join(out_dir, "attainable_auroc.csv"))
    flush_status(status, "[main_attainable_auroc] done")
    return out_dir
