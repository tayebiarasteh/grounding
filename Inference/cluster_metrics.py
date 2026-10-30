"""
Inference/cluster_metrics.py
Created on October 3, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from Inference.stats_utils import BOOT_SEED, N_BOOT

EVIDENCE_PAIR_REGIME = "cluster-once-v1"


def cluster_weights(clusters: Sequence, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    c = pd.Series(clusters)
    bad = c.isna() | c.astype(str).str.strip().str.lower().isin(["", "nan", "none", "<na>"])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have no cluster id")
    _, inv = np.unique(c.to_numpy(), return_inverse=True)
    n_c = int(inv.max()) + 1
    draws = np.random.RandomState(seed).randint(0, n_c, size=(n_boot, n_c))
    offsets = draws + n_c * np.arange(n_boot)[:, None]
    counts = np.bincount(offsets.ravel(), minlength=n_boot * n_c).reshape(n_boot, n_c)
    w = counts[:, inv].astype(np.float64)
    return np.vstack([np.ones((1, len(inv))), w])


def _score_groups(s: np.ndarray):
    _, grp = np.unique(s, return_inverse=True)
    m = sparse.csr_matrix((np.ones(len(s)), (np.arange(len(s)), grp)), shape=(len(s), int(grp.max()) + 1))
    return m


def _per_group(w: np.ndarray, m) -> np.ndarray:
    return np.asarray((m.T @ w.T).T)


def weighted_auroc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    m = _score_groups(np.asarray(s, dtype=np.float64))
    pos = _per_group(w * y, m)
    neg = _per_group(w * (1.0 - y), m)
    below = np.cumsum(neg, axis=1) - neg
    num = (pos * (below + 0.5 * neg)).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def pair_accuracy(groups: List[Dict], w: np.ndarray, within_cluster: bool = False) -> np.ndarray:
    num = np.zeros(w.shape[0])
    den = np.zeros(w.shape[0])
    for g in groups:
        sp, sn = g["s"][g["pos"]], g["s"][g["neg"]]
        cmp = (sp[:, None] > sn[None, :]).astype(np.float64) + 0.5 * (sp[:, None] == sn[None, :])
        wp, wn = w[:, g["pos"]], w[:, g["neg"]]
        if within_cluster:
            count = wp[:, :1]
            if not (np.all(wp == count) and np.all(wn == count)):
                raise ValueError("a within-cluster pair has inconsistent multiplicities")
            num += count[:, 0] * cmp.sum()
            den += count[:, 0] * cmp.size
        else:
            num += ((wp @ cmp) * wn).sum(axis=1)
            den += wp.sum(axis=1) * wn.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def summarize(values: np.ndarray, prefix: str) -> Dict:
    point = float(values[0])
    boot = values[1:][np.isfinite(values[1:])]
    if boot.size < 2 or not np.isfinite(point):
        return {f"{prefix}_point": point, f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                f"{prefix}_ci_low": np.nan, f"{prefix}_ci_high": np.nan, f"{prefix}_n_boot": int(boot.size)}
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    mean = float(boot.mean())
    tol = 1e-9 * max(1.0, abs(point), abs(lo), abs(hi))
    for name, v in (("point", point), ("bootstrap mean", mean)):
        if not (lo - tol <= v <= hi + tol):
            raise AssertionError(f"{prefix}: {name} {v} outside its interval [{lo}, {hi}]")
    return {f"{prefix}_point": point, f"{prefix}_mean": mean, f"{prefix}_std": float(boot.std(ddof=1)),
            f"{prefix}_ci_low": lo, f"{prefix}_ci_high": hi, f"{prefix}_n_boot": int(boot.size)}
