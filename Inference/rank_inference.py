"""
Inference/rank_inference.py
Created on October 30, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json

import numpy as np
import pandas as pd

from Inference.stats_utils import BOOT_SEED, N_BOOT


def joint_patient_weights(groups, n_boot=N_BOOT, seed=BOOT_SEED):
    checked = []
    for cohort, ids in groups:
        c = pd.Series(ids)
        bad = c.isna() | c.astype(str).str.strip().str.lower().isin(["", "nan", "none", "<na>"])
        if bad.any():
            raise ValueError(f"{int(bad.sum())} rows have no patient ID")
        checked.append((cohort, c.astype(str).to_numpy()))
    rng, banks = np.random.RandomState(seed), {}
    h = hashlib.sha256(json.dumps({"n_boot": n_boot, "seed": seed}, sort_keys=True).encode())
    for cohort in sorted({name for name, _ in checked}):
        units = np.unique(np.concatenate([ids for name, ids in checked if name == cohort]))
        n = len(units)
        if not n:
            raise ValueError(f"{cohort}: no patients")
        draws = rng.randint(0, n, size=(n_boot, n))
        counts = np.bincount((draws+n*np.arange(n_boot)[:, None]).ravel(), minlength=n_boot*n).reshape(n_boot, n)
        banks[cohort] = (units, np.vstack([np.ones((1, n)), counts]))
        h.update(json.dumps([cohort, units.tolist()]).encode())
        h.update(counts.astype("<i8").tobytes())
    result = []
    for cohort, ids in checked:
        units, counts = banks[cohort]
        positions = np.searchsorted(units, ids)
        result.append(counts[:, positions])
        h.update(json.dumps([cohort, ids.tolist()]).encode())
    return result, {"resample_hash": h.hexdigest(), "n_boot": n_boot, "boot_seed": seed,
                    "n_patient_clusters": sum(len(u) for u, _ in banks.values())}


def centered_p(point, replicates):
    boot = np.asarray(replicates, float)
    finite = np.isfinite(boot)
    if not np.isfinite(point) or not len(boot):
        return {"p_raw": np.nan, "n_boot_valid": 0, "inference_reason": "undefined_statistic"}
    exceed = int(np.sum(np.abs(boot[finite]-point) >= abs(point))) + int((~finite).sum())
    return {"p_raw": float(exceed/len(boot)), "n_boot_valid": int(finite.sum()),
            "inference_reason": ""}


def weighted_kendall(x, y, weights):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    if len(x) != len(y) or w.ndim != 2 or w.shape[1] != len(x):
        raise ValueError("unaligned Kendall inputs")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[finite], y[finite], w[:, finite]
    dx, dy = np.sign(x[:, None]-x[None, :]), np.sign(y[:, None]-y[None, :])
    sums = [np.einsum("bi,ij,bj->b", w, k, w, optimize=True) for k in (dx*dy, dx*dx, dy*dy)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums[0]/np.sqrt(sums[1]*sums[2])


def patient_kendall(x, y, ids, n_boot=N_BOOT, seed=BOOT_SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) != len(ids):
        raise ValueError("unaligned patient IDs")
    weights, metadata = joint_patient_weights([("patients", ids)], n_boot, seed)
    values = weighted_kendall(x, y, weights[0])
    if metadata["n_patient_clusters"] < 3:
        return {"value": float(values[0]), "p_raw": np.nan, "n_boot_valid": 0, **metadata,
                "inference": "patient_cluster_bootstrap", "inference_reason": "fewer_than_three_patients"}
    return {"value": float(values[0]), **centered_p(values[0], values[1:]), **metadata,
            "inference": "patient_cluster_bootstrap"}
