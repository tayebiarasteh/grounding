"""
Inference/stats_utils.py
Created on July 24, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import numpy as np

N_BOOT    = 1000
BOOT_SEED = 0


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    nan_mask  = np.isnan(p_values)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return p_values.copy()
    valid_p  = p_values[valid_idx]
    m        = len(valid_p)
    order    = np.argsort(valid_p)
    p_sorted = valid_p[order]
    adj = p_sorted * m / (np.arange(m, dtype=float) + 1)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    inv_order = np.empty(m, dtype=int)
    inv_order[order] = np.arange(m)
    result = p_values.copy()
    result[valid_idx] = adj[inv_order]
    return result
