"""
experiments/result_tables_added.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from experiments._common import MissingInput, digest_file, flush_status, get_cfg, read_results_csv, run_units, stats_params, write_table
from experiments.attainable_auroc import bounds
from Inference.cluster_metrics import cluster_weights
from experiments.study import ADDED_CONDITIONS, TEST_SETS, examples_path, out, test_sets, train_set_path
from experiments.score import input_path, is_scored, load_scores, score_path, verifiers
from experiments.result_tables import (TARGETS, _controls, _correlations, _pooled, _thresholds, accepted_share, main_result_tables,
                                    statement_rows, summarize_flagged, threshold_set)
from experiments.statements import statement_name
from experiments.evidence_sets import CITATIONS, FORMS, added_statement_set_names, written_share_rows

_REGIME = "result_tables_added-v1"
EXCLUDED_GENERATOR = "LLaVA-Med"
POOLED_VARIANTS = {"heldout": ("heldout", ()), "retrieved": ("retrieved", ()), "nollavamed": ("", (EXCLUDED_GENERATOR,))}
FLAG_TOLERANCE = 0.05


def class_mean_rows(cfg: Dict, vx: Dict) -> List[Dict]:
    tau = _thresholds(cfg, vx)[0.90]
    rows = []
    for s in test_sets(cfg, vx["domain"], vx["id"]):
        if not is_scored(cfg, vx["id"], s):
            continue
        df = load_scores(cfg, vx["id"], s)
        ds, cond = s.split("__", 1)
        for rt, g in df.groupby("row_type", sort=True):
            p = g["prob_supported"].to_numpy(float)
            rows.append({"table": "class_means", "verifier": vx["id"], "family": vx["family"], "dataset": ds,
                         "condition": cond, "row_type": rt, "n": int(len(p)), "mean_score": float(p.mean()),
                         "median_score": float(np.median(p)), "threshold_point": tau,
                         "accepted_share_tau90_point": float((p >= tau).mean())})
    return rows


def citation_cell_rows(cfg: Dict, vx: Dict, input_set: str, df: Optional[pd.DataFrame] = None,
                       tau: Optional[Dict[float, float]] = None) -> List[Dict]:
    n_boot, seed = stats_params(cfg)
    df = load_scores(cfg, vx["id"], input_set) if df is None else df.reset_index(drop=True)
    tau = _thresholds(cfg, vx) if tau is None else tau
    parts = input_set.split("__")
    source, form = "__".join(parts[1:-1]), parts[-1]
    s = df["prob_supported"].to_numpy(float)
    kappa, hit = df["label_kappa"].to_numpy(float), df["recall_hit"].to_numpy(int)
    w = cluster_weights(df["subject_id"], n_boot, seed)
    rows = []
    for k in (1, 0):
        for h in (1, 0):
            sel = (kappa == k) & (hit == h)
            for q in TARGETS:
                row = {"table": "citation_cells", "verifier": vx["id"], "family": vx["family"], "source": source,
                       "form": form, "report_state_agrees": k, "retrieved_support": h, "sensitivity_target": q,
                       "threshold_point": tau[q], "n": int(sel.sum())}
                if sel.any():
                    row.update(summarize_flagged(accepted_share(s[sel], w[:, sel], np.full(w.shape[0], tau[q])), "accepted"))
                rows.append(row)
    return rows


def operating_flag_rows(test: pd.DataFrame) -> List[Dict]:
    op = test[(test["table"] == "operating") & (test["dataset"] == "mimic") & (test["condition"] == "unchanged")]
    rows = []
    for r in op.to_dict("records"):
        miss = abs(float(r["supported_accept_point"]) - float(r["sensitivity_target"]))
        rows.append({"table": "operating_flags", "verifier": r["verifier"], "sensitivity_target": r["sensitivity_target"],
                     "supported_accept_point": r["supported_accept_point"], "miss": miss, "degenerate": bool(miss > FLAG_TOLERANCE)})
    return rows


def balanced_rows(stmts: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    from experiments.statements import _generators
    gens = _generators(cfg)
    rows = []
    for cit in [""] + [f"_{c}" for c in CITATIONS]:
        sources = {f"gen__{g}{cit}" for g in gens}
        g = stmts[stmts["source"].isin(sources)]
        keys = ["verifier", "family", "form", "definition", "sensitivity_target", "threshold_basis"]
        for key, h in g.groupby(keys):
            if h["source"].nunique() < len(gens):
                continue
            rows.append({"table": "statements_balanced", **dict(zip(keys, key)), "source": f"genbalanced{cit}",
                         "n_generators": int(h["source"].nunique()),
                         "contradicted_accept_mean": float(h["contradicted_accept_mean"].mean()),
                         "consistent_accept_mean": float(h["consistent_accept_mean"].mean())})
    return pd.DataFrame(rows)


def _units(cfg: Dict) -> List[Dict]:
    n_boot, seed = stats_params(cfg)
    units = []
    from experiments.statements import _generators
    retrieved = [s for s in added_statement_set_names(cfg) if "_retrieved__" in s]
    for vx in verifiers(cfg):
        if vx["domain"] != "radiology":
            continue
        sets = [s for s in test_sets(cfg, vx["domain"], vx["id"]) if is_scored(cfg, vx["id"], s)]
        digests = {s: digest_file(score_path(cfg, vx["id"], s)) for s in sets}
        thr = score_path(cfg, vx["id"], threshold_set(vx))
        units.append({"key": f"class_means__{vx['id']}", "kind": "class_means", "vx": vx,
                      "params": {"regime": _REGIME, "scores": digests, "threshold_scores": digest_file(thr) if os.path.exists(thr) else None}})
        for s in retrieved:
            p = score_path(cfg, vx["id"], s)
            units.append({"key": f"cells__{vx['id']}__{s}", "kind": "cells", "vx": vx, "set": s,
                          "params": {"regime": _REGIME, "scores": digest_file(p) if os.path.exists(p) else None,
                                     "input": digest_file(input_path(cfg, s)) if os.path.exists(input_path(cfg, s)) else None,
                                     "threshold_scores": digest_file(thr) if os.path.exists(thr) else None,
                                     "n_boot": n_boot, "boot_seed": seed}})
        for variant, (cit, exclude) in POOLED_VARIANTS.items():
            for form in FORMS:
                ps = [score_path(cfg, vx["id"], statement_name(f"gen__{g}", form, cit)) for g in _generators(cfg) if g not in exclude]
                units.append({"key": f"pooled__{vx['id']}__{variant}__{form}", "kind": "pooled", "vx": vx, "variant": variant,
                              "form": form, "params": {"regime": _REGIME, "scores": [digest_file(p) if os.path.exists(p) else None for p in ps],
                                                       "threshold_scores": digest_file(thr) if os.path.exists(thr) else None,
                                                       "n_boot": n_boot, "boot_seed": seed}})
    from experiments.controls import RETRIEVAL
    vid = cfg["main_verifier_added"]
    ps = [score_path(cfg, vid, f"mimic__{c}") for c in RETRIEVAL] + [score_path(cfg, vid, f"{ds}__unchanged") for ds in TEST_SETS]
    units.append({"key": f"controls__{vid}", "kind": "controls", "vx": None,
                  "params": {"regime": _REGIME, "scores": [digest_file(p) if os.path.exists(p) else None for p in ps],
                             "n_boot": n_boot, "boot_seed": seed}})
    k = int(cfg["max_evidence_sentences"])
    added = [{"key": f"mimic__{c}", "path": examples_path(cfg, "mimic", c), "split": "test"} for c in ADDED_CONDITIONS]
    added += [{"key": f"train__{n}", "path": train_set_path(cfg, n), "split": "train"}
              for n in sorted({r["train_set"] for r in cfg.get("runs_added", [])})]
    for a in added:
        units.append({"key": f"bounds__{a['key']}", "kind": "bounds", "vx": None, "set": a,
                      "params": {"regime": _REGIME, "max_evidence_sentences": k,
                                 "input": digest_file(a["path"]) if os.path.exists(a["path"]) else None}})
    return units


def main_result_tables_added(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    from experiments.score_added import added_main_verifier, pending_added
    added_main_verifier(cfg)
    owed = pending_added(cfg)
    if owed:
        raise MissingInput(f"main_score_added still owes {len(owed)} unit(s), among them {owed[0]}; score them first")
    main_result_tables(cfg_path)
    out14 = out(cfg, "result_tables")
    out_dir = out(cfg, "result_tables_added")
    status = os.path.join(out_dir, "status.txt")
    units = _units(cfg)

    def compute(unit):
        vx = unit["vx"]
        if unit["kind"] == "class_means":
            return class_mean_rows(cfg, vx)
        if unit["kind"] == "cells":
            if not is_scored(cfg, vx["id"], unit["set"]):
                raise MissingInput(f"{vx['id']} has no scores on {unit['set']}")
            return citation_cell_rows(cfg, vx, unit["set"])
        if unit["kind"] == "pooled":
            cit, exclude = POOLED_VARIANTS[unit["variant"]]
            df = _pooled(cfg, vx, unit["form"], citation=cit, exclude=exclude)
            return statement_rows(cfg, vx, statement_name("genpooled", unit["form"], unit["variant"]), df=df)
        if unit["kind"] == "bounds":
            a = unit["set"]
            if unit["params"]["input"] is None:
                raise MissingInput(f"{a['path']} is missing; run main_build_evidence_sets first")
            df = pd.read_parquet(a["path"])
            return [{"table": "bounds", **r} for r in bounds(df[df["split"] == a["split"]], int(cfg["max_evidence_sentences"]), a["key"])]
        return _controls(cfg, vid=cfg["main_verifier_added"])

    rows, refused = run_units("main_result_tables_added", units, compute, out_dir, status)
    if refused:
        raise MissingInput(f"{refused} unit(s) of main_result_tables_added wait for scores")
    df = pd.DataFrame(rows)
    test = read_results_csv(os.path.join(out14, "test_sets.csv"))
    stmts14 = read_results_csv(os.path.join(out14, "statements.csv"))
    stmts19 = df[df["table"] == "statements"].dropna(axis=1, how="all")
    balanced = balanced_rows(pd.concat([stmts14, stmts19], ignore_index=True), cfg)
    stats = _correlations(test, pd.concat([stmts14, stmts19, balanced], ignore_index=True))
    write_table(df[df["table"].isin(["class_means", "citation_cells", "similarity_subsets", "retrieval"])].dropna(axis=1, how="all"),
                os.path.join(out_dir, "additions.csv"))
    write_table(pd.concat([stmts19, balanced], ignore_index=True), os.path.join(out_dir, "statements.csv"))
    write_table(stats, os.path.join(out_dir, "statistics.csv"))
    write_table(pd.DataFrame(operating_flag_rows(test) + written_share_rows(cfg)), os.path.join(out_dir, "flags_and_written.csv"))
    write_table(df[df["table"] == "bounds"].dropna(axis=1, how="all"), os.path.join(out_dir, "bounds.csv"))
    flush_status(status, "[main_result_tables_added] done")
    return out_dir
