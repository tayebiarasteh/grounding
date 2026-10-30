"""
experiments/result_tables.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from experiments._common import (MissingInput, digest_file, flush_status, get_cfg, read_results_csv,
                                 read_source_csv, run_units, stats_params, write_table)
from Inference.cluster_metrics import EVIDENCE_PAIR_REGIME, cluster_weights, pair_accuracy, summarize, weighted_auroc
from Inference.stats_utils import bh_fdr
from Inference.rank_inference import centered_p, patient_kendall
from experiments.study import SUBGROUP_SETS, TEST_SETS, out, patient_groups, read_evidence
from experiments.score import input_path, input_sets, is_scored, load_scores, score_path, verifiers

_REGIME = "result_tables-v5"
TARGETS = (0.90, 0.95)
_PAIR_SOURCE = {"unchanged": "unchanged", "heldout": "heldout", "heldout_written": "heldout_written", "removed": "unchanged",
                "masked_evidence": "masked_evidence", "masked_evidence_statement": "masked_evidence_statement",
                "report_replaced": "report_replaced"}
_DEFINITIONS = ("agree", "kappa", "flag", "full")


def pair_groups(df: pd.DataFrame, kind: str, s: np.ndarray, sel: np.ndarray = None,
                neg_sel: np.ndarray = None) -> List[Dict]:
    sel = np.ones(len(df), bool) if sel is None else sel
    neg_sel = np.ones(len(df), bool) if neg_sel is None else neg_sel
    if kind == "case":
        sub, keys, neg_types = df[df["row_type"].isin(["C", "D"])], ["ev_key", "claim_text"], ["D"]
    else:
        sub, keys, neg_types = df[df["row_type"].isin(["A", "B", "C"])], ["qa_row_id", "claim_text"], ["A", "B"]
    groups = []
    for _, g in sub.groupby(keys, sort=False):
        pos = g.index[(g["row_type"] == "C").to_numpy() & sel[g.index]].to_numpy()
        neg = g.index[g["row_type"].isin(neg_types).to_numpy() & neg_sel[g.index]].to_numpy()
        if len(pos) and len(neg):
            groups.append({"pos": pos, "neg": neg, "s": s})
    return groups


def threshold_at_sensitivity(s_pos: np.ndarray, w_pos: np.ndarray, target: float) -> np.ndarray:
    order = np.argsort(-s_pos, kind="mergesort")
    s_sorted = s_pos[order]
    cum = np.cumsum(w_pos[:, order], axis=1)
    reach = cum >= target * cum[:, -1:] - 1e-9
    idx = reach.argmax(axis=1)
    last_of_value = np.searchsorted(-s_sorted, -s_sorted, side="right") - 1
    return s_sorted[last_of_value[idx]]


def accepted_share(s: np.ndarray, w: np.ndarray, t: np.ndarray) -> np.ndarray:
    acc = (s[None, :] >= t[:, None]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w.sum(axis=1) > 0, (w * acc).sum(axis=1) / w.sum(axis=1), np.nan)


def summarize_flagged(values: np.ndarray, prefix: str, tol: float = 5e-4) -> Dict:
    point = float(values[0])
    boot = values[1:][np.isfinite(values[1:])]
    if boot.size < 2 or not np.isfinite(point):
        return {f"{prefix}_point": point, f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                f"{prefix}_ci_low": np.nan, f"{prefix}_ci_high": np.nan, f"{prefix}_n_boot": int(boot.size),
                f"{prefix}_point_in_ci": np.nan, f"{prefix}_mean_in_ci": np.nan}
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    mean = float(boot.mean())
    return {f"{prefix}_point": point, f"{prefix}_mean": mean, f"{prefix}_std": float(boot.std(ddof=1)),
            f"{prefix}_ci_low": lo, f"{prefix}_ci_high": hi, f"{prefix}_n_boot": int(boot.size),
            f"{prefix}_point_in_ci": bool(lo - tol <= point <= hi + tol),
            f"{prefix}_mean_in_ci": bool(lo - tol <= mean <= hi + tol)}


def summarize_auroc(values: np.ndarray, whole_set: bool) -> Dict:
    return summarize(values, "auroc") if whole_set else summarize_flagged(values, "auroc")


def threshold_set(vx: Dict) -> str:
    return "food__unchanged" if vx["domain"] == "food" else "mimic__unchanged"


def _thresholds(cfg: Dict, vx: Dict, ref: Optional[pd.DataFrame] = None) -> Dict[float, float]:
    ref = load_scores(cfg, vx["id"], threshold_set(vx)) if ref is None else ref
    s = ref.loc[ref["row_type"] == "C", "prob_supported"].to_numpy(float)
    one = np.ones((1, len(s)))
    return {q: float(threshold_at_sensitivity(s, one, q)[0]) for q in TARGETS}


def _ev_key(df: pd.DataFrame, k: int) -> pd.Series:
    return df["evidence_sentences_json"].apply(lambda js: "\n\n".join(read_evidence(js, k)))


KINDS = ("states", "negates", "not_mentioned")
_REPORT_FINDINGS = {}


def report_findings_path(cfg: Dict) -> str:
    return out(cfg, "examples", "report_findings.parquet")


def contradiction_kind(cfg: Dict, ds: str, df: pd.DataFrame) -> np.ndarray:
    p = report_findings_path(cfg)
    key = (p, os.path.getmtime(p))
    if key not in _REPORT_FINDINGS:
        _REPORT_FINDINGS.clear()
        _REPORT_FINDINGS[key] = pd.read_parquet(p, columns=["dataset", "qa_row_id", "report_code"])
    rf = _REPORT_FINDINGS[key]
    code = df["qa_row_id"].map(rf[rf["dataset"] == ds].set_index("qa_row_id")["report_code"]).to_numpy(dtype=object)
    d = (df["row_type"] == "D").to_numpy()
    if pd.isna(code[d]).any():
        raise ValueError(f"{ds}: {int(pd.isna(code[d]).sum())} contradicted examples have no report finding")
    st = df["gold_state"].to_numpy(dtype=object)
    kind = np.where(st == "present", "states", np.where(code == "negative", "negates", "not_mentioned"))
    return np.where(d, kind, "")


def _contradiction_rows(cfg: Dict, ds: str, df: pd.DataFrame, s: np.ndarray, w: np.ndarray, t: np.ndarray,
                        pdf: pd.DataFrame, ident: Dict) -> List[Dict]:
    kind = contradiction_kind(cfg, ds, df)
    rows = []
    for k in KINDS:
        sel = kind == k
        row = {"table": "contradiction", **ident, "finding": "all", "kind": k, "n_contradicted": int(sel.sum())}
        if sel.any():
            row.update(summarize_flagged(accepted_share(s[sel], w[:, sel], t), "contradicted_accept"))
            groups = pair_groups(pdf, "case", s, neg_sel=sel)
            row["n_patient_pairs"] = int(sum(len(g["pos"]) * len(g["neg"]) for g in groups))
            if groups:
                row.update(summarize(pair_accuracy(groups, w), "patient_pair_acc"))
        rows.append(row)
    return rows


def _subgroup_rows(cfg: Dict, ds: str, df: pd.DataFrame, y: np.ndarray, s: np.ndarray, w: np.ndarray, t: np.ndarray,
                   ident: Dict) -> List[Dict]:
    groups = patient_groups(cfg, ds, df)
    min_p = int(cfg["subgroups"]["min_patients"])
    rt, subj = df["row_type"].to_numpy(), df["subject_id"].astype(str).to_numpy()
    rows = []
    for attr, lab in groups.items():
        for g in sorted(set(lab) - {""}):
            sel = lab == g
            row = {"table": "subgroup", **ident, "finding": "all", "attribute": attr, "group": g,
                   "n_patients": int(len(set(subj[sel]))), "n": int(sel.sum()), "n_pos": int(y[sel].sum())}
            if row["n_patients"] >= min_p:
                if 0 < y[sel].sum() < sel.sum():
                    row.update(summarize_auroc(weighted_auroc(y[sel], s[sel], w[:, sel]), whole_set=False))
                for name, m in (("supported_accept", sel & (rt == "C")), ("contradicted_accept", sel & (rt == "D"))):
                    if m.any():
                        row.update(summarize_flagged(accepted_share(s[m], w[:, m], t), name))
            rows.append(row)
    return rows


def report_change_rows(cfg: Dict, vx: Dict, ds: str) -> List[Dict]:
    n_boot, seed = stats_params(cfg)
    b = load_scores(cfg, vx["id"], f"{ds}__report_replaced")
    a = load_scores(cfg, vx["id"], f"{ds}__unchanged").set_index("verifier_row_id").loc[b["verifier_row_id"]]
    tau = _thresholds(cfg, vx)[0.90]
    sa, sb = a["prob_supported"].to_numpy(float), b["prob_supported"].to_numpy(float)
    changed = ((sa >= tau) != (sb >= tau)).astype(np.float64)
    same_text = a["case_context"].to_numpy() == b["case_context"].to_numpy()
    w = cluster_weights(b["subject_id"], n_boot, seed)
    rt = b["row_type"].to_numpy()
    rows = []
    for name, sel in (("supported", rt == "C"), ("contradicted", rt == "D"), ("all", np.ones(len(b), bool))):
        row = {"table": "report_change", "verifier": vx["id"], "family": vx["family"], "dataset": ds,
               "condition": "report_replaced", "finding": "all", "examples": name, "n": int(sel.sum()),
               "n_identical_text": int(same_text[sel].sum()), "threshold_point": tau}
        if sel.any():
            ws = w[:, sel]
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = np.where(ws.sum(axis=1) > 0, (ws * changed[sel]).sum(axis=1) / ws.sum(axis=1), np.nan)
            row.update(summarize_flagged(vals, "decision_change"))
        rows.append(row)
    return rows


def _spearman_across(x: pd.Series, y: pd.Series) -> Tuple[int, float, float]:
    j = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(j) < 4 or j["x"].nunique() < 2 or j["y"].nunique() < 2:
        return int(len(j)), np.nan, np.nan
    rho, p = spearmanr(j["x"], j["y"])
    return int(len(j)), float(rho), float(p)


def _mesh_correlation(test: pd.DataFrame) -> List[Dict]:
    p = test[(test["table"] == "pairs") & (test["condition"] == "unchanged") & (test["finding"] == "all")]
    x = p[p["dataset"] == "openi"].set_index("verifier")["patient_pair_acc_mean"]
    y = p[p["dataset"] == "openi_mesh"].set_index("verifier")["patient_pair_acc_mean"]
    n, rho, pv = _spearman_across(x, y)
    return [{"table": "statistic", "family": "E7", "test": "spearman",
             "comparison": "patient_pairs__openi_chexbert_vs_openi_mesh", "n": n, "value": rho, "p_raw": pv,
             "inference": "analytic_verifier_units"}]


def test_rows(cfg: Dict, vx: Dict, input_set: str, splits: bool = True, df: Optional[pd.DataFrame] = None,
              tau: Optional[Dict[float, float]] = None) -> List[Dict]:
    n_boot, seed = stats_params(cfg)
    k = int(cfg["max_evidence_sentences"])
    ds, cond = input_set.split("__", 1)
    df = load_scores(cfg, vx["id"], input_set) if df is None else df.reset_index(drop=True)
    w = cluster_weights(df["subject_id"], n_boot, seed)
    y, s = df["label_id"].to_numpy(float), df["prob_supported"].to_numpy(float)
    ident = {"verifier": vx["id"], "family": vx["family"], "dataset": ds, "condition": cond}
    rows = []
    findings = ["all"] + sorted(df["concept"].unique())
    pdf = None
    for f in findings:
        sel = np.ones(len(df), bool) if f == "all" else (df["concept"] == f).to_numpy()
        row = {"table": "metrics", **ident, "finding": f, "n": int(sel.sum()), "n_pos": int(y[sel].sum())}
        if 0 < y[sel].sum() < sel.sum():
            row.update(summarize_auroc(weighted_auroc(y[sel], s[sel], w[:, sel]), whole_set=(f == "all")))
        rows.append(row)
    if cond in _PAIR_SOURCE:
        src = df if _PAIR_SOURCE[cond] == cond else pd.read_parquet(input_path(cfg, f"{ds}__{_PAIR_SOURCE[cond]}"))
        ref = src.set_index("verifier_row_id").loc[df["verifier_row_id"]]
        pdf = pd.DataFrame({"row_type": df["row_type"].to_numpy(), "claim_text": ref["claim_text"].to_numpy(),
                            "qa_row_id": df["qa_row_id"].to_numpy(),
                            "ev_key": _ev_key(ref.reset_index(), k).to_numpy()})
        kinds = ("case", "evidence") if cond in ("unchanged", "heldout", "heldout_written") else ("case",)
        per_finding = []
        for f in findings:
            sel = np.ones(len(df), bool) if f == "all" else (df["concept"] == f).to_numpy()
            row = {"table": "pairs", **ident, "finding": f}
            for kind in kinds:
                groups = pair_groups(pdf, kind, s, sel)
                name = "patient" if kind == "case" else "evidence"
                row[f"n_{name}_groups"] = len(groups)
                row[f"n_{name}_pairs"] = int(sum(len(g["pos"]) * len(g["neg"]) for g in groups))
                if groups:
                    vals = pair_accuracy(groups, w, within_cluster=(kind == "evidence"))
                    row.update(summarize(vals, f"{name}_pair_acc"))
                    if kind == "case" and f != "all":
                        per_finding.append(vals)
            rows.append(row)
        if per_finding:
            means = [float(np.mean(v[1:][np.isfinite(v[1:])])) for v in per_finding]
            rows.append({"table": "pairs", **ident, "finding": "median_over_findings",
                         "n_findings": len(per_finding),
                         "patient_pair_acc_median_of_points": float(np.median([v[0] for v in per_finding])),
                         "patient_pair_acc_median_of_means": float(np.median(means))})
    if cond in ("unchanged", "heldout", "heldout_written", "report_replaced"):
        tau = _thresholds(cfg, vx) if tau is None else tau
        rt = df["row_type"].to_numpy()
        pos, con, nos = rt == "C", rt == "D", np.isin(rt, ["A", "B"])
        for q in TARGETS:
            if input_set == threshold_set(vx):
                t = threshold_at_sensitivity(s[pos], w[:, pos], q)
            else:
                t = np.full(w.shape[0], tau[q])
            row = {"table": "operating", **ident, "finding": "all", "sensitivity_target": q,
                   "threshold_point": float(t[0]), "n_supported": int(pos.sum()), "n_contradicted": int(con.sum()),
                   "n_without_support": int(nos.sum())}
            row.update(summarize_flagged(accepted_share(s[pos], w[:, pos], t), "supported_accept"))
            row.update(summarize_flagged(accepted_share(s[con], w[:, con], t), "contradicted_accept"))
            row.update(summarize_flagged(accepted_share(s[nos], w[:, nos], t), "without_support_accept"))
            rows.append(row)
            if splits and cond == "unchanged" and q == 0.90 and ds in TEST_SETS:
                rows += _contradiction_rows(cfg, ds, df, s, w, t, pdf, ident)
                if ds in SUBGROUP_SETS:
                    rows += _subgroup_rows(cfg, ds, df, y, s, w, t, ident)
    return rows


def statement_rows(cfg: Dict, vx: Dict, input_set: str, df: Optional[pd.DataFrame] = None,
                   tau: Optional[Dict[float, float]] = None) -> List[Dict]:
    n_boot, seed = stats_params(cfg)
    df = load_scores(cfg, vx["id"], input_set) if df is None else df
    parts = input_set.split("__")
    source, form = "__".join(parts[1:-1]), parts[-1]
    tau = _thresholds(cfg, vx) if tau is None else tau
    ident = {"verifier": vx["id"], "family": vx["family"], "source": source, "form": form}
    rows = []
    for d in [d for d in _DEFINITIONS if f"label_{d}" in df.columns]:
        g = df[df[f"label_{d}"].notna()].reset_index(drop=True)
        y = g[f"label_{d}"].astype(int).to_numpy()
        if not (0 < y.sum() < len(y)):
            continue
        s = g["prob_supported"].to_numpy(float)
        w = cluster_weights(g["subject_id"], n_boot, seed)
        auroc = summarize(weighted_auroc(y, s, w), "auroc")
        for q in TARGETS:
            for basis in ("test", "statements"):
                t = np.full(w.shape[0], tau[q]) if basis == "test" else threshold_at_sensitivity(s[y == 1], w[:, y == 1], q)
                row = {"table": "statements", **ident, "definition": d, "sensitivity_target": q,
                       "threshold_basis": basis, "threshold_point": float(t[0]), "n_consistent": int((y == 1).sum()),
                       "n_contradicted": int((y == 0).sum()), "n_clusters": int(g["subject_id"].nunique())}
                row.update(summarize_flagged(accepted_share(s[y == 1], w[:, y == 1], t), "consistent_accept"))
                row.update(summarize_flagged(accepted_share(s[y == 0], w[:, y == 0], t), "contradicted_accept"))
                row.update(auroc)
                rows.append(row)
    return rows


def report_error_rows(cfg: Dict, vx: Dict, input_set: str) -> List[Dict]:
    df = load_scores(cfg, vx["id"], input_set)
    tau = _thresholds(cfg, vx)[0.90]
    source = input_set.split("__")[1]
    g = df.groupby("candidate_id")["prob_supported"]
    return [{"table": "report_error", "verifier": vx["id"], "family": vx["family"], "source": source,
             "candidate_id": cid, "n_statements": int(len(x)), "error_score": float((x < tau).mean()),
             "threshold": tau} for cid, x in g]


def _pooled(cfg: Dict, vx: Dict, form: str, citation: str = "", exclude: Tuple[str, ...] = ()) -> pd.DataFrame:
    from experiments.statements import _generators
    from experiments.statements import statement_name
    parts = []
    for gname in [g for g in _generators(cfg) if g not in exclude]:
        name = statement_name(f"gen__{gname}", form, citation)
        if not is_scored(cfg, vx["id"], name):
            raise MissingInput(f"{vx['id']} has no scores on {name}")
        parts.append(load_scores(cfg, vx["id"], name).assign(generator=gname))
    return pd.concat(parts, ignore_index=True)


def agreement_rows(y1: np.ndarray, y2: np.ndarray, name: str, clusters: np.ndarray, cfg: Dict, extra: Dict):
    n_boot, seed = stats_params(cfg)
    y1, y2 = np.asarray(y1), np.asarray(y2)
    n = len(y1)
    cats = np.unique(np.concatenate([y1, y2]))
    p1 = np.array([np.mean(y1 == c) for c in cats])
    p2 = np.array([np.mean(y2 == c) for c in cats])
    po, pe = float(np.mean(y1 == y2)), float((p1 * p2).sum())
    kappa = (po - pe) / (1.0 - pe) if pe < 1 else np.nan
    se0 = np.sqrt(max(pe + pe ** 2 - float((p1 * p2 * (p1 + p2)).sum()), 0.0) / (n * (1.0 - pe) ** 2)) if pe < 1 else np.nan
    p = float(2 * norm.sf(abs(kappa) / se0)) if np.isfinite(kappa) and se0 and np.isfinite(se0) and se0 > 0 else np.nan
    w = cluster_weights(clusters, n_boot, seed)
    totals = w.sum(axis=1)
    agree = (w * (y1 == y2)).sum(axis=1) / totals
    n_clusters = len(np.unique(clusters))
    inference = {"inference": "independent_item_analytic", "n_agreement_clusters": n_clusters}
    if n_clusters < n:
        margins1 = np.stack([w @ (y1 == c) / totals for c in cats], axis=1)
        margins2 = np.stack([w @ (y2 == c) / totals for c in cats], axis=1)
        expected = (margins1 * margins2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            kappas = (agree - expected) / (1.0 - expected)
        inference.update(centered_p(kappa, kappas[1:]))
        inference.update(inference="agreement_cluster_bootstrap", n_boot=n_boot, boot_seed=seed)
        p = inference.pop("p_raw")
    metric = {"table": "agreement", "comparison": name, "n": n, **extra, **summarize(agree, "agreement")}
    stat = {"table": "statistic", "family": "E7", "test": "cohen_kappa", "comparison": name, "n": n, **extra,
            "value": kappa, "p_raw": p, **inference}
    return metric, stat


def _label_agreement(cfg: Dict) -> Tuple[List[Dict], List[Dict]]:
    from Inference.chexbert_labeler import FINDING_HEAD
    from experiments.study import mesh_findings, read_openi
    from experiments.report_labels import load_labels
    from experiments.judges import judge_key, load_judgments
    v = cfg
    code = {1: "present", 0: "absent", 2: "absent"}
    col = {f: FINDING_HEAD[f].lower().replace(" ", "_") for f in v["findings"]}
    col["pulmonary edema"] = "edema"
    metrics, stats = [], []

    def add(pair):
        metrics.append(pair[0])
        stats.append(pair[1])

    for ds, path in (("mimic", v["sources"]["mimic_master_list"]), ("chexpert", v["sources"]["chexpert_master_list"])):
        lab = load_labels(cfg, ds)
        key = "study_id" if ds == "mimic" else "jpg_rel_path"
        m = read_source_csv(path, usecols=[key] + list(col.values()))
        if ds == "mimic":
            m["study_key"] = m["study_id"].astype(str)
        else:
            m["study_key"] = m["jpg_rel_path"].str.replace(r"^CheXpert-v1\.0/", "", regex=True).str.split("/").str[:3].str.join("/")
        m = m[m["study_key"].isin(set(lab["study_key"]))]
        for f, c in col.items():
            codes = m.groupby("study_key")[c].agg(lambda x: x.iloc[0] if x.nunique() == 1 else np.nan)
            a = lab.set_index("study_key")[f"state__{f}"]
            b = codes.map(code)
            j = pd.DataFrame({"a": a, "b": b, "subject": lab.set_index("study_key")["subject_id"]}).dropna()
            j = j[j["a"].isin(["present", "absent"])]
            if len(j) and j["a"].nunique() + j["b"].nunique() > 2:
                add(agreement_rows(j["a"].to_numpy(), j["b"].to_numpy(), f"chexbert_vs_codes__{ds}__{f}",
                                   j["subject"].to_numpy(), cfg, {"dataset": ds, "finding": f}))
    oi = read_openi(v["sources"]["openi_reports_dir"]).set_index("report_id")
    lab = load_labels(cfg, "openi").set_index("study_key")
    mesh = {r: mesh_findings(oi.at[r, "mesh_major_json"]) for r in lab.index}
    mesh = {r: m for r, m in mesh.items() if m is not None}
    for f in v["findings"]:
        a = lab[f"state__{f}"]
        b = pd.Series({r: "present" if f in m else "absent" for r, m in mesh.items()}, dtype=object)
        j = pd.DataFrame({"a": a, "b": b}).dropna()
        j = j[j["a"].isin(["present", "absent"])]
        if len(j) and j["a"].nunique() + j["b"].nunique() > 2:
            add(agreement_rows(j["a"].to_numpy(), j["b"].to_numpy(), f"chexbert_vs_mesh__openi__{f}",
                               j.index.to_numpy(), cfg, {"dataset": "openi", "finding": f}))
    pg = load_labels(cfg, "padchest_gr")
    for f in v["findings"]:
        a, b = f"chexbert_state__{f}", f"state__{f}"
        j = pg[pg[a].isin(["present", "absent"])] if a in pg.columns else pg.iloc[:0]
        if len(j) and j[a].nunique() + j[b].nunique() > 2:
            add(agreement_rows(j[a].to_numpy(), j[b].to_numpy(), f"chexbert_vs_release__padchest_gr__{f}",
                               j["subject_id"].to_numpy(), cfg, {"dataset": "padchest_gr", "finding": f}))
    jd = load_judgments(cfg)
    keys = [judge_key(e["name"]) for e in v["judges"]["models"]]
    if len(keys) != 2:
        raise ValueError(f"the agreement between judges is defined for two judges; the config lists {len(keys)}")
    k1, k2 = keys
    ok = (jd[f"{k1}__status"] == "ok") & (jd[f"{k2}__status"] == "ok")
    add(agreement_rows(jd.loc[ok, f"{k1}__state"].to_numpy(), jd.loc[ok, f"{k2}__state"].to_numpy(),
                       "judge_vs_judge__state", jd.loc[ok, "sentence_id"].to_numpy(), cfg,
                       {"dataset": "radiopaedia", "finding": "all"}))
    return metrics, stats


def _correlations(test: pd.DataFrame, stmts: pd.DataFrame) -> pd.DataFrame:
    pairs = test[(test["table"] == "pairs") & (test["dataset"] == "mimic") & (test["condition"] == "unchanged") &
                 (test["finding"] == "all")].set_index("verifier")["patient_pair_acc_mean"]
    auroc = test[(test["table"] == "metrics") & (test["dataset"] == "mimic") & (test["condition"] == "unchanged") &
                 (test["finding"] == "all")].set_index("verifier")["auroc_mean"]
    out_ = []
    for (src, form, d, q, basis), g in stmts.groupby(["source", "form", "definition", "sensitivity_target",
                                                      "threshold_basis"]):
        y = g.set_index("verifier")["contradicted_accept_mean"]
        for xname, x in (("patient_pairs", pairs), ("auroc", auroc)):
            n, rho, pv = _spearman_across(x, y)
            out_.append({"table": "statistic", "family": "E5", "test": "spearman", "x": xname,
                         "y": "contradicted_accept", "source": src, "form": form, "definition": d,
                         "sensitivity_target": q, "threshold_basis": basis, "n": n, "value": rho, "p_raw": pv,
                         "inference": "analytic_verifier_units"})
    df = pd.DataFrame(out_)
    if len(df):
        df["p_fdr"] = bh_fdr(df["p_raw"].to_numpy(float))
    return df


def report_reference(cfg: Dict, source: str) -> pd.DataFrame:
    from experiments.statements import reference_path
    ref = read_results_csv(reference_path(cfg, source), dtype={"candidate_id": str})
    patients = pd.read_parquet(input_path(cfg, f"e6full__{source}"), columns=["candidate_id", "subject_id"])
    patients["candidate_id"] = patients["candidate_id"].astype(str)
    return ref.merge(patients, on="candidate_id", validate="one_to_one").set_index("candidate_id")


def _report_metrics(cfg: Dict, err: pd.DataFrame, test: pd.DataFrame, green_full: pd.DataFrame) -> pd.DataFrame:
    from experiments.statements import reference_path
    rows = []
    for src in ("rexval", "radevalx"):
        p = reference_path(cfg, src)
        if not os.path.exists(p):
            continue
        ref = report_reference(cfg, src)
        ref.index = ref.index.astype(str)
        for vid, g in err[err["source"] == src].groupby("verifier"):
            j = g.assign(candidate_id=g["candidate_id"].astype(str)).set_index("candidate_id").join(ref, how="inner")
            inference = patient_kendall(j["error_score"], j["reference"], j["subject_id"], *stats_params(cfg))
            rows.append({"table": "statistic", "family": "E6", "test": "kendall_tau_b", "verifier": vid, "source": src,
                         "n": int(len(j)), "n_candidates_without_statements": int(len(ref) - len(j)),
                         **inference})
        gf = green_full[green_full["source"] == src] if len(green_full) else green_full
        if len(gf):
            j = gf.assign(candidate_id=gf["candidate_id"].astype(str)).set_index("candidate_id")
            j = j.drop(columns=["subject_id"], errors="ignore").join(ref, how="inner")
            inference = patient_kendall(j["prob_supported"], j["reference"], j["subject_id"], *stats_params(cfg))
            rows.append({"table": "statistic", "family": "E6", "test": "kendall_tau_b", "verifier": "GREEN_full_report",
                         "source": src, "n": int(len(j)), **inference})
    df = pd.DataFrame(rows)
    pairs = test[(test["table"] == "pairs") & (test["dataset"] == "mimic") & (test["condition"] == "unchanged") &
                 (test["finding"] == "all")].set_index("verifier")["patient_pair_acc_mean"]
    extra = []
    for src, g in (df.groupby("source") if len(df) else []):
        n, rho, pv = _spearman_across(pairs, g[g["verifier"] != "GREEN_full_report"].set_index("verifier")["value"])
        extra.append({"table": "statistic", "family": "E6", "test": "spearman_pairs_vs_tau", "verifier": "all",
                      "source": src, "n": n, "value": rho, "p_raw": pv, "inference": "analytic_verifier_units"})
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    if len(df):
        df["p_fdr"] = bh_fdr(df["p_raw"].to_numpy(float))
    return df


def _paired_row(diff: np.ndarray, a: str, b: str, ds: str, metric: str) -> Dict:
    point, boot = float(diff[0]), diff[1:][np.isfinite(diff[1:])]
    p = float(np.mean(np.abs(boot - point) >= abs(point))) if boot.size and np.isfinite(point) else np.nan
    return {"table": "paired", "family": "Qwen3-8B", "a": a, "b": b, "dataset": ds, "metric": metric,
            **summarize(diff, "difference"), "p_raw": p}


def paired_comparisons(cfg: Dict) -> List[Tuple[str, str]]:
    zs = cfg["llm"]["local_name"]
    lora = [r["id"] for r in cfg["runs"] + list(cfg.get("runs_added", [])) if r["kind"] == "lora"]
    return [(a, zs) for a in lora] + [(a, b) for i, a in enumerate(lora) for b in lora[i + 1:]]


def _paired(cfg: Dict) -> pd.DataFrame:
    from experiments.score import verifier
    n_boot, seed = stats_params(cfg)
    k = int(cfg["max_evidence_sentences"])
    rows = []
    for a, b in paired_comparisons(cfg):
        for ds in TEST_SETS:
            name = f"{ds}__unchanged"
            da, db = load_scores(cfg, a, name), load_scores(cfg, b, name)
            if not (da["verifier_row_id"].to_numpy() == db["verifier_row_id"].to_numpy()).all():
                raise ValueError(f"{a} and {b} scored different rows of {name}")
            w = cluster_weights(da["subject_id"], n_boot, seed)
            y = da["label_id"].to_numpy(float)
            sa, sb = da["prob_supported"].to_numpy(float), db["prob_supported"].to_numpy(float)
            pdf = pd.DataFrame({"row_type": da["row_type"], "claim_text": da["claim_text"], "qa_row_id": da["qa_row_id"],
                                "ev_key": _ev_key(da, k)})
            for metric, fa, fb in (("auroc", weighted_auroc(y, sa, w), weighted_auroc(y, sb, w)),
                                   ("patient_pair_acc", pair_accuracy(pair_groups(pdf, "case", sa), w),
                                    pair_accuracy(pair_groups(pdf, "case", sb), w))):
                rows.append(_paired_row(fa - fb, a, b, ds, metric))
        name = "stmt__rexerr__template"
        da, db = load_scores(cfg, a, name), load_scores(cfg, b, name)
        keep = da["label_agree"].notna().to_numpy()
        da, db = da[keep].reset_index(drop=True), db[keep].reset_index(drop=True)
        w = cluster_weights(da["subject_id"], n_boot, seed)
        con = (da["label_agree"] == 0).to_numpy()
        ta, tb = _thresholds(cfg, verifier(cfg, a))[0.90], _thresholds(cfg, verifier(cfg, b))[0.90]
        fa = accepted_share(da["prob_supported"].to_numpy(float)[con], w[:, con], np.full(w.shape[0], ta))
        fb = accepted_share(db["prob_supported"].to_numpy(float)[con], w[:, con], np.full(w.shape[0], tb))
        rows.append(_paired_row(fa - fb, a, b, "rexerr", "contradicted_accept_template_agree_q90"))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_fdr"] = bh_fdr(df["p_raw"].to_numpy(float))
    return df


def _controls(cfg: Dict, vid: str = "") -> List[Dict]:
    from experiments.controls import RETRIEVAL
    n_boot, seed = stats_params(cfg)
    s = cfg["similarity"]
    vid = vid or cfg["main_verifier"]
    rows = []
    for ds in TEST_SETS:
        sims = read_results_csv(out(cfg, "controls", f"unit_similarity_{ds}.csv"))
        df = load_scores(cfg, vid, f"{ds}__unchanged")
        w = cluster_weights(df["subject_id"], n_boot, seed)
        y, sc, qa = df["label_id"].to_numpy(float), df["prob_supported"].to_numpy(float), df["qa_row_id"].to_numpy()
        for n in s["ranking_ngram_orders"]:
            col = sims[f"sim_{n}gram"].to_numpy()
            for frac in s["partition_fractions"]:
                hi, lo = np.nanpercentile(col, 100 * (1 - frac)), np.nanpercentile(col, 100 * frac)
                for subset, keep_unit in (("most_similar", col >= hi), ("least_similar", col <= lo)):
                    keep = np.isin(qa, sims["qa_row_id"].to_numpy()[keep_unit])
                    row = {"table": "similarity_subsets", "verifier": vid, "dataset": ds, "ranking": f"{n}gram",
                           "subset": f"{subset}_{int(round(frac * 100))}pct", "n_units": int(keep_unit.sum()),
                           "n": int(keep.sum()), "subset_similarity_mean": float(np.nanmean(col[keep_unit]))}
                    if 0 < y[keep].sum() < keep.sum():
                        row.update(summarize_auroc(weighted_auroc(y[keep], sc[keep], w[:, keep]), whole_set=False))
                    rows.append(row)
    for cond in RETRIEVAL:
        df = load_scores(cfg, vid, f"mimic__{cond}")
        y, sc = df["label_id"].to_numpy(float), df["prob_supported"].to_numpy(float)
        own = df[df["claim_state"] == df["gold_state"]]
        row = {"table": "retrieval", "verifier": vid, "condition": cond, "n": len(df), "n_pos": int(y.sum()),
               "combinations_with_support": int(own["recall_hit"].sum()), "combinations": int(len(own)),
               "mean_score_supported": float(sc[y == 1].mean()) if (y == 1).any() else np.nan,
               "mean_score_not_supported": float(sc[y == 0].mean()) if (y == 0).any() else np.nan}
        w = cluster_weights(own["subject_id"], n_boot, seed)
        row.update(summarize((w * own["recall_hit"].to_numpy(float)).sum(axis=1) / w.sum(axis=1), "recall"))
        if y.sum() >= int(cfg["retrieval"]["min_positives_for_auroc"]) and y.sum() < len(y):
            wa = cluster_weights(df["subject_id"], n_boot, seed)
            row.update(summarize(weighted_auroc(y, sc, wa), "auroc"))
        rows.append(row)
    return rows


def metadata_rows(cfg: Dict) -> List[Dict]:
    import platform
    from importlib.metadata import PackageNotFoundError, version
    n_boot, seed = stats_params(cfg)
    rows = [{"key": "n_boot", "value": str(n_boot)}, {"key": "boot_seed", "value": str(seed)},
            {"key": "fdr_alpha", "value": str(cfg["stats"]["fdr_alpha"])},
            {"key": "python", "value": platform.python_version()}]
    for dist in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "transformers", "peft", "accelerate",
                 "huggingface-hub", "openai", "httpx"):
        try:
            rows.append({"key": dist, "value": version(dist)})
        except PackageNotFoundError:
            rows.append({"key": dist, "value": "not installed"})
    return rows


def _units(cfg: Dict) -> List[Dict]:
    from experiments.statements import _generators, statement_name
    units = []
    rf = report_findings_path(cfg)
    rf_digest = digest_file(rf) if os.path.exists(rf) else None
    src = cfg["sources"]
    demo = {n: digest_file(src[n]) if os.path.exists(src[n]) else None
            for n in ("mimic_patients", "mimic_admissions", "chexpert_plus_reports")}
    for vx in verifiers(cfg):
        for s in input_sets(cfg, vx):
            if s.startswith("e6full__"):
                continue
            p = score_path(cfg, vx["id"], s)
            units.append({"key": f"{vx['id']}__{s}", "vx": vx, "set": s,
                          "params": {"regime": _REGIME, "scores": digest_file(p) if os.path.exists(p) else None,
                                     "threshold_scores": digest_file(score_path(cfg, vx["id"], threshold_set(vx)))
                                     if os.path.exists(score_path(cfg, vx["id"], threshold_set(vx))) else None,
                                     "input": digest_file(input_path(cfg, s)) if os.path.exists(input_path(cfg, s)) else None,
                                     "n_boot": stats_params(cfg)[0], "boot_seed": stats_params(cfg)[1]}})
            ds, _, cond = s.partition("__")
            if cond in ("unchanged", "heldout", "heldout_written"):
                units[-1]["params"]["evidence_pair_weighting"] = EVIDENCE_PAIR_REGIME
                units[-1]["legacy"] = {"evidence_pair_weighting": "row-product-v0"}
            if cond == "unchanged" and ds in TEST_SETS:
                units[-1]["params"]["report_findings"] = rf_digest
                if ds in SUBGROUP_SETS:
                    units[-1]["params"]["subgroups"] = {"config": cfg["subgroups"], "sources": demo}
        if vx["domain"] == "radiology":
            for ds in TEST_SETS:
                if "report_replaced" not in cfg["test_sets"].get(ds, []):
                    continue
                ps = [score_path(cfg, vx["id"], n) for n in (f"{ds}__unchanged", f"{ds}__report_replaced",
                                                             threshold_set(vx))]
                units.append({"key": f"{vx['id']}__{ds}__report_change", "vx": vx, "set": f"{ds}__report_change",
                              "params": {"regime": _REGIME, "scores": [digest_file(p) if os.path.exists(p) else None
                                                                       for p in ps],
                                         "n_boot": stats_params(cfg)[0], "boot_seed": stats_params(cfg)[1]}})
            for form in ("template", "raw"):
                ps = [score_path(cfg, vx["id"], statement_name(f"gen__{g}", form)) for g in _generators(cfg)]
                units.append({"key": f"{vx['id']}__stmt__genpooled__{form}", "vx": vx, "set": f"stmt__genpooled__{form}",
                              "params": {"regime": _REGIME, "scores": [digest_file(p) if os.path.exists(p) else None
                                                                       for p in ps],
                                         "n_boot": stats_params(cfg)[0], "boot_seed": stats_params(cfg)[1]}})
    return units


def main_result_tables(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    out_dir = out(cfg, "result_tables")
    status = os.path.join(out_dir, "status.txt")
    units = _units(cfg)

    def compute(unit):
        vx, s = unit["vx"], unit["set"]
        if s.endswith("__report_change"):
            ds = s.split("__")[0]
            for name in (f"{ds}__unchanged", f"{ds}__report_replaced"):
                if not is_scored(cfg, vx["id"], name):
                    raise MissingInput(f"{vx['id']} has no scores on {name}")
            return report_change_rows(cfg, vx, ds)
        if s.startswith("stmt__genpooled__"):
            df = _pooled(cfg, vx, s.split("__")[2])
            return statement_rows(cfg, vx, s, df=df)
        if not is_scored(cfg, vx["id"], s):
            raise MissingInput(f"{vx['id']} has no scores on {s}")
        if s.startswith("stmt__rexval") or s.startswith("stmt__radevalx"):
            return report_error_rows(cfg, vx, s)
        if s.startswith("stmt__"):
            return statement_rows(cfg, vx, s)
        return test_rows(cfg, vx, s)

    rows, refused = run_units("main_result_tables", units, compute, out_dir, status)
    if refused:
        raise MissingInput(f"{refused} unit(s) of main_result_tables wait for scores; the final tables are written once all exist")
    df = pd.DataFrame(rows)
    test = df[df["table"].isin(["metrics", "pairs", "operating", "contradiction", "subgroup",
                                "report_change"])].dropna(axis=1, how="all")
    stmts = df[df["table"] == "statements"].dropna(axis=1, how="all")
    err = df[df["table"] == "report_error"].dropna(axis=1, how="all")
    write_table(pd.concat([test, _paired(cfg)], ignore_index=True), os.path.join(out_dir, "test_sets.csv"))
    write_table(stmts, os.path.join(out_dir, "statements.csv"))
    write_table(err, os.path.join(out_dir, "report_error.csv"))
    green_full = []
    for src in ("rexval", "radevalx"):
        if is_scored(cfg, "GREEN", f"e6full__{src}"):
            green_full.append(load_scores(cfg, "GREEN", f"e6full__{src}").assign(source=src))
    agree, kappas = _label_agreement(cfg)
    kap = pd.concat([pd.DataFrame(kappas), pd.DataFrame(_mesh_correlation(test))], ignore_index=True)
    if len(kap):
        kap["p_fdr"] = bh_fdr(kap["p_raw"].to_numpy(float))
    stats = pd.concat([_correlations(test, stmts),
                       _report_metrics(cfg, err, test, pd.concat(green_full) if green_full else pd.DataFrame()), kap],
                      ignore_index=True)
    write_table(stats, os.path.join(out_dir, "statistics.csv"))
    write_table(pd.DataFrame(_controls(cfg) + agree), os.path.join(out_dir, "controls.csv"))
    write_table(pd.DataFrame(metadata_rows(cfg)), os.path.join(out_dir, "metadata.csv"))
    flush_status(status, "[main_result_tables] done")
    return out_dir
