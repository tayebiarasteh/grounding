"""
experiments/dataset_inputs.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments._common import (MissingInput, digest_file, flush_status, get_cfg, read_results_csv, read_source_csv,
                                 require, run_units, write_table)
from experiments.study import norm_text, read_openi

_REGIME = "dataset_inputs-v1"
REXVAL_FILES = ("50_samples_gt_and_candidates.csv", "6_valid_raters_per_rater_error_categories.csv")
RADEVALX_FILES = ("RadEval_clinically_significant_errors.csv", "RadEval-clinically_insignificant_errors.csv",
                  "metrcis_scores_m2tr.csv")
CANDIDATES = ("radgraph", "bertscore", "s_emb", "bleu")
REXVAL_CATEGORIES = range(1, 7)
RADEVALX_CATEGORIES = range(1, 9)
RADEVALX_METRICS = ("bleu4", "bleu_2", "bertscore", "CheXbert", "radgraph_f1", "radcliq")


def masterlist_name(dataset: str) -> str:
    return {"rexval": "rexval_masterlist.csv", "radevalx": "radevalx_masterlist.csv"}[dataset]


def staging_path(cfg: Dict, dataset: str) -> str:
    folder = {"rexval": "ReXVal", "radevalx": "RadEvalX"}[dataset]
    return os.path.join(cfg["datasets_staging_dir"], "XRay", folder, masterlist_name(dataset))


def _raw(src_dir: str, names) -> List[str]:
    paths = [os.path.join(src_dir, n) for n in names]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise MissingInput(f"{', '.join(missing)} missing; download the release into {src_dir} first")
    return paths


def rexval_master(src_dir: str, mimic_master: str) -> pd.DataFrame:
    gt_path, rater_path = _raw(src_dir, REXVAL_FILES)
    gt = read_source_csv(gt_path).reset_index(drop=True)
    raters = read_source_csv(rater_path)
    need_gt = {"study_id", "gt_report", *CANDIDATES}
    need_r = {"study_number", "candidate_type", "error_category", "rater_index", "clinically_significant", "num_errors"}
    if not (need_gt <= set(gt.columns) and need_r <= set(raters.columns)):
        raise ValueError(f"ReXVal columns differ from the release: missing {sorted(need_gt - set(gt.columns))} and "
                         f"{sorted(need_r - set(raters.columns))}")
    if not set(raters["study_number"].astype(int)) <= set(range(len(gt))):
        raise ValueError(f"ReXVal study_number values fall outside the {len(gt)} rows of the ground-truth file")
    if set(raters["candidate_type"]) != set(CANDIDATES):
        raise ValueError(f"ReXVal candidate types {sorted(set(raters['candidate_type']))} differ from {CANDIDATES}")
    gt["study_number"] = np.arange(len(gt))
    ml = read_source_csv(require(mimic_master, "the MIMIC-CXR download"),
                         usecols=["subject_id", "study_id", "split", "report_rel_path"])
    ml = ml.drop_duplicates("study_id").rename(columns={"split": "mimic_split"})
    gt = gt.merge(ml, on="study_id", how="left", validate="one_to_one")
    if gt["subject_id"].isna().any():
        raise ValueError(f"{int(gt['subject_id'].isna().sum())} ReXVal studies are not in the MIMIC-CXR master list")
    rater_ids = sorted(raters["rater_index"].astype(int).unique())
    r = raters.assign(sig=raters["clinically_significant"].astype(str).str.upper() == "TRUE",
                      num_errors=pd.to_numeric(raters["num_errors"], errors="raise"))
    rows = []
    for g in gt.itertuples():
        for ctype in CANDIDATES:
            sub = r[(r["study_number"] == g.study_number) & (r["candidate_type"] == ctype)]
            row = {"study_number": int(g.study_number), "study_id": int(g.study_id), "subject_id": int(g.subject_id),
                   "mimic_split": g.mimic_split, "report_rel_path": g.report_rel_path, "candidate_type": ctype,
                   "gt_report": norm_text(g.gt_report), "candidate_report": norm_text(getattr(g, ctype)),
                   "n_raters": len(rater_ids)}
            for flag, name in ((True, "sig"), (False, "insig")):
                part = sub[sub["sig"] == flag]
                per_rater = part.groupby("rater_index")["num_errors"].sum().reindex(rater_ids, fill_value=0)
                row[f"{name}_errors_mean"] = float(per_rater.mean())
                for c in REXVAL_CATEGORIES:
                    cat = part[part["error_category"] == c].groupby("rater_index")["num_errors"].sum()
                    row[f"{name}_cat{c}_mean"] = float(cat.reindex(rater_ids, fill_value=0).mean())
                if flag:
                    for k in rater_ids:
                        row[f"sig_errors_rater{k}"] = int(per_rater[k])
            row["all_errors_mean"] = row["sig_errors_mean"] + row["insig_errors_mean"]
            rows.append(row)
    return pd.DataFrame(rows)


def radevalx_master(src_dir: str, openi_reports_dir: str) -> pd.DataFrame:
    sig_path, insig_path, metric_path = _raw(src_dir, RADEVALX_FILES)
    sig, insig, metrics = read_source_csv(sig_path), read_source_csv(insig_path), read_source_csv(metric_path)
    cats = [str(c) for c in RADEVALX_CATEGORIES]
    need = {"report_id", "ground_truth", "M2Tr-Generation", *cats}
    if not (need <= set(sig.columns) and need <= set(insig.columns)
            and {"report_id", *RADEVALX_METRICS} <= set(metrics.columns)):
        raise ValueError("RadEvalX columns differ from the release")
    if sig["report_id"].tolist() != insig["report_id"].tolist() or sig["report_id"].duplicated().any():
        raise ValueError("the RadEvalX significant and insignificant files list different or repeated reports")
    same_text = (sig["ground_truth"].map(norm_text) == insig["ground_truth"].map(norm_text)) & \
        (sig["M2Tr-Generation"].map(norm_text) == insig["M2Tr-Generation"].map(norm_text))
    if not same_text.all():
        raise ValueError("the RadEvalX significant and insignificant files carry different texts")
    oi = read_openi(openi_reports_dir).set_index("report_id")
    ids = sig["report_id"].astype(int).astype(str)
    missing = sorted(set(ids) - set(oi.index))
    if missing:
        raise ValueError(f"RadEvalX reports without an Open-I XML: {missing[:5]}")
    m = metrics.drop_duplicates("report_id").set_index("report_id")
    if not sig["report_id"].isin(m.index).all():
        raise ValueError("RadEvalX reports without metric scores")
    out_ = pd.DataFrame({"report_id": sig["report_id"].astype(int), "openi_report_id": ids,
                         "openi_xml_rel_path": [f"OpenI/NLMCXR_reports/ecgen-radiology/{i}.xml" for i in ids],
                         "openi_findings": [oi.at[i, "findings_text"] for i in ids],
                         "openi_impression": [oi.at[i, "impression_text"] for i in ids],
                         "openi_f1_image": [oi.at[i, "f1_image_id"] for i in ids],
                         "ground_truth": sig["ground_truth"].map(norm_text),
                         "candidate_report": sig["M2Tr-Generation"].map(norm_text)})
    for name, df in (("sig", sig), ("insig", insig)):
        counts = df[cats].apply(pd.to_numeric, errors="raise").fillna(0).astype(int)
        for c in cats:
            out_[f"{name}_cat{c}"] = counts[c].to_numpy()
        out_[f"{name}_errors"] = counts.sum(axis=1).to_numpy()
    out_["all_errors"] = out_["sig_errors"] + out_["insig_errors"]
    for k in RADEVALX_METRICS:
        out_[k] = m.loc[sig["report_id"], k].to_numpy()
    return out_


def main_preprocess_datasets(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    src = cfg["sources"]
    stage_root = cfg["datasets_staging_dir"]
    status = os.path.join(stage_root, "status.txt")
    inputs = {"rexval": [os.path.join(src["rexval_dir"], n) for n in REXVAL_FILES] + [src["mimic_master_list"]],
              "radevalx": [os.path.join(src["radevalx_dir"], n) for n in RADEVALX_FILES]}
    units = [{"key": k, "params": {"regime": _REGIME,
                                   "inputs": {os.path.basename(p): digest_file(p) if os.path.exists(p) else None
                                              for p in paths}}}
             for k, paths in inputs.items()]

    def compute(unit):
        if unit["key"] == "rexval":
            df = rexval_master(src["rexval_dir"], src["mimic_master_list"])
        else:
            df = radevalx_master(src["radevalx_dir"], src["openi_reports_dir"])
        write_table(df, staging_path(cfg, unit["key"]))
        return [{"dataset": unit["key"], "rows": int(len(df)), "path": staging_path(cfg, unit["key"])}]

    def valid(unit, done):
        p = staging_path(cfg, unit["key"])
        return os.path.exists(p) and len(read_results_csv(p, usecols=[0])) == int(done["rows"].iloc[0])

    _, refused = run_units("main_preprocess_datasets", units, compute, stage_root, status, validate=valid)
    if refused:
        raise MissingInput(f"{refused} dataset(s) could not be preprocessed yet")
    flush_status(status, "[main_preprocess_datasets] done")
    return stage_root
