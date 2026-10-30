"""
experiments/study.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments._common import MissingInput, atomic_write, read_source_csv, require

CORE_OF = {"vascular congestion": "pulmonary edema", "interstitial edema": "pulmonary edema",
           "rib fracture": "fracture"}
STATES = ("present", "absent")
TEST_SETS = ("mimic", "chexpert", "rexgradient", "openi", "padchest_gr")
LABEL_VARIANTS = {"openi_mesh": "openi"}
SUBGROUP_SETS = ("mimic", "chexpert", "padchest_gr")
ADDED_FOLDER = "evidence_sets"
ADDED_CONDITIONS = ("heldout_written",)
ADDED_TRAIN_SETS = ("main_uncapped", "main_written", "noD_written")
MESH_HEADINGS = {"Pulmonary Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly",
                 "Consolidation": "consolidation", "Fractures, Bone": "fracture", "Pleural Effusion": "pleural effusion",
                 "Pneumothorax": "pneumothorax", "Pulmonary Edema": "pulmonary edema"}
MESH_NOT_CODED = "No Indexing"
EXAMPLE_COLUMNS = ["verifier_row_id", "qa_row_id", "split", "subject_id", "report_rel_path", "concept", "gold_state",
                   "claim_state", "claim_text", "case_context", "evidence_sentence_ids_json", "evidence_sentences_json",
                   "evidence_article_titles_json", "row_type", "evidence_source_type", "label", "label_id", "dataset",
                   "condition"]


def external_inputs(cfg: Dict) -> Dict[str, List[str]]:
    src = cfg["sources"]
    need = {src["oracle_mimic"]: ["main_chexbert_labels", "main_support_pools", "main_build_examples"], src["oracle_chexpert"]: ["main_chexbert_labels", "main_build_examples"],
            src["mimic_sections"]: ["main_chexbert_labels", "main_statements"], src["chexpert_sections"]: ["main_chexbert_labels"],
            src["rexgradient_masterlist"]: ["main_chexbert_labels"], src["rexgradient_sample"]: ["main_chexbert_labels"],
            src["openi_reports_dir"]: ["main_chexbert_labels", "main_generate_reports", "main_statements", "main_result_tables"], src["openi_images_dir"]: ["main_generate_reports"],
            src["snapshot"]: ["main_judges", "main_support_pools", "main_build_examples"], src["food_trainvaltest"]: ["main_build_examples"],
            src["rexval_masterlist"]: ["main_statements"], src["radevalx_masterlist"]: ["main_statements"],
            src["mimic_master_list"]: ["main_result_tables"], src["chexpert_master_list"]: ["main_result_tables"],
            src["mimic_patients"]: ["main_result_tables"], src["mimic_admissions"]: ["main_result_tables"], src["chexpert_plus_reports"]: ["main_result_tables"],
            src["padchest_gr_reports"]: ["main_chexbert_labels"], src["padchest_gr_table"]: ["main_chexbert_labels", "main_result_tables"]}
    for split in ("train", "val", "test"):
        need[os.path.join(src["rexerr_dir"], f"ReXErr-sentence-level_{split}.csv")] = ["main_statements"]
    for g in cfg["generators"]["models"]:
        if g.get("repo_path"):
            need.setdefault(g["repo_path"], []).append(f"main_generate_reports {g['name']}")
    return need


def check_external_inputs(cfg: Dict, stage: str) -> None:
    missing = [f"{p} (read by {', '.join(st)})" for p, st in external_inputs(cfg).items() if not os.path.exists(p)]
    token = str(cfg.get("hf_token") or "")
    if not token or token.startswith("REPLACE") or set(token) <= {"x"}:
        missing.append("hf_token in config.yaml (MAIRA-2 in main_generate_reports is gated)")
    if missing:
        raise RuntimeError(f"[{stage}] inputs missing on this machine:\n  " + "\n  ".join(missing))


def out(cfg: Dict, *parts: str) -> str:
    return os.path.join(cfg["results_dir"], *parts)


def read_evidence(js: str, k: int) -> List[str]:
    return [str(s).strip() for s in json.loads(js) if str(s).strip()][:k]


def norm_text(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", str(s).replace("\r", " ").replace("\n", " ")).strip()


def opposite(state: str) -> str:
    return "absent" if state == "present" else "present"


def claim_text(finding: str, state: str) -> str:
    return f"{finding} is {state}"


def core_findings(matched_concepts_json: str) -> List[str]:
    try:
        cs = json.loads(matched_concepts_json)
    except (TypeError, ValueError):
        return []
    return sorted({CORE_OF.get(c, c) for c in cs})


def write_parquet(df: pd.DataFrame, path: str) -> None:
    atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))


def read_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def examples_path(cfg: Dict, dataset: str, condition: str) -> str:
    folder = ADDED_FOLDER if condition in ADDED_CONDITIONS else "examples"
    return out(cfg, folder, f"{dataset}__{condition}.parquet")


def train_set_path(cfg: Dict, name: str) -> str:
    return out(cfg, ADDED_FOLDER if name in ADDED_TRAIN_SETS else "examples", f"train__{name}.parquet")


def test_sets(cfg: Dict, domain: str, verifier_id: str = "") -> List[str]:
    v = cfg
    sets = []
    for ds, conds in v["test_sets"].items():
        if (ds == "food") != (domain == "food"):
            continue
        for c in conds:
            if c in v["restricted_conditions"] and verifier_id not in v["restricted_conditions"][c]:
                continue
            sets.append(f"{ds}__{c}")
    return sets


def read_openi(reports_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(reports_dir, "*.xml"))):
        root = ET.parse(f).getroot()
        txt = {a.get("Label"): norm_text(a.text) for a in root.iter("AbstractText")}
        images = []
        for p in root.iter("parentImage"):
            fig = p.find("figureId")
            images.append((fig.text.strip() if fig is not None and fig.text else "", p.get("id")))
        rows.append({"report_id": os.path.splitext(os.path.basename(f))[0], "findings_text": txt.get("FINDINGS", ""),
                     "impression_text": txt.get("IMPRESSION", ""),
                     "mesh_major_json": json.dumps([norm_text(m.text) for m in root.iter("major")]),
                     "image_ids_json": json.dumps([i for _, i in images]),
                     "f1_image_id": next((i for fig, i in images if fig == "F1"), images[0][1] if images else "")})
    if not rows:
        raise MissingInput(f"no Open-I report XML under {reports_dir}")
    return pd.DataFrame(rows)


def mesh_findings(mesh_major_json: str):
    heads = [t.split("/")[0].strip() for t in json.loads(mesh_major_json)]
    if MESH_NOT_CODED in heads:
        return None
    return {MESH_HEADINGS[h] for h in heads if h in MESH_HEADINGS}


_DEMOGRAPHICS = {}


def _sex_group(value) -> str:
    return {"F": "female", "FEMALE": "female", "M": "male", "MALE": "male"}.get(str(value).strip().upper(), "")


def _age_group(value, bins: List[float], labels: List[str]) -> str:
    try:
        a = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(a):
        return ""
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        if lo <= a < hi:
            return str(lab)
    return ""


def _race_group(value, patterns: Dict[str, List[str]]) -> str:
    s = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip().lower()
    for group, pats in patterns.items():
        if any(p.lower() in s for p in pats):
            return group
    return "Other"


def demographics(cfg: Dict, ds: str) -> pd.DataFrame:
    sg, src = cfg["subgroups"], cfg["sources"]
    key = (ds, json.dumps(sg, sort_keys=True), src.get("mimic_patients"), src.get("chexpert_plus_reports"),
           src.get("padchest_gr_table"))
    if key in _DEMOGRAPHICS:
        return _DEMOGRAPHICS[key]
    bins, labels, pats = list(sg["age_bins"]), list(sg["age_labels"]), sg["race_patterns"]
    if ds == "mimic":
        pat = read_source_csv(require(src["mimic_patients"], "the MIMIC-IV hosp module"),
                              usecols=["subject_id", "gender", "anchor_age"])
        adm = read_source_csv(require(src["mimic_admissions"], "the MIMIC-IV hosp module"), usecols=["subject_id", "race"])
        adm = adm[adm["race"].notna() & (adm["race"].astype(str).str.strip() != "")]
        modal = adm.groupby("subject_id")["race"].agg(lambda x: x.value_counts().sort_index().idxmax())
        d = pat.drop_duplicates("subject_id").set_index("subject_id").join(modal, how="left")
        out_ = pd.DataFrame({"sex": d["gender"].map(_sex_group).to_numpy(), "age_group": d["anchor_age"].map(lambda a: _age_group(a, bins, labels)).to_numpy(),
                             "race": d["race"].map(lambda r: _race_group(r, pats)).to_numpy()}, index=d.index.astype(str))
    elif ds == "chexpert":
        c = read_source_csv(require(src["chexpert_plus_reports"], "the CheXpert Plus release"),
                            usecols=["path_to_image", "sex", "age", "race"])
        c["key"] = c["path_to_image"].str.split("/").str[:3].str.join("/")
        c = c.drop_duplicates("key").set_index("key")
        out_ = pd.DataFrame({"sex": c["sex"].map(_sex_group).to_numpy(), "age_group": c["age"].map(lambda a: _age_group(a, bins, labels)).to_numpy(),
                             "race": c["race"].map(lambda r: _race_group(r, pats)).to_numpy()}, index=c.index)
    elif ds == "padchest_gr":
        t = read_source_csv(require(src["padchest_gr_table"], "the PadChest-GR download"),
                            usecols=["StudyID", "PatientSex_DICOM", "PatientAge"], dtype={"StudyID": str})
        t = t.drop_duplicates("StudyID").set_index("StudyID")
        out_ = pd.DataFrame({"sex": t["PatientSex_DICOM"].map(_sex_group).to_numpy(),
                             "age_group": t["PatientAge"].map(lambda a: _age_group(a, bins, labels)).to_numpy(),
                             "race": ""}, index=t.index)
    else:
        raise ValueError(f"no demographics for {ds}")
    _DEMOGRAPHICS[key] = out_
    return out_


def patient_groups(cfg: Dict, ds: str, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    d = demographics(cfg, ds)
    key = (df["subject_id"] if ds == "mimic" else df["report_rel_path"]).astype(str)
    j = d.reindex(key.to_numpy())
    return {a: j[a].fillna("").to_numpy(dtype=object) for a in ("sex", "age_group", "race")}


def openi_eligible(df: pd.DataFrame) -> pd.DataFrame:
    keep = (df["findings_text"] != "") & (df["impression_text"] != "") & (df["f1_image_id"] != "")
    return df[keep].reset_index(drop=True)


PADCHEST_GR_LABELS = {
    "atelectasis": {"atelectasis", "laminar atelectasis", "lobar atelectasis", "segmental atelectasis",
                    "atelectasis basal", "total atelectasis"},
    "cardiomegaly": {"cardiomegaly"},
    "consolidation": {"consolidation"},
    "fracture": {"callus rib fracture", "rib fracture", "humeral fracture", "clavicle fracture", "vertebral fracture",
                 "fracture"},
    "pleural effusion": {"pleural effusion", "loculated pleural effusion", "loculated fissural effusion"},
    "pneumothorax": {"pneumothorax", "hydropneumothorax"},
    "pulmonary edema": {"hilar congestion"},
}


def read_padchest_gr(cfg: Dict) -> pd.DataFrame:
    src = cfg["sources"]
    with open(require(src["padchest_gr_reports"], "the PadChest-GR download"), encoding="utf-8") as fh:
        reports = json.load(fh)
    table = read_source_csv(require(src["padchest_gr_table"], "the PadChest-GR download"),
                            usecols=["StudyID", "PatientID"], dtype={"StudyID": str, "PatientID": str})
    patient = table.drop_duplicates("StudyID").set_index("StudyID")["PatientID"]
    rows = []
    for r in reports:
        sid = str(r["StudyID"])
        text = norm_text(" ".join(str(f.get("sentence_en") or "").strip() for f in r["findings"]))
        labels = {lab for f in r["findings"] if f.get("abnormal") for lab in f.get("labels", [])}
        row = {"study_key": sid, "subject_id": str(patient[sid]), "split": "test", "findings_text": text,
               "impression_text": text}
        for finding in cfg["findings"]:
            row[f"release__{finding}"] = "present" if labels & PADCHEST_GR_LABELS[finding] else "absent"
        rows.append(row)
    return pd.DataFrame(rows)


def read_rexgradient(cfg: Dict) -> pd.DataFrame:
    src = cfg["sources"]
    sample = read_source_csv(require(src["rexgradient_sample"], "the fixed sample of ReXGradient-160K test patients"))
    ml = read_source_csv(require(src["rexgradient_masterlist"], "the ReXGradient-160K download"),
                         usecols=["PatientID", "StudyKey", "split", "Findings", "Impression"])
    ml = ml[ml["StudyKey"].isin(set(sample["StudyKey"]))].drop_duplicates("StudyKey")
    df = sample.merge(ml, on=["PatientID", "StudyKey"], how="left", validate="one_to_one")
    if df["Findings"].isna().any() or (df["split"] != "test").any():
        raise ValueError("a sampled ReXGradient-160K study is missing from the master list or is not a test study")
    return pd.DataFrame({"subject_id": df["PatientID"].astype(str), "study_id": df["StudyKey"].astype(str),
                         "findings_text": df["Findings"].map(norm_text),
                         "impression_text": df["Impression"].map(norm_text)})
