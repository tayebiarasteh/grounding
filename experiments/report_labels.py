"""
experiments/report_labels.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import os
from typing import Dict, List

import pandas as pd

from experiments._common import MissingInput, flush_status, get_cfg, read_results_csv, require, run_units, write_table
from experiments.study import (check_external_inputs, norm_text, out, openi_eligible, read_openi, read_padchest_gr,
                               read_rexgradient)

_REGIME = "report_labels-v1"


def report_studies(cfg: Dict) -> Dict[str, pd.DataFrame]:
    src = cfg["sources"]
    o = pd.read_parquet(require(src["oracle_mimic"], "the earlier data build"), columns=["split", "subject_id",
                                                                                         "report_rel_path"])
    o = o.drop_duplicates("report_rel_path")
    sec = pd.read_parquet(require(src["mimic_sections"], "the MIMIC-CXR report build"),
                          columns=["report_rel_path", "study_id", "findings_text", "impression_text"])
    m = o.merge(sec, on="report_rel_path", how="left", validate="one_to_one")
    mimic = pd.DataFrame({"study_key": m["study_id"].astype(int).astype(str), "subject_id": m["subject_id"].astype(str),
                          "split": m["split"].astype(str), "findings_text": m["findings_text"].map(norm_text),
                          "impression_text": m["impression_text"].map(norm_text)})
    o = pd.read_parquet(require(src["oracle_chexpert"], "the earlier data build"),
                        columns=["split", "subject_id", "report_rel_path", "case_context"])
    o = o[o["split"] == "test"]
    contexts = o.groupby("report_rel_path")["case_context"].agg(lambda s: set(s.map(norm_text)))
    sec = pd.read_parquet(require(src["chexpert_sections"], "the CheXpert Plus report build"),
                          columns=["report_rel_path", "findings_text", "impression_text"])
    sec = sec[sec["report_rel_path"].isin(contexts.index)].copy()
    sec["findings_text"] = sec["findings_text"].map(norm_text)
    sec = sec[[f in contexts[p] for p, f in zip(sec["report_rel_path"], sec["findings_text"])]]
    sec = sec.drop_duplicates("report_rel_path", keep="first")
    o = o.drop_duplicates("report_rel_path")[["split", "subject_id", "report_rel_path"]]
    m = o.merge(sec, on="report_rel_path", how="left", validate="one_to_one")
    if m["findings_text"].isna().any():
        raise ValueError(f"{int(m['findings_text'].isna().sum())} CheXpert Plus test studies have no report whose "
                         f"findings match the earlier build")
    chexpert = pd.DataFrame({"study_key": m["report_rel_path"].astype(str), "subject_id": m["subject_id"].astype(str),
                             "split": "test", "findings_text": m["findings_text"].map(norm_text),
                             "impression_text": m["impression_text"].map(norm_text)})
    rg = read_rexgradient(cfg)
    rexgradient = pd.DataFrame({"study_key": rg["study_id"], "subject_id": rg["subject_id"], "split": "test",
                                "findings_text": rg["findings_text"], "impression_text": rg["impression_text"]})
    oi = openi_eligible(read_openi(src["openi_reports_dir"]))
    openi = pd.DataFrame({"study_key": oi["report_id"], "subject_id": oi["report_id"], "split": "test",
                          "findings_text": oi["findings_text"], "impression_text": oi["impression_text"]})
    out_ = {"mimic": mimic, "chexpert": chexpert, "rexgradient": rexgradient, "openi": openi,
            "padchest_gr": read_padchest_gr(cfg)}
    for name, df in out_.items():
        empty = int(((df["findings_text"] == "") | (df["impression_text"] == "")).sum())
        if empty or df["study_key"].duplicated().any():
            raise ValueError(f"{name}: {empty} studies without both sections or duplicated study keys")
    return out_


def _digest(df: pd.DataFrame) -> str:
    h = hashlib.blake2b(digest_size=16)
    for k, sp, t in zip(df["study_key"], df["split"], df["impression_text"]):
        h.update(f"{k}\t{sp}\t{t}\n".encode("utf-8"))
    return h.hexdigest()


def labels_path(cfg: Dict, name: str) -> str:
    return out(cfg, "report_labels", f"{name}.csv")


def load_labels(cfg: Dict, name: str) -> pd.DataFrame:
    path = labels_path(cfg, name)
    if not os.path.exists(path):
        raise MissingInput(f"{path} is missing; run main_chexbert_labels first")
    return read_results_csv(path, dtype={"study_key": str, "subject_id": str}, keep_default_na=False)


def label_frame(cfg: Dict, df: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
    from Inference.chexbert_labeler import CONDITIONS, FINDING_HEAD, finding_state
    from experiments.chexbert import codes as chexbert_codes
    codes = chexbert_codes(cfg, texts)
    res = df.copy()
    for j, name in enumerate(CONDITIONS):
        res[f"chexbert_{name.replace(' ', '_')}"] = codes[:, j]
    for f in FINDING_HEAD:
        res[f"state__{f}"] = [finding_state(c, f, blank_is_absent=True) for c in codes]
    return res


def main_chexbert_labels(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    check_external_inputs(cfg, "main_chexbert_labels")
    v = cfg
    ch = v["chexbert"]
    out_dir = out(cfg, "report_labels")
    status = os.path.join(out_dir, "status.txt")
    studies = report_studies(cfg)
    units = [{"key": name, "params": {"regime": _REGIME, "chexbert_sha256": ch["sha256"], "n_studies": int(len(df)),
                                      "texts": _digest(df)}} for name, df in studies.items()]
    for u in units:
        release = [c for c in studies[u["key"]].columns if c.startswith("release__")]
        if release:
            u["params"]["release_states"] = hashlib.blake2b(
                studies[u["key"]][["study_key"] + release].to_csv(index=False).encode("utf-8"), digest_size=16).hexdigest()

    def compute(unit):
        df = studies[unit["key"]]
        res = label_frame(cfg, df, df["impression_text"].tolist())
        release = [c for c in df.columns if c.startswith("release__")]
        if release:
            for c in release:
                f = c[len("release__"):]
                res[f"chexbert_state__{f}"] = res[f"state__{f}"]
                res[f"state__{f}"] = res[c]
        return res.drop(columns=["findings_text", "impression_text"] + release).to_dict("records")

    def valid(unit, done):
        return len(done) == unit["params"]["n_studies"]

    rows, refused = run_units("report_labels", units, compute, out_dir, status, validate=valid)
    if refused:
        raise MissingInput(f"{refused} label set(s) could not be built yet")
    table = pd.DataFrame(rows)
    for name, df in studies.items():
        part = read_results_csv(os.path.join(out_dir, "partial", f"{name}.csv"),
                                dtype={"study_key": str, "subject_id": str}, keep_default_na=False)
        write_table(part, labels_path(cfg, name))
    flush_status(status, f"[report_labels] done, {len(table)} studies")
    return out_dir
