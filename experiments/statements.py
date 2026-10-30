"""
experiments/statements.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from data_loader.concept_lexicon import report_findings
from experiments._common import (MissingInput, digest_file, flush_status, get_cfg, read_results_csv, read_source_csv,
                                 require, run_units, write_table)
from experiments.study import STATES, claim_text, examples_path, out, read_evidence, write_parquet

_REGIME = "statements-v2"
SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
_SECTION_LABEL = re.compile(r"^\s*(findings|impressions?)\s*:\s*", re.I)
_ENUMERATOR = re.compile(r"^\s*\(?\d{1,2}[.)]\s+")


def clean_sentence(s) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    for _ in range(2):
        s = _ENUMERATOR.sub("", _SECTION_LABEL.sub("", s.strip()))
    return re.sub(r"\s+", " ", s).strip()


def _generators(cfg: Dict) -> List[str]:
    return [g["name"] for g in cfg["generators"]["models"]]


def sources(cfg: Dict) -> List[str]:
    return ["rexerr"] + [f"gen__{g}" for g in _generators(cfg)] + ["rexval", "radevalx"]


def forms(source: str) -> List[str]:
    return ["template"] if source in ("rexval", "radevalx") else ["template", "raw"]


def statement_name(source: str, form: str, citation: str = "") -> str:
    return f"stmt__{source}{'_' + citation if citation else ''}__{form}"


def statement_set_names(cfg: Dict) -> List[str]:
    return [statement_name(s, f) for s in sources(cfg) for f in forms(s)]


def statement_path(cfg: Dict, name: str) -> str:
    return out(cfg, "statements", f"{name}.parquet")


def reference_path(cfg: Dict, source: str) -> str:
    return out(cfg, "statements", f"{source}_reference.csv")


def support_evidence(cfg: Dict) -> Dict:
    k = int(cfg["max_evidence_sentences"])
    path = require(examples_path(cfg, "mimic", "unchanged"), "main_build_examples")
    c = pd.read_parquet(path)
    c = c[c["row_type"] == "C"].copy()
    c["key"] = c["evidence_sentences_json"].map(lambda js: json.dumps(read_evidence(js, k)))
    outd = {}
    for (f, s), g in c.groupby(["concept", "claim_state"]):
        counts = g.groupby("key").size().reset_index(name="n").sort_values(["n", "key"], ascending=[False, True])
        row = g[g["key"] == counts["key"].iloc[0]].iloc[0]
        outd[(f, s)] = {col: row[col] for col in ("evidence_sentence_ids_json", "evidence_sentences_json",
                                                  "evidence_article_titles_json")}
    return outd


def _sentence_rows(texts: pd.Series, keys: pd.Series) -> pd.DataFrame:
    rows = []
    for key, text in zip(keys, texts):
        for raw in SENT_SPLIT.split(str(text).strip()):
            sent = clean_sentence(raw)
            names = report_findings(sent)
            if sent and len(names) == 1:
                rows.append({"key": key, "sentence": sent, "finding": names[0]})
    return pd.DataFrame(rows, columns=["key", "sentence", "finding"])


def _sentence_states(cfg: Dict, df: pd.DataFrame) -> pd.DataFrame:
    from Inference.chexbert_labeler import finding_state
    from experiments.chexbert import codes
    uniq = sorted(df["sentence"].unique())
    code = dict(zip(uniq, codes(cfg, uniq))) if uniq else {}
    df = df.copy()
    df["state"] = [finding_state(code[s], f, blank_is_absent=False) for s, f in zip(df["sentence"], df["finding"])]
    return df[df["state"].isin(STATES)].reset_index(drop=True)


def _impression_states(cfg: Dict, keys: List[str], impressions: List[str]) -> Dict[str, np.ndarray]:
    from experiments.chexbert import codes
    return dict(zip(keys, codes(cfg, impressions))) if keys else {}


def _with_evidence(df: pd.DataFrame, evidence: Dict) -> pd.DataFrame:
    keep = [(f, s) in evidence for f, s in zip(df["finding"], df["state"])]
    df = df[keep].copy()
    for col in ("evidence_sentence_ids_json", "evidence_sentences_json", "evidence_article_titles_json"):
        df[col] = [evidence[(f, s)][col] for f, s in zip(df["finding"], df["state"])]
    return df


def _assemble(df: pd.DataFrame, source: str, form: str, dedup: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["claim_text"] = [claim_text(f, s) for f, s in zip(df["finding"], df["state"])] if form == "template" \
        else df["sentence"]
    df = df.sort_values(dedup + ["claim_text"]).drop_duplicates(dedup).reset_index(drop=True)
    df["verifier_row_id"] = np.arange(len(df))
    df["qa_row_id"] = pd.factorize(df["report_rel_path"])[0]
    df["concept"], df["claim_state"] = df["finding"], df["state"]
    df["gold_state"] = df.get("report_state", "")
    df["split"], df["row_type"], df["evidence_source_type"] = "test", "S", "statement"
    df["label_id"] = df["label_kappa"].fillna(-1).astype(int)
    df["label"] = np.where(df["label_id"] == 1, "supported", "not_supported")
    df["dataset"], df["condition"], df["source"], df["form"] = source, "statement", source, form
    return df


def _rexerr(cfg: Dict, evidence: Dict) -> Dict[str, pd.DataFrame]:
    v = cfg
    src = v["sources"]
    test = pd.read_parquet(require(examples_path(cfg, "mimic", "unchanged"), "main_build_examples"), columns=["subject_id"])
    subjects = set(test["subject_id"].astype(str))
    frames = []
    for split in ("train", "val", "test"):
        d = read_source_csv(require(os.path.join(src["rexerr_dir"], f"ReXErr-sentence-level_{split}.csv"),
                                    "the ReXErr-v1 download"))
        d["split_file"], d["row_index"] = split, np.arange(len(d))
        frames.append(d)
    r = pd.concat(frames, ignore_index=True)
    r["subject_id"] = pd.to_numeric(r["subject_id"], errors="coerce").astype("Int64").astype(str)
    r = r[r["subject_id"].isin(subjects)].copy()
    orig = r["original_sentence"].fillna("").astype(str).str.strip()
    err = r["error_sentence"].fillna("").astype(str).str.strip()
    ep = pd.to_numeric(r["error_present"], errors="coerce")
    unchanged = (r["error_type"] == "Not applicable") & (ep == 0) & (orig == err) & (err != "")
    error = r["error_type"].isin(v["rexerr_error_types"]) & (ep == 1) & (err != "")
    r["kind"] = np.where(unchanged, "unchanged", np.where(error, "error", ""))
    r = r[r["kind"] != ""].copy()
    r["sentence"] = r["error_sentence"].map(clean_sentence)
    names = r["sentence"].map(report_findings)
    r = r[names.map(len) == 1].copy()
    r["finding"] = names[r.index].str[0]
    sec = pd.read_parquet(require(src["mimic_sections"], "the MIMIC-CXR report build"),
                          columns=["subject_id", "study_id", "findings_text", "impression_text",
                                   "is_main_benchmark_eligible"])
    sec = sec[sec["is_main_benchmark_eligible"].astype(bool)]
    sec = sec.assign(study_key=sec["study_id"].astype(int).astype(str), sub=sec["subject_id"].astype(int).astype(str))
    r["study_key"] = pd.to_numeric(r["study_id"], errors="coerce").astype("Int64").astype(str)
    r = r.merge(sec[["study_key", "sub", "findings_text", "impression_text"]], on="study_key", how="inner",
                validate="many_to_one")
    if not (r["sub"] == r["subject_id"]).all():
        raise ValueError("ReXErr and the report build disagree on the patient of a study")
    r = _sentence_states(cfg, r)
    studies = r.drop_duplicates("study_key")
    imp = _impression_states(cfg, studies["study_key"].tolist(), studies["impression_text"].tolist())
    from Inference.chexbert_labeler import finding_state
    r["report_state"] = [finding_state(imp[k], f, blank_is_absent=True) for k, f in zip(r["study_key"], r["finding"])]
    r = _with_evidence(r[r["report_state"].isin(STATES)], evidence)
    r["label_kappa"] = (r["state"] == r["report_state"]).astype(float)
    r["label_flag"] = (r["kind"] == "unchanged").astype(float)
    r["label_agree"] = np.where(r["label_kappa"] == r["label_flag"], r["label_flag"], np.nan)
    r = r.rename(columns={"study_key": "report_rel_path", "findings_text": "case_context"})
    keep = ["subject_id", "report_rel_path", "case_context", "finding", "state", "report_state", "sentence", "kind",
            "error_type", "split_file", "row_index", "label_kappa", "label_flag", "label_agree",
            "evidence_sentence_ids_json", "evidence_sentences_json", "evidence_article_titles_json"]
    r = r[keep].sort_values(["report_rel_path", "split_file", "row_index"])
    return {"template": _assemble(r, "rexerr", "template", ["report_rel_path", "finding", "state", "kind"]),
            "raw": _assemble(r, "rexerr", "raw", ["report_rel_path", "sentence", "kind"])}


def _generator(cfg: Dict, name: str, evidence: Dict) -> Dict[str, pd.DataFrame]:
    from experiments.study import openi_eligible, read_openi
    from experiments.report_labels import load_labels
    from experiments.generate_reports import load_generations
    gen = load_generations(cfg, name)
    oi = openi_eligible(read_openi(cfg["sources"]["openi_reports_dir"])).set_index("report_id")
    labels = load_labels(cfg, "openi").set_index("study_key")
    s = _sentence_states(cfg, _sentence_rows(gen["text"], gen["report_id"]))
    s["report_rel_path"] = s["key"].astype(str)
    s["case_context"] = s["report_rel_path"].map(oi["findings_text"])
    s["subject_id"] = s["report_rel_path"]
    s["report_state"] = [labels.at[k, f"state__{f}"] for k, f in zip(s["report_rel_path"], s["finding"])]
    s = _with_evidence(s[s["report_state"].isin(STATES)], evidence)
    s["kind"], s["error_type"] = "generated", ""
    s["label_kappa"] = (s["state"] == s["report_state"]).astype(float)
    s["label_flag"], s["label_agree"] = np.nan, np.nan
    s["generator"] = name
    return {"template": _assemble(s, f"gen__{name}", "template", ["report_rel_path", "finding", "state", "kind"]),
            "raw": _assemble(s, f"gen__{name}", "raw", ["report_rel_path", "sentence", "kind"])}


def _masterlist(cfg: Dict, source: str) -> pd.DataFrame:
    path = cfg["sources"][f"{source}_masterlist"]
    if not os.path.exists(path):
        raise MissingInput(f"{path} is missing; build it with main_preprocess_datasets, then set sources.{source}_masterlist to the generated staging file")
    return read_results_csv(path, dtype={"subject_id": str, "study_id": str, "openi_report_id": str},
                            keep_default_na=False)


def _rexval(cfg: Dict, evidence: Dict) -> Dict[str, pd.DataFrame]:
    from experiments.report_labels import load_labels
    m = _masterlist(cfg, "rexval")
    lab = load_labels(cfg, "mimic")
    train_val = set(lab.loc[lab["split"].isin(["train", "val"]), "subject_id"].astype(str))
    excluded = m["subject_id"].isin(train_val)
    m = m[~excluded].copy()
    m["candidate_id"] = m["study_number"].astype(str) + "|" + m["candidate_type"]
    m["report_rel_path"], m["case_context"] = m["study_id"].astype(str), m["gt_report"].astype(str)
    s = _sentence_states(cfg, _sentence_rows(m["candidate_report"], m["candidate_id"]))
    s = s.merge(m[["candidate_id", "study_number", "candidate_type", "subject_id", "report_rel_path", "case_context"]]
                .rename(columns={"candidate_id": "key"}), on="key", how="left").rename(columns={"key": "candidate_id"})
    s = _with_evidence(s, evidence)
    s["kind"], s["error_type"], s["report_state"] = "candidate", "", ""
    s["label_kappa"] = s["label_flag"] = s["label_agree"] = np.nan
    ref = pd.DataFrame({"candidate_id": m["candidate_id"], "reference": m["sig_errors_mean"].astype(float)})
    full = pd.DataFrame({"candidate_id": m["candidate_id"], "subject_id": m["subject_id"],
                         "report_rel_path": m["report_rel_path"], "case_context": m["case_context"],
                         "claim_text": m["candidate_report"].astype(str)})
    return {"template": _assemble(s, "rexval", "template", ["candidate_id", "finding", "state"]),
            "reference": ref.sort_values("candidate_id").reset_index(drop=True), "full": _full_reports(full)}


def _radevalx(cfg: Dict, evidence: Dict) -> Dict[str, pd.DataFrame]:
    m = _masterlist(cfg, "radevalx")
    m["candidate_id"] = m["report_rel_path"] = m["subject_id"] = m["openi_report_id"].astype(str)
    m["case_context"] = m["openi_findings"].astype(str)
    s = _sentence_states(cfg, _sentence_rows(m["candidate_report"], m["candidate_id"]))
    keep = m[["candidate_id", "subject_id", "report_rel_path", "case_context"]].rename(columns={"candidate_id": "key"})
    s = s.merge(keep, on="key", how="left").rename(columns={"key": "candidate_id"})
    s = _with_evidence(s, evidence)
    s["kind"], s["error_type"], s["report_state"] = "candidate", "", ""
    s["label_kappa"] = s["label_flag"] = s["label_agree"] = np.nan
    ref = pd.DataFrame({"candidate_id": m["candidate_id"], "reference": m["sig_errors"].astype(float)})
    full = pd.DataFrame({"candidate_id": m["candidate_id"], "subject_id": m["subject_id"],
                         "report_rel_path": m["report_rel_path"], "case_context": m["case_context"],
                         "claim_text": m["candidate_report"].astype(str)})
    return {"template": _assemble(s, "radevalx", "template", ["candidate_id", "finding", "state"]),
            "reference": ref.sort_values("candidate_id").reset_index(drop=True), "full": _full_reports(full)}


def _full_reports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("candidate_id").reset_index(drop=True)
    return df.assign(verifier_row_id=np.arange(len(df)), qa_row_id=np.arange(len(df)), split="test", concept="",
                     gold_state="", claim_state="", evidence_sentence_ids_json="[]", evidence_sentences_json="[]",
                     evidence_article_titles_json="[]", row_type="F", evidence_source_type="none", label="",
                     label_id=-1, dataset="e6full", condition="full_report")


def full_path(cfg: Dict, source: str) -> str:
    return out(cfg, "statements", f"e6full__{source}.parquet")


def _inputs(cfg: Dict, source: str) -> Dict:
    from experiments.report_labels import labels_path

    def dig(p):
        return digest_file(p) if os.path.exists(p) else None
    rec = {"examples": dig(examples_path(cfg, "mimic", "unchanged"))}
    if source.startswith("gen__"):
        from experiments.generate_reports import generations_path, is_generated
        name = source[len("gen__"):]
        rec["generations"] = dig(generations_path(cfg, name)) if is_generated(cfg, name) else None
        rec["openi_labels"] = dig(labels_path(cfg, "openi"))
    if source == "rexval":
        rec["mimic_labels"] = dig(labels_path(cfg, "mimic"))
    if source in ("rexval", "radevalx"):
        rec["masterlist"] = dig(cfg["sources"][f"{source}_masterlist"])
    return rec


def main_statements(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    v = cfg
    out_dir = out(cfg, "statements")
    status = os.path.join(out_dir, "status.txt")
    units = [{"key": s, "params": {"regime": _REGIME, "inputs": _inputs(cfg, s), "chexbert": v["chexbert"]["sha256"],
                                   "error_types": list(v["rexerr_error_types"]),
                                   "max_evidence_sentences": int(v["max_evidence_sentences"])}} for s in sources(cfg)]
    holder = {}

    def compute(unit):
        src = unit["key"]
        if unit["params"]["inputs"]["examples"] is None:
            raise MissingInput("the MIMIC-CXR test examples are missing; run main_build_examples first")
        if "evidence" not in holder:
            holder["evidence"] = support_evidence(cfg)
        ev = holder["evidence"]
        if src.startswith("gen__"):
            try:
                built = _generator(cfg, src[len("gen__"):], ev)
            except MissingInput:
                for form in forms(src):
                    p = statement_path(cfg, statement_name(src, form))
                    if os.path.exists(p):
                        os.remove(p)
                raise
        else:
            built = (_rexerr(cfg, ev) if src == "rexerr" else _rexval(cfg, ev) if src == "rexval"
                     else _radevalx(cfg, ev))
        rows = []
        for form in forms(src):
            df = built[form]
            write_parquet(df, statement_path(cfg, statement_name(src, form)))
            for (k, st), g in df.groupby(["kind", "claim_state"]):
                rows.append({"source": src, "form": form, "kind": k, "state": st, "n": int(len(g)),
                             "n_reports": int(g["report_rel_path"].nunique()),
                             "label_kappa_1": int((g["label_kappa"] == 1).sum()),
                             "label_agree_defined": int(g["label_agree"].notna().sum())})
        if "reference" in built:
            write_table(built["reference"], reference_path(cfg, src))
            write_parquet(built["full"], full_path(cfg, src))
        return rows

    def valid(unit, done):
        import pyarrow.parquet as pq
        if unit["key"] in ("rexval", "radevalx") and not (os.path.exists(full_path(cfg, unit["key"])) and
                                                          os.path.exists(reference_path(cfg, unit["key"]))):
            return False
        for form in forms(unit["key"]):
            p = statement_path(cfg, statement_name(unit["key"], form))
            n = int(done.loc[done["form"] == form, "n"].sum())
            if not os.path.exists(p) or pq.ParquetFile(p).metadata.num_rows != n:
                return False
        return True

    _, refused = run_units("main_statements", units, compute, out_dir, status, validate=valid)
    from experiments.purge import purge_unused
    purge_unused(cfg)
    if refused:
        raise MissingInput(f"{refused} statement source(s) could not be built yet")
    flush_status(status, "[main_statements] done")
    return out_dir
