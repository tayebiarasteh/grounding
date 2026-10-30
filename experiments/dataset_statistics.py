"""
experiments/dataset_statistics.py
Created on October 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments._common import (MissingInput, digest_file, fingerprint, flush_status, get_cfg, read_results_csv,
                                 read_source_csv, require, run_units, write_table)
from experiments.study import (ADDED_TRAIN_SETS, STATES, examples_path, mesh_findings, openi_eligible, out, read_openi,
                               read_padchest_gr, train_set_path)
from experiments.judges import judge_key, judgments_path, load_judgments
from experiments.support_pools import load_pools, pools_path, snapshot
from experiments.build_examples import TRAIN_SETS, output_path
from experiments.statements import full_path, reference_path, statement_name, statement_path, statement_set_names
from experiments.train_verifiers import encoder_max_tokens
from experiments.written_sentences import candidates_path, load_written_judgments, written_judgments_path
from experiments.evidence_sets import added_statement_path, added_statement_set_names, pools_added_path, written_pools

_REGIME = "dataset_statistics-v5"
FOLDER = "dataset_statistics"
LABELS = ("label_kappa", "label_flag", "label_agree", "label_full")


def _row(table: str, quantity, value, **kw) -> Dict:
    base = {"table": table, "dataset": "", "split": "", "condition": "", "finding": "", "state": "", "row_type": "",
            "quantity": quantity, "value": value, "text": ""}
    base.update(kw)
    return base


def source_rows(cfg: Dict) -> List[Dict]:
    src = cfg["sources"]
    rows = []
    sec = pd.read_parquet(require(src["mimic_sections"], "the MIMIC-CXR report build"), columns=["subject_id", "study_id"])
    rows += [_row("sources", "reports", len(sec), dataset="mimic"),
             _row("sources", "patients", sec["subject_id"].nunique(), dataset="mimic")]
    o = pd.read_parquet(require(src["oracle_mimic"], "the earlier data build"), columns=["split", "subject_id", "report_rel_path"])
    for sp, g in o.groupby("split"):
        rows += [_row("sources", "patients_in_list", g["subject_id"].nunique(), dataset="mimic", split=sp),
                 _row("sources", "studies_in_list", g["report_rel_path"].nunique(), dataset="mimic", split=sp)]
    c = pd.read_parquet(require(src["chexpert_sections"], "the CheXpert Plus report build"), columns=["report_rel_path"])
    rows += [_row("sources", "report_rows", len(c), dataset="chexpert"),
             _row("sources", "studies", c["report_rel_path"].nunique(), dataset="chexpert"),
             _row("sources", "patients", c["report_rel_path"].str.split("/").str[:2].str.join("/").nunique(), dataset="chexpert")]
    oc = pd.read_parquet(require(src["oracle_chexpert"], "the earlier data build"), columns=["split", "subject_id", "report_rel_path"])
    oc = oc[oc["split"] == "test"]
    rows += [_row("sources", "patients_in_list", oc["subject_id"].nunique(), dataset="chexpert", split="test"),
             _row("sources", "studies_in_list", oc["report_rel_path"].nunique(), dataset="chexpert", split="test")]
    ml = read_source_csv(require(src["rexgradient_masterlist"], "the ReXGradient-160K download"),
                         usecols=["PatientID", "StudyKey", "split", "ImageModality", "Findings", "Impression"])
    t = ml[ml["split"] == "test"]
    text_ok = t["Findings"].notna() & (t["Findings"].astype(str).str.strip() != "") & t["Impression"].notna() & \
        (t["Impression"].astype(str).str.strip() != "")
    e = t[t["ImageModality"].isin(["CR", "DX", "DR"]) & text_ok]
    sample = read_source_csv(require(src["rexgradient_sample"], "the fixed sample of ReXGradient-160K test patients"))
    rows += [_row("sources", "image_rows", len(ml), dataset="rexgradient"),
             _row("sources", "studies", ml["StudyKey"].nunique(), dataset="rexgradient"),
             _row("sources", "patients", ml["PatientID"].nunique(), dataset="rexgradient"),
             _row("sources", "test_studies", t["StudyKey"].nunique(), dataset="rexgradient", split="test"),
             _row("sources", "test_patients", t["PatientID"].nunique(), dataset="rexgradient", split="test"),
             _row("sources", "eligible_studies", e["StudyKey"].nunique(), dataset="rexgradient", split="test"),
             _row("sources", "eligible_patients", e["PatientID"].nunique(), dataset="rexgradient", split="test"),
             _row("sources", "sampled_studies", len(sample), dataset="rexgradient", split="test"),
             _row("sources", "sampled_patients", sample["PatientID"].nunique(), dataset="rexgradient", split="test")]
    oi = read_openi(src["openi_reports_dir"])
    both = oi[(oi["findings_text"] != "") & (oi["impression_text"] != "")]
    elig = openi_eligible(oi)
    fr = read_results_csv(require(src["openi_frontal"], "the frontal image selection"), dtype={"report_id": str, "image_id": str})
    fr = fr.merge(elig[["report_id", "f1_image_id"]], on="report_id", how="inner")
    coded = sum(mesh_findings(js) is not None for js in elig["mesh_major_json"])
    rows += [_row("sources", "reports", len(oi), dataset="openi"),
             _row("sources", "reports_with_both_sections", len(both), dataset="openi"),
             _row("sources", "reports_with_image", len(elig), dataset="openi"),
             _row("sources", "reports_with_frontal_image", len(fr), dataset="openi"),
             _row("sources", "reports_with_frontal_as_first_figure", int((fr["image_id"] == fr["f1_image_id"]).sum()), dataset="openi"),
             _row("sources", "reports_mesh_indexed", int(coded), dataset="openi"),
             _row("sources", "images", int(oi["image_ids_json"].map(lambda js: len(json.loads(js))).sum()), dataset="openi")]
    pg = read_padchest_gr(cfg)
    rows += [_row("sources", "studies", len(pg), dataset="padchest_gr"),
             _row("sources", "patients", pg["subject_id"].nunique(), dataset="padchest_gr")]
    for f in cfg["findings"]:
        rows.append(_row("sources", "studies_with_finding_present", int((pg[f"release__{f}"] == "present").sum()),
                         dataset="padchest_gr", finding=f))
    ev = snapshot(cfg)
    chest = ev["clean"] & ev["is_chest_relevant"].astype(bool)
    named = ev[ev["core"].map(len) > 0]
    arts = pd.read_parquet(src["snapshot"], columns=["article_id", "evidence_split"])
    sent_art = pd.read_parquet(src["snapshot"], columns=["sentence_id", "article_id", "sentence_token_count"])
    rows += [_row("sources", "sentences", len(ev), dataset="radiopaedia"),
             _row("sources", "sentence_tokens_min", int(sent_art["sentence_token_count"].min()), dataset="radiopaedia"),
             _row("sources", "sentence_tokens_max", int(sent_art["sentence_token_count"].max()), dataset="radiopaedia"),
             _row("sources", "articles", arts["article_id"].nunique(), dataset="radiopaedia"),
             _row("sources", "articles_naming_a_finding",
                  sent_art[sent_art["sentence_id"].isin(named["sentence_id"])]["article_id"].nunique(), dataset="radiopaedia"),
             _row("sources", "sentences_clean", int(ev["clean"].sum()), dataset="radiopaedia"),
             _row("sources", "sentences_chest", int(ev["is_chest_relevant"].astype(bool).sum()), dataset="radiopaedia"),
             _row("sources", "sentences_naming_a_finding", len(named), dataset="radiopaedia"),
             _row("sources", "combinations", int(named["core"].map(len).sum()), dataset="radiopaedia")]
    for aset, g in ev.groupby("article_set"):
        rows += [_row("sources", "sentences", len(g), dataset="radiopaedia", split=aset),
                 _row("sources", "sentences_chest_clean", int((chest & (ev["article_set"] == aset)).sum()), dataset="radiopaedia", split=aset),
                 _row("sources", "sentences_naming_a_finding", int((g["core"].map(len) > 0).sum()), dataset="radiopaedia", split=aset),
                 _row("sources", "combinations", int(g["core"].map(len).sum()), dataset="radiopaedia", split=aset)]
    for sp, g in arts.groupby("evidence_split"):
        rows.append(_row("sources", "articles", g["article_id"].nunique(), dataset="radiopaedia", split=str(sp)))
    wk = pd.read_parquet(require(_food_evidence_path(cfg), "the food evidence build"), columns=["article_title", "sentence_text"])
    chars = wk["sentence_text"].astype(str).str.len()
    rows += [_row("sources", "sentences", len(wk), dataset="wikipedia"),
             _row("sources", "articles", wk["article_title"].nunique(), dataset="wikipedia"),
             _row("sources", "sentence_chars_min", int(chars.min()), dataset="wikipedia"),
             _row("sources", "sentence_chars_max", int(chars.max()), dataset="wikipedia")]
    return rows


def _food_evidence_path(cfg: Dict) -> str:
    return os.path.join(os.path.dirname(cfg["sources"]["food_trainvaltest"]), "allergen_evidence_sentences.parquet")


def judge_rows(cfg: Dict) -> List[Dict]:
    rows = []
    jd = load_judgments(cfg)
    keys = [judge_key(e["name"]) for e in cfg["judges"]["models"]]
    for k in keys:
        rows.append(_row("judges", "unparsed", int((jd[f"{k}__status"] != "ok").sum()), dataset=k))
        for st, n in jd[f"{k}__state"].value_counts().items():
            rows.append(_row("judges", "assigned", int(n), dataset=k, state=st))
    same = np.ones(len(jd), bool)
    for k in keys[1:]:
        same &= (jd[f"{k}__state"] == jd[f"{keys[0]}__state"]).to_numpy()
    both = same & jd[f"{keys[0]}__state"].isin(STATES).to_numpy()
    rows += [_row("judges", "combinations_judged", len(jd), dataset="both"),
             _row("judges", "combinations_same_answer", int(same.sum()), dataset="both"),
             _row("judges", "combinations_both_assign_a_state", int(both.sum()), dataset="both"),
             _row("judges", "sentences_both_assign_a_state", jd.loc[both, "sentence_id"].nunique(), dataset="both")]
    cand = read_results_csv(require(out(cfg, "support_pools", "candidates.csv"), "main_support_pools"), keep_default_na=False)
    for (aset, f, s), g in cand.groupby(["article_set", "finding", "state"]):
        rows.append(_row("pools", "candidates", len(g), split=aset, finding=f, state=s))
    pools = load_pools(cfg)
    for (p, aset, f, s), g in pools.groupby(["pool", "article_set", "finding", "state"]):
        rows.append(_row("pools", p, len(g), split=aset, finding=f, state=s))
    for (p, aset), g in pools.groupby(["pool", "article_set"]):
        rows += [_row("pools", f"{p}_rows", len(g), split=aset),
                 _row("pools", f"{p}_distinct_sentences", g["sentence_id"].nunique(), split=aset)]
    added = read_results_csv(require(pools_added_path(cfg), "main_build_evidence_sets"), keep_default_na=False)
    for (p, f, s), g in added.groupby(["pool", "finding", "state"]):
        rows.append(_row("pools", p, len(g), split="level", finding=f, state=s))
    for p, g in added.groupby("pool"):
        rows += [_row("pools", f"{p}_rows", len(g), split="level"),
                 _row("pools", f"{p}_distinct_sentences", g["sentence_id"].nunique(), split="level")]
    wc = read_results_csv(require(candidates_path(cfg), "main_write_sentences"), keep_default_na=False)
    for kind, n in wc["kind"].value_counts().items():
        rows.append(_row("written", "candidates", int(n), row_type=kind))
    wj = load_written_judgments(cfg)
    for k in keys:
        rows.append(_row("written", "unparsed", int((wj[f"{k}__status"] != "ok").sum()), dataset=k))
    ok = np.ones(len(wj), bool)
    for k in keys:
        ok &= (wj[f"{k}__status"] == "ok").to_numpy()
    agree = np.ones(len(wj), bool)
    for k in keys[1:]:
        agree &= (wj[f"{k}__state"] == wj[f"{keys[0]}__state"]).to_numpy()
    assigned = np.where(ok & agree, wj[f"{keys[0]}__state"].to_numpy(), "")
    want = np.where(wj["kind"].to_numpy() == "unsupporting", "none", wj["state"].to_numpy())
    for kind, g in wj.assign(accepted=(assigned == want)).groupby("kind"):
        rows.append(_row("written", "accepted_by_both", int(g["accepted"].sum()), row_type=kind))
    frame, _ = written_pools(cfg)
    for p, n in frame["pool"].value_counts().items():
        rows.append(_row("written", "kept", int(n), row_type=p))
    rows.append(_row("written", "kept", len(frame), row_type="all"))
    return rows


def study_rows(cfg: Dict) -> List[Dict]:
    rf = pd.read_parquet(require(output_path(cfg, "report_findings"), "main_build_examples"))
    rows = []
    for (ds, sp), g in rf.groupby(["dataset", "split"]):
        rows += [_row("studies", "studies", g["study_key"].nunique(), dataset=ds, split=sp),
                 _row("studies", "patients", g["subject_id"].nunique(), dataset=ds, split=sp),
                 _row("studies", "report_findings", len(g), dataset=ds, split=sp)]
        for (f, s), h in g.groupby(["finding", "state"]):
            rows.append(_row("studies", "report_findings", len(h), dataset=ds, split=sp, finding=f, state=s))
        for (s, code), h in g.groupby(["state", "report_code"]):
            rows.append(_row("studies", "report_findings", len(h), dataset=ds, split=sp, state=s, row_type=str(code)))
    return rows


def _set_rows(stem: str, path: str, k: int) -> List[Dict]:
    df = pd.read_parquet(path)
    ds, cond = (stem.split("__", 1) if "__" in stem else ("train", stem))
    rows = [_row("examples", "examples", len(df), dataset=ds, condition=cond)]
    for (sp, rt), g in df.groupby(["split", "row_type"]):
        rows.append(_row("examples", "examples", len(g), dataset=ds, condition=cond, split=sp, row_type=rt))
    for sp, g in df.groupby("split"):
        rows += [_row("examples", "patients", g["subject_id"].nunique(), dataset=ds, condition=cond, split=sp),
                 _row("examples", "studies", g["report_rel_path"].nunique(), dataset=ds, condition=cond, split=sp),
                 _row("examples", "examples", len(g), dataset=ds, condition=cond, split=sp)]
    for (sp, f, rt), g in df.groupby(["split", "concept", "row_type"]):
        rows.append(_row("examples", "examples", len(g), dataset=ds, condition=cond, split=sp, finding=f, row_type=rt))
    for (sp, st), g in df[df["row_type"] == "A"].groupby(["split", "gold_state"]):
        rows.append(_row("examples", "report_findings", len(g), dataset=ds, condition=cond, split=sp, state=st))
    sup = df[df["row_type"].isin(["C", "D"])]
    ids = {int(i) for js in sup["evidence_sentence_ids_json"] for i in json.loads(js)[:k]}
    rows += [_row("examples", "evidence_sentences_cd", len(ids), dataset=ds, condition=cond),
             _row("examples", "mixed", int(df["evidence_source_type"].astype(str).str.endswith("_mixed").sum()), dataset=ds, condition=cond)]
    return rows


def replaced_report_rows(cfg: Dict) -> List[Dict]:
    with open(require(out(cfg, "examples", "manifest.json"), "main_build_examples"), encoding="utf-8") as f:
        notes = json.load(f)["notes"]
    rows = []
    for key, counts in notes.items():
        ds = key.split("__", 1)[0]
        for q in ("replaced_within_class", "replaced_from_other_classes", "left_out"):
            rows.append(_row("examples", f"report_{q}", int(counts[q]), dataset=ds, condition="report_replaced"))
    return rows


def example_rows(cfg: Dict) -> List[Dict]:
    k = int(cfg["max_evidence_sentences"])
    stems = []
    for ds, conds in cfg["test_sets"].items():
        for c in conds:
            stems.append((f"{ds}__{c}", examples_path(cfg, ds, c)))
    for name in list(TRAIN_SETS) + list(ADDED_TRAIN_SETS):
        stems.append((f"train__{name}", train_set_path(cfg, name)))
    rows = []
    for stem, path in tqdm(stems, desc="dataset_statistics examples", unit="set"):
        rows += _set_rows(stem, require(path, "main_build_examples or main_build_evidence_sets"), k)
    return rows


def statement_set_rows(cfg: Dict) -> List[Dict]:
    rows = []
    names = statement_set_names(cfg) + added_statement_set_names(cfg)
    for name in tqdm(names, desc="dataset_statistics statements", unit="set"):
        parts = name.split("__")
        source, form = "__".join(parts[1:-1]), parts[-1]
        citation = "training"
        for cit in ("heldout", "retrieved"):
            if source.endswith(f"_{cit}"):
                source, citation = source[:-len(cit) - 1], cit
        path = added_statement_path(cfg, name) if citation != "training" else statement_path(cfg, name)
        df = pd.read_parquet(require(path, "main_statements or main_build_evidence_sets"))
        base = {"dataset": source, "condition": citation, "row_type": form}
        rows += [_row("statements", "statements", len(df), **base),
                 _row("statements", "reports", df["report_rel_path"].nunique(), **base),
                 _row("statements", "patients", df["subject_id"].nunique(), **base)]
        if "kind" in df.columns:
            for kind, n in df["kind"].value_counts().items():
                rows.append(_row("statements", "statements", int(n), state=str(kind), **base))
        for lab in LABELS:
            if lab not in df.columns:
                continue
            v = df[lab]
            rows += [_row("statements", f"{lab}_consistent", int((v == 1).sum()), **base),
                     _row("statements", f"{lab}_contradicted", int((v == 0).sum()), **base),
                     _row("statements", f"{lab}_undefined", int(v.isna().sum()), **base)]
        if "recall_hit" in df.columns:
            rows.append(_row("statements", "retrieved_support", int(df["recall_hit"].astype(int).sum()), **base))
        for f, g in df.groupby("concept"):
            rows.append(_row("statements", "statements", len(g), finding=f, **base))
    return rows


def report_metric_rows(cfg: Dict) -> List[Dict]:
    rows = []
    for source in ("rexval", "radevalx"):
        ref = read_results_csv(require(reference_path(cfg, source), "main_statements"), dtype={"candidate_id": str})
        full = pd.read_parquet(require(full_path(cfg, source), "main_statements"))
        st = pd.read_parquet(require(statement_path(cfg, statement_name(source, "template")), "main_statements"))
        rows += [_row("report_metrics", "candidates", len(ref), dataset=source),
                 _row("report_metrics", "reports", full["report_rel_path"].nunique(), dataset=source),
                 _row("report_metrics", "patients", full["subject_id"].nunique(), dataset=source),
                 _row("report_metrics", "candidates_with_statements", st["candidate_id"].nunique(), dataset=source),
                 _row("report_metrics", "statements", len(st), dataset=source)]
    m = read_results_csv(require(cfg["sources"]["rexval_masterlist"], "main_preprocess_datasets"), dtype={"subject_id": str, "study_number": str})
    rows += [_row("report_metrics", "studies_released", m["study_number"].nunique(), dataset="rexval"),
             _row("report_metrics", "candidates_released", len(m), dataset="rexval")]
    return rows


def _unique_parameters(model) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def model_rows(cfg: Dict) -> List[Dict]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
    rows = []
    names = []
    for r in cfg["runs"] + list(cfg.get("runs_added", [])):
        if r["model_name"] not in names:
            names.append(r["model_name"])
    for name in tqdm(names, desc="dataset_statistics models", unit="model"):
        kinds = {r["kind"] for r in cfg["runs"] + list(cfg.get("runs_added", [])) if r["model_name"] == name}
        if "encoder" in kinds:
            c = AutoConfig.from_pretrained(name, num_labels=2, problem_type="single_label_classification")
            with torch.device("meta"):
                m = AutoModelForSequenceClassification.from_config(c)
            m.tie_weights()
            rows += [_row("models", "parameters", _unique_parameters(m), dataset=name, text=type(m).__name__),
                     _row("models", "max_tokens", encoder_max_tokens(name, int(cfg["training"]["max_tokens"])), dataset=name),
                     _row("models", "max_position_embeddings", int(getattr(c, "max_position_embeddings", 0) or 0), dataset=name)]
        else:
            c = AutoConfig.from_pretrained(name)
            with torch.device("meta"):
                m = AutoModelForCausalLM.from_config(c)
            m.tie_weights()
            rows += [_row("models", "parameters", _unique_parameters(m), dataset=name, text=type(m).__name__),
                     _row("models", "max_tokens", int(cfg["lora"]["max_tokens"]), dataset=name)]
    for vid in ("tfidf_mimic", "tfidf_food"):
        p = os.path.join(out(cfg, "scores", vid), "partial", "fit.csv")
        fit = read_results_csv(require(p, "main_tfidf")).iloc[0]
        for q in ("selected_c", "validation_auroc", "n_train", "n_val", "n_features"):
            rows.append(_row("baselines", q, float(fit[q]), dataset=vid))
    return rows


def generator_rows(cfg: Dict) -> List[Dict]:
    from experiments.generate_reports import generations_path, is_generated
    rows = []
    for g in cfg["generators"]["models"]:
        if not is_generated(cfg, g["name"]):
            raise MissingInput(f"the generations of {g['name']} are not current; run main_generate_reports first")
        df = read_results_csv(generations_path(cfg, g["name"]))
        rows.append(_row("generators", "reports", len(df), dataset=g["name"]))
    return rows


def _inputs(cfg: Dict) -> Dict:
    src = cfg["sources"]

    def dig(p):
        return digest_file(p) if os.path.exists(p) else None
    rec = {"report_findings": dig(output_path(cfg, "report_findings")), "pools": dig(pools_path(cfg)),
           "candidates": dig(out(cfg, "support_pools", "candidates.csv")), "judgments": dig(judgments_path(cfg)),
           "written_candidates": dig(candidates_path(cfg)), "written_judgments": dig(written_judgments_path(cfg)),
           "pools_added": dig(pools_added_path(cfg)), "examples_manifest": dig(out(cfg, "examples", "manifest.json")),
           "evidence_manifest": dig(out(cfg, "evidence_sets", "manifest.json")), "frontal": dig(src["openi_frontal"]),
           "rexval_masterlist": dig(src["rexval_masterlist"]), "radevalx_masterlist": dig(src["radevalx_masterlist"]),
           "rexgradient_sample": dig(src["rexgradient_sample"]), "food_evidence": dig(_food_evidence_path(cfg))}
    for name in ("snapshot", "oracle_mimic", "oracle_chexpert", "mimic_sections", "chexpert_sections",
                 "rexgradient_masterlist", "padchest_gr_reports", "padchest_gr_table"):
        rec[name] = fingerprint(src[name]) if os.path.exists(src[name]) else None
    rec["openi_reports_dir"] = "present" if os.path.isdir(src["openi_reports_dir"]) else None
    rec["statement_sets"] = {n: dig(statement_path(cfg, n)) for n in statement_set_names(cfg)}
    rec["added_statement_sets"] = {n: dig(added_statement_path(cfg, n)) for n in added_statement_set_names(cfg)}
    rec["models"] = sorted({r["model_name"] for r in cfg["runs"] + list(cfg.get("runs_added", []))})
    return rec


def main_dataset_statistics(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    out_dir = out(cfg, FOLDER)
    status = os.path.join(out_dir, "status.txt")
    params = {"regime": _REGIME, "inputs": _inputs(cfg), "findings": list(cfg["findings"]),
              "max_evidence_sentences": int(cfg["max_evidence_sentences"])}

    def compute(unit):
        missing = [n for n, d in unit["params"]["inputs"].items() if d is None]
        missing += [n for n, d in unit["params"]["inputs"]["statement_sets"].items() if d is None]
        missing += [n for n, d in unit["params"]["inputs"]["added_statement_sets"].items() if d is None]
        if missing:
            raise MissingInput(f"inputs of main_dataset_statistics missing: {', '.join(missing[:6])}; compute them first")
        rows = source_rows(cfg) + judge_rows(cfg) + study_rows(cfg) + example_rows(cfg) + replaced_report_rows(cfg)
        rows += statement_set_rows(cfg)
        rows += report_metric_rows(cfg) + model_rows(cfg) + generator_rows(cfg)
        return rows

    def valid(unit, done):
        return bool(len(done) and {"sources", "pools", "examples", "statements", "models"} <= set(done["table"]))

    rows, refused = run_units("main_dataset_statistics", [{"key": "statistics", "params": params}], compute, out_dir, status, validate=valid)
    if refused:
        raise MissingInput("the statistics could not be built yet")
    write_table(pd.DataFrame(rows), os.path.join(out_dir, "dataset_statistics.csv"))
    flush_status(status, f"[main_dataset_statistics] done, {len(rows)} rows")
    return out_dir
