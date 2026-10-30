"""
experiments/reader_analysis.py
Created on October 24, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from Inference.cluster_metrics import EVIDENCE_PAIR_REGIME

from experiments._common import (MissingInput, digest_file, flush_status, get_cfg, read_results_csv, read_source_csv,
                                 require, run_units, stats_params, write_table)
from experiments.statements import _generators, statement_name
from experiments.attainable_auroc import bounds
from experiments.result_tables import (_correlations, _pooled, _thresholds, agreement_rows, statement_rows, test_rows,
                                    threshold_set)
from experiments.result_tables_added import citation_cell_rows

PACKETS = ("A_support", "B_report_labels", "C_generator_statements", "D_rexerr")
PRACTICE = "0_practice"
ANSWERS = {"A_support": ["present", "absent", "neither"],
           "B_report_labels": ["present", "absent", "uncertain", "not mentioned"],
           "C_generator_statements": ["agrees", "contradicts", "not addressed"],
           "D_rexerr": ["agrees", "contradicts", "not addressed"]}
SHEET = {"A_support": ["display_id", "finding", "sentence", "answer", "comment"],
         "B_report_labels": ["display_id", "finding", "report", "answer", "comment"],
         "C_generator_statements": ["display_id", "finding", "sentence", "report", "answer", "comment"],
         "D_rexerr": ["display_id", "finding", "sentence", "report", "answer", "comment"],
         PRACTICE: ["display_id", "task", "finding", "sentence", "report", "answer", "comment"]}
from experiments.score import input_path, is_scored, load_scores, score_path, verifiers
from experiments.study import TEST_SETS, out
from Inference.stats_utils import bh_fdr

_REGIME = "reader_analysis-v2"
NEITHER = "neither"
BOTH = "both"
DISAGREE = "disagree"
SHEET_FILES = ("cases.csv", "cases.xlsx")
CONFIRMED_TEST_SETS = tuple(f"{ds}__unchanged" for ds in TEST_SETS) + ("mimic__heldout", "mimic__heldout_written")
CITATIONS = ("", "heldout", "retrieved")
FORMS = ("template", "raw")
REPORT_POLICY = {"present": "present", "absent": "absent", "not mentioned": "absent"}
CODE_READING = {"positive": "present", "negative": "absent", "blank": "not mentioned", "uncertain": "uncertain"}
JUDGES = {"gpt": "gpt_oss_120b", "qwen": "qwen3_5_122b", "both": "both_judges"}
DEFINITIONS = ("agree", "kappa", "flag")
CONFIRMED_TABLES = {"metrics": "confirmed_metrics", "pairs": "confirmed_pairs", "operating": "confirmed_operating",
                    "statements": "confirmed_statements", "citation_cells": "confirmed_cells"}


def readers(cfg: Dict) -> List[str]:
    rs = [str(r) for r in cfg["reader"]["readers"]]
    if not 1 <= len(rs) <= 2 or len(set(rs)) != len(rs) or BOTH in rs:
        raise ValueError(f"reader.readers must name one or two distinct readers, not {rs}")
    return rs


def support_definitions(cfg: Dict) -> List[str]:
    rs = readers(cfg)
    return rs + [BOTH] if len(rs) == 2 else rs


def members(cfg: Dict, definition: str) -> List[str]:
    return readers(cfg) if definition == BOTH else [definition]


def returned_dir(cfg: Dict, reader: str, packet: str) -> str:
    return out(cfg, "reader_study", "returned", reader, packet)


def _sheet(path: str) -> pd.DataFrame:
    return read_source_csv(path, dtype=str, keep_default_na=False)


def _xlsx(path: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0, dtype=str, keep_default_na=False, engine="openpyxl").fillna("")


def read_packet(cfg: Dict, reader: str, packet: str) -> pd.DataFrame:
    d = returned_dir(cfg, reader, packet)
    if not os.path.isdir(d):
        raise MissingInput(f"the sheet of {packet} from reader {reader} is not back yet ({d})")
    key = _sheet(require(out(cfg, "reader_study", "internal", f"{packet}_key.csv"), "the reader study"))
    sent = _sheet(require(out(cfg, "reader_study", "packet", packet, "cases.csv"), "the reader study"))
    shown = [c for c in SHEET[packet] if c not in ("answer", "comment")]
    filled = {}
    for name in SHEET_FILES:
        path = os.path.join(d, name)
        if not os.path.exists(path):
            continue
        back = _sheet(path) if name.endswith(".csv") else _xlsx(path)
        if list(back.columns) != SHEET[packet]:
            raise ValueError(f"{path}: columns {list(back.columns)} differ from the sheet sent, {SHEET[packet]}")
        if len(back) != len(sent) or not back[shown].equals(sent[shown]):
            raise ValueError(f"{path}: the rows or their visible text differ from the sheet sent")
        back["answer"], back["comment"] = back["answer"].str.strip(), back["comment"].str.strip()
        if back["answer"].ne("").any():
            filled[name] = back
    if not filled:
        raise ValueError(f"{d}: no returned file holds an answer")
    if len(filled) == 2 and not filled[SHEET_FILES[0]]["answer"].equals(filled[SHEET_FILES[1]]["answer"]):
        raise ValueError(f"{d}: the CSV and the Excel file hold different answers")
    back = next(iter(filled.values()))
    bad = back.loc[~back["answer"].isin(ANSWERS[packet] + [""]), "display_id"].tolist()
    if bad:
        raise ValueError(f"{d}: {len(bad)} answers outside {ANSWERS[packet]}, among them {bad[:3]}")
    m = key.merge(back[["display_id", "answer", "comment"]], on="display_id", how="inner", validate="one_to_one")
    if len(m) != len(key):
        raise ValueError(f"{packet}: {len(m)} of {len(key)} key rows matched a returned row of reader {reader}")
    return m


def _name(reader: str, *parts: str) -> str:
    return "__".join(("reader", reader) + parts)


def _counts(df: pd.DataFrame, packet: str, reader: str, rater: str, label: str, by: List[str]) -> List[Dict]:
    g = df.groupby(by + [label, "answer"], dropna=False).size().reset_index(name="n")
    return [{"table": "counts", "packet": packet, "reader": reader, "rater": rater, **{c: r[c] for c in by},
             "label": r[label], "answer": r["answer"], "n": int(r["n"])} for _, r in g.iterrows()]


def _agreement_ids(frame: pd.DataFrame) -> np.ndarray:
    packet = frame["packet"].iloc[0]
    if packet == "A_support":
        ids = "sentence:" + frame["sentence"].str.strip()
    elif packet == "B_report_labels":
        ids = frame["dataset"] + ":" + frame["subject_id"]
    elif packet == "C_generator_statements":
        ids = "openi:" + frame["study_key"]
    elif packet == "D_rexerr":
        ids = "mimic:" + frame["subject_id"]
    else:
        raise ValueError(f"unknown agreement packet {packet}")
    return (ids if ids.duplicated().any() else frame["display_id"]).to_numpy()


def _agree(cfg: Dict, a, b, name: str, ids, extra: Dict) -> Tuple[Dict, Dict]:
    return agreement_rows(np.asarray(a), np.asarray(b), name, np.asarray(ids), cfg, extra)


def _rule_states(cfg: Dict) -> Dict[str, str]:
    pools = read_results_csv(require(out(cfg, "support_pools", "pools.csv"), "main_support_pools"), dtype=str, keep_default_na=False)
    return {f"{int(r.sentence_id)}|{r.finding}": r.state for r in pools[pools["pool"] == "rule"].itertuples()}


def _judged(cfg: Dict, a: pd.DataFrame) -> pd.DataFrame:
    if (a[["gpt_state", "qwen_state"]] == "").any().any():
        raise ValueError("packet A holds a combination without the state of a judge")
    a = a.copy()
    a["rule_state"] = a["item_id"].map(_rule_states(cfg)).fillna("")
    a["gpt"] = a["gpt_state"].replace("none", NEITHER)
    a["qwen"] = a["qwen_state"].replace("none", NEITHER)
    a["both"] = np.where(a["gpt"] == a["qwen"], a["gpt"], NEITHER)
    return a


def confirmation_rows(cfg: Dict, a: pd.DataFrame, reader: str) -> List[Dict]:
    base = {"packet": "A_support", "reader": reader}
    assigned = a[a["state"] != ""].assign(assigned=lambda d: d["state"])
    subsets = {g: assigned[assigned["group"] == g] for g in sorted(assigned["group"].unique())}
    subsets["pool"] = assigned[assigned["group"].isin(["pool_training", "pool_heldout"])]
    subsets["written"] = assigned[assigned["group"].isin(["written_training", "written_heldout"])]
    subsets["rule"] = a[a["rule_state"] != ""].assign(assigned=lambda d: d["rule_state"])
    rows = []
    for name, sub in subsets.items():
        for st in ("all", "present", "absent"):
            x = sub if st == "all" else sub[sub["assigned"] == st]
            if len(x):
                m, _ = _agree(cfg, x["answer"], x["assigned"], _name(reader, "A", "confirmation", name, st),
                              _agreement_ids(x), {**base, "rater": "assigned_state", "group": name, "state": st})
                rows.append({**m, "table": "confirmation"})
    rows += _counts(assigned, "A_support", reader, "assigned_state", "assigned", ["group"])
    rows += _counts(subsets["rule"].assign(group="rule"), "A_support", reader, "assigned_state", "assigned", ["group"])
    return rows


def judge_rows(cfg: Dict, a: pd.DataFrame, reader: str) -> List[Dict]:
    base = {"packet": "A_support", "reader": reader}
    rows = []
    for col, rater in JUDGES.items():
        m, s = _agree(cfg, a[col], a["answer"], _name(reader, "A", rater), _agreement_ids(a),
                      {**base, "rater": rater, "group": "all"})
        rows += [m, s]
        for g, sub in a.groupby("group"):
            m, _ = _agree(cfg, sub[col], sub["answer"], _name(reader, "A", rater, g), _agreement_ids(sub),
                          {**base, "rater": rater, "group": g})
            rows.append(m)
        rows += _counts(a, "A_support", reader, rater, col, ["group"])
    return rows


def _statement_reading(answer: pd.Series, state: pd.Series) -> np.ndarray:
    return np.where(answer == "agrees", 1, np.where(answer == "contradicts", 0, np.where(state == "absent", 1, 0)))


def packet_b_rows(cfg: Dict, b: pd.DataFrame, reader: str) -> List[Dict]:
    b = b.assign(reading=b["answer"].map(REPORT_POLICY), code_reading=b["code"].map(CODE_READING))
    if b["code_reading"].isna().any():
        raise ValueError(f"packet B holds CheXbert classes outside {sorted(CODE_READING)}")
    base = {"packet": "B_report_labels", "reader": reader}
    ok = b[b["reading"].notna()]
    m, s = _agree(cfg, ok["state"], ok["reading"], _name(reader, "B", "test_label"), _agreement_ids(ok),
                  {**base, "rater": "test_label", "group": "all"})
    rows = [m, s]
    for col in ("dataset", "code", "stratum"):
        for v, sub in ok.groupby(col):
            m, _ = _agree(cfg, sub["state"], sub["reading"], _name(reader, "B", "test_label", col, v), _agreement_ids(sub),
                          {**base, "rater": "test_label", "group": f"{col}:{v}"})
            rows.append(m)
    m, s = _agree(cfg, b["code_reading"], b["answer"], _name(reader, "B", "chexbert_class"), _agreement_ids(b),
                  {**base, "rater": "chexbert_class", "group": "all"})
    rows += [m, s]
    rows += _counts(b, "B_report_labels", reader, "test_label", "state", ["dataset", "code", "stratum"])
    return rows


def packet_c_rows(cfg: Dict, c: pd.DataFrame, reader: str) -> List[Dict]:
    c = c.assign(label=c["label_kappa"].astype(float).astype(int), reading=_statement_reading(c["answer"], c["state"]))
    base = {"packet": "C_generator_statements", "reader": reader}
    m, s = _agree(cfg, c["label"], c["reading"], _name(reader, "C", "kappa"), _agreement_ids(c),
                  {**base, "rater": "kappa", "group": "all"})
    rows = [m, s]
    for col in ("kind", "generator"):
        for v, sub in c.groupby(col):
            m, _ = _agree(cfg, sub["label"], sub["reading"], _name(reader, "C", "kappa", col, v), _agreement_ids(sub),
                          {**base, "rater": "kappa", "group": f"{col}:{v}"})
            rows.append(m)
    rows += _counts(c, "C_generator_statements", reader, "kappa", "label", ["generator", "kind", "state"])
    return rows


def packet_d_rows(cfg: Dict, d: pd.DataFrame, reader: str) -> List[Dict]:
    d = d.assign(reading=_statement_reading(d["answer"], d["state"]))
    base = {"packet": "D_rexerr", "reader": reader}
    rows = []
    for dname in DEFINITIONS:
        sub = d[d[f"label_{dname}"] != ""]
        y = sub[f"label_{dname}"].astype(float).astype(int)
        m, s = _agree(cfg, y, sub["reading"], _name(reader, "D", dname), _agreement_ids(sub),
                      {**base, "rater": dname, "group": "all"})
        rows += [m, s]
        for col in ("stratum", "error_type"):
            for v, x in sub.assign(y=y).groupby(col):
                m, _ = _agree(cfg, x["y"], x["reading"], _name(reader, "D", dname, col, v), _agreement_ids(x),
                              {**base, "rater": dname, "group": f"{col}:{v}"})
                rows.append(m)
    rows += _counts(d, "D_rexerr", reader, "kappa", "label_kappa",
                    ["stratum", "error_type", "kind", "label_flag", "label_agree"])
    return rows


def reader_rows(cfg: Dict, sheets: Dict[str, pd.DataFrame], reader: str) -> List[Dict]:
    rows = []
    for p in PACKETS:
        df = sheets[p]
        empty = (df["answer"] == "").to_numpy()
        rows.append({"table": "unanswered", "packet": p, "reader": reader, "n": int(empty.sum()), "n_items": int(len(df)),
                     "display_ids": " ".join(df.loc[empty, "display_id"])})
        df = df[~empty]
        if p == "A_support":
            a = _judged(cfg, df)
            rows += confirmation_rows(cfg, a, reader) + judge_rows(cfg, a, reader)
        elif p == "B_report_labels":
            rows += packet_b_rows(cfg, df, reader)
        elif p == "C_generator_statements":
            rows += packet_c_rows(cfg, df, reader)
        else:
            rows += packet_d_rows(cfg, df, reader)
    return rows


def _paired(packets: Dict[str, Dict[str, pd.DataFrame]], rs: List[str], packet: str) -> pd.DataFrame:
    a, b = (packets[r][packet] for r in rs)
    x = a.rename(columns={"answer": "answer_1"}).merge(b[["display_id", "answer"]].rename(columns={"answer": "answer_2"}),
                                                       on="display_id", validate="one_to_one")
    return x[(x["answer_1"] != "") & (x["answer_2"] != "")]


def consensus_rows(cfg: Dict, packets: Dict[str, Dict[str, pd.DataFrame]], rs: List[str]) -> List[Dict]:
    x = _paired(packets, rs, "A_support")
    a = x.assign(answer=np.where(x["answer_1"] == x["answer_2"], x["answer_1"], DISAGREE))
    return confirmation_rows(cfg, _judged(cfg, a.drop(columns=["answer_1", "answer_2"])), BOTH)


def interreader_rows(cfg: Dict, packets: Dict[str, Dict[str, pd.DataFrame]], rs: List[str]) -> List[Dict]:
    pair = "_".join(rs)
    rows = []
    for p in PACKETS:
        x = _paired(packets, rs, p)
        base = {"packet": p, "reader": pair, "rater": rs[0]}
        m, s = _agree(cfg, x["answer_1"], x["answer_2"], f"readers__{p[0]}__answer", _agreement_ids(x),
                      {**base, "group": "all", "scale": "answer"})
        rows += [m, s]
        if p == "A_support":
            groups = dict(tuple(x.groupby("group")))
            groups["pool"] = x[x["group"].isin(["pool_training", "pool_heldout"])]
            groups["written"] = x[x["group"].isin(["written_training", "written_heldout"])]
            groups["other"] = x[x["group"].str.startswith("other_")]
            for g, sub in groups.items():
                m, _ = _agree(cfg, sub["answer_1"], sub["answer_2"], f"readers__A__answer__{g}", _agreement_ids(sub),
                              {**base, "group": g, "scale": "answer"})
                rows.append(m)
        if p == "B_report_labels":
            y = x.assign(l1=x["answer_1"].map(REPORT_POLICY), l2=x["answer_2"].map(REPORT_POLICY)).dropna(subset=["l1", "l2"])
            m, s = _agree(cfg, y["l1"], y["l2"], "readers__B__label", _agreement_ids(y), {**base, "group": "all", "scale": "label"})
            rows += [m, s]
        if p in ("C_generator_statements", "D_rexerr"):
            l1, l2 = _statement_reading(x["answer_1"], x["state"]), _statement_reading(x["answer_2"], x["state"])
            m, s = _agree(cfg, l1, l2, f"readers__{p[0]}__label", _agreement_ids(x), {**base, "group": "all", "scale": "label"})
            rows += [m, s]
        g = x.groupby(["answer_1", "answer_2"]).size().reset_index(name="n")
        rows += [{"table": "reader_pairs", "packet": p, "reader": pair, "answer": r.answer_1, "answer_2": r.answer_2,
                  "n": int(r.n)} for r in g.itertuples()]
    return rows


def agreement_unit_rows(cfg: Dict, packets: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict]:
    rs = readers(cfg)
    rows = []
    for r in rs:
        rows += reader_rows(cfg, packets[r], r)
    if len(rs) == 2:
        rows += consensus_rows(cfg, packets, rs) + interreader_rows(cfg, packets, rs)
    return rows


def confirmation_map(cfg: Dict, a: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    added = read_results_csv(require(out(cfg, "evidence_sets", "pools_added.csv"), "main_build_evidence_sets"), dtype=str, keep_default_na=False)
    ids = dict(zip(added["item_id"], added["sentence_id"]))
    cmap: Dict[Tuple[int, str], str] = {}
    for item, finding, answer in zip(a["item_id"], a["finding"], a["answer"]):
        sid = int(ids[item]) if item in ids else int(item.split("|")[0])
        if (sid, finding) in cmap:
            raise ValueError(f"packet A holds the sentence {sid} with the finding {finding} twice")
        cmap[(sid, finding)] = answer
    return cmap


def confirmation_maps(cfg: Dict, packets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[Tuple[int, str], str]]:
    rs = readers(cfg)
    maps = {r: confirmation_map(cfg, packets[r]["A_support"]) for r in rs}
    if len(rs) == 2:
        first, second = maps[rs[0]], maps[rs[1]]
        if set(first) != set(second):
            raise ValueError("the two readers' packets A hold different sentences")
        maps[BOTH] = {k: (v if second[k] == v else "") for k, v in first.items()}
    return maps


def heldout_pool(cfg: Dict) -> Set[Tuple[int, str, str]]:
    pools = read_results_csv(require(out(cfg, "support_pools", "pools.csv"), "main_support_pools"), dtype=str, keep_default_na=False)
    h = pools[(pools["pool"] == "support") & (pools["article_set"] == "heldout")]
    return {(int(r.sentence_id), r.finding, r.state) for r in h.itertuples()}


def _evidence_ids(ids_json: str, texts_json: str, k: int) -> List[int]:
    ids, texts = json.loads(ids_json), json.loads(texts_json)
    if len(ids) != len(texts):
        raise ValueError(f"an evidence list has {len(ids)} ids and {len(texts)} sentences")
    return [int(i) for i, t in zip(ids, texts) if str(t).strip()][:k]


def confirmed_mask(df: pd.DataFrame, cmap: Dict[Tuple[int, str], str], k: int, types=("C", "D")) -> np.ndarray:
    keep = np.ones(len(df), bool)
    unread = set()
    for i, (rt, ij, tj, f, st) in enumerate(zip(df["row_type"], df["evidence_sentence_ids_json"],
                                                df["evidence_sentences_json"], df["concept"], df["claim_state"])):
        if rt not in types:
            continue
        sids = _evidence_ids(ij, tj, k)
        unread.update((s, f) for s in sids if (s, f) not in cmap)
        keep[i] = any(cmap.get((s, f)) == st for s in sids)
    if unread:
        raise ValueError(f"{len(unread)} support sentences were not read by the readers, among them {sorted(unread)[:3]}")
    return keep


def retrieved_mask(df: pd.DataFrame, cmap: Dict[Tuple[int, str], str], pool: Set[Tuple[int, str, str]],
                   k: int) -> np.ndarray:
    keep = np.ones(len(df), bool)
    for i, (ij, tj, f, st, hit) in enumerate(zip(df["evidence_sentence_ids_json"], df["evidence_sentences_json"],
                                                 df["concept"], df["claim_state"], df["recall_hit"])):
        in_pool = [s for s in _evidence_ids(ij, tj, k) if (s, f, st) in pool]
        if bool(in_pool) != bool(int(hit)):
            raise ValueError(f"row {i}: the stored retrieved support ({hit}) differs from the held-out pool membership")
        if in_pool:
            keep[i] = any(cmap.get((s, f)) == st for s in in_pool)
    return keep


def _statement_sets(cfg: Dict) -> List[str]:
    return [statement_name(src, form, cit) for src in ["rexerr"] + [f"gen__{g}" for g in _generators(cfg)]
            for cit in CITATIONS for form in FORMS]


def confirmed_rows(cfg: Dict, vx: Dict, cmap: Dict, pool: Set, definition: str) -> List[Dict]:
    k = int(cfg["max_evidence_sentences"])
    for s in CONFIRMED_TEST_SETS + tuple(_statement_sets(cfg)):
        if not is_scored(cfg, vx["id"], s):
            raise MissingInput(f"{vx['id']} has no scores on {s}")
    ref = load_scores(cfg, vx["id"], threshold_set(vx))
    tau = _thresholds(cfg, vx, ref=ref[confirmed_mask(ref, cmap, k)])
    rows = []
    for s in CONFIRMED_TEST_SETS:
        df = load_scores(cfg, vx["id"], s)
        rows += [r for r in test_rows(cfg, vx, s, splits=False, df=df[confirmed_mask(df, cmap, k)], tau=tau)
                 if r.get("finding") == "all"]
    for cit in CITATIONS:
        for form in FORMS:
            name = statement_name("rexerr", form, cit)
            df = load_scores(cfg, vx["id"], name)
            keep = retrieved_mask(df, cmap, pool, k) if cit == "retrieved" else confirmed_mask(df, cmap, k, ("S",))
            rows += statement_rows(cfg, vx, name, df=df[keep], tau=tau)
            if cit == "retrieved":
                rows += citation_cell_rows(cfg, vx, name, df=df[keep], tau=tau)
            pooled = _pooled(cfg, vx, form, citation=cit)
            keep = retrieved_mask(pooled, cmap, pool, k) if cit == "retrieved" else confirmed_mask(pooled, cmap, k, ("S",))
            rows += statement_rows(cfg, vx, statement_name("genpooled", form, cit), df=pooled[keep], tau=tau)
            if cit == "retrieved":
                for g in _generators(cfg):
                    sel = (pooled["generator"] == g).to_numpy()
                    rows += citation_cell_rows(cfg, vx, statement_name(f"gen__{g}", form, cit),
                                               df=pooled[sel & keep], tau=tau)
    return [{**r, "reader": definition} for r in rows]


def confirmed_bound_rows(cfg: Dict, cmap: Dict, definition: str) -> List[Dict]:
    k = int(cfg["max_evidence_sentences"])
    rows = []
    for s in CONFIRMED_TEST_SETS:
        df = pd.read_parquet(require(input_path(cfg, s), "main_build_examples or main_build_evidence_sets"))
        keep = confirmed_mask(df, cmap, k)
        rows += [{"table": "confirmed_bounds", **r} for r in bounds(df[keep], k, s)]
        for (f, st, rt), x in df.assign(kept=keep).groupby(["concept", "claim_state", "row_type"]):
            rows.append({"table": "confirmed_counts", "set": s, "finding": f, "state": st, "row_type": rt,
                         "n": int(len(x)), "n_kept": int(x["kept"].sum())})
    return [{**r, "reader": definition} for r in rows]


def retrieved_support_rows(cfg: Dict, maps: Dict[str, Dict], pool: Set) -> List[Dict]:
    k = int(cfg["max_evidence_sentences"])
    rs = readers(cfg)
    rows = []
    for src in ["rexerr"] + [f"gen__{g}" for g in _generators(cfg)]:
        for form in FORMS:
            df = pd.read_parquet(require(input_path(cfg, statement_name(src, form, "retrieved")), "main_build_evidence_sets"))
            per, n_sup, n_conf = {}, 0, {d: 0 for d in maps}
            for ij, tj, f, st in zip(df["evidence_sentence_ids_json"], df["evidence_sentences_json"], df["concept"],
                                     df["claim_state"]):
                hits = {s for s in _evidence_ids(ij, tj, k) if (s, f, st) in pool}
                n_sup += bool(hits)
                for d, cmap in maps.items():
                    n_conf[d] += any(cmap.get((s, f)) == st for s in hits)
                for s in hits:
                    per[(s, f, st)] = per.get((s, f, st), 0) + 1
            base = {"table": "retrieved_support", "source": f"{src}_retrieved", "form": form}
            rows += [{**base, "kind": "set", "reader": d, "n_statements": int(len(df)), "n_with_support": n_sup,
                      "n_with_confirmed_support": n_conf[d], "n_sentences": len(per)} for d in maps]
            rows += [{**base, "kind": "sentence", "sentence_id": s, "finding": f, "state": st, "n_with_support": n,
                      **{f"answer_{r}": maps[r].get((s, f), "") for r in rs}} for (s, f, st), n in sorted(per.items())]
    return rows


def _digests(d: str) -> Dict[str, str]:
    return {n: digest_file(os.path.join(d, n)) for n in SHEET_FILES if os.path.exists(os.path.join(d, n))}


def _digest(path: str):
    return digest_file(path) if os.path.exists(path) else None


def _units(cfg: Dict) -> List[Dict]:
    n_boot, seed = stats_params(cfg)
    rs = readers(cfg)
    sheets = {r: {p: _digests(returned_dir(cfg, r, p)) for p in PACKETS} for r in rs}
    keys = {p: _digest(out(cfg, "reader_study", "internal", f"{p}_key.csv")) for p in PACKETS}
    sources = {"pools": _digest(out(cfg, "support_pools", "pools.csv")),
               "pools_added": _digest(out(cfg, "evidence_sets", "pools_added.csv"))}

    def common(readers_: List[str]) -> Dict:
        return {"regime": _REGIME, "sheets_A": {r: sheets[r]["A_support"] for r in readers_}, "key_A": keys["A_support"],
                **sources, "max_evidence_sentences": int(cfg["max_evidence_sentences"])}

    units = [{"key": "agreement", "kind": "agreement",
              "legacy": {"agreement_inference": "item-analytic-v0"},
              "params": {"regime": _REGIME, "agreement_inference": "cluster-aware-v1", "sheets": sheets, "keys": keys,
                         "sent": {p: _digest(out(cfg, "reader_study", "packet", p, "cases.csv")) for p in PACKETS},
                         "pools": sources["pools"], "n_boot": n_boot, "boot_seed": seed}},
             {"key": "retrieved_support", "kind": "retrieved_support",
              "params": {**common(rs), "definitions": support_definitions(cfg),
                         "inputs": {s: _digest(input_path(cfg, s)) for s in _statement_sets(cfg) if "_retrieved__" in s}}}]
    sets = CONFIRMED_TEST_SETS + tuple(_statement_sets(cfg))
    for d in support_definitions(cfg):
        units.append({"key": f"bounds__{d}", "kind": "bounds", "definition": d,
                      "params": {**common(members(cfg, d)),
                                 "inputs": {s: _digest(input_path(cfg, s)) for s in CONFIRMED_TEST_SETS}}})
    for d in support_definitions(cfg):
        for vx in verifiers(cfg):
            if vx["domain"] != "radiology":
                continue
            units.append({"key": f"confirmed__{d}__{vx['id']}", "kind": "confirmed", "vx": vx, "definition": d,
                          "legacy": {"evidence_pair_weighting": "row-product-v0"},
                          "params": {**common(members(cfg, d)),
                                     "evidence_pair_weighting": EVIDENCE_PAIR_REGIME,
                                     "scores": {s: _digest(score_path(cfg, vx["id"], s)) for s in sets},
                                     "inputs": {s: _digest(input_path(cfg, s)) for s in sets},
                                     "threshold_scores": _digest(score_path(cfg, vx["id"], threshold_set(vx))),
                                     "n_boot": n_boot, "boot_seed": seed}})
    return units


def main_reader_analysis(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    out_dir = out(cfg, "reader_analysis")
    status = os.path.join(out_dir, "status.txt")
    rs = readers(cfg)
    packets = {r: {p: read_packet(cfg, r, p) for p in PACKETS} for r in rs}
    maps, pool = confirmation_maps(cfg, packets), heldout_pool(cfg)
    units = _units(cfg)

    def compute(unit):
        if unit["kind"] == "agreement":
            return agreement_unit_rows(cfg, packets)
        if unit["kind"] == "bounds":
            return confirmed_bound_rows(cfg, maps[unit["definition"]], unit["definition"])
        if unit["kind"] == "retrieved_support":
            return retrieved_support_rows(cfg, maps, pool)
        vx, d = unit["vx"], unit["definition"]
        return confirmed_rows(cfg, vx, maps[d], pool, d)

    rows, refused = run_units("main_reader_analysis", units, compute, out_dir, status)
    if refused:
        raise MissingInput(f"{refused} unit(s) of main_reader_analysis wait for an input; the table is written once all exist")
    df = pd.DataFrame(rows)
    result_statistics = read_results_csv(require(out(cfg, "result_tables", "statistics.csv"), "main_result_tables"))
    e7 = result_statistics[result_statistics["family"] == "E7"].drop(columns=["p_fdr"]).dropna(axis=1, how="all")
    fam = pd.concat([e7, df[df["table"] == "statistic"].dropna(axis=1, how="all")], ignore_index=True)
    fam["p_fdr"] = bh_fdr(fam["p_raw"].to_numpy(float))
    extra = []
    for d in support_definitions(cfg):
        sel = df["reader"] == d
        test = df[sel & df["table"].isin(["metrics", "pairs", "operating"])].dropna(axis=1, how="all")
        stmts = df[sel & (df["table"] == "statements")].dropna(axis=1, how="all")
        extra.append(_correlations(test, stmts).assign(family="E5_confirmed", reader=d))
    rest = df[df["table"] != "statistic"].copy()
    rest["table"] = rest["table"].replace(CONFIRMED_TABLES)
    final = pd.concat([rest, fam] + extra, ignore_index=True)
    path = os.path.join(out_dir, "reader_analysis.csv")
    wrote = write_table(final, path)
    flush_status(status, f"[main_reader_analysis] done: {len(final)} rows {'written' if wrote else 'unchanged'}")
    return path
