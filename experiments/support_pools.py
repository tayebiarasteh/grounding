"""
experiments/support_pools.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
from typing import Dict, List

import pandas as pd

from experiments._common import (MissingInput, digest_file, fingerprint, flush_status, get_cfg, read_results_csv,
                                 require, run_units, write_table)
from experiments.study import STATES, core_findings, out
from experiments.judges import judge_key, judgments_path, load_judgments

_REGIME = "support_pools-v2"
ARTICLE_SET = {"trainval": "training", "test": "heldout"}


def _judges(cfg: Dict) -> List[str]:
    return [judge_key(e["name"]) for e in cfg["judges"]["models"]]


def snapshot(cfg: Dict) -> pd.DataFrame:
    ev = pd.read_parquet(require(cfg["sources"]["snapshot"], "the Radiopaedia snapshot"),
                         columns=["sentence_id", "article_title", "sentence_text", "matched_concepts_json",
                                  "evidence_split", "is_chest_relevant", "is_metadata_like", "is_fragment_like",
                                  "is_definition_like"])
    ev["sentence_id"] = ev["sentence_id"].astype(int)
    ev["article_set"] = ev["evidence_split"].map(ARTICLE_SET)
    if ev["article_set"].isna().any():
        raise ValueError(f"unknown evidence_split values: {sorted(ev.loc[ev['article_set'].isna(), 'evidence_split'].unique())}")
    ev["core"] = ev["matched_concepts_json"].map(core_findings)
    ev["clean"] = ~(ev["is_metadata_like"].astype(bool) | ev["is_fragment_like"].astype(bool)
                    | ev["is_definition_like"].astype(bool))
    return ev.sort_values("sentence_id").reset_index(drop=True)


def _row(r, pool: str, finding: str, state: str, rank: int) -> Dict:
    return {"pool": pool, "finding": finding, "state": state, "article_set": r.article_set,
            "sentence_id": int(r.sentence_id), "sentence_text": str(r.sentence_text),
            "article_title": str(r.article_title), "rank": rank}


def rule_pools(cfg: Dict) -> pd.DataFrame:
    o = pd.read_parquet(require(cfg["sources"]["oracle_mimic"], "the earlier data build"),
                        columns=["row_type", "concept", "claim_state", "evidence_sentence_ids_json",
                                 "evidence_sentences_json", "evidence_article_titles_json"])
    seen = {}
    for r in o[o["row_type"] == "C"].itertuples():
        for sid, s, t in zip(json.loads(r.evidence_sentence_ids_json), json.loads(r.evidence_sentences_json),
                             json.loads(r.evidence_article_titles_json)):
            seen.setdefault((r.concept, r.claim_state, int(sid)), (str(s), str(t)))
    rows = [{"pool": "rule", "finding": f, "state": s, "article_set": "training", "sentence_id": sid,
             "sentence_text": txt, "article_title": title, "rank": 0}
            for (f, s, sid), (txt, title) in sorted(seen.items())]
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby(["finding", "state"]).cumcount()
    return df


def _set_rule_article_sets(rule: pd.DataFrame, by_id: pd.DataFrame) -> pd.DataFrame:
    sets = rule["sentence_id"].map(by_id["article_set"])
    if sets.isna().any():
        raise ValueError(f"{int(sets.isna().sum())} rule-selected sentences are not in the snapshot")
    return rule.assign(article_set=sets.to_numpy())


def build_pools(cfg: Dict) -> Dict[str, pd.DataFrame]:
    v = cfg
    judges = _judges(cfg)
    jd = load_judgments(cfg)
    ev = snapshot(cfg)
    by_id = ev.set_index("sentence_id")
    jd["sentence_id"] = jd["sentence_id"].astype(int)
    for k in judges:
        if f"{k}__state" not in jd.columns:
            raise MissingInput(f"judge {k} is missing from {judgments_path(cfg)}; run main_judges with every judge")
    both = [jd[f"{k}__state"] for k in judges]
    jd["state"] = [a if all(x == a for x in xs) and a in STATES else "" for a, *xs in zip(*both)]
    unsure = pd.concat([(jd[f"{k}__state"] != "none") | (jd[f"{k}__status"] != "ok") for k in judges], axis=1).any(axis=1)
    jd["article_set"] = jd["sentence_id"].map(by_id["article_set"])
    eligible = jd["sentence_id"].map(by_id["clean"] & by_id["is_chest_relevant"].astype(bool))
    cand = jd[(jd["state"] != "") & eligible].sort_values(["finding", "state", "article_set", "sentence_id"]).copy()
    cand["rank"] = cand.groupby(["finding", "state", "article_set"]).cumcount()
    cand["in_pool"] = cand["rank"] < int(v["max_pool"])
    cand["sentence_text"] = cand["sentence_id"].map(by_id["sentence_text"])
    cand["article_title"] = cand["sentence_id"].map(by_id["article_title"])
    rows = [{"pool": "support", "finding": r.finding, "state": r.state, "article_set": r.article_set,
             "sentence_id": int(r.sentence_id), "sentence_text": r.sentence_text, "article_title": r.article_title,
             "rank": int(r.rank)} for r in cand[cand["in_pool"]].itertuples()]
    excluded = {(int(s), f) for s, f in zip(jd.loc[unsure, "sentence_id"], jd.loc[unsure, "finding"])}
    chest = ev[ev["clean"] & ev["is_chest_relevant"].astype(bool)]
    for f in v["findings"]:
        hard = chest[chest["core"].map(lambda c: f in c)]
        hard = hard[[(int(s), f) not in excluded for s in hard["sentence_id"]]]
        for aset, g in hard.groupby("article_set"):
            rows += [_row(r, "hard", f, "", i) for i, r in enumerate(g.itertuples())]
    easy = chest[chest["core"].map(len) == 0]
    for aset, g in easy.groupby("article_set"):
        rows += [_row(r, "easy", "", "", i) for i, r in enumerate(g.itertuples())]
    pools = pd.concat([pd.DataFrame(rows), _set_rule_article_sets(rule_pools(cfg), by_id)], ignore_index=True)
    k1 = []
    for f in v["findings"]:
        row = {"finding": f}
        for s in STATES:
            for aset in ("training", "heldout"):
                c = cand[(cand["finding"] == f) & (cand["state"] == s) & (cand["article_set"] == aset)]
                row[f"candidates_{s}_{aset}"] = int(len(c))
                row[f"pool_{s}_{aset}"] = int(c["in_pool"].sum())
        row["covered"] = bool(row["candidates_present_training"] + row["candidates_absent_training"] > 0)
        k1.append(row)
    k1 = pd.DataFrame(k1)
    keep = ["item_id", "sentence_id", "finding", "state", "article_set", "rank", "in_pool", "sentence_text",
            "article_title"] + [f"{k}__state" for k in judges]
    return {"pools": pools, "candidates": cand[keep].reset_index(drop=True), "coverage": k1}


def pools_path(cfg: Dict) -> str:
    return out(cfg, "support_pools", "pools.csv")


def main_support_pools(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    v = cfg
    out_dir = out(cfg, "support_pools")
    status = os.path.join(out_dir, "status.txt")
    side = {n: os.path.join(out_dir, f"{n}.csv") for n in ("candidates", "coverage")}
    jp = judgments_path(cfg)
    params = {"regime": _REGIME, "judgments": digest_file(jp) if os.path.exists(jp) else None,
              "snapshot": fingerprint(v["sources"]["snapshot"]), "oracle_mimic": fingerprint(v["sources"]["oracle_mimic"]),
              "max_pool": int(v["max_pool"]), "findings": list(v["findings"]), "judges": _judges(cfg)}

    def compute(unit):
        built = build_pools(cfg)
        for name, path in side.items():
            write_table(built[name], path)
        return built["pools"].to_dict("records")

    def valid(unit, done):
        return all(os.path.exists(p) for p in side.values())

    rows, refused = run_units("support_pools", [{"key": "pools", "params": params}], compute, out_dir, status, validate=valid)
    if refused:
        raise MissingInput("the pools could not be built yet")
    write_table(pd.DataFrame(rows), pools_path(cfg))
    flush_status(status, f"[support_pools] done, {len(rows)} pool rows")
    return out_dir


def load_pools(cfg: Dict) -> pd.DataFrame:
    path = pools_path(cfg)
    if not os.path.exists(path):
        raise MissingInput(f"{path} is missing; run main_support_pools first")
    return read_results_csv(path, keep_default_na=False)
