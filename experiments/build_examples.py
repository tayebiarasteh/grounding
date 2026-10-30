"""
experiments/build_examples.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments._common import (MissingInput, atomic_write_json, digest_file, fingerprint, flush_status, get_cfg,
                                 require, run_units)
from experiments.study import (ADDED_CONDITIONS, EXAMPLE_COLUMNS, LABEL_VARIANTS, STATES, TEST_SETS, claim_text,
                               examples_path, mesh_findings, opposite, out, read_openi, train_set_path, write_parquet)
from experiments.report_labels import labels_path, load_labels, report_studies
from experiments.judges import judge_key, load_judgments
from experiments.support_pools import load_pools, pools_path, snapshot

_REGIME = "build_examples-v4"
TRAIN_SETS = ("main", "noD", "noD_easy", "noD_random", "comparison", "food")


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=4).digest(), "big")


def _recs(df: pd.DataFrame) -> List[Dict]:
    return [{"sentence_id": int(r.sentence_id), "sentence_text": str(r.sentence_text),
             "article_title": str(r.article_title)} for r in df.sort_values("rank").itertuples()]


class Pools:

    def __init__(self, pools: pd.DataFrame, article_set: str, support_pool: str = "support"):
        p = pools[pools["article_set"] == article_set]
        sup = p[p["pool"] == support_pool]
        self.support = {(f, s): _recs(g) for (f, s), g in sup.groupby(["finding", "state"])}
        drop = set(sup["sentence_id"].astype(int))
        hard = p[(p["pool"] == "hard") & ~p["sentence_id"].astype(int).isin(drop)]
        self.hard = {f: _recs(g) for f, g in hard.groupby("finding")}
        self.easy = _recs(p[p["pool"] == "easy"])


def _sample_n(rng: random.Random, rows: List[Dict], n: int) -> List[Dict]:
    return list(rows) if len(rows) <= n else rng.sample(rows, n)


def _take(rng: random.Random, cands: List[Dict], k: int, source: str) -> Optional[Tuple[List[Dict], str]]:
    if not cands:
        return None
    return sorted(_sample_n(rng, cands, k), key=lambda x: x["sentence_id"]), source


def _nonsupport(rng: random.Random, hard: List[Dict], easy: List[Dict], k: int, hard_first: bool):
    first, second = (hard, easy) if hard_first else (easy, hard)
    mixed = _sample_n(rng, first, min(len(first), max(1, k)))
    if len(mixed) < k:
        mixed += _sample_n(rng, second, min(len(second), k - len(mixed)))
    seen, uniq = set(), []
    for r in mixed:
        key = (r["sentence_id"], r["sentence_text"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return _take(rng, uniq, k, "nonsupport_hard" if hard_first else "nonsupport_easy") if uniq else None


def _example(rf: Dict, row_type: str, state: str, pack, label: int, condition: str) -> Dict:
    sents, source = pack
    return {"qa_row_id": rf["qa_row_id"], "split": rf["split"], "subject_id": rf["subject_id"],
            "report_rel_path": rf["study_key"], "concept": rf["finding"], "gold_state": rf["state"],
            "claim_state": state, "claim_text": claim_text(rf["finding"], state), "case_context": rf["case_context"],
            "evidence_sentence_ids_json": json.dumps([r["sentence_id"] for r in sents]),
            "evidence_sentences_json": json.dumps([r["sentence_text"] for r in sents], ensure_ascii=False),
            "evidence_article_titles_json": json.dumps([r["article_title"] for r in sents], ensure_ascii=False),
            "row_type": row_type, "evidence_source_type": source, "label": "supported" if label else "not_supported",
            "label_id": int(label), "condition": condition}


def build_examples(findings: pd.DataFrame, pools: Pools, rng: random.Random, k: int, condition: str) -> List[Dict]:
    rows = []
    for rf in findings.to_dict("records"):
        f, s = rf["finding"], rf["state"]
        c = _take(rng, pools.support.get((f, s), []), k, f"support_{s}")
        if c:
            rows.append(_example(rf, "C", s, c, 1, condition))
        d = _take(rng, pools.support.get((f, opposite(s)), []), k, f"support_{opposite(s)}")
        if d:
            rows.append(_example(rf, "D", opposite(s), d, 0, condition))
        a = _nonsupport(rng, pools.hard.get(f, []), pools.easy, k, hard_first=True)
        if a:
            rows.append(_example(rf, "A", "present", a, 0, condition))
        b = _nonsupport(rng, pools.hard.get(f, []), pools.easy, k, hard_first=False)
        if b:
            rows.append(_example(rf, "B", "absent", b, 0, condition))
    return rows


def negative_examples(findings: pd.DataFrame, negatives, condition: str) -> List[Dict]:
    rows = []
    for rf in findings.to_dict("records"):
        a, b = negatives(rf)
        if a:
            rows.append(_example(rf, "A", "present", a, 0, condition))
        if b:
            rows.append(_example(rf, "B", "absent", b, 0, condition))
    return rows


def old_absent_counts(cfg: Dict, name: str) -> Dict[Tuple[str, str], int]:
    o = pd.read_parquet(require(cfg["sources"][f"oracle_{name}"], "the earlier data build"),
                        columns=["qa_row_id", "split", "concept", "gold_state"]).drop_duplicates("qa_row_id")
    o = o[o["gold_state"] == "absent"]
    return {(sp, f): int(n) for (sp, f), n in o.groupby(["split", "concept"]).size().items()}


def report_findings(cfg: Dict, name: str, labels: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    from Inference.chexbert_labeler import FINDING_HEAD
    v = cfg
    m = labels.merge(texts[["study_key", "findings_text"]], on="study_key", how="left", validate="one_to_one")
    if m["findings_text"].isna().any() or (m["findings_text"] == "").any():
        raise ValueError(f"{name}: a labeled study has no findings section")
    rows = []
    for r in m.to_dict("records"):
        for f in v["findings"]:
            st = r[f"state__{f}"]
            if st in STATES:
                code = int(r[f"chexbert_{FINDING_HEAD[f].replace(' ', '_')}"])
                rows.append({"dataset": name, "split": r["split"], "study_key": r["study_key"],
                             "subject_id": r["subject_id"], "finding": f, "state": st,
                             "report_code": {0: "blank", 1: "positive", 2: "negative"}.get(code, str(code)),
                             "case_context": r["findings_text"]})
    df = pd.DataFrame(rows).sort_values(["study_key", "finding"]).reset_index(drop=True)
    present, absent = df[df["state"] == "present"], df[df["state"] == "absent"]
    if name in ("mimic", "chexpert"):
        target = old_absent_counts(cfg, name)
    else:
        target = {(sp, f): int(n) for (sp, f), n in present.groupby(["split", "finding"]).size().items()}
    kept, short = [present], []
    for (sp, f), g in absent.groupby(["split", "finding"]):
        n = target.get((sp, f), 0)
        if len(g) < n:
            short.append(f"{sp} {f}: {len(g)} of {n}")
        if len(g) > n:
            rng = np.random.RandomState(stable_seed(v["seed"], name, sp, f))
            neg, blank = g[g["report_code"] == "negative"], g[g["report_code"] != "negative"]
            take_neg = min(n, len(neg))
            parts = [neg.iloc[np.sort(rng.choice(len(neg), take_neg, replace=False))],
                     blank.iloc[np.sort(rng.choice(len(blank), n - take_neg, replace=False))]]
            g = pd.concat(parts)
        kept.append(g)
    df = pd.concat(kept).sort_values(["study_key", "finding"]).reset_index(drop=True)
    df["qa_row_id"] = np.arange(len(df))
    return df


def mesh_report_findings(cfg: Dict, texts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    oi = read_openi(cfg["sources"]["openi_reports_dir"]).set_index("report_id")
    m = labels[["study_key", "subject_id", "split"]].merge(texts[["study_key", "findings_text"]], on="study_key",
                                                           how="left", validate="one_to_one")
    rows = []
    for r in m.to_dict("records"):
        coded = mesh_findings(oi.at[r["study_key"], "mesh_major_json"])
        if coded is None:
            continue
        for f in cfg["findings"]:
            st = "present" if f in coded else "absent"
            rows.append({"dataset": "openi_mesh", "split": r["split"], "study_key": r["study_key"],
                         "subject_id": r["subject_id"], "finding": f, "state": st,
                         "report_code": "coded" if st == "present" else "not_coded",
                         "case_context": r["findings_text"]})
    df = pd.DataFrame(rows).sort_values(["study_key", "finding"]).reset_index(drop=True)
    present, absent = df[df["state"] == "present"], df[df["state"] == "absent"]
    target = present.groupby(["split", "finding"]).size()
    kept = [present]
    for (sp, f), g in absent.groupby(["split", "finding"]):
        n = int(target.get((sp, f), 0))
        if len(g) > n:
            rng = np.random.RandomState(stable_seed(cfg["seed"], "openi_mesh", sp, f))
            g = g.iloc[np.sort(rng.choice(len(g), n, replace=False))]
        kept.append(g)
    df = pd.concat(kept).sort_values(["study_key", "finding"]).reset_index(drop=True)
    df["qa_row_id"] = np.arange(len(df))
    return df


def derangement(keys: List[str], rng: np.random.RandomState) -> np.ndarray:
    keys = np.asarray(keys, dtype=object)
    n = len(keys)
    top = max(Counter(keys.tolist()).values())
    if 2 * top > n:
        raise ValueError(f"one evidence list holds {top} of {n} examples, so no derangement on the text exists")
    perm = rng.permutation(n)
    for _ in range(100):
        bad = np.flatnonzero(keys[perm] == keys)
        if not len(bad):
            return perm
        for i in rng.permutation(bad):
            if keys[perm[i]] != keys[i]:
                continue
            for j in rng.permutation(n):
                if keys[perm[j]] != keys[i] and keys[perm[i]] != keys[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    raise RuntimeError("the derangement repair did not converge")


def replaced(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    keys = df["evidence_sentences_json"].tolist()
    perm = derangement(keys, np.random.RandomState(seed))
    out_ = df.copy()
    for c in ("evidence_sentence_ids_json", "evidence_sentences_json", "evidence_article_titles_json"):
        out_[c] = df[c].to_numpy()[perm]
    out_["evidence_source_type"] = "replaced"
    out_["condition"] = "replaced"
    if (out_["evidence_sentences_json"].to_numpy() == df["evidence_sentences_json"].to_numpy()).any():
        raise AssertionError("an example kept its own evidence text under replacement")
    return out_


def report_replaced(test: pd.DataFrame, rf_test: pd.DataFrame, seed: int, name: str) -> Tuple[pd.DataFrame, Dict]:
    new, donor = {}, {}
    n_other, left = 0, []
    for (f, st), g in rf_test.groupby(["finding", "state"], sort=True):
        for code, h in g.groupby("report_code", sort=True):
            keys = h["subject_id"].astype(str).tolist()
            if len(h) >= 2 and 2 * max(Counter(keys).values()) <= len(h):
                perm = derangement(keys, np.random.RandomState(stable_seed(seed, name, "report", f, st, code)))
                new.update(zip(h["qa_row_id"], h["case_context"].to_numpy()[perm]))
                donor.update(zip(h["qa_row_id"], np.asarray(keys, dtype=object)[perm]))
                continue
            rng = np.random.RandomState(stable_seed(seed, name, "report", f, st, code, "other_classes"))
            for r in h.itertuples(index=False):
                pool = g[g["subject_id"].astype(str) != str(r.subject_id)]
                if not len(pool):
                    left.append(r.qa_row_id)
                    continue
                pick = pool.iloc[rng.randint(len(pool))]
                new[r.qa_row_id], donor[r.qa_row_id] = pick["case_context"], str(pick["subject_id"])
                n_other += 1
    own = dict(zip(rf_test["qa_row_id"], rf_test["subject_id"].astype(str)))
    if any(donor[q] == own[q] for q in donor):
        raise AssertionError(f"{name}: a report finding received a report of its own patient")
    out_ = test[test["qa_row_id"].isin(set(new))].copy()
    before = out_["case_context"].to_numpy()
    out_["case_context"] = out_["qa_row_id"].map(new).to_numpy()
    out_["condition"] = "report_replaced"
    counts = {"report_findings": int(len(rf_test)), "replaced_within_class": int(len(new) - n_other),
              "replaced_from_other_classes": int(n_other), "left_out": int(len(left)),
              "examples_with_identical_text": int((out_["case_context"].to_numpy() == before).sum())}
    return out_.reset_index(drop=True), counts


def removed(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(evidence_sentence_ids_json="[]", evidence_sentences_json="[]", evidence_article_titles_json="[]",
                     evidence_source_type="removed", condition="removed")


def masked(df: pd.DataFrame, token: str, statement: bool) -> pd.DataFrame:
    from data_loader.verifier_dataset import mask_concepts
    out_ = df.copy()
    out_["evidence_sentences_json"] = [json.dumps([mask_concepts(x, token) for x in json.loads(js)], ensure_ascii=False)
                                       for js in df["evidence_sentences_json"]]
    if statement:
        out_["claim_text"] = [mask_concepts(c, token) for c in df["claim_text"]]
    out_["condition"] = "masked_evidence_statement" if statement else "masked_evidence"
    return out_


def _ids(frames: List[pd.DataFrame]) -> Dict[Tuple, int]:
    keys = sorted({(r, f, t) for d in frames for r, f, t in zip(d["report_rel_path"], d["concept"], d["row_type"])})
    return {k: i for i, k in enumerate(keys)}


def _finish(df: pd.DataFrame, dataset: str, ids: Dict) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset
    df["verifier_row_id"] = [ids[(r, f, t)] for r, f, t in zip(df["report_rel_path"], df["concept"], df["row_type"])]
    df = df.sort_values("verifier_row_id").reset_index(drop=True)
    if df["verifier_row_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicated example ids within one condition")
    return df[EXAMPLE_COLUMNS]


def _random_negatives(cfg: Dict, rng: random.Random, k: int):
    ev = snapshot(cfg)
    ev = ev[ev["clean"] & (ev["article_set"] == "training")]
    jd = load_judgments(cfg)
    keys = [judge_key(e["name"]) for e in cfg["judges"]["models"]]
    flagged = jd[pd.concat([(jd[f"{j}__state"] != "none") | (jd[f"{j}__status"] != "ok") for j in keys], axis=1)
                 .any(axis=1)]
    drop = {f: set(g["sentence_id"].astype(int)) for f, g in flagged.groupby("finding")}
    universe = _recs(ev.assign(rank=np.arange(len(ev))))
    by_f = {f: [r for r in universe if r["sentence_id"] not in drop.get(f, set())] for f in cfg["findings"]}

    def neg(rf):
        u = by_f[rf["finding"]]
        return (_take(rng, _sample_n(rng, u, k), k, "random_universe"),
                _take(rng, _sample_n(rng, u, k), k, "random_universe"))
    return neg


def build_all(cfg: Dict) -> Dict[str, pd.DataFrame]:
    v = cfg
    if set(TEST_SETS) | {"food"} | set(LABEL_VARIANTS) != set(v["test_sets"]):
        raise ValueError(f"the config's test sets {sorted(v['test_sets'])} differ from the sets built here")
    unknown = {r["train_set"] for r in v["runs"]} - set(TRAIN_SETS)
    if unknown:
        raise ValueError(f"runs name training sets that are not built here: {sorted(unknown)}")
    k, seed = int(v["max_evidence_sentences"]), int(v["seed"])
    pools = load_pools(cfg)
    train_p, held_p, rule_p = Pools(pools, "training"), Pools(pools, "heldout"), Pools(pools, "training", "rule")
    texts = report_studies(cfg)
    outs, findings, notes = {}, [], {}
    for name in TEST_SETS:
        rf = report_findings(cfg, name, load_labels(cfg, name), texts[name])
        findings.append(rf)
        rng = random.Random(f"{seed}|{name}|training")
        unchanged = pd.DataFrame(build_examples(rf, train_p, rng, k, "unchanged"))
        test_rf = rf[rf["split"] == "test"]
        held = pd.DataFrame(build_examples(test_rf, held_p, random.Random(f"{seed}|{name}|heldout"), k, "heldout"))
        tv_rf = rf[rf["split"].isin(["train", "val"])]
        comp = pd.DataFrame(build_examples(tv_rf, rule_p, random.Random(f"{seed}|{name}|training|comparison"), k,
                                           "unchanged")) if name == "mimic" else None
        ids = _ids([d for d in (unchanged, held, comp) if d is not None])
        unchanged, held = _finish(unchanged, name, ids), _finish(held, name, ids)
        test = unchanged[unchanged["split"] == "test"].reset_index(drop=True)
        outs[f"{name}__unchanged"] = test
        outs[f"{name}__heldout"] = held
        outs[f"{name}__replaced"] = replaced(test, seed)
        outs[f"{name}__removed"] = removed(test)
        if "report_replaced" in v["test_sets"][name]:
            rr, notes[f"{name}__report_replaced"] = report_replaced(test, rf[rf["split"] == "test"], seed, name)
            outs[f"{name}__report_replaced"] = rr
        if name != "mimic":
            continue
        outs["mimic__masked_evidence"] = masked(test, v["mask_token"], statement=False)
        outs["mimic__masked_evidence_statement"] = masked(test, v["mask_token"], statement=True)
        main = unchanged[unchanged["split"].isin(["train", "val"])].reset_index(drop=True)
        c_rows = main[main["row_type"] == "C"]
        outs["train__main"] = main
        outs["train__noD"] = main[main["row_type"] != "D"].reset_index(drop=True)
        easy_rng = random.Random(f"{seed}|mimic|training|noD_easy")
        easy = negative_examples(tv_rf, lambda r: (_take(easy_rng, train_p.easy, k, "nonsupport_easy"),
                                                   _take(easy_rng, train_p.easy, k, "nonsupport_easy")), "unchanged")
        easy = _finish(pd.DataFrame(easy), name, ids)
        outs["train__noD_easy"] = pd.concat([c_rows, easy]).sort_values("verifier_row_id").reset_index(drop=True)
        rnd_rng = random.Random(f"{seed}|mimic|training|noD_random")
        rnd = _finish(pd.DataFrame(negative_examples(tv_rf, _random_negatives(cfg, rnd_rng, k), "unchanged")), name,
                      ids)
        outs["train__noD_random"] = pd.concat([c_rows, rnd]).sort_values("verifier_row_id").reset_index(drop=True)
        outs["train__comparison"] = _finish(comp, name, ids)
    rf_mesh = mesh_report_findings(cfg, texts["openi"], load_labels(cfg, "openi"))
    findings.append(rf_mesh)
    mesh = pd.DataFrame(build_examples(rf_mesh, train_p, random.Random(f"{seed}|openi_mesh|training"), k, "unchanged"))
    outs["openi_mesh__unchanged"] = _finish(mesh, "openi_mesh", _ids([mesh]))
    food = pd.read_parquet(require(v["sources"]["food_trainvaltest"], "the food data build of the earlier version"))
    food = food.assign(subject_id=food["subject_id"].astype(str), report_rel_path=food["report_rel_path"].astype(str),
                       dataset="food", condition="unchanged")[EXAMPLE_COLUMNS]
    outs["train__food"] = food[food["split"].isin(["train", "val"])].reset_index(drop=True)
    ftest = food[food["split"] == "test"].sort_values("verifier_row_id").reset_index(drop=True)
    outs["food__unchanged"] = ftest
    outs["food__replaced"] = replaced(ftest, seed)
    outs["food__removed"] = removed(ftest)
    outs["report_findings"] = pd.concat(findings, ignore_index=True)
    missing = [t for t in TRAIN_SETS if f"train__{t}" not in outs]
    if missing:
        raise RuntimeError(f"the build wrote no training set for {missing}")
    unbuilt = [f"{d}__{c}" for d, cs in v["test_sets"].items() for c in cs
               if c not in ADDED_CONDITIONS and f"{d}__{c}" not in outs]
    if unbuilt:
        raise RuntimeError(f"the build wrote no example set for {unbuilt}")
    return outs, notes


def output_path(cfg: Dict, stem: str) -> str:
    if stem == "report_findings":
        return out(cfg, "examples", "report_findings.parquet")
    if stem.startswith("train__"):
        return train_set_path(cfg, stem[len("train__"):])
    ds, cond = stem.split("__", 1)
    return examples_path(cfg, ds, cond)


def _counts(stem: str, df: pd.DataFrame) -> List[Dict]:
    if stem == "report_findings":
        g = df.groupby(["dataset", "split", "finding", "state", "report_code"]).size()
        return [{"file": stem, "dataset": d, "split": sp, "finding": f, "state": s, "report_code": c, "row_type": "",
                 "n": int(n)} for (d, sp, f, s, c), n in g.items()]
    g = df.groupby(["dataset", "split", "concept", "gold_state", "row_type"]).size()
    return [{"file": stem, "dataset": d, "split": sp, "finding": f, "state": s, "report_code": "", "row_type": t,
             "n": int(n)} for (d, sp, f, s, t), n in g.items()]


def _inputs(cfg: Dict) -> Dict:
    src = cfg["sources"]
    files = {name: labels_path(cfg, name) for name in TEST_SETS}
    files["pools"] = pools_path(cfg)
    files["judgments"] = out(cfg, "judges", "judgments.csv")
    files["food"] = src["food_trainvaltest"]
    rec = {n: digest_file(p) if os.path.exists(p) else None for n, p in files.items()}
    oi = src["openi_reports_dir"]
    mesh = read_openi(oi)[["report_id", "mesh_major_json"]].to_json(orient="values") if os.path.isdir(oi) else None
    rec["openi_mesh"] = hashlib.blake2b(mesh.encode("utf-8"), digest_size=16).hexdigest() if mesh else None
    for n in ("snapshot", "oracle_mimic", "oracle_chexpert", "mimic_sections", "chexpert_sections",
              "rexgradient_masterlist", "rexgradient_sample"):
        rec[n] = fingerprint(src[n]) if os.path.exists(src[n]) else None
    return rec


def main_build_examples(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    v = cfg
    out_dir = out(cfg, "examples")
    status = os.path.join(out_dir, "status.txt")
    params = {"regime": _REGIME, "inputs": _inputs(cfg), "seed": int(v["seed"]), "max_pool": int(v["max_pool"]),
              "max_evidence_sentences": int(v["max_evidence_sentences"]), "findings": list(v["findings"]),
              "mask_token": v["mask_token"]}

    def compute(unit):
        for n, d in unit["params"]["inputs"].items():
            if d is None:
                raise MissingInput(f"the input '{n}' of main_build_examples is missing; run main_chexbert_labels to main_support_pools first")
        built, notes = build_all(cfg)
        rows = []
        for stem, df in tqdm(built.items(), desc="build_examples write", unit="file"):
            write_parquet(df, output_path(cfg, stem))
            rows += _counts(stem, df)
        manifest = {"params": unit["params"], "files": {s: {"path": output_path(cfg, s), "rows": int(len(d))}
                                                        for s, d in built.items()}, "notes": notes}
        atomic_write_json(manifest, os.path.join(out_dir, "manifest.json"))
        return rows

    def valid(unit, done):
        for stem, g in done.groupby("file"):
            path = output_path(cfg, stem)
            if not os.path.exists(path):
                return False
            import pyarrow.parquet as pq
            if pq.ParquetFile(path).metadata.num_rows != int(g["n"].sum()):
                return False
        return True

    _, refused = run_units("build_examples", [{"key": "build", "params": params}], compute, out_dir, status, validate=valid)
    if refused:
        raise MissingInput("the example sets could not be built yet")
    flush_status(status, "[build_examples] done")
    return out_dir
