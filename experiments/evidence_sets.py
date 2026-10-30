"""
experiments/evidence_sets.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import math
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments._common import (MissingInput, atomic_write_json, digest_file, flush_status, get_cfg, read_results_csv,
                                 require, run_units, write_table)
from experiments.study import (ADDED_CONDITIONS, ADDED_FOLDER, ADDED_TRAIN_SETS, STATES, examples_path, out,
                               read_evidence, train_set_path, write_parquet)
from experiments.judges import judge_key
from experiments.support_pools import load_pools, pools_path
from experiments.build_examples import Pools, _finish, build_examples, output_path
from experiments.statements import _generators, _with_evidence, statement_name, statement_path
from experiments.written_sentences import candidates_path, load_written_judgments, norm_sentence, written_judgments_path

_REGIME = "evidence_sets-v1"
WRITTEN_ID_BASE = 10_000_000
CITATIONS = ("heldout", "retrieved")
FORMS = ("template", "raw")
_WORD = re.compile(r"[a-z0-9]+")


def added_statement_sources(cfg: Dict) -> List[str]:
    return ["rexerr"] + [f"gen__{g}" for g in _generators(cfg)]


def added_statement_set_names(cfg: Dict) -> List[str]:
    return [statement_name(src, form, cit) for src in added_statement_sources(cfg) for cit in CITATIONS for form in FORMS]


def is_added_statement(name: str) -> bool:
    if not name.startswith("stmt__"):
        return False
    source = "__".join(name.split("__")[1:-1])
    return any(source.endswith(f"_{cit}") for cit in CITATIONS)


def added_statement_path(cfg: Dict, name: str) -> str:
    return out(cfg, ADDED_FOLDER, f"{name}.parquet")


def pools_added_path(cfg: Dict) -> str:
    return out(cfg, ADDED_FOLDER, "pools_added.csv")


def words(text: str) -> set:
    return set(_WORD.findall(str(text).lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def max_jaccard(w: set, others: List[set]) -> float:
    return max((jaccard(w, o) for o in others), default=0.0)


def _rows_of(df: pd.DataFrame, pool: str, article_set: str) -> List[Dict]:
    return [{"pool": pool, "finding": r["finding"], "state": r["state"], "article_set": article_set,
             "sentence_id": int(r["sentence_id"]), "sentence_text": str(r["sentence_text"]),
             "article_title": str(r["article_title"]), "rank": int(r["rank"])} for r in df.to_dict("records")]


def uncapped_support(cfg: Dict) -> pd.DataFrame:
    c = read_results_csv(require(out(cfg, "support_pools", "candidates.csv"), "main_support_pools"), keep_default_na=False)
    c = c[c["article_set"] == "training"].copy()
    c["sentence_id"] = c["sentence_id"].astype(int)
    c["rank"] = c["rank"].astype(int)
    return pd.DataFrame(_rows_of(c.sort_values(["finding", "state", "rank"]), "support", "training"))


def written_pools(cfg: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    from experiments.support_pools import snapshot
    from experiments.controls import _corpus
    v, ws = cfg, cfg["written"]
    jd = load_written_judgments(cfg)
    keys = [judge_key(e["name"]) for e in v["judges"]["models"]]
    for k in keys:
        if f"{k}__state" not in jd.columns:
            raise MissingInput(f"judge {k} is missing from {written_judgments_path(cfg)}; run main_write_sentences with every judge")
    ok = np.ones(len(jd), bool)
    for k in keys:
        ok &= (jd[f"{k}__status"] == "ok").to_numpy()
    states = [jd[f"{k}__state"].to_numpy() for k in keys]
    agree = np.ones(len(jd), bool)
    for s in states[1:]:
        agree &= s == states[0]
    assigned = np.where(ok & agree, states[0], "")
    target = jd["state"].to_numpy()
    kind = jd["kind"].to_numpy()
    sup = jd[(kind != "unsupporting") & (assigned == target) & np.isin(target, STATES)].copy()
    hard = jd[(kind == "unsupporting") & (assigned == "none")].copy()
    snap = snapshot(cfg)
    snap_norm = set(snap["sentence_text"].map(norm_sentence))
    held = _corpus(cfg)
    pools = load_pools(cfg)
    held_pool = pools[(pools["pool"] == "support") & (pools["article_set"] == "heldout")]
    held_words = [words(t) for t in pd.concat([held["sentence_text"], held_pool["sentence_text"]]).astype(str)]
    leak, near = float(ws["leak_jaccard"]), float(ws["near_duplicate_jaccard"])
    kept_words: List[set] = []
    counts = {"copies": 0, "leaks": 0, "near_duplicates": 0}
    keep = []
    frame = pd.concat([sup.assign(_pool="support_written"), hard.assign(_pool="hard_written")], ignore_index=True)
    frame = frame.sort_values(["_pool", "finding", "state", "kind", "sentence_text"]).reset_index(drop=True)
    for text in tqdm(frame["sentence_text"].astype(str), desc="written leakage filter", unit="sentence"):
        w = words(text)
        if norm_sentence(text) in snap_norm:
            counts["copies"] += 1
            keep.append(False)
        elif max_jaccard(w, held_words) >= leak:
            counts["leaks"] += 1
            keep.append(False)
        elif max_jaccard(w, kept_words) >= near:
            counts["near_duplicates"] += 1
            keep.append(False)
        else:
            kept_words.append(w)
            keep.append(True)
    frame = frame[np.asarray(keep)].reset_index(drop=True)
    frame["sentence_id"] = WRITTEN_ID_BASE + np.arange(len(frame))
    frame["article_title"] = f"written by {v['writer']['served']}"
    frac, min_h, min_n = float(ws["heldout_fraction"]), int(ws["min_heldout"]), int(ws["min_for_heldout"])
    rows, per_state = [], []
    for (f, s), g in frame[frame["_pool"] == "support_written"].groupby(["finding", "state"], sort=True):
        g = g.sort_values("sentence_id")
        n = len(g)
        held_idx = set()
        if n >= min_n:
            rng = random.Random(f"{int(v['seed'])}|written_heldout|{f}|{s}")
            held_idx = set(rng.sample(range(n), max(min_h, int(round(frac * n)))))
        for i, r in enumerate(g.to_dict("records")):
            pool = "support_written_heldout" if i in held_idx else "support_written"
            rows.append({"pool": pool, "finding": f, "state": s, "article_set": "written", "sentence_id": int(r["sentence_id"]),
                         "sentence_text": str(r["sentence_text"]), "article_title": r["article_title"], "rank": i,
                         "kind": r["kind"], "item_id": r["item_id"]})
        per_state.append({"finding": f, "state": s, "written_training": n - len(held_idx), "written_heldout": len(held_idx)})
    for f, g in frame[frame["_pool"] == "hard_written"].groupby("finding", sort=True):
        for i, r in enumerate(g.sort_values("sentence_id").to_dict("records")):
            rows.append({"pool": "hard_written", "finding": f, "state": "", "article_set": "written",
                         "sentence_id": int(r["sentence_id"]), "sentence_text": str(r["sentence_text"]),
                         "article_title": r["article_title"], "rank": i, "kind": r["kind"], "item_id": r["item_id"]})
    return pd.DataFrame(rows), per_state


def level_pools(cfg: Dict, level: int, written: pd.DataFrame) -> Pools:
    pools = load_pools(cfg)
    base = pools[(pools["article_set"] == "training") & pools["pool"].isin(["hard", "easy"])]
    parts = [base[["pool", "finding", "state", "article_set", "sentence_id", "sentence_text", "article_title", "rank"]],
             uncapped_support(cfg)]
    if level == 3:
        w = written.copy()
        w = w[w["pool"].isin(["support_written", "hard_written"])]
        w["pool"] = w["pool"].map({"support_written": "support", "hard_written": "hard"})
        w["article_set"] = "training"
        offset = pd.concat(parts).groupby(["pool", "finding", "state"])["rank"].max().add(1)
        w["rank"] = [int(r["rank"]) + int(offset.get((r["pool"], r["finding"], r["state"]), 0)) for r in w.to_dict("records")]
        parts.append(w[["pool", "finding", "state", "article_set", "sentence_id", "sentence_text", "article_title", "rank"]])
    return Pools(pd.concat(parts, ignore_index=True), "training")


def written_heldout_pools(cfg: Dict, written: pd.DataFrame) -> Pools:
    pools = load_pools(cfg)
    base = pools[(pools["article_set"] == "heldout") & pools["pool"].isin(["hard", "easy"])].copy()
    w = written[written["pool"] == "support_written_heldout"].copy()
    w["pool"] = "support"
    cols = ["pool", "finding", "state", "article_set", "sentence_id", "sentence_text", "article_title", "rank"]
    frame = pd.concat([base[cols], w[cols]], ignore_index=True)
    frame["article_set"] = "written"
    return Pools(frame, "written")


def _evidence_json(sents: List[Dict]) -> Dict:
    return {"evidence_sentence_ids_json": json.dumps([r["sentence_id"] for r in sents]),
            "evidence_sentences_json": json.dumps([r["sentence_text"] for r in sents], ensure_ascii=False),
            "evidence_article_titles_json": json.dumps([r["article_title"] for r in sents], ensure_ascii=False)}


def mix_evidence(rows: List[Dict], pools: Pools, rng: random.Random, share: float) -> int:
    n = 0
    for row in rows:
        if row["row_type"] not in ("C", "D") or rng.random() >= share:
            continue
        hard = pools.hard.get(row["concept"], [])
        if not hard:
            continue
        ids = json.loads(row["evidence_sentence_ids_json"])
        texts, titles = json.loads(row["evidence_sentences_json"]), json.loads(row["evidence_article_titles_json"])
        pair = [{"sentence_id": ids[0], "sentence_text": texts[0], "article_title": titles[0]}, rng.choice(hard)]
        rng.shuffle(pair)
        row.update(_evidence_json(pair))
        row["evidence_source_type"] = row["evidence_source_type"] + "_mixed"
        n += 1
    return n


def _ids_like(base: pd.DataFrame, frames: List[pd.DataFrame]) -> Dict[Tuple, int]:
    ids = {(r, f, t): int(i) for r, f, t, i in zip(base["report_rel_path"], base["concept"], base["row_type"],
                                                   base["verifier_row_id"])}
    new = sorted({(r, f, t) for d in frames for r, f, t in zip(d["report_rel_path"], d["concept"], d["row_type"])}
                 - set(ids))
    nxt = max(ids.values()) + 1 if ids else 0
    for key in new:
        ids[key] = nxt
        nxt += 1
    return ids


def report_findings_of(cfg: Dict, splits: List[str]) -> pd.DataFrame:
    rf = pd.read_parquet(require(output_path(cfg, "report_findings"), "main_build_examples"))
    rf = rf[(rf["dataset"] == "mimic") & rf["split"].isin(splits)]
    return rf.sort_values("qa_row_id").reset_index(drop=True)


def build_training_sets(cfg: Dict, written: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    v = cfg
    k, seed, share = int(v["max_evidence_sentences"]), int(v["seed"]), float(v["written"]["mixed_share"])
    rf = report_findings_of(cfg, ["train", "val"])
    base = pd.read_parquet(require(train_set_path(cfg, "main"), "main_build_examples"))
    outs, notes = {}, {}
    for name, level, mixed in (("main_uncapped", 2, 0.0), ("main_written", 3, share)):
        pools = level_pools(cfg, level, written)
        rng = random.Random(f"{seed}|mimic|training|{name}")
        rows = build_examples(rf, pools, rng, k, "unchanged")
        n_mixed = mix_evidence(rows, pools, rng, mixed) if mixed > 0 else 0
        df = pd.DataFrame(rows)
        df = _finish(df, "mimic", _ids_like(base, [df]))
        outs[f"train__{name}"] = df
        notes[name] = {"rows": int(len(df)), "mixed": n_mixed,
                       "support_sentences": int(sum(len(x) for x in pools.support.values())),
                       "hard_sentences": int(sum(len(x) for x in pools.hard.values())),
                       "states_with_support": sorted(f"{f}|{s}" for (f, s) in pools.support)}
    main_w = outs["train__main_written"]
    outs["train__noD_written"] = main_w[main_w["row_type"] != "D"].reset_index(drop=True)
    return outs, notes


def build_heldout_written(cfg: Dict, written: pd.DataFrame) -> pd.DataFrame:
    v = cfg
    k, seed = int(v["max_evidence_sentences"]), int(v["seed"])
    rf = report_findings_of(cfg, ["test"])
    pools = written_heldout_pools(cfg, written)
    rows = build_examples(rf, pools, random.Random(f"{seed}|mimic|heldout_written"), k, "heldout_written")
    df = pd.DataFrame(rows)
    unchanged = pd.read_parquet(require(examples_path(cfg, "mimic", "unchanged"), "main_build_examples"))
    held = pd.read_parquet(require(examples_path(cfg, "mimic", "heldout"), "main_build_examples"))
    ids = _ids_like(pd.concat([unchanged, held]).drop_duplicates(["report_rel_path", "concept", "row_type"]), [df])
    return _finish(df, "mimic", ids)


def heldout_evidence(cfg: Dict) -> Dict:
    k = int(cfg["max_evidence_sentences"])
    c = pd.read_parquet(require(examples_path(cfg, "mimic", "heldout"), "main_build_examples"))
    c = c[c["row_type"] == "C"].copy()
    c["key"] = c["evidence_sentences_json"].map(lambda js: json.dumps(read_evidence(js, k)))
    outd = {}
    for (f, s), g in c.groupby(["concept", "claim_state"]):
        counts = g.groupby("key").size().reset_index(name="n").sort_values(["n", "key"], ascending=[False, True])
        row = g[g["key"] == counts["key"].iloc[0]].iloc[0]
        outd[(f, s)] = {col: row[col] for col in ("evidence_sentence_ids_json", "evidence_sentences_json",
                                                  "evidence_article_titles_json")}
    return outd


class BM25Index:

    def __init__(self, docs: List[str], k1: float = 1.2, b: float = 0.75):
        import scipy.sparse as sp
        toks = [_WORD.findall(d.lower()) for d in docs]
        self.vocab: Dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, ts in enumerate(toks):
            counts: Dict[str, int] = {}
            for t in ts:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                rows.append(i)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
                vals.append(float(c))
        n = len(docs)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)))
        dl = np.asarray([len(ts) for ts in toks], dtype=float)
        avgdl = float(dl.mean())
        dfreq = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.asarray([math.log(1.0 + (n - int(d) + 0.5) / (int(d) + 0.5)) for d in dfreq])
        norm = k1 * (1.0 - b + b * dl / avgdl)
        tf = tf.tocoo()
        w = idf[tf.col] * tf.data * (k1 + 1.0) / (tf.data + norm[tf.row])
        self.weights = sp.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.n = n

    def rank(self, query: str) -> List[int]:
        cols = [self.vocab[t] for t in _WORD.findall(query.lower()) if t in self.vocab]
        if not cols:
            return []
        scores = np.zeros(self.n)
        for c in cols:
            scores += self.weights[:, c].toarray().ravel()
        order = np.lexsort((np.arange(self.n), -scores))
        return [int(i) for i in order if scores[i] > 0]


def build_statement_sets(cfg: Dict) -> Dict[str, pd.DataFrame]:
    from experiments.controls import _corpus
    hev = heldout_evidence(cfg)
    corpus = _corpus(cfg)
    texts = corpus["sentence_text"].astype(str).tolist()
    sids = corpus["sentence_id"].astype(int).to_numpy()
    titles = corpus["article_title"].astype(str).to_numpy()
    index = BM25Index(texts)
    from experiments.study import claim_text
    templates = sorted({claim_text(f, s) for f in cfg["findings"] for s in STATES})
    bad = rank_check(index, texts, templates)
    if bad:
        raise RuntimeError(f"[main_build_evidence_sets] the sparse BM25 index disagrees with bm25_rank on {bad} of {len(templates)} queries")
    pools = load_pools(cfg)
    held = pools[(pools["pool"] == "support") & (pools["article_set"] == "heldout")]
    pool_ids = {(f, s): set(g["sentence_id"].astype(int)) for (f, s), g in held.groupby(["finding", "state"])}
    outs = {}
    for src in added_statement_sources(cfg):
        for form in FORMS:
            base = pd.read_parquet(require(statement_path(cfg, statement_name(src, form)), "main_statements"))
            h = _with_evidence(base, hev)
            h["evidence_source_type"], h["condition"] = "heldout_pool", "statement_heldout"
            outs[statement_name(src, form, "heldout")] = h.reset_index(drop=True)
            r = base.copy()
            picks = {q: index.rank(q)[:2] for q in sorted(r["claim_text"].astype(str).unique())}
            ev = [picks[q] for q in r["claim_text"].astype(str)]
            r["evidence_sentence_ids_json"] = [json.dumps([int(sids[i]) for i in idx]) for idx in ev]
            r["evidence_sentences_json"] = [json.dumps([texts[i] for i in idx], ensure_ascii=False) for idx in ev]
            r["evidence_article_titles_json"] = [json.dumps([titles[i] for i in idx], ensure_ascii=False) for idx in ev]
            hit = np.asarray([int(any(int(sids[i]) in pool_ids.get((f, s), set()) for i in idx))
                              for idx, f, s in zip(ev, r["concept"], r["claim_state"])])
            r["recall_hit"] = hit
            r["label_hit"] = hit.astype(float)
            r["label_full"] = r["label_kappa"].to_numpy(float) * hit
            r["evidence_source_type"], r["condition"] = "retrieval_bm25_top2", "statement_retrieved"
            outs[statement_name(src, form, "retrieved")] = r.reset_index(drop=True)
    return outs


def _inputs(cfg: Dict) -> Dict:
    def dig(p):
        return digest_file(p) if os.path.exists(p) else None
    rec = {"pools": dig(pools_path(cfg)), "candidates": dig(out(cfg, "support_pools", "candidates.csv")),
           "written_candidates": dig(candidates_path(cfg)), "written_judgments": dig(written_judgments_path(cfg)),
           "report_findings": dig(output_path(cfg, "report_findings")), "train_main": dig(train_set_path(cfg, "main")),
           "mimic_unchanged": dig(examples_path(cfg, "mimic", "unchanged")),
           "mimic_heldout": dig(examples_path(cfg, "mimic", "heldout"))}
    for src in added_statement_sources(cfg):
        for form in FORMS:
            rec[statement_name(src, form)] = dig(statement_path(cfg, statement_name(src, form)))
    return rec


def main_build_evidence_sets(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    v = cfg
    from experiments.build_examples import TRAIN_SETS
    unknown = {r["train_set"] for r in v.get("runs_added", [])} - set(ADDED_TRAIN_SETS) - set(TRAIN_SETS)
    if unknown:
        raise ValueError(f"runs_added name training sets that are not built: {sorted(unknown)}")
    out_dir = out(cfg, ADDED_FOLDER)
    status = os.path.join(out_dir, "status.txt")
    params = {"regime": _REGIME, "inputs": _inputs(cfg), "seed": int(v["seed"]),
              "max_evidence_sentences": int(v["max_evidence_sentences"]), "written": v["written"],
              "writer": v["writer"]["served"], "findings": list(v["findings"])}

    def compute(unit):
        for n, d in unit["params"]["inputs"].items():
            if d is None:
                raise MissingInput(f"the input '{n}' of main_build_evidence_sets is missing; run main_support_pools, main_build_examples, main_statements and main_write_sentences first")
        written, per_state = written_pools(cfg)
        rows = []
        pools_table = pd.concat([uncapped_support(cfg).assign(kind="real", item_id=""), written], ignore_index=True)
        write_table(pools_table, pools_added_path(cfg))
        rows += [{"file": "pools_added", "pool": p, "finding": f, "state": s, "n": int(n)}
                 for (p, f, s), n in pools_table.groupby(["pool", "finding", "state"]).size().items()]
        built, notes = build_training_sets(cfg, written)
        built["mimic__heldout_written"] = build_heldout_written(cfg, written)
        built.update(build_statement_sets(cfg))
        missing = [t for t in ADDED_TRAIN_SETS if f"train__{t}" not in built] + [s for s in added_sets(cfg) if s not in built]
        if missing:
            raise RuntimeError(f"the build wrote no file for {missing}")
        for stem, df in tqdm(built.items(), desc="evidence_sets write", unit="file"):
            write_parquet(df, output_path_added(cfg, stem))
            rows.append({"file": stem, "pool": "", "finding": "", "state": "", "n": int(len(df))})
        manifest = {"params": unit["params"], "files": {s: {"path": output_path_added(cfg, s), "rows": int(len(d))}
                                                        for s, d in built.items()}, "notes": notes,
                    "written_per_state": per_state}
        atomic_write_json(manifest, os.path.join(out_dir, "manifest.json"))
        return rows

    def valid(unit, done):
        import pyarrow.parquet as pq
        if not os.path.exists(pools_added_path(cfg)):
            return False
        for stem, g in done[done["file"] != "pools_added"].groupby("file"):
            path = output_path_added(cfg, stem)
            if not os.path.exists(path) or pq.ParquetFile(path).metadata.num_rows != int(g["n"].sum()):
                return False
        return True

    _, refused = run_units("main_build_evidence_sets", [{"key": "build", "params": params}], compute, out_dir, status, validate=valid)
    if refused:
        raise MissingInput("the additions could not be built yet")
    flush_status(status, "[main_build_evidence_sets] done")
    return out_dir


def output_path_added(cfg: Dict, stem: str) -> str:
    if stem.startswith("train__"):
        name = stem[len("train__"):]
        if name not in ADDED_TRAIN_SETS:
            raise ValueError(f"{stem} is not a training set of main_build_evidence_sets")
        return train_set_path(cfg, name)
    if stem.startswith("stmt__"):
        return added_statement_path(cfg, stem)
    ds, cond = stem.split("__", 1)
    if cond not in ADDED_CONDITIONS:
        raise ValueError(f"{stem} is not a condition of main_build_evidence_sets")
    return examples_path(cfg, ds, cond)


def added_test_sets(cfg: Dict) -> List[str]:
    return [f"{ds}__{c}" for ds, conds in cfg["test_sets"].items() for c in conds if c in ADDED_CONDITIONS]


def added_sets(cfg: Dict) -> List[str]:
    return added_test_sets(cfg) + added_statement_set_names(cfg)


def rank_check(index: BM25Index, texts: List[str], queries: List[str]) -> int:
    from experiments.controls import bm25_rank
    return int(sum(index.rank(q) != bm25_rank(texts, q) for q in queries))


def written_share_rows(cfg: Dict) -> List[Dict]:
    jd = load_written_judgments(cfg)
    keys = [judge_key(e["name"]) for e in cfg["judges"]["models"]]
    rows = []
    for (kind, f, s), g in jd.groupby(["kind", "finding", "state"]):
        want = "none" if kind == "unsupporting" else s
        each = {k: int((g[f"{k}__state"] == want).sum()) for k in keys}
        both = int(np.all([(g[f"{k}__state"] == want).to_numpy() for k in keys], axis=0).sum())
        rows.append({"table": "written_acceptance", "kind": kind, "finding": f, "state": s, "n_candidates": int(len(g)),
                     **{f"accepted_by_{k}": n for k, n in each.items()}, "accepted_by_both": both,
                     "share_accepted_by_both": both / len(g) if len(g) else math.nan})
    return rows
