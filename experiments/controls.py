"""
experiments/controls.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import math
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments._common import (MissingInput, atomic_write_csv, digest_file, flush_status, get_cfg, require,
                                 run_units, write_table)
from experiments.study import TEST_SETS, claim_text, examples_path, opposite, out, train_set_path, write_parquet

_REGIME = "controls-v1"
RETRIEVAL = {"retrieval_bm25_top2": 2, "retrieval_bm25_top5": 5, "retrieval_dense_top2": 2, "retrieval_rerank_top2": 2}
RETRIEVAL_MODEL = {"retrieval_bm25_top2": "embedding_model", "retrieval_bm25_top5": "embedding_model",
                   "retrieval_dense_top2": "embedding_model", "retrieval_rerank_top2": "reranker_model"}
QUERY_INSTRUCTION = "Given a statement about a chest radiograph, retrieve reference sentences that support it"
_WORD = re.compile(r"[a-z0-9]+")


def retrieval_path(cfg: Dict, condition: str) -> str:
    return out(cfg, "controls", f"mimic__{condition}.parquet")


def _corpus(cfg: Dict) -> pd.DataFrame:
    from experiments.support_pools import snapshot
    ev = snapshot(cfg)
    ev = ev[(ev["article_set"] == "heldout") & ev["clean"] & ev["is_chest_relevant"].astype(bool)]
    return ev.reset_index(drop=True)


def bm25_rank(corpus: List[str], query: str, k1: float = 1.2, b: float = 0.75) -> List[int]:
    docs = [_WORD.findall(t.lower()) for t in corpus]
    n, avgdl = len(docs), float(np.mean([len(d) for d in docs]))
    dfreq: Dict[str, int] = {}
    for d in docs:
        for t in set(d):
            dfreq[t] = dfreq.get(t, 0) + 1
    scores = np.zeros(n)
    for t in _WORD.findall(query.lower()):
        if t not in dfreq:
            continue
        idf = math.log(1.0 + (n - dfreq[t] + 0.5) / (dfreq[t] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(t)
            if tf:
                scores[i] += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len(d) / avgdl))
    order = np.lexsort((np.arange(n), -scores))
    return [int(i) for i in order if scores[i] > 0]


class ServerRetrieval:

    def __init__(self, cfg: Dict):
        import httpx
        from Inference.llm_client import resolve_secret
        sv = cfg["serving"]
        self.base = resolve_secret(sv["api_base_url"], sv.get("api_base_url_env")).rstrip("/")
        key = resolve_secret(sv["api_key"], sv.get("api_key_env"))
        self.client = httpx.Client(verify=sv.get("verify_ssl", False), timeout=300,
                                   headers={"Authorization": f"Bearer {key}"})
        self.r = cfg["retrieval"]

    def _post(self, path: str, payload: Dict) -> Dict:
        from Inference.llm_client import call_with_retry

        def _call():
            resp = self.client.post(self.base + path, json=payload)
            resp.raise_for_status()
            return resp.json()
        return call_with_retry(_call, 5, 900.0, f"retrieval {path}", 300.0)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 64):
            data = self._post("/embeddings", {"model": self.r["embedding_model"], "input": texts[i:i + 64]})["data"]
            vecs += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        x = np.asarray(vecs, dtype=np.float64)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def rerank(self, query: str, docs: List[str]) -> List[int]:
        res = self._post("/rerank", {"model": self.r["reranker_model"], "query": query, "documents": docs})["results"]
        return [int(r["index"]) for r in sorted(res, key=lambda r: (-r["relevance_score"], r["index"]))]


def build_retrieval(cfg: Dict, conds: List[str]) -> Dict[str, pd.DataFrame]:
    from experiments.support_pools import load_pools
    from experiments.build_examples import output_path
    v = cfg
    rf = pd.read_parquet(require(output_path(cfg, "report_findings"), "main_build_examples"))
    rf = rf[(rf["dataset"] == "mimic") & (rf["split"] == "test")].reset_index(drop=True)
    corpus = _corpus(cfg)
    texts = corpus["sentence_text"].tolist()
    pools = load_pools(cfg)
    held = pools[(pools["pool"] == "support") & (pools["article_set"] == "heldout")]
    pool_ids = {(f, s): set(g["sentence_id"].astype(int)) for (f, s), g in held.groupby(["finding", "state"])}
    statements = sorted({claim_text(f, s) for f in v["findings"] for s in ("present", "absent")})
    srv = ServerRetrieval(cfg)
    dense, rerank = "retrieval_dense_top2" in conds, "retrieval_rerank_top2" in conds
    q = srv.embed([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}" for t in statements]) if dense else None
    d = srv.embed(texts) if dense else None
    picks: Dict[str, Dict[str, List[int]]] = {c: {} for c in conds}
    for qi, st in enumerate(tqdm(statements, desc="retrieval", unit="statement")):
        bm = bm25_rank(texts, st)
        for c, k in (("retrieval_bm25_top2", 2), ("retrieval_bm25_top5", 5)):
            if c in picks:
                picks[c][st] = bm[:k]
        if dense:
            picks["retrieval_dense_top2"][st] = [int(i) for i in np.lexsort((np.arange(len(texts)), -(d @ q[qi])))[:2]]
        if rerank:
            pool80 = bm[:int(v["retrieval"]["rerank_pool"])]
            order = srv.rerank(st, [texts[i] for i in pool80]) if pool80 else []
            picks["retrieval_rerank_top2"][st] = [pool80[j] for j in order[:2]]
    outs = {}
    for cond in conds:
        k = RETRIEVAL[cond]
        rows = []
        for r in rf.itertuples():
            for state in (r.state, opposite(r.state)):
                st = claim_text(r.finding, state)
                idx = picks[cond][st]
                sids = [int(corpus.at[i, "sentence_id"]) for i in idx]
                hit = any(sid in pool_ids.get((r.finding, state), set()) for sid in sids)
                kappa = int(state == r.state)
                rows.append({"qa_row_id": int(r.qa_row_id), "split": "test", "subject_id": r.subject_id,
                             "report_rel_path": r.study_key, "concept": r.finding, "gold_state": r.state,
                             "claim_state": state, "claim_text": st, "case_context": r.case_context,
                             "evidence_sentence_ids_json": json.dumps(sids),
                             "evidence_sentences_json": json.dumps([texts[i] for i in idx], ensure_ascii=False),
                             "evidence_article_titles_json": json.dumps([str(corpus.at[i, "article_title"]) for i in idx],
                                                                        ensure_ascii=False),
                             "row_type": "R", "evidence_source_type": cond, "label_id": kappa * int(hit),
                             "label": "supported" if kappa * int(hit) else "not_supported", "dataset": "mimic",
                             "condition": cond, "recall_hit": int(hit), "evidence_k": k})
        df = pd.DataFrame(rows)
        df["verifier_row_id"] = np.arange(len(df))
        outs[cond] = df
    return outs


def main_retrieval(cfg_path: str, model: str = "") -> str:
    cfg = get_cfg(cfg_path)
    from experiments.support_pools import pools_path
    from experiments.build_examples import output_path
    out_dir = out(cfg, "controls")
    status = os.path.join(out_dir, "status.txt")
    rfp, pp = output_path(cfg, "report_findings"), pools_path(cfg)
    params = {"regime": _REGIME, "report_findings": digest_file(rfp) if os.path.exists(rfp) else None,
              "pools": digest_file(pp) if os.path.exists(pp) else None, "retrieval": cfg["retrieval"],
              "instruction": QUERY_INSTRUCTION}

    groups: Dict[str, List[str]] = {}
    for cond in RETRIEVAL:
        groups.setdefault(cfg["retrieval"][RETRIEVAL_MODEL[cond]], []).append(cond)
    if model and model not in groups:
        raise ValueError(f"{model} is not a retrieval model: {', '.join(groups)}")
    units = [{"key": f"retrieval__{m.split('/')[-1]}", "model": m, "conds": cs,
              "params": {**params, "model": m, "conditions": cs}} for m, cs in groups.items()]
    waiting: List[str] = []

    def compute(unit):
        if None in (params["report_findings"], params["pools"]):
            raise MissingInput("main_support_pools and main_build_examples have not run yet")
        waiting.append(unit["model"])
        if model and unit["model"] != model:
            raise MissingInput(f"{unit['model']} runs in its own call")
        from experiments.judges import _served
        if unit["model"] not in set(_served(cfg["serving"])):
            raise MissingInput(f"{unit['model']} is not served; load it and run it again")
        waiting.remove(unit["model"])
        built = build_retrieval(cfg, unit["conds"])
        rows = []
        for cond, df in built.items():
            write_parquet(df, retrieval_path(cfg, cond))
            g = df[df["claim_state"] == df["gold_state"]]
            rows.append({"condition": cond, "n_examples": len(df), "n_supported": int(df["label_id"].sum()),
                         "n_combinations": int(len(df)), "n_with_support_retrieved": int(df["recall_hit"].sum()),
                         "n_report_state_combinations": int(len(g)),
                         "n_report_state_with_support": int(g["recall_hit"].sum())})
        return rows

    def valid(unit, done):
        return all(os.path.exists(retrieval_path(cfg, c)) for c in unit["conds"])

    rows, refused = run_units("main_retrieval", units, compute, out_dir, status, validate=valid)
    if refused and model and model not in waiting and waiting:
        return out_dir
    if refused:
        raise MissingInput(f"retrieval sets not built yet for {', '.join(waiting) or 'main_support_pools and main_build_examples'}; load each model "
                           f"and run it again")
    write_table(pd.DataFrame(rows), os.path.join(out_dir, "retrieval.csv"))
    flush_status(status, "[main_retrieval] done")
    return out_dir


def main_score_retrieval(cfg_path: str) -> str:
    from experiments.score import EncoderScorer, run_scoring, trained_identity, verifier
    cfg = get_cfg(cfg_path)
    vx = verifier(cfg, cfg["main_verifier"])
    sets = [f"mimic__{c}" for c in RETRIEVAL]
    n = run_scoring(cfg, "main_score_retrieval", vx, lambda: EncoderScorer(cfg, vx), trained_identity(cfg, vx), sets=sets)
    from experiments.purge import purge_unused
    purge_unused(cfg)
    if n:
        raise MissingInput(f"{n} retrieval set(s) could not be scored yet")
    return out(cfg, "scores", vx["id"])


def _vectorizer(n: int):
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(analyzer="word", ngram_range=(n, n), lowercase=True, stop_words="english",
                           token_pattern=r"[a-z0-9]+", dtype=np.float32)


def nearest_train_cosine(eval_texts: List[str], train_texts: List[str], n: int) -> np.ndarray:
    from sklearn.preprocessing import normalize
    vec = _vectorizer(n)
    vec.fit(list(train_texts) + list(eval_texts))
    a, b = normalize(vec.transform(eval_texts)), normalize(vec.transform(train_texts))
    sim = np.zeros(a.shape[0])
    for i in range(0, a.shape[0], 512):
        sim[i:i + 512] = np.asarray((a[i:i + 512] @ b.T).todense()).max(axis=1)
    sim[np.asarray(a.sum(axis=1)).ravel() == 0] = np.nan
    return sim


def nearest_other_patient_cosine(texts: List[str], subjects: np.ndarray, n: int):
    from sklearn.preprocessing import normalize
    vec = _vectorizer(n)
    x = normalize(vec.fit_transform(texts))
    best = np.zeros(x.shape[0])
    for i in tqdm(range(0, x.shape[0], 256), desc=f"reference n={n}", unit="block", leave=False):
        block = np.asarray((x[i:i + 256] @ x.T).todense())
        block[subjects[i:i + 256][:, None] == subjects[None, :]] = -1.0
        best[i:i + 256] = block.max(axis=1)
    best[(best < 0) | (np.asarray(x.sum(axis=1)).ravel() == 0)] = np.nan
    return best, int(x.shape[1])


def _units_text(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("qa_row_id")[["qa_row_id", "subject_id", "case_context"]].reset_index(drop=True)


def main_similarity(cfg_path: str) -> str:
    from experiments.build_examples import output_path
    cfg = get_cfg(cfg_path)
    orders = list(cfg["similarity"]["ngram_orders"])
    out_dir = out(cfg, "controls")
    status = os.path.join(out_dir, "status.txt")
    rfp = output_path(cfg, "report_findings")

    def dig(p):
        return digest_file(p) if os.path.exists(p) else None
    inputs = {d: {"report_findings": dig(rfp)} for d in TEST_SETS}
    inputs["food"] = {"train": dig(train_set_path(cfg, "food")), "test": dig(examples_path(cfg, "food", "unchanged"))}
    units = [{"key": f"similarity__{d}", "domain": d, "params": {"regime": _REGIME, "inputs": inputs[d], "orders": orders}}
             for d in list(TEST_SETS) + ["food"]]
    units += [{"key": f"reference__{d}", "domain": d, "params": {"regime": _REGIME, "inputs": inputs[d], "orders": orders}}
              for d in ("mimic", "food")]

    def corpora(domain):
        if None in inputs[domain].values():
            raise MissingInput("main_build_examples has not run yet")
        if domain == "food":
            tr = pd.read_parquet(train_set_path(cfg, "food"))
            te = pd.read_parquet(require(examples_path(cfg, "food", "unchanged"), "main_build_examples"))
            return _units_text(tr[tr["split"] == "train"]), _units_text(te)
        rf = pd.read_parquet(rfp)
        train = rf[(rf["dataset"] == "mimic") & (rf["split"] == "train")]
        test = rf[(rf["dataset"] == domain) & (rf["split"] == "test")]
        return (train.drop_duplicates("qa_row_id")[["qa_row_id", "subject_id", "case_context"]],
                test.drop_duplicates("qa_row_id")[["qa_row_id", "subject_id", "case_context"]].reset_index(drop=True))

    def compute(unit):
        train, test = corpora(unit["domain"])
        rows = []
        if unit["key"].startswith("similarity__"):
            docs = train["case_context"].astype(str).unique().tolist()
            per = test[["qa_row_id", "subject_id"]].copy()
            for n in orders:
                sim = nearest_train_cosine(test["case_context"].astype(str).tolist(), docs, n)
                per[f"sim_{n}gram"] = sim
                rows.append({"table": "similarity", "domain": unit["domain"], "ngram_order": n,
                             "mean": float(np.nanmean(sim)), "median": float(np.nanmedian(sim)),
                             "max": float(np.nanmax(sim)), "fraction_above_0p8": float(np.nanmean(sim > 0.8)),
                             "n_test_units": int(np.isfinite(sim).sum()), "n_train_documents": len(docs)})
            atomic_write_csv(per, out(cfg, "controls", f"unit_similarity_{unit['domain']}.csv"))
            return rows
        texts, subs = train["case_context"].astype(str).tolist(), train["subject_id"].astype(str).to_numpy()
        per_text = train.groupby("case_context")["subject_id"].nunique()
        shared = train["case_context"].map(per_text).to_numpy() > 1
        for variant, keep in (("all_training_findings", np.ones(len(texts), bool)),
                              ("texts_of_several_patients_dropped", ~shared)):
            for n in orders:
                best, n_types = nearest_other_patient_cosine([t for t, k in zip(texts, keep) if k], subs[keep], n)
                rows.append({"table": "reference", "domain": unit["domain"], "variant": variant, "ngram_order": n,
                             "mean": float(np.nanmean(best)), "median": float(np.nanmedian(best)),
                             "vocabulary_size": n_types, "n_documents": int(np.isfinite(best).sum())})
        return rows

    def valid(unit, done):
        return not unit["key"].startswith("similarity__") or \
            os.path.exists(out(cfg, "controls", f"unit_similarity_{unit['domain']}.csv"))

    rows, refused = run_units("main_similarity", units, compute, out_dir, status, validate=valid)
    if refused:
        raise MissingInput(f"{refused} similarity unit(s) could not run yet")
    write_table(pd.DataFrame(rows), os.path.join(out_dir, "similarity.csv"))
    flush_status(status, "[main_similarity] done")
    return out_dir
