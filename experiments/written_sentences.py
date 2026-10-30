"""
experiments/written_sentences.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import os
import re
from typing import Dict, List

import pandas as pd

from experiments._common import (MissingInput, digest_file, flush_status, get_cfg, read_results_csv, require, run_units,
                                 write_table)
from Inference.llm_client import call_with_retry, resolve_secret
from Inference.llm_verifier import append_record
from experiments.study import STATES, out
from experiments.judges import _served, judge_table

_REGIME = "written_sentences-v1"
FOLDER = "written_sentences"
SYSTEM = ("You write sentences for a radiology reference text about chest radiography, in the style of an encyclopedia "
          "article: general statements about a finding, never about a particular patient.")
SYSTEM_DIGEST = hashlib.blake2b(SYSTEM.encode("utf-8"), digest_size=4).hexdigest()
KINDS = ("support", "unsupporting", "rewrite")
_LINE_PREFIX = re.compile(r"^\s*(?:[-*•]|\(?\d{1,2}[.)])\s*")
_ATTEMPTS = 3
_TAIL = "\nVary the sentence structure and the vocabulary. Output only the sentences, one per line, without numbering."


def slug(finding: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", finding.lower()).strip("_")


def candidates_path(cfg: Dict) -> str:
    return out(cfg, FOLDER, "candidates.csv")


def written_judgments_path(cfg: Dict) -> str:
    return out(cfg, FOLDER, "judgments.csv")


def real_support(cfg: Dict) -> pd.DataFrame:
    c = read_results_csv(require(out(cfg, "support_pools", "candidates.csv"), "main_support_pools"), keep_default_na=False)
    c = c[c["article_set"] == "training"].copy()
    c["sentence_id"] = c["sentence_id"].astype(int)
    return c.sort_values(["finding", "state", "sentence_id"]).reset_index(drop=True)


def support_request(finding: str, state: str, n: int, exemplars: List[str]) -> str:
    if state == "present":
        what = (f"that each state or describe that {finding} is present on a chest radiograph, as a reference article "
                f"would")
    else:
        what = (f"that each describe the absence of {finding} on a chest radiograph: its exclusion, a normal appearance, "
                f"or a sign that rules it out, as a reference article would")
    text = f"Write {n} different sentences, one per line, {what}."
    if exemplars:
        text += "\nExamples of such sentences from the reference text:\n" + "\n".join(f"- {e}" for e in exemplars)
    return text + _TAIL


def unsupporting_request(finding: str, n: int) -> str:
    return (f"Write {n} different sentences, one per line, that mention {finding} without saying whether it is present "
            f"or absent on a chest radiograph, for example its history, causes, epidemiology, a differential list or an "
            f"imaging technique." + _TAIL)


def rewrite_request(sentence: str, k: int) -> str:
    return (f"Rewrite the following sentence in {k} different ways that keep its meaning and its style, one per line. "
            f"Output only the rewritten sentences, without numbering.\n\nSentence: {sentence}")


def requests(cfg: Dict) -> List[Dict]:
    w = cfg["writer"]
    n, calls, k = int(w["sentences_per_call"]), int(w["calls_per_request"]), int(w["rewrites_per_sentence"])
    real = real_support(cfg)
    ex = {(f, s): g["sentence_text"].astype(str).tolist()[:int(w["exemplars"])]
          for (f, s), g in real.groupby(["finding", "state"])}
    units = []
    for f in cfg["findings"]:
        for s in STATES:
            for b in range(calls):
                units.append({"key": f"support__{slug(f)}__{s}__{b}", "kind": "support", "finding": f, "state": s,
                              "batch": b, "source_sentence_id": -1, "exemplars": ex.get((f, s), []),
                              "message": support_request(f, s, n, ex.get((f, s), []))})
        for b in range(calls):
            units.append({"key": f"unsupporting__{slug(f)}__{b}", "kind": "unsupporting", "finding": f, "state": "none",
                          "batch": b, "source_sentence_id": -1, "exemplars": [], "message": unsupporting_request(f, n)})
    for r in real.itertuples():
        units.append({"key": f"rewrite__{int(r.sentence_id)}__{slug(r.finding)}", "kind": "rewrite",
                      "finding": r.finding, "state": r.state, "batch": 0, "source_sentence_id": int(r.sentence_id),
                      "exemplars": [str(r.sentence_text)], "message": rewrite_request(str(r.sentence_text), k)})
    return units


def norm_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def parse_lines(text: str, min_words: int, exemplars: List[str]) -> List[str]:
    seen = {norm_sentence(e) for e in exemplars}
    outl = []
    for raw in str(text).splitlines():
        s = re.sub(r"\s+", " ", _LINE_PREFIX.sub("", raw)).strip().strip('"“”').strip()
        if len(s.split()) < min_words or not s.endswith("."):
            continue
        key = norm_sentence(s)
        if key in seen:
            continue
        seen.add(key)
        outl.append(s)
    return outl


def unit_seed(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest(), "big") % (2 ** 31)


class WriterClient:

    def __init__(self, cfg: Dict):
        import httpx
        from openai import OpenAI
        sv, w = cfg["serving"], cfg["writer"]
        conc = int(w["concurrency"])
        self.served, self.temperature, self.top_p = w["served"], float(w["temperature"]), float(w["top_p"])
        self.max_tokens = int(w["max_tokens"])
        self.kwargs = w.get("chat_template_kwargs") or None
        self.timeout_s = min(1200.0, 120.0 + 0.12 * self.max_tokens)
        self.max_retries = int(sv.get("max_retries", 5))
        self.connect_retry_max_s = float(sv.get("connect_retry_max_s", 900))
        http = httpx.Client(verify=sv.get("verify_ssl", False), timeout=self.timeout_s,
                            limits=httpx.Limits(max_connections=conc, max_keepalive_connections=conc))
        self.client = OpenAI(base_url=resolve_secret(sv["api_base_url"], sv.get("api_base_url_env")),
                             api_key=resolve_secret(sv["api_key"], sv.get("api_key_env")), http_client=http,
                             max_retries=0)

    def identity(self) -> Dict:
        return {"model": self.served, "system": SYSTEM_DIGEST,
                "decoding": {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
                             "chat_template_kwargs": self.kwargs}}

    def ask(self, message: str, seed: int) -> Dict:
        extra = {"extra_body": {"chat_template_kwargs": self.kwargs}} if self.kwargs else {}

        def _call():
            return self.client.chat.completions.create(
                model=self.served, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}],
                temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens, seed=seed,
                timeout=self.timeout_s, **extra)
        resp = call_with_retry(_call, self.max_retries, self.connect_retry_max_s, f"writer {self.served}", self.timeout_s)
        choice = resp.choices[0]
        return {"text": choice.message.content or "", "finish_reason": choice.finish_reason, "response_id": resp.id,
                "seed": seed}


def item_id(kind: str, finding: str, state: str, text: str) -> str:
    h = hashlib.blake2b(f"{kind}|{finding}|{state}|{norm_sentence(text)}".encode("utf-8"), digest_size=6).hexdigest()
    return f"w{h}|{finding}"


def _dedup(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not len(df):
        raise RuntimeError("[main_write_sentences] the writer gave no candidate sentence")
    df["norm"] = df["sentence_text"].map(norm_sentence)
    df = df.drop_duplicates(["finding", "norm"]).drop(columns="norm").reset_index(drop=True)
    return df


def main_write_sentences(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    w = cfg["writer"]
    out_dir = out(cfg, FOLDER)
    status = os.path.join(out_dir, "status.txt")
    units = requests(cfg)
    min_words = int(w["min_words"])
    cand = out(cfg, "support_pools", "candidates.csv")
    base = {"regime": _REGIME, "model": w["served"], "system": SYSTEM_DIGEST, "candidates": digest_file(cand),
            "decoding": {"temperature": float(w["temperature"]), "top_p": float(w["top_p"]),
                         "max_tokens": int(w["max_tokens"]), "chat_template_kwargs": w.get("chat_template_kwargs") or None},
            "min_words": min_words}
    for u in units:
        u["params"] = {**base, "message": hashlib.blake2b(u["message"].encode("utf-8"), digest_size=8).hexdigest(),
                       "seed": unit_seed(u["key"])}
    holder: Dict = {}
    raw_path = os.path.join(out_dir, "jsonl", "writer.jsonl")

    def compute(unit):
        if "client" not in holder:
            if w["served"] not in set(_served(cfg["serving"])):
                raise MissingInput(f"{w['served']} is not served; load it and run it again")
            holder["client"] = WriterClient(cfg)
        client = holder["client"]
        seed = int(unit["params"]["seed"])
        lines, res = [], {"text": "", "finish_reason": "", "response_id": ""}
        for attempt in range(_ATTEMPTS):
            res = client.ask(unit["message"], seed + attempt)
            append_record(raw_path, {"item_id": unit["key"], **client.identity(), "attempt": attempt, **res})
            lines = parse_lines(res["text"], min_words, unit["exemplars"])
            if lines:
                break
        if not lines:
            raise RuntimeError(f"[main_write_sentences] {unit['key']}: no sentence passed the filter in {_ATTEMPTS} answers "
                               f"({res['finish_reason']}): {res['text'][:200]!r}")
        return [{"item_id": item_id(unit["kind"], unit["finding"], unit["state"], s), "kind": unit["kind"],
                 "finding": unit["finding"], "state": unit["state"], "batch": int(unit["batch"]),
                 "source_sentence_id": int(unit["source_sentence_id"]), "sentence_text": s,
                 "response_id": res["response_id"]} for s in lines]

    def valid(unit, done):
        texts = done["sentence_text"].astype(str)
        return bool(len(done) and (texts.str.split().str.len() >= min_words).all())

    rows, refused = run_units("main_write_sentences", units, compute, out_dir, status, validate=valid)
    if refused:
        raise MissingInput(f"{refused} writer call(s) could not run yet; load {w['served']} and run it again")
    table = _dedup(rows).sort_values(["kind", "finding", "state", "sentence_text"]).reset_index(drop=True)
    write_table(table, candidates_path(cfg))
    flush_status(status, f"[main_write_sentences] done, {len(table)} candidates")
    return out_dir


def main_judge_written(cfg_path: str, judge_name: str = "") -> str:
    cfg = get_cfg(cfg_path)
    out_dir = out(cfg, FOLDER)
    status = os.path.join(out_dir, "status.txt")
    cand = read_results_csv(require(candidates_path(cfg), "main_write_sentences"), keep_default_na=False)
    items_df = cand[["item_id", "finding", "sentence_text"]].reset_index(drop=True)
    table = judge_table(cfg, items_df, out_dir, status, judge_name, _REGIME, "main_judge_written")
    if table is None:
        return out_dir
    table = table.merge(cand[["item_id", "kind", "state", "batch", "source_sentence_id", "sentence_text"]], on="item_id",
                        how="left", validate="one_to_one")
    write_table(table, written_judgments_path(cfg))
    flush_status(status, f"[main_judge_written] done, {len(table)} candidates judged")
    return out_dir


def load_written_judgments(cfg: Dict) -> pd.DataFrame:
    path = written_judgments_path(cfg)
    if not os.path.exists(path):
        raise MissingInput(f"{path} is missing; run main_write_sentences first")
    return read_results_csv(path, keep_default_na=False)
