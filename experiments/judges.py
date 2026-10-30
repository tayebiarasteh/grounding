"""
experiments/judges.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from experiments._common import (GateFailure, MissingInput, flush_status, get_cfg, read_results_csv, require, run_units,
                                 write_table)
from Inference.llm_client import _extract_json, _split_harmony, call_with_retry, resolve_secret
from Inference.llm_verifier import append_record, read_records
from experiments.study import core_findings, out

_REGIME = "judges-v1"
SYSTEM = (
    "You are a radiology evidence auditor. You are given a concept, a claim about that concept, and one or more "
    "external evidence sentences. Apply exactly two judgments. First, does the evidence CONCERN the claim's concept? "
    "Second, if it does, does it point toward the claim's state (present or absent), the opposite state, or neither? "
    "Judge only what the evidence says. The claim describes a finding on a patient's chest radiograph, so evidence "
    "about the same finding outside the chest does not concern the claim's concept. Answer with a single JSON object "
    'and nothing else: {"concerns_concept": "yes" | "no", "direction": "claim" | "opposite" | "neither"}.')
RUBRIC = hashlib.blake2b(SYSTEM.encode("utf-8"), digest_size=4).hexdigest()
_TIMEOUT_FLOOR_S, _TIMEOUT_PER_TOKEN_S, _TIMEOUT_CAP_S = 120.0, 0.12, 1200.0
_ATTEMPTS = 3
_BREAKER = 30


def user_message(finding: str, sentence: str) -> str:
    return f'Concept: "{finding}"\nClaim: "{finding} is present"\n\nEvidence:\n"""\n{sentence}\n"""'


def judge_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def combinations(cfg: Dict) -> pd.DataFrame:
    ev = pd.read_parquet(require(cfg["sources"]["snapshot"], "the Radiopaedia snapshot"),
                         columns=["sentence_id", "article_title", "sentence_text", "matched_concepts_json",
                                  "evidence_split"])
    rows = []
    for r in ev.itertuples():
        for f in core_findings(r.matched_concepts_json):
            rows.append({"item_id": f"{int(r.sentence_id)}|{f}", "sentence_id": int(r.sentence_id), "finding": f,
                         "article_title": str(r.article_title), "evidence_split": str(r.evidence_split),
                         "sentence_text": str(r.sentence_text)})
    df = pd.DataFrame(rows).sort_values(["sentence_id", "finding"]).reset_index(drop=True)
    if df["item_id"].duplicated().any():
        raise ValueError("duplicated sentence and finding combinations")
    return df


def parse(text: str) -> Optional[Dict]:
    v = _extract_json(_split_harmony(text)) or {}
    concerns, direction = str(v.get("concerns_concept", "")).strip().lower(), str(v.get("direction", "")).strip().lower()
    if concerns not in ("yes", "no") or (concerns == "yes" and direction not in ("claim", "opposite", "neither")):
        return None
    return {"concerns": concerns, "direction": direction if concerns == "yes" else "neither"}


def judge_identity(jcfg: Dict, entry: Dict) -> Dict:
    return {"model": entry["served"], "rubric": RUBRIC,
            "decoding": {"temperature": 0.0, "max_tokens": int(jcfg["max_tokens"]),
                         "chat_template_kwargs": entry.get("chat_template_kwargs") or None}}


def assigned_state(concerns: str, direction: str) -> str:
    if concerns != "yes":
        return "none"
    return {"claim": "present", "opposite": "absent"}.get(direction, "none")


class JudgeClient:

    def __init__(self, serving: Dict, jcfg: Dict, entry: Dict):
        import httpx
        from openai import OpenAI
        conc = int(jcfg["concurrency"])
        self.jcfg, self.entry = jcfg, entry
        self.served, self.name = entry["served"], entry["name"]
        self.max_tokens = int(jcfg["max_tokens"])
        self.timeout_s = min(_TIMEOUT_CAP_S, _TIMEOUT_FLOOR_S + self.max_tokens * _TIMEOUT_PER_TOKEN_S)
        self.kwargs = entry.get("chat_template_kwargs") or None
        self.max_retries = int(serving.get("max_retries", 5))
        self.connect_retry_max_s = float(serving.get("connect_retry_max_s", 900))
        http = httpx.Client(verify=serving.get("verify_ssl", False), timeout=self.timeout_s,
                            limits=httpx.Limits(max_connections=conc, max_keepalive_connections=conc))
        self.client = OpenAI(base_url=resolve_secret(serving["api_base_url"], serving.get("api_base_url_env")),
                             api_key=resolve_secret(serving["api_key"], serving.get("api_key_env")),
                             http_client=http, max_retries=0)

    def identity(self) -> Dict:
        return judge_identity(self.jcfg, self.entry)

    def ask(self, user: str) -> Dict:
        extra = {"extra_body": {"chat_template_kwargs": self.kwargs}} if self.kwargs else {}

        def _call():
            return self.client.chat.completions.create(
                model=self.served, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=self.max_tokens, timeout=self.timeout_s, **extra)
        resp = call_with_retry(_call, self.max_retries, self.connect_retry_max_s, f"judge {self.served}",
                               self.timeout_s)
        choice = resp.choices[0]
        return {"text": choice.message.content or "", "finish_reason": choice.finish_reason, "response_id": resp.id}


def _served(serving: Dict) -> List[str]:
    from Inference.llm_client import served_models
    try:
        return served_models(serving)
    except Exception as e:
        raise MissingInput(f"the serving proxy did not answer ({type(e).__name__}); run it again once the proxy answers")


def _check(judge: JudgeClient, item: Dict) -> Optional[str]:
    a, b = judge.ask(user_message(item["finding"], item["sentence_text"])), \
        judge.ask(user_message(item["finding"], item["sentence_text"]))
    if parse(a["text"]) is None:
        return f"its answer does not parse ({a['finish_reason']}): {a['text'][:160]!r}"
    if a["response_id"] == b["response_id"]:
        return "two identical calls returned one response id, so the proxy serves a cache"
    return None


def _ask_all(judge: JudgeClient, items: List[Dict], path: str, ident: Dict, conc: int, tag: str) -> Dict:
    done = read_records(path, ident, compare=("concerns", "direction"))
    for rec in done.values():
        p = parse(rec.get("text", ""))
        rec.update(p if p else {"concerns": "", "direction": ""})
        rec["status"] = "ok" if p else "unparsed"
    todo = [it for it in items if it["item_id"] not in done]
    state = {"fails_in_row": 0}

    def one(it):
        last = None
        for _ in range(_ATTEMPTS):
            try:
                res = judge.ask(user_message(it["finding"], it["sentence_text"]))
            except Exception as e:
                return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            last = res
            if parse(res["text"]) is not None:
                break
        return last

    failed = []
    chunk = conc * 4
    with ThreadPoolExecutor(max_workers=conc) as pool, tqdm(total=len(todo), desc=tag, unit="item") as bar:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            for it, res in zip(part, pool.map(one, part)):
                if "error" in res:
                    failed.append((it, res["error"]))
                    state["fails_in_row"] += 1
                    if state["fails_in_row"] >= _BREAKER:
                        raise RuntimeError(f"[judges] {tag}: {_BREAKER} calls failed in a row ({res['error']}); the "
                                           f"server is down or rejects the model")
                    continue
                state["fails_in_row"] = 0
                p = parse(res["text"])
                rec = {"item_id": it["item_id"], **ident, "text": res["text"][:4000],
                       "finish_reason": res["finish_reason"], "response_id": res["response_id"]}
                append_record(path, rec)
                rec.update(p if p else {"concerns": "", "direction": ""})
                rec["status"] = "ok" if p else "unparsed"
                done[it["item_id"]] = rec
            bar.update(len(part))
    if failed:
        raise RuntimeError(f"[judges] {tag}: {len(failed)} calls failed (first: {failed[0][1]}); run it again to "
                           f"ask them")
    return done


def judgments_path(cfg: Dict) -> str:
    return out(cfg, "judges", "judgments.csv")


def judge_table(cfg: Dict, items_df: pd.DataFrame, out_dir: str, status: str, judge_name: str, regime: str,
                stage: str) -> Optional[pd.DataFrame]:
    jcfg = cfg["judges"]
    if judge_name and judge_name not in [e["name"] for e in jcfg["models"]]:
        raise ValueError(f"{judge_name} is not a configured judge")
    items = items_df.to_dict("records")
    items_digest = hashlib.blake2b("\n".join(f"{r['item_id']}\t{r['sentence_text']}" for r in items).encode(),
                                   digest_size=16).hexdigest()
    served: List[str] = []
    results, refused_names, failed = {}, [], []
    for entry in jcfg["models"]:
        ident = judge_identity(jcfg, entry)
        unit = {"key": judge_key(entry["name"]), "params": {"regime": regime, **ident, "items": items_digest}}

        def compute(u, ident=ident, entry=entry):
            if judge_name and entry["name"] != judge_name:
                raise MissingInput(f"{entry['name']} runs in its own call")
            if not served:
                served.extend(_served(cfg["serving"]))
            if entry["served"] not in served:
                raise MissingInput(f"{entry['served']} is not served; load it and run it again")
            judge = JudgeClient(cfg["serving"], jcfg, entry)
            problem = _check(judge, items[0])
            if problem:
                raise MissingInput(f"{entry['served']} was not run because {problem}")
            path = os.path.join(out_dir, "jsonl", f"{u['key']}.jsonl")
            done = _ask_all(judge, items, path, ident, int(jcfg["concurrency"]), entry["name"])
            rows = [{"item_id": it["item_id"], "judge": entry["name"], "concerns": done[it["item_id"]]["concerns"],
                     "direction": done[it["item_id"]]["direction"], "status": done[it["item_id"]]["status"],
                     "finish_reason": done[it["item_id"]]["finish_reason"],
                     "response_id": done[it["item_id"]]["response_id"]} for it in items]
            share = sum(r["status"] != "ok" for r in rows) / len(rows)
            if share > float(jcfg["max_unparsed_share"]):
                raise GateFailure(f"judge {entry['name']}: {share:.2%} of the answers do not parse, above "
                                  f"{float(jcfg['max_unparsed_share']):.0%}; the raw answers are in {path}")
            return rows

        def valid(u, done):
            return len(done) == len(items) and set(done["item_id"]) == set(items_df["item_id"])

        try:
            rows, refused = run_units(f"{stage} {entry['name']}", [unit], compute, out_dir, status, validate=valid)
        except GateFailure as e:
            failed.append(str(e))
            continue
        if refused:
            refused_names.append(entry["served"])
        else:
            results[entry["name"]] = pd.DataFrame(rows)
    if failed:
        raise GateFailure("; ".join(failed))
    if refused_names and judge_name and judge_name in results:
        return None
    if refused_names:
        raise MissingInput(f"judges not run yet: {', '.join(refused_names)}; load each and run it again")
    table = items_df.drop(columns="sentence_text").copy()
    for entry in jcfg["models"]:
        j = results[entry["name"]].set_index("item_id").loc[table["item_id"]]
        k = judge_key(entry["name"])
        table[f"{k}__concerns"] = j["concerns"].fillna("").to_numpy()
        table[f"{k}__direction"] = j["direction"].fillna("").to_numpy()
        table[f"{k}__status"] = j["status"].to_numpy()
        table[f"{k}__state"] = [assigned_state(c, d) if s == "ok" else "none"
                                for c, d, s in zip(table[f"{k}__concerns"], table[f"{k}__direction"],
                                                   table[f"{k}__status"])]
    return table


def main_judges(cfg_path: str, judge_name: str = "") -> str:
    cfg = get_cfg(cfg_path)
    out_dir = out(cfg, "judges")
    status = os.path.join(out_dir, "status.txt")
    items_df = combinations(cfg)
    table = judge_table(cfg, items_df, out_dir, status, judge_name, _REGIME, "judges")
    if table is None:
        return out_dir
    write_table(table, judgments_path(cfg))
    flush_status(status, f"[judges] done, {len(table)} combinations")
    return out_dir


def load_judgments(cfg: Dict) -> pd.DataFrame:
    path = judgments_path(cfg)
    if not os.path.exists(path):
        raise MissingInput(f"{path} is missing; run main_judges first")
    return read_results_csv(path, keep_default_na=False)
