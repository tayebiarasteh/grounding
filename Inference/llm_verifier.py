"""
Inference/llm_verifier.py
Created on October 10, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
import math
import os
from typing import Dict, List, Optional

from Inference.llm_client import call_with_retry, resolve_secret

PROMPT = ("You check statements about a patient's chest radiograph. A statement is supported only if the patient's "
          "report agrees with it and the cited evidence supports it.\n\n"
          "Patient's report:\n{report}\n\n"
          "Cited evidence:\n{evidence}\n\n"
          "Statement: {statement}\n\n"
          "Is the statement supported? Answer with Yes or No only.")
PROMPT_DIGEST = hashlib.blake2b(PROMPT.encode("utf-8"), digest_size=4).hexdigest()
_PER_TOKEN_S = 0.12


def build_prompt(report: str, evidence: List[str], statement: str) -> str:
    return PROMPT.format(report=str(report).strip(), evidence="\n".join(evidence) if evidence else "None",
                         statement=str(statement).strip())


def yes_no_score(content) -> Dict:
    for pos, tok in enumerate(content or []):
        if not any(ch.isalnum() for ch in tok.token):
            continue
        cands = [[c.token, float(c.logprob)] for c in (tok.top_logprobs or [])]
        p_yes = sum(math.exp(lp) for t, lp in cands if t.strip().lower() == "yes")
        p_no = sum(math.exp(lp) for t, lp in cands if t.strip().lower() == "no")
        ok = p_yes + p_no > 0
        return {"position": pos, "token": tok.token, "candidates": cands, "p_yes": p_yes, "p_no": p_no,
                "score": p_yes / (p_yes + p_no) if ok else 0.5, "status": "ok" if ok else "fallback_no_yes_no"}
    return {"position": -1, "token": "", "candidates": [], "p_yes": 0.0, "p_no": 0.0, "score": 0.5,
            "status": "fallback_no_answer_token"}


class LLMVerifier:

    def __init__(self, serving_cfg: Dict, llm_cfg: Dict, entry: Dict):
        import httpx
        from openai import OpenAI
        conc = int(llm_cfg["concurrency"])
        self.max_tokens = int(llm_cfg["max_tokens"])
        self.top_logprobs = int(llm_cfg["top_logprobs"])
        self.timeout_s = float(llm_cfg["timeout_s"]) + self.max_tokens * _PER_TOKEN_S
        self.max_retries = int(serving_cfg.get("max_retries", 5))
        self.connect_retry_max_s = float(serving_cfg.get("connect_retry_max_s", 900))
        self.served = entry["served"]
        self.extra = {"reasoning_effort": entry["reasoning_effort"]} if entry.get("reasoning_effort") else {}
        if entry.get("chat_template_kwargs"):
            self.extra["extra_body"] = {"chat_template_kwargs": dict(entry["chat_template_kwargs"])}
        http = httpx.Client(verify=serving_cfg.get("verify_ssl", False), timeout=self.timeout_s,
                            limits=httpx.Limits(max_connections=conc, max_keepalive_connections=conc))
        self.client = OpenAI(base_url=resolve_secret(serving_cfg["api_base_url"], serving_cfg.get("api_base_url_env")),
                             api_key=resolve_secret(serving_cfg["api_key"], serving_cfg.get("api_key_env")),
                             http_client=http, max_retries=0)

    def identity(self) -> Dict:
        return {"model": self.served, "prompt": PROMPT_DIGEST,
                "decoding": {"temperature": 0.0, "max_tokens": self.max_tokens, "top_logprobs": self.top_logprobs,
                             **self.extra}}

    def ask(self, prompt: str) -> Dict:
        def _call():
            return self.client.chat.completions.create(
                model=self.served, messages=[{"role": "user", "content": prompt}], temperature=0.0,
                max_tokens=self.max_tokens, logprobs=True, top_logprobs=self.top_logprobs, timeout=self.timeout_s,
                **self.extra)
        resp = call_with_retry(_call, self.max_retries, self.connect_retry_max_s, f"verifier {self.served}",
                               self.timeout_s)
        choice = resp.choices[0]
        content = choice.logprobs.content if choice.logprobs is not None else None
        out = yes_no_score(content)
        out.update({"response_id": resp.id, "text": choice.message.content or "",
                    "finish_reason": choice.finish_reason, "has_logprobs": content is not None})
        return out

    def check(self, prompt: str) -> Optional[str]:
        a, b = self.ask(prompt), self.ask(prompt)
        if not a["has_logprobs"]:
            return "it returns no log probabilities"
        if a["status"] != "ok":
            return f"neither Yes nor No is among the candidates at the answer position ({a['token']!r})"
        if len(a["candidates"]) < min(5, self.top_logprobs):
            return f"only {len(a['candidates'])} candidates at the answer position"
        if a["response_id"] == b["response_id"]:
            return "two identical calls returned one response id, so the proxy serves a cache"
        return None


def append_record(path: str, rec: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = (json.dumps(rec, ensure_ascii=True) + "\n").encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _differs(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > 1e-6
    return a != b


def read_records(path: str, ident: Dict, compare=("score",)) -> Dict:
    out, bad, other, disagree = {}, 0, 0, 0
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        for raw in f:
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad += 1
                continue
            if any(rec.get(k) != v for k, v in ident.items()):
                other += 1
                continue
            key = rec["item_id"]
            if key in out:
                disagree += int(any(_differs(out[key].get(c), rec.get(c)) for c in compare))
                continue
            out[key] = rec
    return out


def ask_all(verifier, items: List[Dict], path: str, ident: Dict, conc: int, chunk: int, tag: str,
            stage: str = "llm") -> Dict:
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    done = read_records(path, ident)
    todo = [it for it in items if it["item_id"] not in done]
    failed = []

    def safe(it):
        try:
            return verifier.ask(it["prompt"])
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    for attempt in (1, 2):
        failed, in_row = [], 0
        with ThreadPoolExecutor(max_workers=conc) as pool, tqdm(total=len(todo), desc=tag, unit="item") as bar:
            for i in range(0, len(todo), chunk):
                part = todo[i:i + chunk]
                for it, res in zip(part, pool.map(safe, part)):
                    if "error" in res:
                        failed.append((it, res["error"]))
                        in_row += 1
                        if in_row >= 30:
                            raise RuntimeError(f"[{stage}] {tag}: 30 calls failed in a row ({res['error']}); the "
                                               f"server is down or rejects the model")
                        continue
                    in_row = 0
                    rec = {"item_id": it["item_id"], **ident, **res}
                    append_record(path, rec)
                    done[it["item_id"]] = rec
                bar.update(len(part))
        todo = [it for it, _ in failed]
        if not todo:
            return done
    raise RuntimeError(f"[{stage}] {tag}: {len(failed)} items failed twice (first: {failed[0][1]}); run it "
                       f"again to ask them")
