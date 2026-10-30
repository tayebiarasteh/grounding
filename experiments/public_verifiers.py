"""
experiments/public_verifiers.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from experiments._common import GateFailure, MissingInput, atomic_write_json, cuda_device, get_cfg, oom_batches
from experiments.overlay import package_overlay
from Inference.llm_verifier import append_record, read_records
from experiments.study import out, read_evidence
from experiments.score import run_scoring, score_dir, verifier

QUESTION = "Is this statement about the patient supported?"
MINICHECK_SYSTEM = ('Determine whether the provided claim is consistent with the corresponding document. Consistency in '
                    'this context implies that all information presented in the claim is substantiated by the document. '
                    'If not, it should be considered inconsistent. Please assess the claim\'s consistency with the '
                    'document by responding with either "Yes" or "No".')
LYNX_PROMPT = (
    "\nGiven the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether "
    "it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context "
    "provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your "
    "final verdict by strictly following this format: \"PASS\" if the answer is faithful to the DOCUMENT and \"FAIL\" "
    "if the answer is not faithful to the DOCUMENT. Show your reasoning.\n\n--\nQUESTION (THIS DOES NOT COUNT AS "
    "BACKGROUND INFORMATION):\n{question}\n\n--\nDOCUMENT:\n{context}\n\n--\nANSWER:\n{answer}\n\n--\n\nYour output "
    "should be in JSON FORMAT with the keys \"REASONING\" and \"SCORE\":\n{{\"REASONING\": <your reasoning as bullet "
    "points>, \"SCORE\": <your final score>}}\n")


def document(report: str, evidence: List[str]) -> str:
    return f"Patient's report:\n{report}\n\nCited evidence:\n" + ("\n".join(evidence) if evidence else "None")


def _sentences(text: str) -> List[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]


def minicheck_chunks(doc: str, size: int = 500) -> List[str]:
    if len(doc.split()) <= size:
        return [doc]
    sents = []
    for block in doc.split("\n"):
        sents += _sentences(block) + ["\n"]
    sents = sents[:-1]
    chunks, cur, n = [], [], 0
    for sent in sents:
        w = len(sent.split())
        if n + w > size:
            chunks.append(" ".join(cur))
            cur, n = [sent], w
        else:
            cur.append(sent)
            n += w
    chunks.append(" ".join(cur))
    return [c for c in (x.replace(" \n ", "\n").strip() for x in chunks) if c]


def alignscore_chunks(premise: str) -> List[str]:
    sents = _sentences(premise) or [""]
    per = max(len(sents) // (len(premise.strip().split()) // 350 + 1), 1)
    return [" ".join(sents[i:i + per]) for i in range(0, len(sents), per)]


def _rows(cfg: Dict, df: pd.DataFrame):
    k = int(cfg["max_evidence_sentences"])
    for r, c, e, s in zip(df["verifier_row_id"], df["case_context"], df["evidence_sentences_json"], df["claim_text"]):
        yield int(r), str(c), read_evidence(e, k), str(s)


def _gpu() -> torch.device:
    return cuda_device("main_public_verifiers")


class MiniCheckT5:

    def __init__(self, cfg: Dict, spec: Dict):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.cfg, self.spec, self.device = cfg, spec, _gpu()
        self.tok = AutoTokenizer.from_pretrained(spec["hf_repos"][0])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(spec["hf_repos"][0]).to(self.device).eval()

    @torch.no_grad()
    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        ids, texts, owner = [], [], []
        for rid, rep, ev, st in _rows(self.cfg, df):
            for chunk in minicheck_chunks(document(rep, ev)):
                owner.append(len(ids))
                texts.append("predict: " + self.tok.eos_token.join([chunk, st]))
            ids.append(rid)
        probs = np.empty(len(texts))
        order = np.argsort([-len(t) for t in texts], kind="mergesort")

        def run(idx):
            enc = self.tok([texts[j] for j in idx], max_length=2048, truncation=True, padding=True,
                           return_tensors="pt").to(self.device)
            dec = torch.zeros((len(idx), 1), dtype=torch.long, device=self.device)
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                decoder_input_ids=dec).logits.squeeze(1)
            probs[idx] = torch.softmax(logits[:, [3, 209]].float(), dim=-1)[:, 1].cpu().numpy()
        oom_batches(order, int(self.spec["batch_size"]), run, f"{self.spec['id']} {input_set}")
        best = np.full(len(ids), -np.inf)
        np.maximum.at(best, np.asarray(owner), probs)
        return pd.DataFrame({"verifier_row_id": ids, "prob_supported": best})


def _load_lightning_state(path: str) -> Dict:

    class _Stub(dict):
        def __init__(self, *a, **k):
            super().__init__()

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.update(state)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("pytorch_lightning", "lightning", "lightning_fabric"):
                return _Stub
            return super().find_class(module, name)

    class _Module:
        Unpickler = _Unpickler

        @staticmethod
        def load(f, **kw):
            return _Unpickler(f, **kw).load()

    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_Module)
    return ck["state_dict"]


class AlignScoreLarge:

    def __init__(self, cfg: Dict, spec: Dict):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoTokenizer, RobertaModel
        self.cfg, self.spec, self.device = cfg, spec, _gpu()
        ckpt = hf_hub_download(spec["hf_repos"][0], spec["ckpt"])
        state = _load_lightning_state(ckpt)
        base = spec["hf_repos"][1]
        self.tok = AutoTokenizer.from_pretrained(base)
        self.base = RobertaModel(AutoConfig.from_pretrained(base))
        own = {k[len("base_model."):]: v for k, v in state.items() if k.startswith("base_model.")}
        missing, unexpected = self.base.load_state_dict(own, strict=False)
        bad = [k for k in list(missing) + list(unexpected) if not k.endswith("position_ids")]
        if bad:
            raise RuntimeError(f"[main_public_verifiers] AlignScore: base weights do not match RoBERTa-large: {bad[:5]}")
        self.tri = torch.nn.Linear(self.base.config.hidden_size, 3)
        self.tri.load_state_dict({"weight": state["tri_layer.weight"], "bias": state["tri_layer.bias"]})
        self.base.to(self.device).eval()
        self.tri.to(self.device).eval()

    @torch.no_grad()
    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        ids, prem, hyp, owner = [], [], [], []
        for rid, rep, ev, st in _rows(self.cfg, df):
            hyps = _sentences(st) or [st]
            for chunk in alignscore_chunks(document(rep, ev)):
                for h, sent in enumerate(hyps):
                    owner.append((len(ids), h))
                    prem.append(chunk)
                    hyp.append(sent)
            ids.append(rid)
        probs = np.empty(len(prem))

        def run(idx):
            enc = self.tok([prem[j] for j in idx], [hyp[j] for j in idx], truncation="only_first", padding="max_length",
                           max_length=self.tok.model_max_length, return_tensors="pt").to(self.device)
            pooled = self.base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).pooler_output
            probs[idx] = torch.softmax(self.tri(pooled).float(), dim=-1)[:, 0].cpu().numpy()
        oom_batches(np.arange(len(prem)), int(self.spec["batch_size"]), run, f"{self.spec['id']} {input_set}")
        best: Dict = {}
        for key, p in zip(owner, probs):
            best[key] = max(best.get(key, -np.inf), float(p))
        per_row: Dict[int, List[float]] = {}
        for (row, _), p in best.items():
            per_row.setdefault(row, []).append(p)
        return pd.DataFrame({"verifier_row_id": ids, "prob_supported": [float(np.mean(per_row[r])) for r in range(len(ids))]})


class HHEM:

    def __init__(self, cfg: Dict, spec: Dict):
        from transformers import AutoModelForSequenceClassification
        self.cfg, self.spec, self.device = cfg, spec, _gpu()
        self.model = AutoModelForSequenceClassification.from_pretrained(spec["hf_repos"][0], trust_remote_code=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        pairs, ids = [], []
        for rid, rep, ev, st in _rows(self.cfg, df):
            ids.append(rid)
            pairs.append((document(rep, ev), st))
        probs = np.empty(len(pairs))

        def run(idx):
            probs[idx] = self.model.predict([pairs[j] for j in idx]).float().cpu().numpy()
        oom_batches(np.arange(len(pairs)), int(self.spec["batch_size"]), run, f"{self.spec['id']} {input_set}")
        return pd.DataFrame({"verifier_row_id": ids, "prob_supported": probs})


class LettuceDetect:

    def __init__(self, cfg: Dict, spec: Dict):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.cfg, self.spec, self.device = cfg, spec, _gpu()
        self.tok = AutoTokenizer.from_pretrained(spec["hf_repos"][0])
        self.model = AutoModelForTokenClassification.from_pretrained(spec["hf_repos"][0]).to(self.device).eval()

    @staticmethod
    def prompt(report: str, evidence: List[str]) -> str:
        ctx = [report, "\n".join(evidence) if evidence else "None"]
        block = "\n".join(f"passage {i + 1}: {p}" for i, p in enumerate(ctx))
        return ("Briefly answer the following question:\n" + QUESTION + "\nBear in mind that your response should be "
                f"strictly based on the following {len(ctx)} passages:\n" + block + "\nIn case the passages do not "
                "contain the necessary information to answer the question, please reply with: \"Unable to answer "
                "based on given passages.\"\noutput:\n")

    @torch.no_grad()
    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        ids, prompts, answers = [], [], []
        for rid, rep, ev, st in _rows(self.cfg, df):
            ids.append(rid)
            prompts.append(self.prompt(rep, ev))
            answers.append(st)
        out_ = np.empty(len(ids))

        def run(idx):
            enc = self.tok([prompts[j] for j in idx], [answers[j] for j in idx], truncation="only_first",
                           max_length=4096, padding=True, return_tensors="pt")
            if int(enc["attention_mask"].sum(1).max()) >= 4096:
                raise RuntimeError(f"[main_public_verifiers] {self.spec['id']}: an input of {input_set} reached the length limit")
            probs = torch.softmax(self.model(input_ids=enc["input_ids"].to(self.device),
                                             attention_mask=enc["attention_mask"].to(self.device)).logits.float(),
                                  dim=-1)[..., 1].cpu()
            for b, j in enumerate(idx):
                answer_pos = [t for t, sid in enumerate(enc.sequence_ids(b)) if sid == 1]
                out_[j] = 1.0 - float(probs[b, answer_pos].max())
        oom_batches(np.arange(len(ids)), int(self.spec["batch_size"]), run, f"{self.spec['id']} {input_set}")
        return pd.DataFrame({"verifier_row_id": ids, "prob_supported": out_})


def green_prompt(reference: str, candidate: str, max_len: int = 300) -> str:
    from experiments.green_prompt import make_prompt
    return make_prompt(reference, candidate, max_len)


class GREEN:

    def __init__(self, cfg: Dict, spec: Dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from experiments.green_prompt import CHAT_TEMPLATE
        self.cfg, self.spec, self.device = cfg, spec, _gpu()
        repo = spec["hf_repos"][0]
        self.model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True,
                                                          dtype=torch.float16).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(repo, add_eos_token=True, use_fast=True, trust_remote_code=True,
                                                 padding_side="left")
        self.tok.chat_template = CHAT_TEMPLATE
        self.tok.pad_token = self.tok.eos_token
        self.tok.clean_up_tokenization_spaces = True
        self.cache = os.path.join(score_dir(cfg, spec["id"]), "jsonl", "pairs.jsonl")
        self.ident = {"model": repo, "release": "green_score main 2026-10-11", "max_length": 2048}

    @staticmethod
    def key(reference: str, candidate: str) -> str:
        return hashlib.blake2b(f"{reference}\x00{candidate}".encode("utf-8"), digest_size=12).hexdigest()

    @torch.no_grad()
    def _generate(self, prompts: List[str]) -> List[str]:
        from experiments.green_prompt import clean_responses
        chats = [self.tok.apply_chat_template([{"from": "human", "value": p}, {"from": "gpt", "value": ""}],
                                              tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
        outs = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                   eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id,
                                   max_length=2048, do_sample=False, temperature=None, top_p=None)
        return [clean_responses(r) for r in self.tok.batch_decode(outs, skip_special_tokens=True)]

    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        from experiments.green_prompt import compute_green
        pairs = {}
        rows = []
        for rid, rep, _, st in _rows(self.cfg, df):
            k = self.key(rep, st)
            pairs.setdefault(k, (rep, st))
            rows.append((rid, k))
        done = read_records(self.cache, self.ident, compare=("analysis",))
        todo = [k for k in pairs if k not in done]
        order = sorted(todo, key=lambda k: -len(pairs[k][0]))

        def run(idx):
            part = [order[j] for j in idx]
            texts = self._generate([green_prompt(*pairs[k]) for k in part])
            for k, t in zip(part, texts):
                rec = {"item_id": k, **self.ident, "analysis": t}
                append_record(self.cache, rec)
                done[k] = rec
        oom_batches(np.arange(len(order)), int(self.spec["batch_size"]), run, f"GREEN {input_set}")
        scores = {k: compute_green(done[k]["analysis"]) for k in pairs}
        return pd.DataFrame({"verifier_row_id": [r for r, _ in rows],
                             "prob_supported": [float(scores[k]) for _, k in rows],
                             "pair_key": [k for _, k in rows]})


def minicheck_messages(report: str, evidence: List[str], statement: str) -> List[Dict]:
    return [{"role": "system", "content": MINICHECK_SYSTEM},
            {"role": "user", "content": f"Document: {document(report, evidence)}\nClaim: {statement}"}]


BESPOKE_PINS = ("transformers==4.51.3", "tokenizers==0.21.1", "huggingface_hub==0.35.3", "einops==0.8.1")


class LocalClaimChecker:

    def __init__(self, cfg: Dict, spec: Dict):
        from experiments.generate_reports import _worker_transformers, worker_entry, worker_env
        _gpu()
        self.cfg, self.spec = cfg, spec
        env = worker_env(cfg, package_overlay(out(cfg, "public_worker_overlay"), BESPOKE_PINS, "main_public_verifiers"))
        tv = _worker_transformers(env)
        if tv != BESPOKE_PINS[0].split("==")[1]:
            raise MissingInput(f"{spec['id']} needs {BESPOKE_PINS[0]} in its worker, which imports {tv}")
        prompt = hashlib.blake2b(json.dumps(minicheck_messages("R", ["E"], "S")).encode(), digest_size=8).hexdigest()
        self.ident = {"model": spec["hf_id"], "reading": "p_yes_top5_first_token", "prompt": prompt,
                      "dtype": "bfloat16", "worker_pins": list(BESPOKE_PINS)}
        work = os.path.join(score_dir(cfg, spec["id"]), "worker")
        spec_path = os.path.join(work, "spec.json")
        atomic_write_json({**spec, "ident": self.ident}, spec_path)
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bespoke_worker.py")
        self.proc = subprocess.Popen([sys.executable, "-c", worker_entry(worker), "--spec", spec_path], env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.expect("READY")

    def expect(self, kind: str) -> Dict:
        for line in self.proc.stdout:
            if line.startswith(kind + " "):
                return json.loads(line[len(kind) + 1:])
        rc = self.proc.wait()
        raise GateFailure(f"the {self.spec['id']} worker exited with code {rc} (traceback above); its finished items "
                          f"are kept in the jsonl folder of its scores")

    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        folder = os.path.join(score_dir(self.cfg, self.spec["id"]), "jsonl")
        path, items_path = os.path.join(folder, f"{input_set}.jsonl"), os.path.join(folder, f"{input_set}.items.json")
        items = [{"item_id": rid, "messages": minicheck_messages(rep, ev, st)} for rid, rep, ev, st in _rows(self.cfg, df)]
        atomic_write_json(items, items_path)
        self.proc.stdin.write(json.dumps({"items": items_path, "out": path, "tag": input_set}) + "\n")
        self.proc.stdin.flush()
        self.expect("DONE")
        done = read_records(path, self.ident)
        missing = [it["item_id"] for it in items if it["item_id"] not in done]
        if missing:
            raise RuntimeError(f"[main_public_verifiers] {self.spec['id']} {input_set}: {len(missing)} items have no record")
        recs = [done[it["item_id"]] for it in items]
        from experiments.score import check_answer_share
        check_answer_share(self.cfg, "main_public_verifiers", self.spec["id"], input_set, [r["status"] for r in recs])
        return pd.DataFrame({"verifier_row_id": [it["item_id"] for it in items],
                             "prob_supported": [float(r["score"]) for r in recs], "status": [r["status"] for r in recs]})

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait()


class ServedClaimChecker:

    def __init__(self, cfg: Dict, spec: Dict):
        import httpx
        from openai import OpenAI
        from Inference.llm_client import resolve_secret
        sv, self.cfg, self.spec = cfg["serving"], cfg, spec
        conc = int(spec["concurrency"])
        self.max_tokens = int(spec["max_tokens"])
        self.timeout_s = min(1200.0, 120.0 + 0.12 * self.max_tokens)
        http = httpx.Client(verify=sv.get("verify_ssl", False), timeout=self.timeout_s,
                            limits=httpx.Limits(max_connections=conc, max_keepalive_connections=conc))
        self.client = OpenAI(base_url=resolve_secret(sv["api_base_url"], sv.get("api_base_url_env")),
                             api_key=resolve_secret(sv["api_key"], sv.get("api_key_env")), http_client=http,
                             max_retries=0)
        self.ident = {"model": spec["served"], "kind": spec["kind"],
                      "decoding": {"temperature": 0.0, "max_tokens": self.max_tokens,
                                   "top_logprobs": int(spec["top_logprobs"])}}

    def messages(self, report: str, evidence: List[str], statement: str) -> List[Dict]:
        doc = document(report, evidence)
        return [{"role": "user", "content": LYNX_PROMPT.format(question=QUESTION, context=doc, answer=statement)}]

    def ask(self, msgs: List[Dict]) -> Dict:
        from Inference.llm_client import call_with_retry
        sv = self.cfg["serving"]

        def _call():
            return self.client.chat.completions.create(
                model=self.spec["served"], messages=msgs, temperature=0.0, max_tokens=self.max_tokens, logprobs=True,
                top_logprobs=int(self.spec["top_logprobs"]), timeout=self.timeout_s)
        resp = call_with_retry(_call, int(sv.get("max_retries", 5)), float(sv.get("connect_retry_max_s", 900)),
                               f"public {self.spec['served']}", self.timeout_s)
        ch = resp.choices[0]
        content = ch.logprobs.content if ch.logprobs is not None else None
        rec = lynx_score(content)
        rec.update({"response_id": resp.id, "text": (ch.message.content or "")[:3000],
                    "finish_reason": ch.finish_reason})
        return rec

    def check(self, df: pd.DataFrame):
        rid, rep, ev, st = next(_rows(self.cfg, df))
        a, b = self.ask(self.messages(rep, ev, st)), self.ask(self.messages(rep, ev, st))
        if a["status"] != "ok":
            return f"no answer token could be read ({a['status']}): {a['text'][:160]!r}"
        if a["response_id"] == b["response_id"]:
            return "two identical calls returned one response id, so the proxy serves a cache"
        return None

    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        from Inference.llm_verifier import ask_all
        path = os.path.join(score_dir(self.cfg, self.spec["id"]), "jsonl", f"{input_set}.jsonl")
        items = [{"item_id": rid, "prompt": self.messages(rep, ev, st)} for rid, rep, ev, st in _rows(self.cfg, df)]
        conc = int(self.spec["concurrency"])
        done = ask_all(self, items, path, self.ident, conc, conc * 4, f"{self.spec['id']} {input_set}", "main_public_verifiers")
        recs = [done[it["item_id"]] for it in items]
        from experiments.score import check_answer_share
        check_answer_share(self.cfg, "main_public_verifiers", self.spec["id"], input_set, [r["status"] for r in recs])
        return pd.DataFrame({"verifier_row_id": [it["item_id"] for it in items],
                             "prob_supported": [float(r["score"]) for r in recs], "status": [r["status"] for r in recs]})


def lynx_score(content) -> Dict:
    text, seen_key = "", False
    for pos, tok in enumerate(content or []):
        if seen_key and re.sub(r"[\s\"':]", "", tok.token):
            cands = [[c.token, float(c.logprob)] for c in (tok.top_logprobs or [])]
            norm = lambda t: re.sub(r"[\s\"':]", "", t).upper()
            p_pass = sum(math.exp(lp) for t, lp in cands if norm(t).startswith("PASS"))
            p_fail = sum(math.exp(lp) for t, lp in cands if norm(t).startswith("FAIL"))
            ok = p_pass + p_fail > 0
            return {"score": p_pass / (p_pass + p_fail) if ok else 0.5, "status": "ok" if ok else "fallback_no_verdict",
                    "position": pos, "token": tok.token, "candidates": cands}
        text += tok.token
        seen_key = seen_key or '"SCORE"' in text
    return {"score": 0.5, "status": "fallback_no_score_key", "position": -1, "token": "", "candidates": []}


_LOCAL = {"minicheck_t5": MiniCheckT5, "alignscore": AlignScoreLarge, "hhem": HHEM, "lettuce": LettuceDetect,
          "bespoke_local": LocalClaimChecker,
          "green": GREEN}


def public_identity(spec: Dict) -> Dict:
    keep = {k: v for k, v in spec.items() if k not in ("batch_size", "concurrency", "sweep")}
    return {"regime": "public_verifiers-v1", "spec": keep}


def main_public_verifiers(cfg_path: str, model: str = "") -> str:
    cfg = get_cfg(cfg_path)
    names = [model] if model else [p["id"] for p in cfg["public"]["models"] if p.get("sweep")]
    refused, failed = [], []
    for name in names:
        vx = verifier(cfg, name)
        spec = vx["spec"]
        if spec["kind"] in _LOCAL:
            cls = _LOCAL[spec["kind"]]
            n = run_scoring(cfg, "main_public_verifiers", vx, lambda cls=cls, spec=spec: cls(cfg, spec), public_identity(spec))
        else:
            from experiments.judges import _served
            from experiments.score import all_scored, load_input
            if all_scored(cfg, vx, public_identity(spec)):
                continue
            if spec["served"] not in set(_served(cfg["serving"])):
                refused.append(f"{name} (load {spec['served']})")
                continue
            checker = ServedClaimChecker(cfg, spec)
            problem = checker.check(load_input(cfg, "mimic__unchanged"))
            if problem:
                refused.append(f"{name} ({problem})")
                continue
            try:
                n = run_scoring(cfg, "main_public_verifiers", vx, lambda checker=checker: checker, public_identity(spec))
            except GateFailure as e:
                failed.append(str(e))
                continue
        if n:
            refused.append(f"{name} ({n} sets)")
    from experiments.purge import purge_unused
    purge_unused(cfg)
    if failed:
        raise GateFailure("; ".join(failed + [f"not scored yet: {r}" for r in refused]))
    if refused:
        raise MissingInput(f"not scored yet: {', '.join(refused)}")
    return out(cfg, "scores")
