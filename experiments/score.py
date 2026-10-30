"""
experiments/score.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from experiments._common import (GateFailure, MissingInput, digest_file, flush_status, get_cfg, read_results_csv,
                                 run_units)
from experiments.study import ADDED_CONDITIONS, examples_path, out, read_json, test_sets


def verifiers(cfg: Dict) -> List[Dict]:
    v = cfg
    vs = []
    for r in v["runs"]:
        vs.append({"id": r["id"], "family": r["kind"], "domain": "food" if r["train_set"] == "food" else "radiology",
                   "model_name": r["model_name"], "stage": "main_score_trained"})
    for r in v.get("runs_added", []):
        vs.append({"id": r["id"], "family": r["kind"], "domain": "radiology", "model_name": r["model_name"],
                   "stage": "main_score_added"})
    vs += [{"id": "tfidf_mimic", "family": "tfidf", "domain": "radiology", "train_set": "main", "stage": "main_tfidf"},
           {"id": "tfidf_food", "family": "tfidf", "domain": "food", "train_set": "food", "stage": "main_tfidf"}]
    for p in v["public"]["models"]:
        vs.append({"id": p["id"], "family": "public", "domain": "radiology", "spec": p, "stage": "main_public_verifiers"})
    for m in v["llm"]["models"]:
        vs.append({"id": m["name"], "family": "served", "domain": "radiology", "entry": m, "stage": "main_llm_verifiers"})
    vs.append({"id": v["llm"]["local_name"], "family": "local_llm", "domain": "radiology",
               "model_name": v["llm"]["local_model"], "stage": "main_llm_verifiers"})
    ids = [x["id"] for x in vs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicated verifier ids: {ids}")
    return vs


def verifier(cfg: Dict, vid: str) -> Dict:
    for x in verifiers(cfg):
        if x["id"] == vid:
            return x
    raise ValueError(f"unknown verifier {vid!r}")


def statement_sets(cfg: Dict) -> List[str]:
    from experiments.statements import statement_set_names
    from experiments.evidence_sets import added_statement_set_names
    return statement_set_names(cfg) + added_statement_set_names(cfg)


def input_sets(cfg: Dict, vx: Dict) -> List[str]:
    sets = test_sets(cfg, vx["domain"], vx["id"])
    if vx["domain"] == "radiology":
        sets += statement_sets(cfg)
    if vx["family"] == "public" and vx["spec"]["kind"] == "green":
        sets += ["e6full__rexval", "e6full__radevalx"]
    return sets


def input_path(cfg: Dict, input_set: str) -> str:
    if input_set.startswith("e6full__"):
        return out(cfg, "statements", f"{input_set}.parquet")
    if input_set.startswith("stmt__"):
        from experiments.statements import statement_path
        from experiments.evidence_sets import added_statement_path, is_added_statement
        return added_statement_path(cfg, input_set) if is_added_statement(input_set) else statement_path(cfg, input_set)
    ds, cond = input_set.split("__", 1)
    if cond.startswith("retrieval_"):
        from experiments.controls import retrieval_path
        return retrieval_path(cfg, cond)
    return examples_path(cfg, ds, cond)


def producer_of(input_set: str) -> str:
    from experiments.evidence_sets import is_added_statement
    if input_set.startswith("stmt__"):
        return "main_build_evidence_sets" if is_added_statement(input_set) else "main_statements"
    if input_set.startswith("e6full__"):
        return "main_statements"
    cond = input_set.split("__", 1)[1]
    return "main_retrieval" if cond.startswith("retrieval_") else "main_build_evidence_sets" if cond in ADDED_CONDITIONS else "main_build_examples"


def load_input(cfg: Dict, input_set: str) -> pd.DataFrame:
    path = input_path(cfg, input_set)
    if not os.path.exists(path):
        raise MissingInput(f"{path} is missing; run {producer_of(input_set)} first")
    df = pd.read_parquet(path)
    if df["verifier_row_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated row ids")
    return df


def score_dir(cfg: Dict, vid: str) -> str:
    return out(cfg, "scores", vid)


def score_path(cfg: Dict, vid: str, input_set: str) -> str:
    return os.path.join(score_dir(cfg, vid), "partial", f"{input_set}.csv")


def is_scored(cfg: Dict, vid: str, input_set: str) -> bool:
    p = score_path(cfg, vid, input_set)
    return os.path.exists(p) and os.path.exists(p[:-len(".csv")] + ".params.json")


def load_scores(cfg: Dict, vid: str, input_set: str) -> pd.DataFrame:
    if not is_scored(cfg, vid, input_set):
        raise MissingInput(f"{vid} has no finished scores on {input_set}; score it first")
    p = score_path(cfg, vid, input_set)
    stored = read_json(p[:-len(".csv")] + ".params.json").get("input")
    if stored is not None and stored != digest_file(input_path(cfg, input_set)):
        raise MissingInput(f"the scores of {vid} on {input_set} were computed on an earlier version of that set; run "
                           f"its scoring again")
    s = read_results_csv(p)
    df = load_input(cfg, input_set)
    m = df.merge(s[["verifier_row_id", "prob_supported"]], on="verifier_row_id", how="inner", validate="one_to_one")
    if len(m) != len(df):
        raise ValueError(f"{vid} {input_set}: {len(s)} scores against {len(df)} rows, {len(m)} matched")
    return m.sort_values("verifier_row_id").reset_index(drop=True)


def check_answer_share(cfg: Dict, stage: str, vid: str, input_set: str, status: List[str]) -> None:
    share = float(np.mean([st != "ok" for st in status])) if len(status) else 0.0
    limit = float(cfg["llm"]["max_fallback_share"])
    if share > limit:
        raise GateFailure(f"{vid} gave no answer token on {share:.1%} of {input_set}, above {limit:.0%}; its raw "
                          f"answers are in the jsonl folder of its scores")


def all_scored(cfg: Dict, vx: Dict, identity: Dict) -> bool:
    for s in input_sets(cfg, vx):
        p, ip = score_path(cfg, vx["id"], s), input_path(cfg, s)
        rec = p[:-len(".csv")] + ".params.json"
        if not (os.path.exists(p) and os.path.exists(rec) and os.path.exists(ip)):
            return False
        if read_json(rec) != {**identity, "input": digest_file(ip)}:
            return False
    return True


def run_scoring(cfg: Dict, stage: str, vx: Dict, make_scorer: Callable[[], object], identity: Dict,
                sets: List[str] = None) -> int:
    sets = input_sets(cfg, vx) if sets is None else sets
    units = []
    for s in sets:
        p = input_path(cfg, s)
        units.append({"key": s, "params": {**identity, "input": digest_file(p) if os.path.exists(p) else None}})
    holder = {}

    def compute(unit):
        df = load_input(cfg, unit["key"])
        if "scorer" not in holder:
            holder["scorer"] = make_scorer()
        res = holder["scorer"].score(df, unit["key"])
        if len(res) != len(df):
            raise RuntimeError(f"[{stage}] {vx['id']} {unit['key']}: {len(res)} scores for {len(df)} rows")
        return res.to_dict("records")

    def valid(unit, done):
        p = done["prob_supported"].to_numpy(float)
        return bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())

    status = os.path.join(score_dir(cfg, vx["id"]), f"{stage.lower()}_status.txt")
    _, refused = run_units(f"{stage} {vx['id']}", units, compute, score_dir(cfg, vx["id"]), status, validate=valid)
    if "scorer" in holder:
        close = getattr(holder["scorer"], "close", None)
        if close:
            close()
    if not refused:
        flush_status(status, f"[{stage}] {vx['id']} done, {len(units)} input sets")
    return refused


def load_verifier_from_config(model_name: str, ckpt_path: str, device):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    config = AutoConfig.from_pretrained(model_name, num_labels=2, problem_type="single_label_classification")
    model = AutoModelForSequenceClassification.from_config(config)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in list(missing) + list(unexpected) if not k.endswith(("position_ids", "embed_tokens.weight"))]
    if bad:
        raise RuntimeError(f"{ckpt_path}: state dict keys do not match {model_name}: {bad[:6]}")
    del state
    return model.to(device).eval(), AutoTokenizer.from_pretrained(model_name, use_fast=True)


def score_sorted(model, tok, texts: List[str], device, max_tokens: int, batch_size: int) -> np.ndarray:
    import torch
    from experiments._common import oom_batches
    order = np.argsort([-len(t) for t in texts], kind="mergesort")
    probs = np.empty(len(texts), dtype=np.float64)

    @torch.no_grad()
    def run(idx):
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        probs[idx] = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
    oom_batches(order, batch_size, run, "verifier forward")
    return probs


class EncoderScorer:

    def __init__(self, cfg: Dict, vx: Dict):
        import torch
        from experiments._common import node_and_gpu_line
        from experiments.train_verifiers import done_recipe, model_dir
        self.cfg, self.vx = cfg, vx
        d = model_dir(cfg, vx["id"])
        rec = done_recipe(d)
        if rec is None or not os.path.exists(os.path.join(d, "best_model.pth")):
            raise MissingInput(f"run {vx['id']} is not trained; train it first")
        if not torch.cuda.is_available():
            raise RuntimeError(f"[main_score_trained] no usable GPU: {node_and_gpu_line()}")
        self.device = torch.device("cuda")
        self.max_tokens, self.fit_report = int(rec["max_tokens"]), bool(rec.get("fit_report", False))
        self.model, self.tok = load_verifier_from_config(vx["model_name"], os.path.join(d, "best_model.pth"),
                                                         self.device)

    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        from data_loader.verifier_dataset import build_input_text, fit_inputs, materialize_evidence
        v = self.cfg
        k = int(df["evidence_k"].iloc[0]) if "evidence_k" in df.columns else int(v["max_evidence_sentences"])
        chars = int(v["max_evidence_chars"])
        evidence = [materialize_evidence(e, k, chars) for e in df["evidence_sentences_json"]]
        cases, claims = df["case_context"].astype(str).tolist(), df["claim_text"].astype(str).tolist()
        if self.fit_report:
            texts, _ = fit_inputs(self.tok, cases, evidence, claims, self.max_tokens)
        else:
            texts = [build_input_text(c, e, s) for c, e, s in zip(cases, evidence, claims)]
        p = score_sorted(self.model, self.tok, texts, self.device, self.max_tokens, int(v["training"]["eval_batch_size"]))
        return pd.DataFrame({"verifier_row_id": df["verifier_row_id"].to_numpy(), "prob_supported": p})

    def close(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


class LocalLLMScorer:

    def __init__(self, cfg: Dict, vx: Dict, adapter_dir: str = None):
        import torch
        from experiments._common import node_and_gpu_line
        from experiments.llm_local import load_base, with_adapter
        from experiments.train_verifiers import lora_cfg
        if adapter_dir is not None:
            from experiments.llm_local import lora_is_done
            if not lora_is_done(adapter_dir):
                raise MissingInput(f"run {vx['id']} is not trained; train it first")
        if not torch.cuda.is_available():
            raise RuntimeError(f"[{vx['stage']}] no usable GPU: {node_and_gpu_line()}")
        self.cfg, self.vx = cfg, vx
        model, self.tok = load_base(vx["model_name"], torch.device("cuda"))
        if adapter_dir is not None:
            model = with_adapter(model, lora_cfg(cfg), os.path.join(adapter_dir, "best_adapter.pt"))
        self.model = model.eval()

    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        from experiments.llm_local import prompts_of, score_prompts
        llm = self.cfg["llm"]
        recs = score_prompts(self.model, self.tok, prompts_of(df, int(self.cfg["max_evidence_sentences"])),
                             int(llm["local_batch_size"]), int(llm["top_logprobs"]), int(llm["max_tokens"]),
                             desc=f"{self.vx['id']} {input_set}")
        check_answer_share(self.cfg, self.vx["stage"], self.vx["id"], input_set, [r["status"] for r in recs])
        return pd.DataFrame({"verifier_row_id": df["verifier_row_id"].to_numpy(),
                             "prob_supported": [r["score"] for r in recs], "status": [r["status"] for r in recs],
                             "p_yes": [r["p_yes"] for r in recs], "p_no": [r["p_no"] for r in recs],
                             "token": [r["token"] for r in recs]})

    def close(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


def trained_identity(cfg: Dict, vx: Dict) -> Dict:
    from experiments.train_verifiers import model_dir
    flag = os.path.join(model_dir(cfg, vx["id"]), "done.flag")
    ident = {"regime": "score_trained-v1", "verifier": vx["id"], "done_flag": digest_file(flag) if os.path.exists(flag) else None}
    if vx["family"] == "encoder":
        ident.update({"precision": "float32", "max_evidence_sentences": int(cfg["max_evidence_sentences"]),
                      "max_evidence_chars": int(cfg["max_evidence_chars"])})
    else:
        from Inference.llm_verifier import PROMPT_DIGEST
        llm = cfg["llm"]
        ident.update({"precision": "bf16", "max_new_tokens": int(llm["max_tokens"]),
                      "top_logprobs": int(llm["top_logprobs"]), "prompt": PROMPT_DIGEST})
    return ident


def main_score_trained(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    from experiments.train_verifiers import model_dir
    refused, failed = [], []
    for vx in [x for x in verifiers(cfg) if x["stage"] == "main_score_trained"]:
        if vx["family"] == "encoder":
            make = lambda vx=vx: EncoderScorer(cfg, vx)
        else:
            make = lambda vx=vx: LocalLLMScorer(cfg, vx, adapter_dir=model_dir(cfg, vx["id"]))
        try:
            n = run_scoring(cfg, "main_score_trained", vx, make, trained_identity(cfg, vx))
        except GateFailure as e:
            failed.append(str(e))
            continue
        if n:
            refused.append(f"{vx['id']} ({n} sets)")
    from experiments.purge import purge_unused
    purge_unused(cfg)
    if failed:
        raise GateFailure("; ".join(failed))
    if refused:
        raise MissingInput(f"not scored yet: {', '.join(refused)}")
    return out(cfg, "scores")
