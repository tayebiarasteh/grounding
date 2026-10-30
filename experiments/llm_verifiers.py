"""
experiments/llm_verifiers.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import pandas as pd

from experiments._common import GateFailure, MissingInput, get_cfg
from Inference.llm_verifier import LLMVerifier, ask_all, build_prompt
from experiments.study import out, read_evidence
from experiments.score import LocalLLMScorer, all_scored, check_answer_share, run_scoring, score_dir, verifier, verifiers


class ServedScorer:

    def __init__(self, cfg: Dict, vx: Dict, llm: LLMVerifier):
        self.cfg, self.vx, self.llm = cfg, vx, llm
        lc = cfg["llm"]
        self.conc, self.chunk = int(lc["concurrency"]), int(lc["concurrency"]) * int(lc["chunk_per_worker"])

    def score(self, df: pd.DataFrame, input_set: str) -> pd.DataFrame:
        k = int(self.cfg["max_evidence_sentences"])
        items = [{"item_id": int(r), "prompt": build_prompt(c, read_evidence(e, k), s)}
                 for r, c, e, s in zip(df["verifier_row_id"], df["case_context"], df["evidence_sentences_json"],
                                       df["claim_text"])]
        path = os.path.join(score_dir(self.cfg, self.vx["id"]), "jsonl", f"{input_set}.jsonl")
        done = ask_all(self.llm, items, path, self.llm.identity(), self.conc, self.chunk,
                       f"{self.vx['id']} {input_set}", "main_llm_verifiers")
        recs = [done[it["item_id"]] for it in items]
        check_answer_share(self.cfg, "main_llm_verifiers", self.vx["id"], input_set, [r["status"] for r in recs])
        return pd.DataFrame({"verifier_row_id": df["verifier_row_id"].to_numpy(),
                             "prob_supported": [float(r["score"]) for r in recs],
                             "status": [r["status"] for r in recs], "p_yes": [float(r["p_yes"]) for r in recs],
                             "p_no": [float(r["p_no"]) for r in recs], "response_id": [r["response_id"] for r in recs]})


def _served_identity(cfg: Dict, vx: Dict) -> Dict:
    return {"regime": "llm_verifiers-v1", **LLMVerifier(cfg["serving"], cfg["llm"], vx["entry"]).identity()}


def local_identity(cfg: Dict, vx: Dict) -> Dict:
    from Inference.llm_verifier import PROMPT_DIGEST
    lc = cfg["llm"]
    return {"regime": "llm_verifiers-v1", "model": vx["model_name"], "precision": "bf16", "prompt": PROMPT_DIGEST,
            "max_new_tokens": int(lc["max_tokens"]), "top_logprobs": int(lc["top_logprobs"])}


def served_identity(cfg: Dict, vx: Dict) -> Dict:
    return _served_identity(cfg, vx)


def _one(cfg: Dict, vx: Dict) -> int:
    lc = cfg["llm"]
    if vx["family"] == "local_llm":
        return run_scoring(cfg, "main_llm_verifiers", vx, lambda: LocalLLMScorer(cfg, vx), local_identity(cfg, vx))
    llm = LLMVerifier(cfg["serving"], lc, vx["entry"])
    from experiments.score import load_input
    if all_scored(cfg, vx, _served_identity(cfg, vx)):
        return 0
    first = load_input(cfg, "mimic__unchanged").iloc[0]
    problem = llm.check(build_prompt(first["case_context"],
                                     read_evidence(first["evidence_sentences_json"],
                                                   int(cfg["max_evidence_sentences"])),
                                     first["claim_text"]))
    if problem:
        raise MissingInput(f"{vx['id']} stopped: {problem}")
    return run_scoring(cfg, "main_llm_verifiers", vx, lambda: ServedScorer(cfg, vx, llm), _served_identity(cfg, vx))


def main_llm_verifiers(cfg_path: str, model: str = "") -> str:
    from experiments.judges import _served
    cfg = get_cfg(cfg_path)
    if model:
        names = [model]
    else:
        names = [vx["id"] for vx in verifiers(cfg) if vx["family"] == "served"]
    served: List[str] = []
    refused, failed = [], []
    for name in names:
        vx = verifier(cfg, name)
        if vx["family"] not in ("served", "local_llm"):
            raise ValueError(f"{name} is not an LLM verifier")
        if vx["family"] == "served" and not all_scored(cfg, vx, _served_identity(cfg, vx)):
            if not served:
                served.extend(_served(cfg["serving"]))
            if vx["entry"]["served"] not in served:
                refused.append(f"{name} (load {vx['entry']['served']})")
                continue
        try:
            n = _one(cfg, vx)
        except MissingInput as e:
            refused.append(f"{name} ({e})")
            continue
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
        raise MissingInput(f"not scored yet: {'; '.join(refused)}")
    return out(cfg, "scores")
