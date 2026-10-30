"""
experiments/score_added.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

from experiments._common import GateFailure, MissingInput, digest_file, get_cfg
from experiments.study import out
from experiments.score import (EncoderScorer, LocalLLMScorer, input_path, input_sets, is_scored, read_json,
                               run_scoring, score_path, trained_identity, verifier, verifiers)
from experiments.evidence_sets import added_sets

PUBLIC_ARGUMENT = "public"


def added_main_verifier(cfg: Dict) -> Dict:
    vx = verifier(cfg, cfg["main_verifier_added"])
    if vx["stage"] != "main_score_added":
        raise ValueError(f"main_verifier_added {vx['id']!r} is not a run of runs_added")
    return vx


def sets_of(cfg: Dict, vx: Dict) -> List[str]:
    from experiments.controls import RETRIEVAL
    if vx["stage"] == "main_score_added":
        sets = input_sets(cfg, vx)
        if vx["id"] == cfg["main_verifier_added"]:
            sets += [f"mimic__{c}" for c in RETRIEVAL]
        return sets
    added = set(added_sets(cfg))
    return [s for s in input_sets(cfg, vx) if s in added]


def sets_scored(cfg: Dict, vx: Dict, identity: Dict, sets: List[str]) -> bool:
    for s in sets:
        p, ip = score_path(cfg, vx["id"], s), input_path(cfg, s)
        rec = p[:-len(".csv")] + ".params.json"
        if not (os.path.exists(p) and os.path.exists(rec) and os.path.exists(ip)):
            return False
        if read_json(rec) != {**identity, "input": digest_file(ip)}:
            return False
    return True


def _trained(cfg: Dict, refused: List[str], failed: List[str]) -> None:
    from experiments.train_verifiers import model_dir
    from experiments.tfidf_baseline import TfidfScorer, model_path, tfidf_identity
    for vx in verifiers(cfg):
        if vx["family"] not in ("encoder", "lora", "tfidf") or vx["domain"] != "radiology":
            continue
        sets = sets_of(cfg, vx)
        if not sets:
            continue
        try:
            if vx["family"] == "tfidf":
                if not os.path.exists(model_path(cfg, vx["id"])):
                    raise MissingInput(f"{vx['id']} is not fitted; run main_tfidf first")
                ident, make = tfidf_identity(cfg, vx["id"]), (lambda vx=vx: TfidfScorer(cfg, vx["id"]))
            elif vx["family"] == "encoder":
                ident, make = trained_identity(cfg, vx), (lambda vx=vx: EncoderScorer(cfg, vx))
            else:
                ident = trained_identity(cfg, vx)
                make = lambda vx=vx: LocalLLMScorer(cfg, vx, adapter_dir=model_dir(cfg, vx["id"]))
            n = run_scoring(cfg, "main_score_added", vx, make, ident, sets=sets)
        except MissingInput as e:
            refused.append(f"{vx['id']} ({e})")
            continue
        except GateFailure as e:
            failed.append(str(e))
            continue
        if n:
            refused.append(f"{vx['id']} ({n} sets)")


def _public(cfg: Dict, refused: List[str], failed: List[str]) -> None:
    from experiments.public_verifiers import _LOCAL, public_identity
    for vx in verifiers(cfg):
        if vx["family"] != "public" or vx["spec"]["kind"] not in _LOCAL:
            continue
        spec, cls = vx["spec"], _LOCAL[vx["spec"]["kind"]]
        try:
            n = run_scoring(cfg, "main_score_added", vx, lambda cls=cls, spec=spec: cls(cfg, spec), public_identity(spec),
                            sets=sets_of(cfg, vx))
        except MissingInput as e:
            refused.append(f"{vx['id']} ({e})")
            continue
        except GateFailure as e:
            failed.append(str(e))
            continue
        if n:
            refused.append(f"{vx['id']} ({n} sets)")


def _served(cfg: Dict, vx: Dict, refused: List[str], failed: List[str]) -> None:
    from experiments.judges import _served as served_models
    from experiments.score import load_input
    sets = sets_of(cfg, vx)
    if vx["family"] == "served":
        from experiments.llm_verifiers import served_identity
        ident, name = served_identity(cfg, vx), vx["entry"]["served"]
    else:
        from experiments.public_verifiers import public_identity
        ident, name = public_identity(vx["spec"]), vx["spec"]["served"]
    if sets_scored(cfg, vx, ident, sets):
        return
    if name not in set(served_models(cfg["serving"])):
        refused.append(f"{vx['id']} (load {name})")
        return
    first = load_input(cfg, "mimic__unchanged")
    if vx["family"] == "served":
        from experiments.llm_verifiers import ServedScorer
        from Inference.llm_verifier import LLMVerifier, build_prompt
        from experiments.study import read_evidence
        llm = LLMVerifier(cfg["serving"], cfg["llm"], vx["entry"])
        row = first.iloc[0]
        problem = llm.check(build_prompt(row["case_context"], read_evidence(row["evidence_sentences_json"],
                                                                            int(cfg["max_evidence_sentences"])),
                                         row["claim_text"]))
        make = lambda: ServedScorer(cfg, vx, llm)
    else:
        from experiments.public_verifiers import ServedClaimChecker
        checker = ServedClaimChecker(cfg, vx["spec"])
        problem = checker.check(first)
        make = lambda: checker
    if problem:
        refused.append(f"{vx['id']} ({problem})")
        return
    try:
        n = run_scoring(cfg, "main_score_added", vx, make, ident, sets=sets)
    except GateFailure as e:
        failed.append(str(e))
        return
    if n:
        refused.append(f"{vx['id']} ({n} sets)")


def main_score_added(cfg_path: str, model: str = "") -> str:
    cfg = get_cfg(cfg_path)
    added_main_verifier(cfg)
    refused, failed = [], []
    if not model:
        _trained(cfg, refused, failed)
    elif model == PUBLIC_ARGUMENT:
        _public(cfg, refused, failed)
    else:
        vx = verifier(cfg, model)
        if vx["family"] == "local_llm":
            from experiments.llm_verifiers import local_identity
            try:
                n = run_scoring(cfg, "main_score_added", vx, lambda: LocalLLMScorer(cfg, vx), local_identity(cfg, vx), sets=sets_of(cfg, vx))
            except GateFailure as e:
                failed.append(str(e))
                n = 0
            if n:
                refused.append(f"{vx['id']} ({n} sets)")
        elif vx["family"] == "served" or (vx["family"] == "public" and vx["spec"].get("served")):
            _served(cfg, vx, refused, failed)
        else:
            raise ValueError(f"{model} cannot be scored here; use no model, '{PUBLIC_ARGUMENT}', the in-process "
                             f"LLM or a served verifier")
    from experiments.purge import purge_unused
    purge_unused(cfg)
    if failed:
        raise GateFailure("; ".join(failed + [f"not scored yet: {r}" for r in refused]))
    if refused:
        raise MissingInput(f"not scored yet: {'; '.join(refused)}")
    return out(cfg, "scores")


def pending_added(cfg: Dict) -> List[str]:
    return [f"{vx['id']} {s}" for vx in verifiers(cfg) for s in sets_of(cfg, vx) if not is_scored(cfg, vx["id"], s)]
