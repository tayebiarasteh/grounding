"""
experiments/train_verifiers.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
import time
from typing import Dict, Optional

from experiments._common import MissingInput, claim_beat, claim_unit, get_cfg, release_claim, require
from experiments.study import out, train_set_path


def run_spec(cfg: Dict, run_id: str) -> Dict:
    runs = {r["id"]: r for r in cfg["runs"] + list(cfg.get("runs_added", []))}
    if run_id not in runs:
        raise ValueError(f"unknown run {run_id!r}; the config lists {sorted(runs)}")
    return runs[run_id]


def model_dir(cfg: Dict, run_id: str) -> str:
    return out(cfg, "models", run_id)


def encoder_max_tokens(model_name: str, requested: int) -> int:
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_name)
    limit = getattr(c, "max_position_embeddings", None)
    if limit is None or c.model_type == "t5":
        return int(requested)
    usable = limit - 2 if c.model_type in ("roberta", "longformer") else limit
    return int(min(requested, usable))


def encoder_cfg(cfg: Dict, spec: Dict) -> Dict:
    v = cfg
    tr = dict(v["training"])
    if "lr" in spec:
        tr["lr"] = float(spec["lr"])
    tr["max_tokens"] = encoder_max_tokens(spec["model_name"], int(tr["max_tokens"]))
    return {"training": tr, "seed": int(spec["seed"]),
            "verifier": {"model_name": spec["model_name"], "max_evidence_sentences": int(v["max_evidence_sentences"]),
                         "max_evidence_chars": int(v["max_evidence_chars"]),
                         "eval_batch_size": int(tr["eval_batch_size"])}}


def lora_cfg(cfg: Dict) -> Dict:
    v = cfg
    return {**v["lora"], "max_evidence_sentences": int(v["max_evidence_sentences"])}


def done_recipe(out_dir: str) -> Optional[Dict]:
    path = os.path.join(out_dir, "done.flag")
    if not os.path.exists(path):
        return None
    for line in open(path).read().splitlines():
        if line.startswith("recipe="):
            return json.loads(line[len("recipe="):])
    return {}


def refuse_stale(out_dir: str, current: Dict, tag: str) -> None:
    stored = done_recipe(out_dir)
    if stored is None:
        return
    moved = [f"{k}: {stored.get(k)!r} -> {v!r}" for k, v in current.items() if stored.get(k) != v]
    if moved:
        aside = f"{out_dir}.refused_{time.strftime('%Y%m%d_%H%M%S')}"
        os.replace(out_dir, aside)


def main_train(cfg_path: str, run: str) -> str:
    cfg = get_cfg(cfg_path)
    spec = run_spec(cfg, run)
    from experiments.study import ADDED_TRAIN_SETS
    parquet = require(train_set_path(cfg, spec["train_set"]), "main_build_evidence_sets" if spec["train_set"] in ADDED_TRAIN_SETS else "main_build_examples")
    out_dir = model_dir(cfg, run)
    claims = out(cfg, "models", "claims")
    if not claim_unit(claims, run):
        raise MissingInput(f"another job holds run {run}; run it again once that job has finished")
    try:
        with claim_beat(claims, run):
            path = _train(cfg, spec, parquet, out_dir)
    finally:
        release_claim(claims, run)
    from experiments.purge import purge_unused
    purge_unused(cfg)
    return path


def main_train_added(cfg_path: str, run: str) -> str:
    return main_train(cfg_path, run)


def _train(cfg: Dict, spec: Dict, parquet: str, out_dir: str) -> str:
    run = spec["id"]
    if spec["kind"] == "encoder":
        from controlled.train_verifier import recipe, train_verifier
        tcfg = encoder_cfg(cfg, spec)
        refuse_stale(out_dir, recipe(tcfg, parquet), f"train_verifiers {run}")
        return train_verifier(tcfg, parquet, out_dir)
    if spec["kind"] == "lora":
        from experiments.llm_local import lora_recipe, train_lora
        lc = lora_cfg(cfg)
        refuse_stale(out_dir, lora_recipe(lc, spec["model_name"], parquet, int(spec["seed"])), f"train_verifiers {run}")
        return train_lora(lc, spec["model_name"], parquet, out_dir, int(spec["seed"]))
    raise ValueError(f"run {run}: unknown kind {spec['kind']!r}")
