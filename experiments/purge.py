"""
experiments/purge.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import gc
import os
import shutil
import sys
from typing import Callable, Dict, List


def _repo_dir(cache_dir: str, repo_id: str) -> str:
    try:
        from huggingface_hub import repo_folder_name
        return os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model"))
    except ImportError:
        return os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))


def _cache_dirs(cfg: Dict) -> List[str]:
    dirs = [cfg["hf_cache_dir"]]
    try:
        from huggingface_hub import constants
        dirs.append(constants.HF_HUB_CACHE)
    except ImportError:
        pass
    seen, out = set(), []
    for d in dirs:
        key = os.path.realpath(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _uses(cfg: Dict) -> Dict[str, List[Callable[[], bool]]]:
    from experiments.llm_local import lora_is_done
    from experiments.train_verifiers import model_dir
    from experiments.score import input_sets, is_scored, verifiers
    uses: Dict[str, List[Callable[[], bool]]] = {}

    def pending(vx):
        return lambda: any(not is_scored(cfg, vx["id"], s) for s in input_sets(cfg, vx))

    for vx in verifiers(cfg):
        if vx["family"] == "encoder":
            d = model_dir(cfg, vx["id"])
            uses.setdefault(vx["model_name"], []).append(lambda d=d: not os.path.exists(os.path.join(d, "done.flag")))
        elif vx["family"] == "lora":
            d = model_dir(cfg, vx["id"])
            uses.setdefault(vx["model_name"], []).append(lambda d=d: not lora_is_done(d))
            uses[vx["model_name"]].append(pending(vx))
        elif vx["family"] == "local_llm":
            uses.setdefault(vx["model_name"], []).append(pending(vx))
        elif vx["family"] == "public":
            for repo in vx["spec"]["hf_repos"]:
                uses.setdefault(repo, []).append(pending(vx))
    from experiments.generate_reports import generators, is_generated
    for g in generators(cfg):
        for repo in g.get("hf_repos", []):
            uses.setdefault(repo, []).append(lambda name=g["name"]: not is_generated(cfg, name))
    from experiments.study import TEST_SETS, out
    from experiments.statements import sources

    def labeling_pending():
        label_params = [out(cfg, "report_labels", "partial", f"{n}.params.json") for n in TEST_SETS]
        statement_params = [out(cfg, "statements", "partial", f"{n}.params.json") for n in sources(cfg)]
        return not all(os.path.exists(p) for p in label_params + statement_params)
    for repo in (cfg["chexbert"]["repo_id"], cfg["chexbert"]["tokenizer"]):
        uses.setdefault(repo, []).append(labeling_pending)
    return uses


def purge_unused(cfg: Dict) -> None:
    v = cfg
    if not v.get("purge_hf_cache_after_use", False):
        return
    gc.collect()
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dirs = _cache_dirs(cfg)
    for repo, checks in _uses(cfg).items():
        folders = [f for f in (_repo_dir(d, repo) for d in dirs) if os.path.isdir(f)]
        if not folders or any(check() for check in checks):
            continue
        for folder in folders:
            try:
                shutil.rmtree(folder)
            except OSError:
                pass
