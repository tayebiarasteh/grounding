"""
experiments/generate_reports.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments._common import (GateFailure, MissingInput, atomic_write_json, flush_status, get_cfg,
                                 read_results_csv, run_units)
from experiments.overlay import package_overlay
from Inference.llm_verifier import append_record, read_records
from experiments.study import openi_eligible, out, read_json, read_openi
from experiments.generators_inprocess import REPORT_PROMPT

PROMPT_DIGEST = hashlib.blake2b(REPORT_PROMPT.encode("utf-8"), digest_size=4).hexdigest()


def generators(cfg: Dict) -> List[Dict]:
    return list(cfg["generators"]["models"])


def generator(cfg: Dict, name: str) -> Dict:
    for g in generators(cfg):
        if g["name"] == name:
            return g
    raise ValueError(f"unknown generator {name!r}")


def image_tree(cfg: Dict) -> str:
    return os.path.basename(os.path.normpath(cfg["sources"]["openi_images_dir"]))


_ITEMS: Dict[tuple, List[Dict]] = {}


def items(cfg: Dict) -> List[Dict]:
    src = cfg["sources"]
    key = (src["openi_reports_dir"], src["openi_frontal"], src["openi_images_dir"])
    if key not in _ITEMS:
        oi = openi_eligible(read_openi(src["openi_reports_dir"]))
        fr = read_results_csv(src["openi_frontal"], dtype={"report_id": str, "image_id": str})
        fr = dict(zip(fr["report_id"], fr["image_id"]))
        out_items = []
        for r, f1 in zip(oi["report_id"], oi["f1_image_id"]):
            p = os.path.join(src["openi_images_dir"], f"{fr.get(r, '')}.png")
            if r in fr and os.path.exists(p):
                out_items.append({"item_id": r, "image_id": fr[r], "f1_image_id": f1, "image": p})
        _ITEMS[key] = out_items
    return _ITEMS[key]


def jsonl_path(cfg: Dict, name: str) -> str:
    return out(cfg, "generated_reports", "jsonl", f"{name}.frontal.jsonl")


def generations_path(cfg: Dict, name: str) -> str:
    return out(cfg, "generated_reports", "partial", f"{name}.csv")


def unit_params(cfg: Dict, spec: Dict, its: List[Dict]) -> Dict:
    what = {k: spec[k] for k in ("name", "kind", "hf_id", "served", "model_base", "model_name") if k in spec}
    params = {"regime": "generate_reports-v1", "spec": what, "prompt": PROMPT_DIGEST, "images": image_tree(cfg),
              "max_new_tokens": int(cfg["generators"]["max_new_tokens"]),
              "items": hashlib.blake2b("|".join(f"{i['item_id']}:{i['image_id']}" for i in its).encode(),
                                       digest_size=8).hexdigest()}
    if spec["kind"] != "served":
        params["worker_pins"] = list(WORKER_PINS)
    return params


def is_generated(cfg: Dict, name: str) -> bool:
    p = generations_path(cfg, name)
    rec = p[:-len(".csv")] + ".params.json"
    if not (os.path.exists(p) and os.path.exists(rec)):
        return False
    return read_json(rec) == json.loads(json.dumps(unit_params(cfg, generator(cfg, name), items(cfg))))


def _seed_from_first_figure(cfg: Dict, spec: Dict, ident: Dict, its: List[Dict]) -> None:
    new, old = jsonl_path(cfg, spec["name"]), out(cfg, "generated_reports", "jsonl", f"{spec['name']}.jsonl")
    if os.path.exists(new) or not os.path.exists(old):
        return
    prev = read_records(old, ident, compare=("text",))
    keep = [{**prev[it["item_id"]], "image_id": it["image_id"]} for it in its
            if it["item_id"] in prev and it["image_id"] == it["f1_image_id"]]
    tmp = f"{new}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="ascii") as f:
        f.writelines(json.dumps(r, ensure_ascii=True) + "\n" for r in keep)
    os.replace(tmp, new)


def _check_images(done: Dict, its: List[Dict], path: str) -> Dict:
    want = {it["item_id"]: it["image_id"] for it in its}
    wrong = [k for k, r in done.items() if k in want and r.get("image_id") != want[k]]
    if wrong:
        raise RuntimeError(f"[main_generate_reports] {path} holds {len(wrong)} texts of other images (e.g. report {wrong[0]}); the frontal "
                           f"selection changed, so delete that file and run it again")
    return {k: r for k, r in done.items() if k in want}


def load_generations(cfg: Dict, name: str) -> pd.DataFrame:
    if not is_generated(cfg, name):
        raise MissingInput(f"generator {name} has no finished findings; generate them first")
    return read_results_csv(generations_path(cfg, name), keep_default_na=False, dtype={"report_id": str})


class ServedGenerator:

    def __init__(self, cfg: Dict, spec: Dict):
        import httpx
        from openai import OpenAI
        from Inference.llm_client import resolve_secret
        g, sv = cfg["generators"], cfg["serving"]
        self.spec, self.sv = spec, sv
        self.max_tokens = int(g["max_new_tokens"])
        self.timeout_s = min(1200.0, 120.0 + 0.12 * self.max_tokens)
        conc = int(g["concurrency"])
        http = httpx.Client(verify=sv.get("verify_ssl", False), timeout=self.timeout_s,
                            limits=httpx.Limits(max_connections=conc, max_keepalive_connections=conc))
        self.client = OpenAI(base_url=resolve_secret(sv["api_base_url"], sv.get("api_base_url_env")),
                             api_key=resolve_secret(sv["api_key"], sv.get("api_key_env")), http_client=http,
                             max_retries=0)
        self.ident = {"model": spec["served"], "prompt": PROMPT_DIGEST, "images": image_tree(cfg),
                      "decoding": {"temperature": 0.0, "max_tokens": self.max_tokens}}

    def ask(self, image_path: str) -> Dict:
        from PIL import Image
        from Inference.llm_client import call_with_retry
        buf = io.BytesIO()
        Image.open(image_path).convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                             {"type": "text", "text": REPORT_PROMPT}]}]

        def _call():
            return self.client.chat.completions.create(model=self.spec["served"], messages=msgs, temperature=0.0,
                                                       max_tokens=self.max_tokens, timeout=self.timeout_s)
        resp = call_with_retry(_call, int(self.sv.get("max_retries", 5)), float(self.sv.get("connect_retry_max_s", 900)),
                               f"generator {self.spec['served']}", self.timeout_s)
        ch = resp.choices[0]
        return {"text": (ch.message.content or "").strip(), "finish_reason": ch.finish_reason,
                "response_id": resp.id}

    def check(self, image_path: str):
        a, b = self.ask(image_path), self.ask(image_path)
        if not a["text"]:
            return f"it returned no text ({a['finish_reason']})"
        if a["finish_reason"] == "length":
            return f"its text ran to the {self.max_tokens}-token limit: {a['text'][:160]!r}"
        if a["response_id"] == b["response_id"]:
            return "two identical calls returned one response id, so the proxy serves a cache"
        return None


def _run_served(cfg: Dict, spec: Dict, its: List[Dict]) -> Dict:
    gen = ServedGenerator(cfg, spec)
    path = jsonl_path(cfg, spec["name"])
    _seed_from_first_figure(cfg, spec, gen.ident, its)
    done = _check_images(read_records(path, gen.ident, compare=("text",)), its, path)
    image_of = {it["item_id"]: it["image_id"] for it in its}
    todo = [it for it in its if it["item_id"] not in done]
    if todo:
        problem = gen.check(todo[0]["image"])
        if problem:
            raise MissingInput(f"{spec['served']} was not run because {problem}")
    conc = int(cfg["generators"]["concurrency"])

    def one(it):
        try:
            return it["item_id"], gen.ask(it["image"])
        except Exception as e:
            return it["item_id"], {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    failed, in_row = [], 0
    with ThreadPoolExecutor(max_workers=conc) as pool, tqdm(total=len(todo), desc=spec["name"], unit="report") as bar:
        for i in range(0, len(todo), conc * 4):
            for rid, res in pool.map(one, todo[i:i + conc * 4]):
                if "error" in res:
                    failed.append(res["error"])
                    in_row += 1
                    if in_row >= 30:
                        raise RuntimeError(f"[main_generate_reports] {spec['name']}: 30 calls failed in a row ({res['error']})")
                    continue
                in_row = 0
                rec = {"item_id": rid, **gen.ident, **res, "image_id": image_of[rid]}
                append_record(path, rec)
                done[rid] = rec
            bar.update(len(todo[i:i + conc * 4]))
    if failed:
        raise RuntimeError(f"[main_generate_reports] {spec['name']}: {len(failed)} calls failed ({failed[0]}); run it again")
    return done


INPROCESS_TRANSFORMERS = ("4.48", "4.52")
WORKER_PINS = ("transformers==4.51.3", "tokenizers==0.21.1", "huggingface_hub==0.35.3")


def worker_overlay(cfg: Dict) -> str:
    return package_overlay(out(cfg, "generated_reports", "worker_overlay"), WORKER_PINS, "main_generate_reports")


def worker_env(cfg: Dict, overlay: str) -> Dict[str, str]:
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(p for p in (overlay, os.environ.get("PYTHONPATH", "")) if p))
    if cfg.get("hf_token"):
        env["HF_TOKEN"] = str(cfg["hf_token"])
    return env


def worker_entry(path: str) -> str:
    return f"import runpy, sys; sys.path[0] = {os.path.dirname(path)!r}; runpy.run_path({path!r})['main']()"


def _worker_transformers(env: Dict[str, str]) -> str:
    r = subprocess.run([sys.executable, "-c", "import transformers; print(transformers.__version__)"], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"none ({(r.stderr.strip().splitlines() or ['no message'])[-1]})"
    return r.stdout.strip()


def _run_inprocess(cfg: Dict, spec: Dict, its: List[Dict]) -> Dict:
    from packaging.version import Version
    if spec["kind"] == "llava_rad" and not os.path.isdir(spec["repo_path"]):
        raise MissingInput(f"the LLaVA-Rad clone {spec['repo_path']} is missing; clone microsoft/LLaVA-Rad there and "
                           f"run it again")
    env = worker_env(cfg, worker_overlay(cfg))
    tv = _worker_transformers(env)
    if not (tv[:1].isdigit() and Version(INPROCESS_TRANSFORMERS[0]) <= Version(tv) < Version(INPROCESS_TRANSFORMERS[1])):
        raise MissingInput(f"{spec['name']} needs transformers >={INPROCESS_TRANSFORMERS[0]},<{INPROCESS_TRANSFORMERS[1]}, "
                           f"because its code subclasses the LLaVA classes that 4.52 rewrote; its worker imports {tv}")
    work = out(cfg, "generated_reports", "worker")
    os.makedirs(work, exist_ok=True)
    spec_path, items_path = os.path.join(work, f"{spec['name']}.spec.json"), os.path.join(work, f"{spec['name']}.items.json")
    atomic_write_json({**spec, "prompt_digest": PROMPT_DIGEST, "images": image_tree(cfg),
                       "max_new_tokens": int(cfg["generators"]["max_new_tokens"]), "worker_pins": list(WORKER_PINS)},
                      spec_path)
    atomic_write_json(its, items_path)
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generators_inprocess.py")
    rc = subprocess.run([sys.executable, "-c", worker_entry(worker), "--spec", spec_path, "--items", items_path,
                         "--out", jsonl_path(cfg, spec["name"])], env=env).returncode
    if rc != 0:
        raise GateFailure(f"the {spec['name']} worker exited with code {rc} (traceback above); its finished reports "
                          f"are kept in {jsonl_path(cfg, spec['name'])}")
    ident = {"model": spec["hf_id"], "prompt": PROMPT_DIGEST, "images": image_tree(cfg),
             "max_new_tokens": int(cfg["generators"]["max_new_tokens"]), "worker_pins": list(WORKER_PINS)}
    path = jsonl_path(cfg, spec["name"])
    return _check_images(read_records(path, ident, compare=("text",)), its, path)


def main_generate_reports(cfg_path: str, generator_name: str = "") -> str:
    cfg = get_cfg(cfg_path)
    g = cfg["generators"]
    its = items(cfg)
    if generator_name:
        specs = [generator(cfg, generator_name)]
    else:
        specs = [s for s in generators(cfg) if s["kind"] == "served"]
    out_dir = out(cfg, "generated_reports")
    status = os.path.join(out_dir, "status.txt")
    missing: List[str] = []
    served: List[str] = []
    failed: List[str] = []
    for spec in specs:
        unit = {"key": spec["name"], "params": unit_params(cfg, spec, its)}

        def compute(u, spec=spec):
            if spec["kind"] == "served":
                if not served:
                    from experiments.judges import _served
                    served.extend(_served(cfg["serving"]))
                if spec["served"] not in served:
                    raise MissingInput(f"{spec['served']} is not served; load it and run it again")
            done = (_run_served if spec["kind"] == "served" else _run_inprocess)(cfg, spec, its)
            rows = [{"report_id": it["item_id"], "image_id": it["image_id"], "generator": spec["name"],
                     "text": done[it["item_id"]]["text"] if it["item_id"] in done else ""} for it in its]
            unanswered = float(np.mean([not r["text"].strip() for r in rows]))
            if unanswered > float(g["max_empty_share"]):
                raise GateFailure(f"generator {spec['name']}: {unanswered:.1%} empty generations, above "
                                  f"{float(g['max_empty_share']):.0%}; the texts are in {jsonl_path(cfg, spec['name'])}")
            return rows

        def valid(u, done):
            return len(done) == len(its)

        try:
            _, refused = run_units("main_generate_reports", [unit], compute, out_dir, status, validate=valid)
        except GateFailure as e:
            failed.append(str(e))
            continue
        if refused:
            missing.append(spec["served"] if spec["kind"] == "served" else spec["name"])
    from experiments.purge import purge_unused
    purge_unused(cfg)
    if failed:
        raise GateFailure("; ".join(failed + [f"not run: {m}" for m in missing]))
    if missing:
        raise MissingInput(f"generators not run: {', '.join(missing)}; load or set them up and run it again")
    flush_status(status, f"[main_generate_reports] done for {', '.join(s['name'] for s in specs)}")
    return out_dir
