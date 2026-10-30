"""
experiments/generators_inprocess.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import argparse
import importlib
import json
import os
import sys


REPORT_PROMPT = ("You are an experienced radiologist. Write the FINDINGS section of a chest X-ray report for the image "
                 "provided. Write only the FINDINGS section, in concise clinical radiology style. Do not write an "
                 "IMPRESSION section, a heading, or any text besides the findings themselves.")


def read_done(path: str, ident: dict) -> dict:
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "rb") as f:
        for raw in f:
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if all(rec.get(k) == v for k, v in ident.items()):
                done.setdefault(rec["item_id"], rec)
    return done


def no_brotli() -> None:
    try:
        import requests.utils
        requests.utils.DEFAULT_ACCEPT_ENCODING = "gzip, deflate"
    except ImportError:
        pass


def append(path: str, rec: dict) -> None:
    data = (json.dumps(rec, ensure_ascii=True) + "\n").encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


class Maira2:
    def __init__(self, spec: dict, token: str):
        import torch
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        hf_id = spec["hf_id"]
        self.processor = AutoProcessor.from_pretrained(hf_id, token=token, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(hf_id, token=token, trust_remote_code=True)
        cls = get_class_from_dynamic_module(cfg.auto_map["AutoModelForCausalLM"], hf_id, token=token)
        import inspect
        dtype_kw = "dtype" if "dtype" in inspect.signature(cls.from_pretrained).parameters else "torch_dtype"
        self.model = cls.from_pretrained(hf_id, token=token, trust_remote_code=True,
                                         **{dtype_kw: torch.bfloat16}).to("cuda").eval()

    def generate(self, image, max_new_tokens: int) -> str:
        import torch
        p = self.processor
        x = p.format_and_preprocess_reporting_input(
            current_frontal=image, current_lateral=None, prior_frontal=None, indication=None, technique=None,
            comparison=None, prior_report=None, return_tensors="pt", get_grounding=False).to("cuda")
        grounding = [p.phrase_start_token, p.phrase_end_token, p.box_start_token, p.box_end_token]
        stop = [p.tokenizer.eos_token_id] + [p.tokenizer.convert_tokens_to_ids(t) for t in grounding]
        with torch.no_grad():
            ids = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                      eos_token_id=stop)
        text = p.decode(ids[0][x["input_ids"].shape[1]:], skip_special_tokens=True)
        cut = min([i for i in (text.find(t) for t in grounding) if i >= 0], default=len(text))
        out = p.convert_output_to_plaintext_or_grounded_sequence(text[:cut])
        return out if isinstance(out, str) else str(out)


def import_llava_clone() -> None:
    import types
    from transformers import AutoConfig, AutoModelForCausalLM
    stub = types.ModuleType("llava.model.language_model.llava_mpt")
    stub.LlavaMPTForCausalLM = stub.LlavaMPTConfig = None
    sys.modules.setdefault(stub.__name__, stub)
    config_register = AutoConfig.__dict__["register"]
    own_model_register = AutoModelForCausalLM.__dict__.get("register")
    model_register = AutoModelForCausalLM.register
    AutoConfig.register = staticmethod(lambda model_type, config, exist_ok=False:
                                       config_register.__func__(model_type, config, exist_ok=True))
    AutoModelForCausalLM.register = lambda config_class, model_class, exist_ok=False: \
        model_register(config_class, model_class, exist_ok=True)
    try:
        importlib.import_module("llava.model")
    finally:
        AutoConfig.register = config_register
        if own_model_register is None:
            del AutoModelForCausalLM.register
        else:
            AutoModelForCausalLM.register = own_model_register


class LlavaRad:
    def __init__(self, spec: dict, token: str):
        import torch
        if spec["repo_path"] not in sys.path:
            sys.path.insert(0, spec["repo_path"])
        import_llava_clone()
        from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        self.tokenizer_image_token, self.conv_templates = tokenizer_image_token, conv_templates
        self.image_token, self.im_start, self.im_end = DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        original = torch.load

        def trusting(*a, **k):
            k["weights_only"] = False
            return original(*a, **k)
        torch.load = trusting
        try:
            self.tok, self.model, self.proc, _ = load_pretrained_model(
                model_path=spec["hf_id"], model_base=spec["model_base"], model_name=spec["model_name"])
        finally:
            torch.load = original
        self.model.eval()

    def generate(self, image, max_new_tokens: int) -> str:
        import torch
        block = (self.im_start + self.image_token + self.im_end
                 if getattr(self.model.config, "mm_use_im_start_end", False) else self.image_token)
        conv = self.conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], block + "\n" + REPORT_PROMPT)
        conv.append_message(conv.roles[1], None)
        pv = self.proc.preprocess(image, return_tensors="pt")["pixel_values"]
        pv = torch.stack(pv) if isinstance(pv, list) else pv
        ids = self.tokenizer_image_token(conv.get_prompt(), self.tok, return_tensors="pt").unsqueeze(0)
        with torch.no_grad():
            out = self.model.generate(ids.to(self.model.device), images=pv.to(self.model.device, dtype=torch.float16),
                                      do_sample=False, max_new_tokens=max_new_tokens, use_cache=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from PIL import Image
    from tqdm import tqdm
    with open(args.spec) as f:
        spec = json.load(f)
    with open(args.items) as f:
        items = json.load(f)
    ident = {"model": spec["hf_id"], "prompt": spec["prompt_digest"], "images": spec["images"],
             "max_new_tokens": int(spec["max_new_tokens"]), "worker_pins": spec["worker_pins"]}
    done = read_done(args.out, ident)
    todo = [it for it in items if it["item_id"] not in done]
    if not todo:
        return
    no_brotli()
    import socket
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"[main_generate_reports worker] no usable GPU on node {os.environ.get('SLURMD_NODENAME') or socket.gethostname()}"
                           f" (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    model = {"maira2": Maira2, "llava_rad": LlavaRad}[spec["kind"]](spec, os.environ.get("HF_TOKEN"))
    for it in tqdm(todo, desc=spec["name"], unit="report"):
        image = Image.open(it["image"]).convert("RGB")
        text = model.generate(image, int(spec["max_new_tokens"]))
        append(args.out, {"item_id": it["item_id"], **ident, "text": text, "image_id": it["image_id"]})
