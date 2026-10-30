"""
experiments/llm_local.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import gc
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from experiments._common import (MissingInput, atomic_write, claim_unit, digest_file, flush_status, heartbeat_claim,
                                 node_and_gpu_line, release_claim, require)
from experiments.study import read_evidence
from Inference.llm_verifier import build_prompt

_SAMPLER = "permutation_seed1000epoch"


def prompts_of(df, k: int) -> List[str]:
    return [build_prompt(c, read_evidence(e, k), s) for c, e, s in
            zip(df["case_context"].astype(str), df["evidence_sentences_json"], df["claim_text"].astype(str))]


def chat_text(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def answer_token_ids(tok) -> Dict[str, int]:
    ids = {w: tok.encode(w, add_special_tokens=False) for w in ("Yes", "No")}
    if any(len(v) != 1 for v in ids.values()):
        raise RuntimeError(f"'Yes' or 'No' is not one token for this tokenizer: {ids}")
    return {w: v[0] for w, v in ids.items()}


def load_base(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    model.eval()
    return model, tok


def lora_config(lcfg: Dict):
    from peft import LoraConfig
    return LoraConfig(r=int(lcfg["rank"]), lora_alpha=int(lcfg["alpha"]), lora_dropout=float(lcfg["dropout"]),
                      target_modules=list(lcfg["target_modules"]), task_type="CAUSAL_LM", bias="none")


def with_adapter(model, lcfg: Dict, adapter_path: Optional[str] = None):
    from peft import get_peft_model, set_peft_model_state_dict
    model = get_peft_model(model, lora_config(lcfg))
    if adapter_path is not None:
        state = torch.load(adapter_path, map_location="cpu", weights_only=False)
        res = set_peft_model_state_dict(model, state)
        missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        if missing or getattr(res, "unexpected_keys", []):
            raise RuntimeError(f"{adapter_path}: adapter keys do not match ({missing[:3]}, {res.unexpected_keys[:3]})")
    return model


@torch.no_grad()
def score_prompts(model, tok, prompts: List[str], batch_size: int, top_logprobs: int = 20,
                  max_new_tokens: int = 4, desc: str = "llm") -> List[Dict]:
    from experiments._common import oom_batches
    device = next(model.parameters()).device
    texts = [chat_text(tok, p) for p in prompts]
    order = np.argsort([-len(t) for t in texts], kind="mergesort")
    out: List[Optional[Dict]] = [None] * len(texts)

    def run(idx):
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, output_logits=True,
                             return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        steps = torch.stack(gen.logits, dim=1).float()
        new = gen.sequences[:, enc["input_ids"].shape[1]:]
        for row, j in enumerate(idx):
            rec = {"position": -1, "token": "", "candidates": [], "p_yes": 0.0, "p_no": 0.0, "score": 0.5,
                   "status": "fallback_no_answer_token"}
            for pos in range(new.shape[1]):
                piece = tok.decode([int(new[row, pos])])
                if not any(ch.isalnum() for ch in piece):
                    continue
                lp = torch.log_softmax(steps[row, pos], dim=-1)
                top = torch.topk(lp, top_logprobs)
                cands = [[tok.decode([int(t)]), float(v)] for v, t in zip(top.values, top.indices)]
                p_yes = sum(math.exp(v) for t, v in cands if t.strip().lower() == "yes")
                p_no = sum(math.exp(v) for t, v in cands if t.strip().lower() == "no")
                ok = p_yes + p_no > 0
                rec = {"position": pos, "token": piece, "candidates": cands, "p_yes": p_yes, "p_no": p_no,
                       "score": p_yes / (p_yes + p_no) if ok else 0.5, "status": "ok" if ok else "fallback_no_yes_no"}
                break
            out[j] = rec
    oom_batches(order, batch_size, run, desc)
    return out


def _answer_logits(model, tok, texts: List[str], ids: Dict[str, int], device) -> torch.Tensor:
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos,
                       logits_to_keep=1).logits[:, -1, :]
    return logits.float()[:, [ids["No"], ids["Yes"]]]


@torch.no_grad()
def _val_auroc(model, tok, texts: List[str], labels: np.ndarray, ids, device, batch_size: int) -> float:
    model.eval()
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="validation", unit="batch", leave=False):
        lg = _answer_logits(model, tok, texts[i:i + batch_size], ids, device)
        scores += torch.softmax(lg, dim=-1)[:, 1].cpu().tolist()
    model.train()
    if len(np.unique(labels)) < 2:
        raise ValueError("the validation split holds one class, so no AUROC can select a checkpoint")
    return float(roc_auc_score(labels, np.asarray(scores)))


def lora_recipe(lcfg: Dict, model_name: str, train_parquet: str, seed: int) -> Dict:
    eff = int(lcfg["micro_batch"]) * int(lcfg["grad_accum"])
    return {"model_name": model_name, "rank": int(lcfg["rank"]), "alpha": int(lcfg["alpha"]),
            "dropout": float(lcfg["dropout"]), "target_modules": list(lcfg["target_modules"]),
            "lr": float(lcfg["lr"]), "weight_decay": float(lcfg["weight_decay"]), "effective_batch": eff,
            "num_epochs": int(lcfg["num_epochs"]), "grad_clip_norm": float(lcfg["grad_clip_norm"]),
            "max_tokens": int(lcfg["max_tokens"]), "max_evidence_sentences": int(lcfg["max_evidence_sentences"]),
            "amp": "bf16", "base_dtype": "bf16", "seed": int(seed), "sampler": _SAMPLER,
            "loss": "weighted_ce_yes_no_over_effective_batch", "selection": "best_val_auroc",
            "train_parquet": digest_file(train_parquet)}


def _save(obj, path: str) -> None:
    atomic_write(path, lambda tmp: torch.save(obj, tmp))


def _rng_state() -> Dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def lora_is_done(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "done.flag")) and os.path.exists(os.path.join(out_dir, "best_adapter.pt"))


def read_done(out_dir: str) -> Dict:
    out = {}
    for line in open(os.path.join(out_dir, "done.flag")).read().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def train_lora(lcfg: Dict, model_name: str, train_parquet: str, out_dir: str, seed: int) -> str:
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    best_path, resume_path = os.path.join(out_dir, "best_adapter.pt"), os.path.join(out_dir, "resume.pt")
    status, claims = os.path.join(out_dir, "train_status.txt"), os.path.join(out_dir, "claims")
    if lora_is_done(out_dir):
        return best_path
    require(train_parquet, "main_build_examples")
    if not claim_unit(claims, "train"):
        raise MissingInput(f"another job is training {out_dir}; run it again once that job has finished")
    try:
        if lora_is_done(out_dir):
            return best_path
        if not torch.cuda.is_available():
            raise RuntimeError(f"[lora] no usable GPU for {out_dir}: {node_and_gpu_line()}")
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        micro = int(lcfg["micro_batch"])
        if props.total_memory < 40 * 1024 ** 3:
            micro = min(micro, int(lcfg["micro_batch_small_gpu"]))
        rec = lora_recipe(lcfg, model_name, train_parquet, seed)
        accum = rec["effective_batch"] // micro
        if accum * micro != rec["effective_batch"]:
            raise ValueError(f"micro-batch {micro} does not divide the effective batch {rec['effective_batch']}")
        df = pd.read_parquet(train_parquet)
        k = rec["max_evidence_sentences"]
        model, tok = load_base(model_name, device)
        ids = answer_token_ids(tok)
        data = {}
        for split in ("train", "val"):
            part = df[df["split"] == split].reset_index(drop=True)
            texts = [chat_text(tok, p) for p in prompts_of(part, k)]
            longest = max(len(tok(t)["input_ids"]) for t in texts)
            if longest > rec["max_tokens"]:
                raise RuntimeError(f"[lora] a {split} prompt has {longest} tokens, above max_tokens {rec['max_tokens']}")
            data[split] = (texts, part["label_id"].to_numpy(int))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        torch.manual_seed(rec["seed"])
        model = with_adapter(model, lcfg)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        bad = [n for n, p in trainable if p.dtype != torch.float32]
        if bad or not trainable:
            raise RuntimeError(f"[lora] trainable parameters must be fp32 LoRA weights; offending: {bad[:5]}")
        opt = torch.optim.AdamW([p for _, p in trainable], lr=rec["lr"], weight_decay=rec["weight_decay"])
        y_tr = data["train"][1]
        w = torch.tensor([1.0, max((y_tr == 0).sum(), 1) / max((y_tr == 1).sum(), 1)], dtype=torch.float32,
                         device=device)
        n_steps = len(y_tr) // rec["effective_batch"]
        if n_steps == 0:
            raise RuntimeError(f"[lora] {len(y_tr)} training examples give no batch of {rec['effective_batch']}")
        start_epoch, start_step, best, best_epoch = 0, 0, -1.0, 0
        ck = None
        if os.path.exists(resume_path):
            try:
                ck = torch.load(resume_path, map_location="cpu", weights_only=False)
            except Exception:
                os.replace(resume_path, f"{resume_path}.refused_{time.strftime('%Y%m%d_%H%M%S')}")
        if ck is not None:
            if ck.get("recipe") != rec:
                os.replace(resume_path, f"{resume_path}.refused_{time.strftime('%Y%m%d_%H%M%S')}")
            else:
                from peft import set_peft_model_state_dict
                try:
                    set_peft_model_state_dict(model, ck["adapter"])
                    opt.load_state_dict(ck["opt"])
                    _restore_rng(ck["rng"])
                except Exception as e:
                    raise RuntimeError(f"[lora] {resume_path} was applied only partly ({type(e).__name__}: {e}); "
                                       f"delete it and train the run again") from e
                start_epoch, start_step = int(ck["epoch"]), int(ck["step"])
                best, best_epoch = float(ck["best"]), int(ck["best_epoch"])
            del ck
            gc.collect()
        model.train()
        checked = False
        save_every = int(lcfg["save_every_steps"])
        from peft import get_peft_model_state_dict
        for epoch in tqdm(range(start_epoch, rec["num_epochs"]), desc=f"{os.path.basename(out_dir)} epochs"):
            order = np.random.RandomState(rec["seed"] * 1000 + epoch).permutation(len(y_tr))
            first = start_step if epoch == start_epoch else 0
            bar = tqdm(range(first, n_steps), desc=f"epoch {epoch + 1}/{rec['num_epochs']}", unit="step")
            running = 0.0
            for step in bar:
                batch = order[step * rec["effective_batch"]:(step + 1) * rec["effective_batch"]]
                yb = torch.tensor(y_tr[batch], device=device)
                total_w = float(w[yb].sum())
                step_loss = 0.0
                while True:
                    opt.zero_grad(set_to_none=True)
                    failed, step_loss = False, 0.0
                    try:
                        for m0 in range(0, len(batch), micro):
                            mb = batch[m0:m0 + micro]
                            lg = _answer_logits(model, tok, [data["train"][0][j] for j in mb], ids, device)
                            nll = torch.nn.functional.cross_entropy(lg, yb[m0:m0 + micro], reduction="none")
                            loss = (w[yb[m0:m0 + micro]] * nll).sum() / total_w
                            loss.backward()
                            step_loss += float(loss.detach())
                    except torch.OutOfMemoryError:
                        failed = True
                    if not failed:
                        break
                    opt.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise RuntimeError(f"[lora] {out_dir}: one example does not fit on the "
                                           f"{props.total_memory / 1024 ** 3:.0f} GiB card")
                    micro = max(1, micro // 2)
                running += step_loss
                if not checked:
                    grads = [p.grad for _, p in trainable if p.grad is not None]
                    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads) or \
                            not any(float(g.abs().sum()) > 0 for g in grads):
                        raise RuntimeError("[lora] the LoRA weights did not receive a finite, non-zero gradient")
                    checked = True
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], rec["grad_clip_norm"])
                opt.step()
                heartbeat_claim(claims, "train")
                if (step + 1) % 20 == 0:
                    bar.set_postfix(loss=f"{running / (step + 1 - first):.4f}")
                if (step + 1) % save_every == 0 and step + 1 < n_steps:
                    _save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(), "epoch": epoch,
                           "step": step + 1, "best": best, "best_epoch": best_epoch, "recipe": rec,
                           "rng": _rng_state()}, resume_path)
            auroc = _val_auroc(model, tok, data["val"][0], data["val"][1], ids, device, int(lcfg["eval_batch_size"]))
            msg = f"epoch {epoch + 1}: val_auroc={auroc:.4f}"
            flush_status(status, msg)
            if auroc > best:
                best, best_epoch = auroc, epoch + 1
                _save(get_peft_model_state_dict(model), best_path)
            _save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(), "epoch": epoch + 1, "step": 0,
                   "best": best, "best_epoch": best_epoch, "recipe": rec, "rng": _rng_state()}, resume_path)
        text = (f"best_val_auroc={best:.4f}\nbest_epoch={best_epoch}\nrecipe={json.dumps(rec, sort_keys=True)}\n"
                f"finished_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

        def _write(tmp):
            with open(tmp, "w") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        atomic_write(os.path.join(out_dir, "done.flag"), _write)
        if os.path.exists(resume_path):
            os.remove(resume_path)
        del model
        torch.cuda.empty_cache()
        return best_path
    finally:
        release_claim(claims, "train")
