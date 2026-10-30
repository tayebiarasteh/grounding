"""
controlled/train_verifier.py
Created on July 24, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import gc
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loader.verifier_dataset import load_verifier_records
from experiments._common import (MissingInput, atomic_write, claim_unit, digest_file, flush_status,
                                 heartbeat_claim, node_and_gpu_line, release_claim, require)

_SAMPLER = "seeded_per_epoch_w0"
_LEGACY = {"sampler": "unseeded_shuffle_w2"}


class _RecordDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.texts = [r["input_text"] for r in records]
        self.labels = [int(r["label_id"]) for r in records]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def _collate(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def _save(obj, path: str) -> None:
    atomic_write(path, lambda tmp: torch.save(obj, tmp))


def _class_weights(records: List[Dict]) -> torch.Tensor:
    n_pos = max(sum(1 for r in records if int(r["label_id"]) == 1), 1)
    n_neg = max(sum(1 for r in records if int(r["label_id"]) == 0), 1)
    return torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)


def _resolve_amp(tr: Dict) -> str:
    amp = str(tr.get("amp_dtype", "bf16")).lower()
    if amp != "bf16":
        raise ValueError(f"training.amp_dtype must be bf16 (autocast over fp32 weights); got {amp!r}")
    return amp


def recipe(cfg: Dict, train_parquet: str) -> Dict:
    tr, v = cfg["training"], cfg["verifier"]
    return {"model_name": v["model_name"], "lr": float(tr["lr"]), "weight_decay": float(tr["weight_decay"]),
            "batch_size": int(tr["batch_size"]), "num_epochs": int(tr["num_epochs"]),
            "grad_clip_norm": float(tr["grad_clip_norm"]), "max_tokens": int(tr["max_tokens"]),
            "max_evidence_sentences": int(v["max_evidence_sentences"]),
            "max_evidence_chars": int(v["max_evidence_chars"]), "amp": _resolve_amp(tr),
            "fit_report": bool(tr.get("fit_report", False)),
            "seed": int(cfg["seed"]), "sampler": _SAMPLER, "selection": "best_val_auroc",
            "train_parquet": digest_file(train_parquet)}


def _recipe_moved(stored: Optional[Dict], current: Dict) -> List[str]:
    stored = stored or {}
    moved = []
    for k, v in current.items():
        old = stored.get(k, _LEGACY.get(k, v))
        if old != v:
            moved.append(f"{k}: {old!r} -> {v!r}")
    return moved


def _rng_state() -> Dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: Optional[Dict]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _move_aside(path: str) -> None:
    os.replace(path, f"{path}.refused_{time.strftime('%Y%m%d_%H%M%S')}")


def _load_resume(path: str, rec: Dict, model, optimizer) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        _move_aside(path)
        return None
    moved = _recipe_moved(ck.get("recipe"), rec)
    own = model.state_dict()
    if not moved and (set(ck["model"]) != set(own) or any(ck["model"][k].shape != own[k].shape for k in own)):
        moved = ["the model's parameter names or shapes"]
    groups = ck["opt"].get("param_groups", [])
    if not moved and [len(g["params"]) for g in groups] != [len(g["params"]) for g in optimizer.param_groups]:
        moved = ["the optimizer's parameter groups"]
    if moved:
        _move_aside(path)
        return None
    model.load_state_dict(ck["model"])
    try:
        optimizer.load_state_dict(ck["opt"])
        _restore_rng(ck.get("rng"))
    except Exception as e:
        raise RuntimeError(f"[train] {path} was applied only partly ({type(e).__name__}: {e}); delete it and "
                           f"train the run again") from e
    counters = {k: ck[k] for k in ("epoch", "best_auroc", "best_epoch") if k in ck}
    del ck
    gc.collect()
    return counters


def _train_loader(records: List[Dict], batch_size: int, seed: int, epoch: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed * 1000 + epoch)
    return DataLoader(_RecordDataset(records), batch_size=batch_size, shuffle=True, num_workers=0,
                      collate_fn=_collate, drop_last=True, generator=g)


@torch.no_grad()
def _val_auroc(model, tokenizer, loader, device, max_tokens) -> float:
    model.eval()
    probs, targets = [], []
    for texts, labels in tqdm(loader, desc="validation", unit="batch", leave=False):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        targets.extend(labels.tolist())
    targets = np.asarray(targets)
    if len(np.unique(targets)) < 2:
        raise ValueError("the validation split holds one class, so no AUROC can select a checkpoint")
    return float(roc_auc_score(targets, np.asarray(probs)))


def _step(model, tokenizer, texts, labels, class_w, max_tokens: int, device, micro: int) -> float:
    total = float(class_w[labels].sum())
    loss_sum = 0.0
    for a in range(0, len(texts), micro):
        lab = labels[a:a + micro]
        enc = tokenizer(texts[a:a + micro], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
        enc = {k: v_.to(device) for k, v_ in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()
        nll = torch.nn.functional.cross_entropy(logits, lab, reduction="none")
        loss = (class_w[lab] * nll).sum() / total
        loss.backward()
        loss_sum += float(loss.detach())
    return loss_sum


def _persistent_gib(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) * 4 / 1024 ** 3


def _assert_backbone_trains(model) -> None:
    prefix = getattr(model, "base_model_prefix", "") + "."
    grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
    if not any(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in grads):
        raise RuntimeError(f"[train] no parameter under '{prefix}' received a finite non-zero gradient")


def is_done(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "done.flag")) and os.path.exists(os.path.join(out_dir, "best_model.pth"))


def train_verifier(cfg: dict, train_parquet: str, out_dir: str, force: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    done_flag, best_path = os.path.join(out_dir, "done.flag"), os.path.join(out_dir, "best_model.pth")
    resume_path, status_path = os.path.join(out_dir, "resume.pt"), os.path.join(out_dir, "train_status.txt")
    claims_dir = os.path.join(out_dir, "claims")
    if is_done(out_dir) and not force:
        return best_path
    require(train_parquet, "the dataset build of this run")
    if not claim_unit(claims_dir, "train"):
        raise MissingInput(f"another job is training {out_dir}; run it again once that job has finished")
    try:
        if is_done(out_dir) and not force:
            return best_path
        if not torch.cuda.is_available():
            raise RuntimeError(f"[train] no usable GPU for {out_dir}: {node_and_gpu_line()}")
        if force and os.path.exists(resume_path):
            _move_aside(resume_path)
        return _train(cfg, train_parquet, out_dir, done_flag, best_path, resume_path, status_path, claims_dir)
    finally:
        release_claim(claims_dir, "train")


def _train(cfg, train_parquet, out_dir, done_flag, best_path, resume_path, status_path, claims_dir) -> str:
    rec = recipe(cfg, train_parquet)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(rec["model_name"], use_fast=True)
    fit = tokenizer if rec["fit_report"] else None
    load = lambda split: load_verifier_records(train_parquet, split=split,
                                               max_evidence_sentences=rec["max_evidence_sentences"],
                                               max_evidence_chars=rec["max_evidence_chars"],
                                               fit_tokenizer=fit, max_tokens=rec["max_tokens"])
    train_records, val_records = load("train"), load("val")

    torch.manual_seed(rec["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(
        rec["model_name"], num_labels=2, problem_type="single_label_classification").to(device)
    bad = [n for n, p in model.named_parameters() if p.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"[train] parameters must stay fp32 under bf16 autocast; not fp32: {bad[:5]}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=rec["lr"], weight_decay=rec["weight_decay"])
    class_w = _class_weights(train_records).to(device)
    val_loader = DataLoader(_RecordDataset(val_records), batch_size=int(cfg["verifier"]["eval_batch_size"]),
                            shuffle=False, num_workers=0, collate_fn=_collate)

    start_epoch, best_auroc, best_epoch = 0, -1.0, 0
    ck = _load_resume(resume_path, rec, model, optimizer)
    if ck is not None:
        start_epoch, best_auroc, best_epoch = int(ck["epoch"]), float(ck["best_auroc"]), int(ck.get("best_epoch", 0))
    checked = False
    micro = rec["batch_size"]
    for epoch in tqdm(range(start_epoch, rec["num_epochs"]), desc=f"{os.path.basename(out_dir)} epochs", unit="epoch"):
        loader = _train_loader(train_records, rec["batch_size"], rec["seed"], epoch)
        if len(loader) == 0:
            raise RuntimeError(f"[train] {len(train_records)} train rows give no batch of {rec['batch_size']}")
        model.train()
        running = 0.0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{rec['num_epochs']}", unit="batch")
        for step, (texts, labels) in enumerate(bar):
            labels = labels.to(device)
            loss = 0.0
            while True:
                optimizer.zero_grad(set_to_none=True)
                failed = False
                try:
                    loss = _step(model, tokenizer, list(texts), labels, class_w, rec["max_tokens"], device, micro)
                except torch.OutOfMemoryError:
                    failed = True
                if not failed:
                    break
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if micro == 1:
                    free, total = torch.cuda.mem_get_info()
                    raise RuntimeError(f"[train] {out_dir}: one example does not fit; the weights, gradients and AdamW "
                                       f"moments take {_persistent_gib(model):.1f} GiB of a {total / 1024 ** 3:.1f} GiB "
                                       f"card ({free / 1024 ** 3:.1f} GiB free), so no batch size can fix this")
                micro = max(1, micro // 2)
            if not checked:
                _assert_backbone_trains(model)
                checked = True
            if rec["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), rec["grad_clip_norm"])
            optimizer.step()
            running += loss
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running / (step + 1):.4f}")
            heartbeat_claim(claims_dir, "train")
        val_auroc = _val_auroc(model, tokenizer, val_loader, device, rec["max_tokens"])
        msg = f"epoch {epoch + 1}: train_loss={running / len(loader):.4f} val_auroc={val_auroc:.4f}"
        flush_status(status_path, msg)
        if val_auroc > best_auroc:
            best_auroc, best_epoch = val_auroc, epoch + 1
            _save(model.state_dict(), best_path)
        _save({"model": model.state_dict(), "opt": optimizer.state_dict(), "epoch": epoch + 1,
               "best_auroc": best_auroc, "best_epoch": best_epoch, "recipe": rec, "rng": _rng_state()}, resume_path)

    text = (f"best_val_auroc={best_auroc:.4f}\nbest_epoch={best_epoch}\nrecipe={json.dumps(rec, sort_keys=True)}\n"
            f"finished_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

    def _write(tmp):
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    atomic_write(done_flag, _write)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    return best_path
