"""
Inference/chexbert_labeler.py
Created on October 10, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import os
import re
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertModel

CONDITIONS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
              "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
              "Support Devices", "No Finding"]
FINDING_HEAD = {"atelectasis": "Atelectasis", "cardiomegaly": "Cardiomegaly", "consolidation": "Consolidation",
                "fracture": "Fracture", "pleural effusion": "Pleural Effusion", "pneumothorax": "Pneumothorax",
                "pulmonary edema": "Edema"}
CLASS_NAME = {0: "blank", 1: "positive", 2: "negative", 3: "uncertain"}


class BertLabeler(nn.Module):

    def __init__(self, base: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel(BertConfig.from_pretrained(base))
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.linear_heads = nn.ModuleList([nn.Linear(hidden, 4, bias=True) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden, 2, bias=True))

    def forward(self, ids, mask):
        cls = self.dropout(self.bert(ids, attention_mask=mask)[0][:, 0, :])
        return [head(cls) for head in self.linear_heads]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch_checkpoint(ch: Dict, cache_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=ch["repo_id"], filename=ch["filename"], cache_dir=cache_dir)
    stamp = f"{path}.sha256"
    if not (os.path.exists(stamp) and open(stamp).read().strip() == ch["sha256"]):
        got = sha256_file(path)
        if got != ch["sha256"]:
            raise RuntimeError(f"[chexbert] {path} has SHA-256 {got}, expected {ch['sha256']}")
        with open(stamp, "w") as f:
            f.write(got)
    return path


def load_labeler(ckpt_path: str, base: str, device: torch.device) -> BertLabeler:
    model = BertLabeler(base)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"[chexbert] state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.to(device).eval()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().replace("\n", " ")).strip()


def encode(tokenizer, text: str) -> List[int]:
    pieces = tokenizer.tokenize(clean_text(text))
    if not pieces:
        return [tokenizer.cls_token_id, tokenizer.sep_token_id]
    ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(pieces) + [tokenizer.sep_token_id]
    return ids[:511] + [tokenizer.sep_token_id] if len(ids) > 512 else ids


@torch.no_grad()
def label_texts(model: BertLabeler, tokenizer, texts: List[str], device: torch.device,
                batch_size: int = 64) -> np.ndarray:
    encoded = [encode(tokenizer, t) for t in texts]
    order = np.argsort([len(e) for e in encoded], kind="mergesort")
    out = np.zeros((len(texts), len(CONDITIONS)), dtype=np.int64)
    for i in tqdm(range(0, len(order), batch_size), desc="chexbert", unit="batch"):
        idx = order[i:i + batch_size]
        width = max(len(encoded[j]) for j in idx)
        ids = torch.zeros((len(idx), width), dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, j in enumerate(idx):
            ids[row, :len(encoded[j])] = torch.tensor(encoded[j])
            mask[row, :len(encoded[j])] = 1
        heads = model(ids.to(device), mask.to(device))
        out[idx] = torch.stack([h.argmax(dim=1) for h in heads], dim=1).cpu().numpy()
    return out


def finding_state(codes: np.ndarray, finding: str, blank_is_absent: bool) -> str:
    c = int(codes[CONDITIONS.index(FINDING_HEAD[finding])])
    if c == 1:
        return "present"
    if c == 2:
        return "absent"
    if c == 3:
        return "uncertain"
    return "absent" if blank_is_absent else ""
