"""
experiments/bespoke_worker.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.generators_inprocess import append, no_brotli, read_done


class Bespoke:

    def __init__(self, spec: dict):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(spec["hf_id"], trust_remote_code=True)
        config = AutoConfig.from_pretrained(spec["hf_id"], trust_remote_code=True)
        self.model = self.load(AutoModelForCausalLM, spec["hf_id"], config, torch.bfloat16)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.top_k = int(spec["top_logprobs"])

    def load(self, auto, hf_id: str, config, dtype):
        return auto.from_pretrained(hf_id, config=config, trust_remote_code=True, torch_dtype=dtype).to("cuda").eval()

    def encode(self, messages: list) -> list:
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    def decoded(self, ids: list, tid: int) -> str:
        before = self.tok.decode(ids[-6:], skip_special_tokens=True)
        return self.tok.decode(ids[-6:] + [tid], skip_special_tokens=True)[len(before):]

    def score(self, batch: list) -> list:
        import torch
        n = max(len(x) for x in batch)
        ids = torch.full((len(batch), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, x in enumerate(batch):
            ids[i, n - len(x):] = torch.tensor(x, dtype=torch.long)
            mask[i, n - len(x):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        dev = self.model.output.weight.device
        with torch.no_grad():
            hidden = self.model.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                                      use_cache=False)[0][:, -1, :]
            top = torch.topk(torch.log_softmax(self.model.output(hidden).float(), dim=-1), self.top_k, dim=-1)
        recs = []
        for i, x in enumerate(batch):
            cands = [[self.decoded(x, int(t)), float(v)] for v, t in zip(top.values[i].tolist(), top.indices[i].tolist())]
            recs.append({"score": sum(math.exp(v) for t, v in cands if t.lower() == "yes"), "status": "ok",
                         "candidates": cands, "n_tokens": len(x)})
        return recs


def serve(model: Bespoke, spec: dict, requests, write) -> None:
    from experiments._common import oom_batches
    ident = spec["ident"]
    for line in requests:
        if not line.strip():
            continue
        req = json.loads(line)
        with open(req["items"]) as f:
            items = json.load(f)
        done = read_done(req["out"], ident)
        todo = [it for it in items if it["item_id"] not in done]
        encoded = [model.encode(it["messages"]) for it in todo]
        order = sorted(range(len(todo)), key=lambda i: -len(encoded[i]))

        def run(idx):
            for i, rec in zip(idx, model.score([encoded[i] for i in idx])):
                append(req["out"], {"item_id": todo[i]["item_id"], **ident, **rec})
        if todo:
            oom_batches(order, int(spec["batch_size"]), run, f"{spec['id']} {req['tag']}")
        write("DONE " + json.dumps({"tag": req["tag"], "scored": len(todo), "before": len(items) - len(todo)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    no_brotli()
    with open(args.spec) as f:
        spec = json.load(f)
    import transformers
    model = Bespoke(spec)

    def write(msg: str) -> None:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    write("READY " + json.dumps({"transformers": transformers.__version__}))
    serve(model, spec, sys.stdin, write)
