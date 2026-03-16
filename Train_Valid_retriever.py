"""
Created on October 27, 2025.
Train_Valid_retriever.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os.path
import time
import pdb
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import os
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from tqdm import tqdm

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from config.serde import read_config, write_config

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15



class Training:
    def __init__(self, cfg_path, resume=False):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path

        if resume is False:
            self.model_info = self.params["Network"]
            self.epoch = 0
            self.best_loss = float("inf")
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.params["target_dir"], self.params["tb_logs_path"])
            )

        self.setup_cuda()

        self.grad_clip = float(self.params["Network"]["grad_clip_norm"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.params["retriever"]["model_name"],
            use_fast=True,
        )

        self.max_q = int(self.params["retriever"]["max_query_tokens"])
        self.max_d = int(self.params["retriever"]["max_doc_tokens"])
        self.tau = float(self.params["retriever"]["temperature"])

        self.num_hard_negatives = int(self.params["retriever"].get("num_hard_negatives", 8))

        self.use_amp = bool(self.params.get("runtime", {}).get("use_amp", True))
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.use_amp and self.device.type == "cuda")
        )


    def setup_cuda(self, cuda_device_id=0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def time_duration(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = elapsed_time - (elapsed_mins * 60)
        return elapsed_hours, elapsed_mins, elapsed_secs


    def _tokenize(self, texts, max_len: int):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}


    def setup_model(self, model, optimiser, loss_function):
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal # of trainable parameters: {total_param_num:,}")
        print("----------------------------------------------------\n")

        self.model = model.to(self.device)
        self.loss_function = loss_function().to(self.device)
        self.optimiser = optimiser

        self.model_info["total_param_num"] = total_param_num
        self.model_info["loss_function"] = loss_function.__name__
        self.params["Network"] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)



    def load_checkpoint(self, model, optimiser, loss_function):
        checkpoint = torch.load(
            os.path.join(
                self.params["target_dir"],
                self.params["network_output_path"],
                self.params["checkpoint_name"],
            ),
            map_location="cpu",
        )

        self.device = None
        self.model_info = checkpoint["model_info"]
        self.setup_cuda()

        self.model = model.to(self.device)
        self.loss_function = loss_function().to(self.device)
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception:
                pass

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.params["target_dir"], self.params["tb_logs_path"]),
            purge_step=self.epoch + 1,
        )


    def train_epoch(self, train_loader, valid_loader=None, num_epochs=1000):
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for _ in range(num_epochs - self.epoch):
            self.epoch += 1

            self.model.train()
            batch_loss = 0.0
            total_q = 0

            start_time = time.time()

            for idx, batch in enumerate(tqdm(train_loader)):
                # filter invalid items
                keep_idx = [
                    i for i, (q, p) in enumerate(zip(batch.queries, batch.pos_docs))
                    if str(q).strip() and str(p).strip()
                ]
                if len(keep_idx) == 0:
                    continue

                queries = [batch.queries[i] for i in keep_idx]
                pos_docs = [batch.pos_docs[i] for i in keep_idx]
                neg_docs_per_q = [batch.neg_docs[i] for i in keep_idx]

                B = len(queries)

                # flatten explicit negatives
                flat_neg_docs = []
                neg_offsets = []
                for negs in neg_docs_per_q:
                    start = len(flat_neg_docs)
                    flat_neg_docs.extend([x for x in negs if str(x).strip()])
                    end = len(flat_neg_docs)
                    neg_offsets.append((start, end))

                q_tok = self._tokenize(queries, self.max_q)
                p_tok = self._tokenize(pos_docs, self.max_d)

                if len(flat_neg_docs) > 0:
                    n_tok = self._tokenize(flat_neg_docs, self.max_d)
                else:
                    n_tok = None

                self.optimiser.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(self.use_amp and self.device.type == "cuda")):
                    q_emb = self.model.encode_queries(q_tok["input_ids"], q_tok["attention_mask"])  # [B, D]
                    p_emb = self.model.encode_docs(p_tok["input_ids"], p_tok["attention_mask"])  # [B, D]

                    if n_tok is not None:
                        n_emb = self.model.encode_docs(n_tok["input_ids"], n_tok["attention_mask"])  # [Nneg, D]
                        cand_emb = torch.cat([p_emb, n_emb], dim=0)  # [B+Nneg, D]
                    else:
                        cand_emb = p_emb  # [B, D]

                    logits = torch.matmul(q_emb, cand_emb.T) / self.tau  # [B, B+Nneg]
                    targets = torch.arange(B, device=self.device, dtype=torch.long)  # positive is at index i

                    loss = self.loss_function(logits, targets)

                batch_loss += float(loss.item()) * B
                total_q += B

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimiser)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimiser.step()

            train_loss = batch_loss / max(total_q, 1)
            self.writer.add_scalar("Train_loss_avg", train_loss, self.epoch)

            self.params["Network"]["num_epoch"] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimiser.state_dict(),
                    "loss_state_dict": self.loss_function.state_dict(),
                    "model_info": self.model_info,
                    "best_loss": self.best_loss,
                },
                os.path.join(
                    self.params["target_dir"],
                    self.params["network_output_path"],
                    self.params["checkpoint_name"],
                ),
            )

            if (self.epoch) % int(self.params["display_stats_freq"]) == 0:
                if valid_loader is not None:
                    valid_loss, valid_recall_1, valid_recall_5, valid_mrr = self.valid_epoch(valid_loader)

                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(
                        valid_loss=valid_loss,
                        valid_recall_1=valid_recall_1,
                        valid_recall_5=valid_recall_5,
                        valid_mrr=valid_mrr,
                    )

                    self.savings_prints(
                        iteration_hours,
                        iteration_mins,
                        iteration_secs,
                        total_hours,
                        total_mins,
                        total_secs,
                        train_loss,
                        total_time,
                        valid_loss=valid_loss,
                        valid_recall_1=valid_recall_1,
                        valid_recall_5=valid_recall_5,
                        valid_mrr=valid_mrr,
                    )
                else:
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.savings_prints(
                        iteration_hours,
                        iteration_mins,
                        iteration_secs,
                        total_hours,
                        total_mins,
                        total_secs,
                        train_loss,
                        total_time,
                    )




    def valid_epoch(self, valid_loader):
        self.model.eval()

        total_loss = 0.0
        total_q = 0

        hit1 = 0
        hit5 = 0
        rr_sum = 0.0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_loader)):
                keep_idx = [
                    i for i, (q, p) in enumerate(zip(batch.queries, batch.pos_docs))
                    if str(q).strip() and str(p).strip()
                ]
                if len(keep_idx) == 0:
                    continue

                queries = [batch.queries[i] for i in keep_idx]
                pos_docs = [batch.pos_docs[i] for i in keep_idx]
                neg_docs_per_q = [batch.neg_docs[i] for i in keep_idx]

                B = len(queries)

                flat_neg_docs = []
                for negs in neg_docs_per_q:
                    flat_neg_docs.extend([x for x in negs if str(x).strip()])

                q_tok = self._tokenize(queries, self.max_q)
                p_tok = self._tokenize(pos_docs, self.max_d)

                if len(flat_neg_docs) > 0:
                    n_tok = self._tokenize(flat_neg_docs, self.max_d)
                else:
                    n_tok = None

                with torch.cuda.amp.autocast(enabled=(self.use_amp and self.device.type == "cuda")):
                    q_emb = self.model.encode_queries(q_tok["input_ids"], q_tok["attention_mask"])
                    p_emb = self.model.encode_docs(p_tok["input_ids"], p_tok["attention_mask"])

                    if n_tok is not None:
                        n_emb = self.model.encode_docs(n_tok["input_ids"], n_tok["attention_mask"])
                        cand_emb = torch.cat([p_emb, n_emb], dim=0)
                    else:
                        cand_emb = p_emb

                    logits = torch.matmul(q_emb, cand_emb.T) / self.tau
                    targets = torch.arange(B, device=self.device, dtype=torch.long)

                    loss = self.loss_function(logits, targets)

                total_loss += float(loss.item()) * B
                total_q += B

                # ranks of true positive
                sorted_idx = torch.argsort(logits, dim=1, descending=True)

                for i in range(B):
                    ranking = sorted_idx[i].detach().cpu().tolist()
                    true_idx = i

                    rank = ranking.index(true_idx) + 1

                    if rank == 1:
                        hit1 += 1
                    if rank <= 5:
                        hit5 += 1
                    rr_sum += 1.0 / float(rank)

        valid_loss = total_loss / max(total_q, 1)
        valid_recall_1 = hit1 / max(total_q, 1)
        valid_recall_5 = hit5 / max(total_q, 1)
        valid_mrr = rr_sum / max(total_q, 1)

        return valid_loss, valid_recall_1, valid_recall_5, valid_mrr




    def savings_prints(
            self,
            iteration_hours,
            iteration_mins,
            iteration_secs,
            total_hours,
            total_mins,
            total_secs,
            train_loss,
            total_time,
            total_overhead_time=0,
            total_datacopy_time=0,
            valid_loss=None,
            valid_recall_1=None,
            valid_recall_5=None,
            valid_mrr=None,
    ):
        self.params["Network"]["num_epoch"] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        if valid_loss is not None:
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.params["target_dir"], self.params["network_output_path"],
                                 self.params["trained_model_name"]),
                )
        else:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.params["target_dir"], self.params["network_output_path"],
                                 self.params["trained_model_name"]),
                )

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimiser.state_dict(),
                "loss_state_dict": self.loss_function.state_dict(),
                "model_info": self.model_info,
                "best_loss": self.best_loss,
            },
            os.path.join(self.params["target_dir"], self.params["network_output_path"], self.params["checkpoint_name"]),
        )

        if (self.epoch) % self.params["display_stats_freq"] == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.params["target_dir"],
                    self.params["network_output_path"],
                    f"epoch{self.epoch}_" + self.params["trained_model_name"],
                ),
            )

        print("--------------------------------------------------------------------------------")
        print(
            f"epoch: {self.epoch} | "
            f"epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s | "
            f"total time: {total_hours}h {total_mins}m {total_secs:.2f}s"
        )
        print(f"\n\tTrain loss: {train_loss:.4f}\n")

        if valid_loss is not None:
            print(
                f"\tVal. loss: {valid_loss:.4f} | "
                f"Recall@1: {valid_recall_1 * 100:.2f}% | "
                f"Recall@5: {valid_recall_5 * 100:.2f}% | "
                f"MRR: {valid_mrr:.4f}\n"
            )

            msg = (
                f"\n\n----------------------------------------------------------------------------------------\n"
                f"epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s"
                f" | total time: {total_hours}h {total_mins}m {total_secs:.2f}s | "
                f"\n\n\tTrain loss: {train_loss:.4f} | "
                f"Val. loss: {valid_loss:.4f} | Recall@1: {valid_recall_1 * 100:.2f}% | "
                f"Recall@5: {valid_recall_5 * 100:.2f}% | MRR: {valid_mrr:.4f}\n\n"
            )
        else:
            msg = (
                f"----------------------------------------------------------------------------------------\n"
                f"epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s"
                f" | total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n\n\ttrain loss: {train_loss:.4f}\n\n"
            )

        with open(os.path.join(self.params["target_dir"], self.params["stat_log_path"]) + "/Stats", "a") as f:
            f.write(msg)





    def calculate_tb_stats(self, valid_loss=None, valid_recall_1=None, valid_recall_5=None, valid_mrr=None):
        if valid_loss is not None:
            self.writer.add_scalar("Val_loss", valid_loss, self.epoch)
        if valid_recall_1 is not None:
            self.writer.add_scalar("Recall@1", valid_recall_1, self.epoch)
        if valid_recall_5 is not None:
            self.writer.add_scalar("Recall@5", valid_recall_5, self.epoch)
        if valid_mrr is not None:
            self.writer.add_scalar("MRR", valid_mrr, self.epoch)