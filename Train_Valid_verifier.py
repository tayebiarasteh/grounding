"""
Created on Feb 24, 2026.
Train_Valid_verifier.py

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



class TrainingVerifier:
    def __init__(self, cfg_path, resume: bool = False):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        # AMP config
        self.use_amp = bool(self.params["Verifier"]["use_amp"])
        self.amp_dtype = str(self.params["Verifier"]["amp_dtype"]).lower()  # "bf16" or "fp16"
        self.grad_scaler_mode = self.params["Verifier"]["grad_scaler_mode"]      # "auto"/bool
        self.best_auroc = -1.0

        # Speed knobs
        self.tf32 = bool(self.params["Verifier"]["tf32"])

        if resume is False:
            self.model_info = self.params["Network"]
            self.epoch = 0
            self.best_loss = float("inf")
            self.writer = SummaryWriter(log_dir=os.path.join(self.params["target_dir"], self.params["tb_logs_path"]))

        self.setup_cuda()
        self.grad_clip = float(self.params["Network"]["grad_clip_norm"])


        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["Verifier"]["model_name"], use_fast=True)
        self.max_len = int(self.params["Verifier"].get("max_tokens", 512))

        # will be set in setup_cuda
        self.autocast_dtype = torch.bfloat16
        self.scaler = None


    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # TF32 speeds matmul/conv on Ampere+; does not affect bf16/fp16 autocast choice
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = bool(self.tf32)
                torch.backends.cudnn.allow_tf32 = bool(self.tf32)
            except Exception:
                pass

        # Decide autocast dtype + GradScaler usage
        if self.amp_dtype == "fp16":
            self.autocast_dtype = torch.float16
        else:
            # default bf16
            self.autocast_dtype = torch.bfloat16

        # GradScaler rules:
        # - fp16 autocast: scaler should be ON (recommended)
        # - bf16 autocast: scaler should be OFF (bf16 is stable; scaler often unnecessary)
        if self.grad_scaler_mode == "auto":
            enable_scaler = (self.use_amp and self.device.type == "cuda" and self.autocast_dtype == torch.float16)
        else:
            enable_scaler = bool(self.grad_scaler_mode)

        self.scaler = torch.cuda.amp.GradScaler(enabled=enable_scaler)

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



    def setup_model(self, model, optimiser, loss_function, class_weights):
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal # of trainable parameters: {total_param_num:,}")
        print("----------------------------------------------------\n")

        # IMPORTANT: keep weights in fp32 for stable training with AMP
        self.model = model.to(self.device).float()
        self.loss_function = loss_function(weight=class_weights.to(self.device)).to(self.device)
        self.optimiser = optimiser

        self.model_info["total_param_num"] = total_param_num
        self.model_info["loss_function"] = loss_function.__name__
        self.params["Network"] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)



    def load_checkpoint(self, model, optimiser, loss_function, class_weights):
        checkpoint = torch.load(
            os.path.join(self.params["target_dir"], self.params["network_output_path"], self.params["checkpoint_name"]),
            map_location="cpu",
        )
        self.device = None
        self.model_info = checkpoint["model_info"]
        self.setup_cuda()

        self.model = model.to(self.device).float()
        self.loss_function = loss_function(weight=class_weights.to(self.device)).to(self.device)
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.best_auroc = checkpoint.get("best_auroc", -1.0)

        # restore scaler state if present (optional)
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception:
                pass

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.params["target_dir"], self.params["tb_logs_path"]),
            purge_step=self.epoch + 1,
        )



    def train_epoch_randomp(self, train_loader, valid_loader=None, num_epochs=1000):
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for _ in range(num_epochs - self.epoch):
            self.epoch += 1

            self.model.train()
            batch_loss = 0.0
            total_n = 0
            correct = 0

            start_time = time.time()

            for idx, batch in enumerate(train_loader):
                tok = self._tokenize(batch.inputs, self.max_len)
                targets = batch.targets.to(self.device, non_blocking=True).long()

                self.optimiser.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(
                    enabled=(self.use_amp and self.device.type == "cuda"),
                    dtype=self.autocast_dtype,
                ):
                    out = self.model(**tok)
                    logits = out.logits
                    loss = self.loss_function(logits, targets)

                B = int(targets.shape[0])
                batch_loss += float(loss.item()) * B
                total_n += B

                preds = logits.argmax(dim=-1)
                correct += int((preds == targets).sum().item())

                if self.scaler is not None and self.scaler.is_enabled():
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

                # optional logging of sampled batch top-p
                if idx % 100 == 0:
                    print(f"[train] epoch={self.epoch} batch={idx} sampled_top_p={batch.batch_top_p}")

            train_loss = batch_loss / max(total_n, 1)
            train_acc = correct / max(total_n, 1)

            self.writer.add_scalar("Train_loss_avg", train_loss, self.epoch)
            self.writer.add_scalar("Train_accuracy", train_acc, self.epoch)

            self.params["Network"]["num_epoch"] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            ckpt = {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimiser.state_dict(),
                "loss_state_dict": self.loss_function.state_dict(),
                "model_info": self.model_info,
                "best_loss": self.best_loss,
            }
            if self.scaler is not None and self.scaler.is_enabled():
                ckpt["scaler_state_dict"] = self.scaler.state_dict()

            torch.save(
                ckpt,
                os.path.join(
                    self.params["target_dir"],
                    self.params["network_output_path"],
                    self.params["checkpoint_name"],
                ),
            )

            if (self.epoch) % int(self.params["display_stats_freq"]) == 0:
                end_time = time.time()
                total_time = end_time - total_start_time
                iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                if valid_loader is not None:
                    valid_loss, valid_acc, valid_f1, valid_auroc, valid_specificity, valid_sensitivity, valid_precision, optimal_threshold = self.valid_epoch(valid_loader)

                    self.calculate_tb_stats(
                        valid_loss=valid_loss,
                        valid_acc=valid_acc,
                        valid_f1=valid_f1,
                        valid_auroc=valid_auroc,
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
                        train_acc=train_acc,
                        valid_loss=valid_loss,
                        valid_acc=valid_acc,
                        valid_f1=valid_f1,
                        valid_auroc=valid_auroc,
                    )
                else:
                    self.savings_prints(
                        iteration_hours,
                        iteration_mins,
                        iteration_secs,
                        total_hours,
                        total_mins,
                        total_secs,
                        train_loss,
                        total_time,
                        train_acc=train_acc,
                    )


    def train_epoch(self, train_loader, valid_loader=None, num_epochs=1000):
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for _ in range(num_epochs - self.epoch):
            self.epoch += 1

            self.model.train()
            batch_loss = 0.0
            total_n = 0
            correct = 0

            start_time = time.time()

            for idx, batch in enumerate(train_loader):
                tok = self._tokenize(batch.inputs, self.max_len)
                targets = batch.targets.to(self.device, non_blocking=True).long()

                self.optimiser.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(self.use_amp and self.device.type == "cuda"), dtype=self.autocast_dtype):

                    out = self.model(**tok)
                    logits = out.logits
                    loss = self.loss_function(logits, targets)

                B = int(targets.shape[0])
                batch_loss += float(loss.item()) * B
                total_n += B

                preds = logits.argmax(dim=-1)
                correct += int((preds == targets).sum().item())

                if self.scaler is not None and self.scaler.is_enabled():
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

            train_loss = batch_loss / max(total_n, 1)
            train_acc = correct / max(total_n, 1)

            self.writer.add_scalar("Train_loss_avg", train_loss, self.epoch)
            self.writer.add_scalar("Train_accuracy", train_acc, self.epoch)

            self.params["Network"]["num_epoch"] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            ckpt = {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimiser.state_dict(),
                "loss_state_dict": self.loss_function.state_dict(),
                "model_info": self.model_info,
                "best_loss": self.best_loss,
            }
            if self.scaler is not None and self.scaler.is_enabled():
                ckpt["scaler_state_dict"] = self.scaler.state_dict()

            torch.save(ckpt, os.path.join(self.params["target_dir"], self.params["network_output_path"],
                             self.params["checkpoint_name"]))

            if (self.epoch) % int(self.params["display_stats_freq"]) == 0:
                end_time = time.time()
                total_time = end_time - total_start_time
                iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                if valid_loader is not None:
                    valid_loss, valid_acc, valid_f1, valid_auroc, valid_specificity, valid_sensitivity, valid_precision, optimal_threshold = self.valid_epoch(valid_loader)

                    self.calculate_tb_stats(
                        valid_loss=valid_loss,
                        valid_acc=valid_acc,
                        valid_f1=valid_f1,
                        valid_auroc=valid_auroc,
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
                        train_acc=train_acc,
                        valid_loss=valid_loss,
                        valid_acc=valid_acc,
                        valid_f1=valid_f1,
                        valid_auroc=valid_auroc,
                    )
                else:
                    self.savings_prints(
                        iteration_hours,
                        iteration_mins,
                        iteration_secs,
                        total_hours,
                        total_mins,
                        total_secs,
                        train_loss,
                        total_time,
                        train_acc=train_acc,
                    )


    def valid_epoch(self, valid_loader):
        self.model.eval()

        total_loss = 0.0
        total_n = 0

        preds_all = []
        targets_all = []
        probs_all = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_loader)):
                tok = self._tokenize(batch.inputs, self.max_len)
                targets = batch.targets.to(self.device, non_blocking=True).long()

                with torch.cuda.amp.autocast(
                        enabled=(self.use_amp and self.device.type == "cuda"),
                        dtype=self.autocast_dtype,
                ):
                    out = self.model(**tok)
                    logits = out.logits
                    loss = self.loss_function(logits, targets)

                B = int(targets.shape[0])
                total_loss += float(loss.item()) * B
                total_n += B

                probs = torch.softmax(logits, dim=-1)[:, 1]

                targets_all.extend(targets.detach().cpu().tolist())
                probs_all.extend(probs.detach().cpu().tolist())

        valid_loss = total_loss / max(total_n, 1)

        if total_n == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "auroc": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0,
                "precision": 0.0,
                "threshold": 0.5,
            }

        targets_all = np.asarray(targets_all, dtype=np.int32)
        probs_all = np.asarray(probs_all, dtype=np.float32)

        try:
            valid_auroc = float(metrics.roc_auc_score(targets_all, probs_all))
        except Exception:
            valid_auroc = 0.0

        # Youden threshold
        try:
            fpr, tpr, thresholds = metrics.roc_curve(targets_all, probs_all, pos_label=1)
            youden = tpr - fpr
            best_idx = int(np.argmax(youden))
            optimal_threshold = float(thresholds[best_idx])

            # sometimes sklearn may return inf threshold first
            if not np.isfinite(optimal_threshold):
                finite_thresholds = thresholds[np.isfinite(thresholds)]
                if len(finite_thresholds) > 0:
                    optimal_threshold = float(finite_thresholds.max())
                else:
                    optimal_threshold = 0.5
        except Exception:
            optimal_threshold = 0.5

        preds_all = (probs_all >= optimal_threshold).astype(np.int32)

        valid_acc = float(metrics.accuracy_score(targets_all, preds_all))
        valid_f1 = float(metrics.f1_score(targets_all, preds_all, average="binary", zero_division=0))
        valid_precision = float(metrics.precision_score(targets_all, preds_all, average="binary", zero_division=0))
        valid_sensitivity = float(metrics.recall_score(targets_all, preds_all, average="binary", zero_division=0))

        tn, fp, fn, tp = metrics.confusion_matrix(targets_all, preds_all, labels=[0, 1]).ravel()
        valid_specificity = float(tn / (tn + fp + epsilon))

        return valid_loss, valid_acc, valid_f1, valid_auroc, valid_specificity, valid_sensitivity, valid_precision, optimal_threshold


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
            train_acc=None,
            valid_loss=None,
            valid_acc=None,
            valid_f1=None,
            valid_auroc=None,
    ):
        self.params["Network"]["num_epoch"] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        # save best by validation macro-F1 if available, else by train loss
        save_best = False
        if valid_auroc is not None:
            if not hasattr(self, "best_auroc"):
                self.best_auroc = -1.0
            if valid_auroc > self.best_auroc:
                self.best_auroc = valid_auroc
                save_best = True
        else:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                save_best = True

        if save_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.params["target_dir"], self.params["network_output_path"],
                             self.params["trained_model_name"]),
            )

        ckpt = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            "loss_state_dict": self.loss_function.state_dict(),
            "model_info": self.model_info,
            "best_loss": self.best_loss,
        }

        if hasattr(self, "best_auroc"):
            ckpt["best_auroc"] = self.best_auroc
        if self.scaler is not None and self.scaler.is_enabled():
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(
            ckpt,
            os.path.join(self.params["target_dir"], self.params["network_output_path"], self.params["checkpoint_name"]),
        )

        if (self.epoch) % int(self.params["display_stats_freq"]) == 0:
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

        if train_acc is not None:
            print(f"\n\tTrain loss: {train_loss:.4f} | Train Accuracy: {float(train_acc)*100:.1f}%\n")
        else:
            print(f"\n\tTrain loss: {train_loss:.4f}\n")

        if valid_loss is not None:
            print(
                f"\tVal. loss: {valid_loss:.4f} | "
                f"Accuracy: {float(valid_acc)*100:.1f}% | "
                f"F1: {float(valid_f1)*100:.1f}% | "
                f"AUROC: {float(valid_auroc)*100:.1f}%\n"
            )
            msg = (
                f"\n\n----------------------------------------------------------------------------------------\n"
                f"epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s"
                f" | total time: {total_hours}h {total_mins}m {total_secs:.2f}s | "
                f"\n\n\tTrain loss: {train_loss:.4f} | Train acc: {float(train_acc)*100:.1f}% | "
                f"Val. loss: {valid_loss:.4f} | Accuracy: {float(valid_acc)*100:.1f}% | "
                f"F1: {float(valid_f1)*100:.1f}% | AUROC: {float(valid_auroc)*100:.1f}%\n\n"
            )

        else:
            msg = (
                f"----------------------------------------------------------------------------------------\n"
                f"epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s"
                f" | total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n\n"
                f"\ttrain loss: {train_loss:.4f} | train acc: {float(train_acc):.4f}\n\n"
            )

        with open(os.path.join(self.params["target_dir"], self.params["stat_log_path"]) + "/Stats", "a") as f:
            f.write(msg)



    def calculate_tb_stats(self, valid_loss=None, valid_acc=None, valid_f1=None, valid_auroc=None):
        if valid_loss is not None:
            self.writer.add_scalar("Val_loss", valid_loss, self.epoch)
        if valid_acc is not None:
            self.writer.add_scalar("Val_accuracy", valid_acc, self.epoch)
        if valid_f1 is not None:
            self.writer.add_scalar("Val_F1", valid_f1, self.epoch)
        if valid_auroc is not None:
            self.writer.add_scalar("Val_AUROC", valid_auroc, self.epoch)