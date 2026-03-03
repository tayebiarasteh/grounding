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

    def setup_model(self, model, optimiser, loss_function):
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal # of trainable parameters: {total_param_num:,}")
        print("----------------------------------------------------\n")

        # IMPORTANT: keep weights in fp32 for stable training with AMP
        self.model = model.to(self.device).float()
        self.loss_function = loss_function().to(self.device)
        self.optimiser = optimiser

        self.model_info["total_param_num"] = total_param_num
        self.model_info["loss_function"] = loss_function.__name__
        self.params["Network"] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)



    def load_checkpoint(self, model, optimiser, loss_function):
        checkpoint = torch.load(
            os.path.join(self.params["target_dir"], self.params["network_output_path"], self.params["checkpoint_name"]),
            map_location="cpu",
        )
        self.device = None
        self.model_info = checkpoint["model_info"]
        self.setup_cuda()

        self.model = model.to(self.device).float()
        self.loss_function = loss_function().to(self.device)
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

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




    def train_epoch(self, train_loader, valid_loader=None, num_epochs=1000):
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for _ in range(num_epochs - self.epoch):
            self.epoch += 1

            batch_loss = 0.0
            start_time = time.time()

            for idx, batch in enumerate(tqdm(train_loader)):
                # remove this sanity-limit in real training
                # if idx >= 20:
                #     break

                self.model.train()

                tok = self._tokenize(batch.inputs, self.max_len)
                targets = batch.targets.to(self.device, non_blocking=True).float()

                self.optimiser.zero_grad(set_to_none=True)

                # AMP forward
                with torch.cuda.amp.autocast(
                    enabled=(self.use_amp and self.device.type == "cuda"),
                    dtype=self.autocast_dtype,
                ):
                    out = self.model(**tok)
                    preds = out.logits.squeeze(-1)
                    loss = self.loss_function(preds, targets)

                batch_loss += float(loss.item())

                # Backward + step
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

            train_loss = batch_loss / max(len(train_loader), 1)
            self.writer.add_scalar("Train_loss_avg", train_loss, self.epoch)

            # Saves information about training to config file
            self.params["Network"]["num_epoch"] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            # checkpoint each epoch (your style)
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
                os.path.join(self.params["target_dir"], self.params["network_output_path"], self.params["checkpoint_name"]),
            )

            # Validation
            if (self.epoch) % int(self.params["display_stats_freq"]) == 0:
                if valid_loader is not None:
                    valid_loss, valid_mae, valid_pearson = self.valid_epoch(valid_loader)

                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(valid_loss=valid_loss, valid_mae=valid_mae, valid_pearson=valid_pearson)
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
                        valid_mae=valid_mae,
                        valid_pearson=valid_pearson,
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




    def valid_epoch(self, valid_loader) -> Tuple[float, float, float]:
        self.model.eval()

        total_loss = 0.0
        total_n = 0

        preds_all = []
        targets_all = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_loader)):
                # remove this sanity-limit in real validation
                # if idx >= 50:
                #     break

                tok = self._tokenize(batch.inputs, self.max_len)
                targets = batch.targets.to(self.device, non_blocking=True).float()

                with torch.cuda.amp.autocast(
                    enabled=(self.use_amp and self.device.type == "cuda"),
                    dtype=self.autocast_dtype,
                ):
                    out = self.model(**tok)
                    preds = out.logits.squeeze(-1)
                    loss = self.loss_function(preds, targets)

                B = int(targets.shape[0])
                total_loss += float(loss.item()) * B
                total_n += B

                preds_all.extend(preds.detach().float().cpu().tolist())
                targets_all.extend(targets.detach().float().cpu().tolist())

        valid_loss = total_loss / max(total_n, 1)

        if total_n > 0:
            arr_p = np.asarray(preds_all, dtype=np.float32)
            arr_t = np.asarray(targets_all, dtype=np.float32)
            valid_mae = float(np.mean(np.abs(arr_p - arr_t)))
            if float(np.std(arr_p)) > 1e-8 and float(np.std(arr_t)) > 1e-8:
                valid_pearson = float(np.corrcoef(arr_p, arr_t)[0, 1])
            else:
                valid_pearson = 0.0
        else:
            valid_mae = 0.0
            valid_pearson = 0.0

        return valid_loss, valid_mae, valid_pearson





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
        valid_loss=None,
        valid_mae=None,
        valid_pearson=None,
    ):
        self.params["Network"]["num_epoch"] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        # Save best model by validation loss if available, else train loss
        if valid_loss is not None:
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.params["target_dir"], self.params["network_output_path"], self.params["trained_model_name"]),
                )
        else:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.params["target_dir"], self.params["network_output_path"], self.params["trained_model_name"]),
                )

        # checkpoint every epoch (already done in train loop, keep your style)
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

        print("------------------------------------------------------" "----------------------------------")
        print(
            f"epoch: {self.epoch} | "
            f"epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s | "
            f"total time: {total_hours}h {total_mins}m {total_secs:.2f}s"
        )
        print(f"\n\tTrain loss: {train_loss:.4f}\n")

        if valid_loss is not None:
            print(f"\t Val. loss: {valid_loss:.4f} | MAE: {float(valid_mae):.4f} | Pearson-r: {float(valid_pearson):.4f}\n")
            msg = (
                f"\n\n----------------------------------------------------------------------------------------\n"
                f"epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s"
                f" | total time: {total_hours}h {total_mins}m {total_secs:.2f}s | "
                f"\n\n\tTrain loss: {train_loss:.4f} | "
                f"Val. loss: {valid_loss:.4f} | MAE: {float(valid_mae):.4f} | Pearson-r: {float(valid_pearson):.4f}\n\n"
            )
        else:
            msg = (
                f"----------------------------------------------------------------------------------------\n"
                f"epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s"
                f" | total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n\n\ttrain loss: {train_loss:.4f}\n\n"
            )

        with open(os.path.join(self.params["target_dir"], self.params["stat_log_path"]) + "/Stats", "a") as f:
            f.write(msg)





    def calculate_tb_stats(self, valid_loss=None, valid_mae=None, valid_pearson=None):
        if valid_loss is not None:
            self.writer.add_scalar("Val_loss", valid_loss, self.epoch)
        if valid_mae is not None:
            self.writer.add_scalar("Val_MAE", valid_mae, self.epoch)
        if valid_pearson is not None:
            self.writer.add_scalar("Val_PearsonR", valid_pearson, self.epoch)