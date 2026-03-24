"""
Created on March 18, 2026.
Prediction.py

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
import pandas as pd

from config.serde import read_config, write_config


import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15







class PredictionVerifier:
    def __init__(self, cfg_path: str, split_name: str, scenario_name: str):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.split_name = str(split_name)

        self.setup_cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.params["Verifier"]["model_name"],
            use_fast=True,
        )
        self.max_len = int(self.params["Verifier"].get("max_tokens", 1024))

        os.makedirs(os.path.join(self.params["target_dir"], self.params["stat_log_path"]), exist_ok=True)
        os.makedirs(os.path.join(self.params["target_dir"], self.params["stat_log_path"], f"{scenario_name}_split_{self.split_name}"), exist_ok=True)

    def setup_cuda(self, cuda_device_id: int = 0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _tokenize(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

    def setup_model(self, model, model_file_name=None, epoch_num=None):
        if model_file_name is None:
            model_file_name = self.params["trained_model_name"]

        self.model = model.to(self.device)

        if epoch_num is None:
            ckpt_path = os.path.join(
                self.params["target_dir"],
                self.params["network_output_path"],
                model_file_name,
            )
        else:
            ckpt_path = os.path.join(
                self.params["target_dir"],
                self.params["network_output_path"],
                f"epoch{epoch_num}_{model_file_name}",
            )

        state = torch.load(ckpt_path, map_location="cpu")

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        self.model.load_state_dict(state)
        self.model.eval()

    def predict_only(self, test_loader):
        self.model.eval()

        probs_cache = []
        labels_cache = []
        qa_ids_cache = []
        verifier_ids_cache = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                tok = self._tokenize(batch.inputs)
                targets = batch.targets.to(self.device, non_blocking=True).long()

                out = self.model(**tok)
                logits = out.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]

                probs_cache.append(probs.detach().cpu())
                labels_cache.append(targets.detach().cpu())
                qa_ids_cache.extend(batch.qa_row_ids)
                verifier_ids_cache.extend(batch.verifier_row_ids)

        probs_all = torch.cat(probs_cache, dim=0).numpy()
        labels_all = torch.cat(labels_cache, dim=0).numpy()

        return {
            "probs": probs_all,
            "targets": labels_all,
            "qa_row_ids": np.asarray(qa_ids_cache),
            "verifier_row_ids": np.asarray(verifier_ids_cache),
        }

    def _compute_youden_threshold(self, probs, targets):
        probs = np.asarray(probs).astype(np.float32)
        targets = np.asarray(targets).astype(np.int32)

        try:
            fpr, tpr, thresholds = metrics.roc_curve(targets, probs, pos_label=1)
            youden = tpr - fpr
            best_idx = int(np.argmax(youden))
            optimal_threshold = float(thresholds[best_idx])

            if not np.isfinite(optimal_threshold):
                finite_thresholds = thresholds[np.isfinite(thresholds)]
                if len(finite_thresholds) > 0:
                    optimal_threshold = float(finite_thresholds.max())
                else:
                    optimal_threshold = 0.5
        except Exception:
            optimal_threshold = 0.5

        return optimal_threshold


    def _compute_metrics(self, probs, targets):
        probs = np.asarray(probs).astype(np.float32)
        targets = np.asarray(targets).astype(np.int32)

        try:
            auc = float(metrics.roc_auc_score(targets, probs))
        except Exception:
            auc = 0.0

        try:
            auprc = float(metrics.average_precision_score(targets, probs))
        except Exception:
            auprc = 0.0

        try:
            brier = float(metrics.brier_score_loss(targets, probs))
        except Exception:
            brier = 0.0

        threshold = self._compute_youden_threshold(probs, targets)
        preds = (probs >= threshold).astype(np.int32)

        acc = float(metrics.accuracy_score(targets, preds))
        bal_acc = float(metrics.balanced_accuracy_score(targets, preds))
        f1 = float(metrics.f1_score(targets, preds, average="binary", zero_division=0))
        precision = float(metrics.precision_score(targets, preds, average="binary", zero_division=0))
        sensitivity = float(metrics.recall_score(targets, preds, average="binary", zero_division=0))

        tn, fp, fn, tp = metrics.confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp + epsilon))

        return {
            "auc": auc,
            "auprc": auprc,
            "brier": brier,
            "f1": f1,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "threshold": threshold,
            "preds": preds,
        }


    def _format_pct(self, x: float) -> str:
        return f"{x * 100:.2f}"

    def ci95(self, x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    def bootstrap_metrics(self, probs, targets, n_bootstrap=1000, seed=13):
        rng = np.random.RandomState(seed)
        n = len(targets)

        auc_list = []
        auprc_list = []
        brier_list = []
        f1_list = []
        acc_list = []
        bal_acc_list = []
        sens_list = []
        spec_list = []
        prec_list = []
        thr_list = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)

            probs_bs = probs[idx]
            targets_bs = targets[idx]

            m = self._compute_metrics(probs_bs, targets_bs)

            auc_list.append(m["auc"])
            auprc_list.append(m["auprc"])
            brier_list.append(m["brier"])
            f1_list.append(m["f1"])
            acc_list.append(m["accuracy"])
            bal_acc_list.append(m["balanced_accuracy"])
            sens_list.append(m["sensitivity"])
            spec_list.append(m["specificity"])
            prec_list.append(m["precision"])
            thr_list.append(m["threshold"])

        return {
            "auc": np.asarray(auc_list, dtype=np.float32),
            "auprc": np.asarray(auprc_list, dtype=np.float32),
            "brier": np.asarray(brier_list, dtype=np.float32),
            "f1": np.asarray(f1_list, dtype=np.float32),
            "accuracy": np.asarray(acc_list, dtype=np.float32),
            "balanced_accuracy": np.asarray(bal_acc_list, dtype=np.float32),
            "sensitivity": np.asarray(sens_list, dtype=np.float32),
            "specificity": np.asarray(spec_list, dtype=np.float32),
            "precision": np.asarray(prec_list, dtype=np.float32),
            "threshold": np.asarray(thr_list, dtype=np.float32),
        }

    def save_predictions_csv(self, pred_dict, metrics_dict, scenario_name):
        df = pd.DataFrame({
            "verifier_row_id": pred_dict["verifier_row_ids"],
            "qa_row_id": pred_dict["qa_row_ids"],
            "target": pred_dict["targets"],
            "pred_label": metrics_dict["preds"],
            "prob_supported": pred_dict["probs"],
        })

        out_path = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"], f"{scenario_name}_split_{self.split_name}",
            f"predictions_{scenario_name}_split_{self.split_name}.csv",
        )
        df.to_csv(out_path, index=False)

    def save_bootstrap_csv(self, boot_dict, scenario_name):
        df = pd.DataFrame({
            "auc": boot_dict["auc"],
            "auprc": boot_dict["auprc"],
            "brier": boot_dict["brier"],
            "f1": boot_dict["f1"],
            "accuracy": boot_dict["accuracy"],
            "balanced_accuracy": boot_dict["balanced_accuracy"],
            "sensitivity": boot_dict["sensitivity"],
            "specificity": boot_dict["specificity"],
            "precision": boot_dict["precision"],
            "threshold": boot_dict["threshold"],
        })

        out_path = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"], f"{scenario_name}_split_{self.split_name}",
            f"bootstrap_{scenario_name}_split_{self.split_name}.csv",
        )
        df.to_csv(out_path, index=False)

    def _summary_str_pct(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean()) * 100.0
        std = float(arr.std()) * 100.0
        lo, hi = self.ci95(arr)
        lo *= 100.0
        hi *= 100.0
        return f"{mean:.2f} ± {std:.2f} [{lo:.2f}, {hi:.2f}]"

    def _summary_str_raw(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        lo, hi = self.ci95(arr)
        return f"{mean:.8f} ± {std:.8f} [{lo:.8f}, {hi:.8f}]"

    def _summary_str_raw_short(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        lo, hi = self.ci95(arr)
        return f"{mean:.3f} ± {std:.3f} [{lo:.3f}, {hi:.3f}]"

    def print_and_save_stats(self, metrics_dict, boot_dict, scenario_name):
        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"experiment: {self.params['experiment_name']}\n"
            f"scenario: {scenario_name}\n"
            f"split: {self.split_name}\n\n"
            f"Threshold (Youden): {self._summary_str_raw(boot_dict['threshold'])}\n"
            f"AUROC: {self._summary_str_pct(boot_dict['auc'])}\n"
            f"AUPRC: {self._summary_str_pct(boot_dict['auprc'])}\n"
            f"F1: {self._summary_str_pct(boot_dict['f1'])}\n"
            f"Accuracy: {self._summary_str_pct(boot_dict['accuracy'])}\n"
            f"Balanced Accuracy: {self._summary_str_pct(boot_dict['balanced_accuracy'])}\n"
            f"Sensitivity: {self._summary_str_pct(boot_dict['sensitivity'])}\n"
            f"Specificity: {self._summary_str_pct(boot_dict['specificity'])}\n"
            f"Precision: {self._summary_str_pct(boot_dict['precision'])}\n"
            f"Brier score: {self._summary_str_raw(boot_dict['brier'])}\n"
        )

        print(msg)

        out_path = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"], f"{scenario_name}_split_{self.split_name}",
            f"Test_on_{scenario_name}_split_{self.split_name}",
        )
        with open(out_path, "a") as f:
            f.write(msg)




    def predict_only_mcq(self, test_loader):
        self.model.eval()

        probs_cache = []
        labels_cache = []
        qa_ids_cache = []
        verifier_ids_cache = []

        question_uids_cache = []
        option_ids_cache = []
        correct_option_ids_cache = []
        mcq_labels_cache = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                tok = self._tokenize(batch.inputs)
                targets = batch.targets.to(self.device, non_blocking=True).long()

                out = self.model(**tok)
                logits = out.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]

                probs_cache.append(probs.detach().cpu())
                labels_cache.append(targets.detach().cpu())
                qa_ids_cache.extend(batch.qa_row_ids)
                verifier_ids_cache.extend(batch.verifier_row_ids)

                question_uids_cache.extend(batch.question_uids)
                option_ids_cache.extend(batch.option_ids)
                correct_option_ids_cache.extend(batch.correct_option_ids)
                mcq_labels_cache.extend(batch.mcq_labels)

        probs_all = torch.cat(probs_cache, dim=0).numpy()
        labels_all = torch.cat(labels_cache, dim=0).numpy()

        return {
            "probs": probs_all,
            "targets": labels_all,
            "qa_row_ids": np.asarray(qa_ids_cache),
            "verifier_row_ids": np.asarray(verifier_ids_cache),
            "question_uids": np.asarray(question_uids_cache),
            "option_ids": np.asarray(option_ids_cache),
            "correct_option_ids": np.asarray(correct_option_ids_cache),
            "mcq_labels": np.asarray(mcq_labels_cache),
        }

    def save_and_report_mcq_stats(self, pred_dict, scenario_name, n_bootstrap=1000, seed=13):
        """
        Aggregates row-level option scores into question-level MCQ metrics.

        Expected pred_dict keys:
            probs
            targets
            qa_row_ids
            verifier_row_ids
            question_uids
            option_ids
            correct_option_ids
            mcq_labels
        """

        out_dir = os.path.join(
            self.params["target_dir"],
            self.params["stat_log_path"],
            f"{scenario_name}_split_{self.split_name}",
        )
        os.makedirs(out_dir, exist_ok=True)

        # --------------------------------------------------
        # row-level predictions
        # --------------------------------------------------
        row_df = pd.DataFrame({
            "verifier_row_id": pred_dict["verifier_row_ids"],
            "qa_row_id": pred_dict["qa_row_ids"],
            "question_uid": pred_dict["question_uids"],
            "option_id": pred_dict["option_ids"],
            "correct_option_id": pred_dict["correct_option_ids"],
            "is_correct_option": pred_dict["targets"],
            "mcq_label": pred_dict["mcq_labels"],
            "prob_supported": pred_dict["probs"],
        })

        row_out_path = os.path.join(
            out_dir,
            f"predictions_mcq_rows_{scenario_name}_split_{self.split_name}.csv",
        )
        row_df.to_csv(row_out_path, index=False)

        # --------------------------------------------------
        # question-level aggregation
        # --------------------------------------------------
        question_rows = []

        grouped = row_df.groupby("question_uid", sort=False)
        for question_uid, g in grouped:
            g = g.copy().reset_index(drop=True)

            if g.empty:
                continue

            if g["correct_option_id"].nunique() != 1:
                continue

            correct_option_id = int(g["correct_option_id"].iloc[0])

            # sort options by predicted support score descending
            g_sorted = g.sort_values(
                by=["prob_supported", "option_id"],
                ascending=[False, True],
            ).reset_index(drop=True)

            pred_option_id = int(g_sorted["option_id"].iloc[0])

            correct_match = g_sorted[g_sorted["option_id"].astype(int) == correct_option_id]
            if len(correct_match) == 0:
                continue

            correct_rank = int(correct_match.index[0]) + 1

            top1_correct = int(pred_option_id == correct_option_id)
            top2_correct = int(correct_rank <= min(2, len(g_sorted)))
            reciprocal_rank = 1.0 / float(correct_rank)

            top1_prob = float(g_sorted["prob_supported"].iloc[0])
            top2_prob = float(g_sorted["prob_supported"].iloc[1]) if len(g_sorted) >= 2 else np.nan
            margin = float(top1_prob - top2_prob) if len(g_sorted) >= 2 else np.nan

            pred_label = ""
            correct_label = ""

            pred_label_match = g_sorted[g_sorted["option_id"].astype(int) == pred_option_id]
            if len(pred_label_match) > 0 and pd.notna(pred_label_match["mcq_label"].iloc[0]):
                pred_label = str(pred_label_match["mcq_label"].iloc[0])

            correct_label_match = g_sorted[g_sorted["option_id"].astype(int) == correct_option_id]
            if len(correct_label_match) > 0 and pd.notna(correct_label_match["mcq_label"].iloc[0]):
                correct_label = str(correct_label_match["mcq_label"].iloc[0])

            question_rows.append({
                "question_uid": str(question_uid),
                "num_options": int(len(g_sorted)),
                "correct_option_id": int(correct_option_id),
                "pred_option_id": int(pred_option_id),
                "correct_label": correct_label,
                "pred_label": pred_label,
                "correct_rank": int(correct_rank),
                "mcq_accuracy": int(top1_correct),
                "top2_accuracy": int(top2_correct),
                "mrr": float(reciprocal_rank),
                "margin_top1_top2": float(margin) if np.isfinite(margin) else np.nan,
                "top1_prob": float(top1_prob),
                "top2_prob": float(top2_prob) if np.isfinite(top2_prob) else np.nan,
            })

        question_df = pd.DataFrame(question_rows)
        if question_df.empty:
            raise ValueError("Question-level MCQ aggregation produced no valid rows.")

        question_out_path = os.path.join(
            out_dir,
            f"predictions_mcq_questions_{scenario_name}_split_{self.split_name}.csv",
        )
        question_df.to_csv(question_out_path, index=False)

        # --------------------------------------------------
        # point estimates
        # --------------------------------------------------
        metrics_dict = {
            "mcq_accuracy": float(question_df["mcq_accuracy"].mean()),
            "top2_accuracy": float(question_df["top2_accuracy"].mean()),
            "mrr": float(question_df["mrr"].mean()),
            "margin_top1_top2": float(question_df["margin_top1_top2"].dropna().mean())
            if question_df["margin_top1_top2"].notna().any() else 0.0,
            "num_questions": int(len(question_df)),
        }

        # --------------------------------------------------
        # bootstrap over questions
        # --------------------------------------------------
        rng = np.random.RandomState(seed)
        n_questions = len(question_df)

        acc_list = []
        top2_list = []
        mrr_list = []
        margin_list = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n_questions, n_questions, replace=True)
            bs = question_df.iloc[idx]

            acc_list.append(float(bs["mcq_accuracy"].mean()))
            top2_list.append(float(bs["top2_accuracy"].mean()))
            mrr_list.append(float(bs["mrr"].mean()))

            margin_vals = bs["margin_top1_top2"].dropna().values
            if len(margin_vals) > 0:
                margin_list.append(float(np.mean(margin_vals)))
            else:
                margin_list.append(0.0)

        boot_dict = {
            "mcq_accuracy": np.asarray(acc_list, dtype=np.float32),
            "top2_accuracy": np.asarray(top2_list, dtype=np.float32),
            "mrr": np.asarray(mrr_list, dtype=np.float32),
            "margin_top1_top2": np.asarray(margin_list, dtype=np.float32),
        }

        boot_df = pd.DataFrame({
            "mcq_accuracy": boot_dict["mcq_accuracy"],
            "top2_accuracy": boot_dict["top2_accuracy"],
            "mrr": boot_dict["mrr"],
            "margin_top1_top2": boot_dict["margin_top1_top2"],
        })

        boot_out_path = os.path.join(
            out_dir,
            f"bootstrap_mcq_{scenario_name}_split_{self.split_name}.csv",
        )
        boot_df.to_csv(boot_out_path, index=False)

        # --------------------------------------------------
        # reporting
        # --------------------------------------------------
        msg = (
            f"\n\n----------------------------------------------------------------------------------------\n"
            f"experiment: {self.params['experiment_name']}\n"
            f"scenario: {scenario_name}\n"
            f"split: {self.split_name}\n\n"
            f"Num questions: {metrics_dict['num_questions']}\n"
            f"MCQ Accuracy: {self._summary_str_pct(boot_dict['mcq_accuracy'])}\n"
            f"Top-2 Accuracy: {self._summary_str_pct(boot_dict['top2_accuracy'])}\n"
            f"MRR: {self._summary_str_raw_short(boot_dict['mrr'])}\n"
            f"Margin (best-second): {self._summary_str_raw_short(boot_dict['margin_top1_top2'])}\n"
        )

        print(msg)

        txt_out_path = os.path.join(
            out_dir,
            f"Test_on_{scenario_name}_split_{self.split_name}",
        )
        with open(txt_out_path, "a") as f:
            f.write(msg)

        return metrics_dict, boot_dict, question_df