"""
Created on October 27, 2025.
Train_Valid_grounding.py

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
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        resume: bool
            if we are resuming training from a checkpoint
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path


        if resume == False:
            self.model_info = self.params['Network']
            self.epoch = 0
            self.best_loss = float('inf')
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path']))

        self.setup_cuda()
        self.grad_clip = float(self.params["Network"]["grad_clip_norm"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["retriever"]["model_name"], use_fast=True)
        self.max_q = int(self.params["retriever"]["max_query_tokens"])
        self.max_d = int(self.params["retriever"]["max_doc_tokens"])
        self.tau = float(self.params["retriever"]["temperature"])
        self.K = int(self.params["retriever"]["top_k_positives"])
        self.use_amp = bool(self.params.get("runtime", {}).get("use_amp", True))
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and self.device.type == "cuda"))



    def setup_cuda(self, cuda_device_id=0):
        """setup the device.

        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def time_duration(self, start_time, end_time):
        """calculating the duration of training or one iteration

        Parameters
        ----------
        start_time: float
            starting time of the operation

        end_time: float
            ending time of the operation

        Returns
        -------
        elapsed_hours: int
            total hours part of the elapsed time

        elapsed_mins: int
            total minutes part of the elapsed time

        elapsed_secs: int
            total seconds part of the elapsed time
        """
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
        return {k: v.to(self.device) for k, v in enc.items()}



    def setup_model(self, model, optimiser, loss_function):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function

        weight: 1D tensor of float
            class weights
        """

        # prints the network's total number of trainable parameters and
        # stores it to the experiment config
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal # of trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        self.model = model.to(self.device)

        self.loss_function = loss_function(temperature=self.tau).to(self.device)
        self.optimiser = optimiser

        # Saves the model, optimiser,loss function name for writing to config file
        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = loss_function.__name__
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)



    def load_checkpoint(self, model, optimiser, loss_function):
        """In case of resuming training from a checkpoint,
        loads the weights for all the models, optimizers, and
        loss functions, and device, tensorboard events, number
        of iterations (epochs), and every info from checkpoint.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function
        """
        checkpoint = torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                self.params['checkpoint_name']))
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.loss_function = loss_function(temperature=self.tau).to(self.device)
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.epoch + 1)




    def train_epoch(self, train_loader, valid_loader=None, num_epochs=1000):
        """Training epoch
        """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for epoch in range(num_epochs - self.epoch):
            self.epoch += 1

            # initializing the loss list
            batch_loss = 0
            start_time = time.time()

            for idx, batch in enumerate(train_loader):
                self.model.train()
                # if idx >= 100:
                #     break

                # If docs are empty strings, tokenizer will still work but embeddings will be garbage.
                # Keep as-is; weights are 0 for empties so they only act as negatives.
                q_tok = self._tokenize(batch.queries, self.max_q)
                d_tok = self._tokenize(batch.docs, self.max_d)

                pos_mask = batch.pos_mask.to(self.device)
                pos_w = batch.pos_weights.to(self.device)

                self.optimiser.zero_grad()

                with torch.cuda.amp.autocast(enabled=(self.use_amp and self.device.type == "cuda")):
                    q_emb = self.model.encode_queries(q_tok["input_ids"], q_tok["attention_mask"])
                    d_emb = self.model.encode_docs(d_tok["input_ids"], d_tok["attention_mask"])

                    loss, sim = self.loss_function(q_emb, d_emb, pos_mask, pos_w)
                    batch_loss += loss.item()

                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimiser)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimiser)
                self.scaler.update()

            train_loss = batch_loss / len(train_loader)
            self.writer.add_scalar('Train_loss_avg', train_loss, self.epoch)

            # Saves information about training to config file
            self.params['Network']['num_epoch'] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            ######## Save a checkpoint every epoch ########
            torch.save({'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'loss_state_dict': self.loss_function.state_dict(),
                        'model_info': self.model_info, 'best_loss': self.best_loss},
                       os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                    self.params['checkpoint_name']))
            ######## Save a checkpoint every epoch ########

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:

                    valid_loss, valid_recall_1, valid_recall_5 = self.valid_epoch(valid_loader)

                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(valid_loss=valid_loss, valid_recall_1=valid_recall_1, valid_recall_5=valid_recall_5)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time, valid_loss=valid_loss, valid_recall_1=valid_recall_1, valid_recall_5=valid_recall_5)
                else:
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time)



    def valid_epoch(self, valid_loader):
        """
        Run 1 full validation epoch.

        Returns:
          valid_loss_avg, recall@1, recall@5

        Assumptions (based on your train loop):
          - batch.queries: List[str] length B
          - batch.docs:    List[str] length B (flattened per-query docs, typically B*K)
          - batch.pos_mask: Tensor [B, B*K] with 1 where doc is a positive for query
          - batch.pos_weights: Tensor [B, B*K] with weights for positives (0 for non-pos)
          - self._tokenize(texts, max_len) -> dict with input_ids, attention_mask
          - self.model.encode_queries(ids, mask) -> [B, D]
          - self.model.encode_docs(ids, mask) -> [B*K, D]
            where sim is [B, B*K] (higher is better)
        """
        self.model.eval()

        total_loss = 0.0
        total_q = 0  # number of queries evaluated

        # for micro-averaged recall (count queries with >=1 hit)
        hit1 = 0
        hit5 = 0

        with torch.no_grad():
            for idx, batch in enumerate(valid_loader):
                # if idx >= 100:
                #     break

                # tokenize
                q_tok = self._tokenize(batch.queries, self.max_q)
                d_tok = self._tokenize(batch.docs, self.max_d)

                # masks/weights
                pos_mask = batch.pos_mask.to(self.device)  # [B, Ndocs]
                pos_w = batch.pos_weights.to(self.device)  # [B, Ndocs]

                # forward
                with torch.cuda.amp.autocast(enabled=(self.use_amp and self.device.type == "cuda")):
                    q_emb = self.model.encode_queries(q_tok["input_ids"], q_tok["attention_mask"])
                    d_emb = self.model.encode_docs(d_tok["input_ids"], d_tok["attention_mask"])

                    loss, sim = self.loss_function(q_emb, d_emb, pos_mask, pos_w)

                # accumulate loss (weighted by batch size)
                B = int(pos_mask.shape[0])
                total_loss += float(loss.item()) * B
                total_q += B

                # -------- recall@k computation (within this batch's doc pool) --------
                # A query is a "hit@k" if any of its positive docs appears in top-k ranked docs.
                # If a query has no positives in this batch (should be rare), we count it as 0 hit.

                # sim: [B, Ndocs]
                # pos_mask: [B, Ndocs] binary
                # topk indices
                # (handle small Ndocs safely)
                Ndocs = int(sim.shape[1])
                k1 = 1
                k5 = 5 if Ndocs >= 5 else Ndocs

                # top-1
                top1_idx = torch.topk(sim, k=k1, dim=1, largest=True).indices  # [B,1]
                # gather pos flags at those indices
                top1_pos = pos_mask.gather(1, top1_idx)  # [B,1]
                # hit if any positive in top1 (same as top1_pos==1)
                hit1_batch = (top1_pos.max(dim=1).values > 0).sum().item()

                # top-5
                if k5 > 0:
                    top5_idx = torch.topk(sim, k=k5, dim=1, largest=True).indices  # [B,k5]
                    top5_pos = pos_mask.gather(1, top5_idx)  # [B,k5]
                    hit5_batch = (top5_pos.max(dim=1).values > 0).sum().item()
                else:
                    hit5_batch = 0

                hit1 += int(hit1_batch)
                hit5 += int(hit5_batch)

        valid_loss = (total_loss / max(total_q, 1)) if total_q > 0 else 0.0
        valid_recall_1 = (hit1 / max(total_q, 1)) if total_q > 0 else 0.0
        valid_recall_5 = (hit5 / max(total_q, 1)) if total_q > 0 else 0.0


        return valid_loss, valid_recall_1, valid_recall_5




    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs, total_hours,
                       total_mins, total_secs, train_loss, total_time, total_overhead_time=0, total_datacopy_time=0, valid_loss=None, valid_recall_1=None, valid_recall_5=None):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        Parameters
        ----------
        iteration_hours: int
            hours part of the elapsed time of each iteration

        iteration_mins: int
            minutes part of the elapsed time of each iteration

        iteration_secs: int
            seconds part of the elapsed time of each iteration

        total_hours: int
            hours part of the total elapsed time

        total_mins: int
            minutes part of the total elapsed time

        total_secs: int
            seconds part of the total elapsed time

        train_loss: float
            training loss of the model

        valid_loss: float
            validation loss of the model
        """

        # Saves information about training to config file
        self.params['Network']['num_epoch'] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        # Saving the model based on the best loss
        if valid_loss:
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'],
                                                                 self.params['network_output_path'], self.params['trained_model_name']))
        else:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'],
                                                                 self.params['network_output_path'], self.params['trained_model_name']))

        # Save a checkpoint every epoch
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'loss_state_dict': self.loss_function.state_dict(),
                    'model_info': self.model_info, 'best_loss': self.best_loss},
                   os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))
        # Saving every couple of epochs
        if (self.epoch) % self.params['display_stats_freq'] == 0:
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path'],
                       'epoch{}_'.format(self.epoch) + self.params['trained_model_name']))

        print('------------------------------------------------------'
              '----------------------------------')
        print(f'epoch: {self.epoch} | '
              f'epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s | '
              f'total time: {total_hours}h {total_mins}m {total_secs:.2f}s')
        print(f'\n\tTrain loss: {train_loss:.4f}\n')

        if valid_loss:
            print(f'\t Val. loss: {valid_loss:.4f} | avg Recall@1: {valid_recall_1 * 100:.2f}% | avg Recall@5: {valid_recall_5 * 100:.2f}%\n')

            # saving the training and validation stats
            msg = f'\n\n----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s | ' \
                  f'\n\n\tTrain loss: {train_loss:.4f} | ' \
                   f'Val. loss: {valid_loss:.4f} | avg Recall@1: {valid_recall_1 * 100:.2f}% | avg Recall@5: {valid_recall_5 * 100:.2f}\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n\n\ttrain loss: {train_loss:.4f}\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def calculate_tb_stats(self, valid_loss=None, valid_recall_1=None, valid_recall_5=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        Parameters
        ----------
        valid_loss: float
            validation loss of the model
        """
        if valid_loss is not None:
            self.writer.add_scalar('Recall@1', valid_recall_1, self.epoch)

            self.writer.add_scalar('Recall@5', valid_recall_5, self.epoch)