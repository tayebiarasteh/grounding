"""
Created on Feb 23, 2026.
model.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel





class MeanPooler(nn.Module):
    def forward(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom





class BiEncoder(nn.Module):
    def __init__(self, model_name: str, tied: bool = True, pooling: str = "mean"):
        super().__init__()
        self.tied = bool(tied)

        self.query_encoder = AutoModel.from_pretrained(model_name)
        if self.tied:
            self.doc_encoder = self.query_encoder
        else:
            self.doc_encoder = AutoModel.from_pretrained(model_name)

        self.pooler = MeanPooler() if pooling == "mean" else MeanPooler()

    def encode(self, encoder: AutoModel, input_ids, attention_mask):
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(out.last_hidden_state, attention_mask)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    def encode_queries(self, input_ids, attention_mask):
        return self.encode(self.query_encoder, input_ids, attention_mask)

    def encode_docs(self, input_ids, attention_mask):
        return self.encode(self.doc_encoder, input_ids, attention_mask)
