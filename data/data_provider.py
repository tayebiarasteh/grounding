"""
Created on Feb 23, 2026.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import json
import os.path
import pdb
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import math
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional
import random
import re

from config.serde import read_config





@dataclass(frozen=True)
class VerifierBatch:
    inputs: List[str]
    targets: torch.Tensor
    qa_row_ids: List[int]
    verifier_row_ids: List[int]


class VerifierBinaryDataset(Dataset):
    """
    Reads binary verifier supervision parquet.

    Input format:
        CASE + EVIDENCE + CLAIM

    Expected parquet columns:
        verifier_row_id
        qa_row_id
        split
        case_context
        claim_text
        evidence_sentences_json
        label_id
    """

    def __init__(
        self,
        cfg_path: str,
        split_name: str,
        supervision_relpath: str = "verifier_binary_dataset/verifier_binary_trainvaltest.parquet",
        max_evidence_sentences: int = 2,
        max_evidence_chars: int = 12000,
    ) -> None:
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.base_path = self.params["Preprocessing"]["base_path_evidenceData"]

        self.split_name = str(split_name)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.max_evidence_chars = int(max_evidence_chars)

        self.supervision_path = os.path.join(self.base_path, supervision_relpath)
        if not os.path.exists(self.supervision_path):
            raise FileNotFoundError(f"Verifier supervision parquet not found: {self.supervision_path}")

        self.pf = pq.ParquetFile(self.supervision_path)

        required = [
            "verifier_row_id",
            "qa_row_id",
            "split",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        schema_names = set(self.pf.schema.names)
        for c in required:
            if c not in schema_names:
                raise ValueError(f"Missing required column in verifier parquet: {c}")

        self._rg_row_index: List[tuple[int, int]] = []
        self._build_split_index()

        self._rg_cache: Dict[int, Dict[str, List]] = {}
        self._label_counts_cache: Optional[Dict[int, int]] = None

    def _build_split_index(self) -> None:
        for rg in range(self.pf.num_row_groups):
            t = self.pf.read_row_group(rg, columns=["split"])
            split_list = t.column(0).to_pylist()
            for i, s in enumerate(split_list):
                if str(s) == self.split_name:
                    self._rg_row_index.append((rg, i))

        if len(self._rg_row_index) == 0:
            print(f"[WARN] No rows found for split='{self.split_name}' in {self.supervision_path}")

    def __len__(self) -> int:
        return len(self._rg_row_index)

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(self, s: str) -> List:
        try:
            out = json.loads(s)
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _truncate_text(self, s: str) -> str:
        s = self._safe_str(s)
        if self.max_evidence_chars > 0 and len(s) > self.max_evidence_chars:
            return s[: self.max_evidence_chars]
        return s

    def _materialize_evidence(self, evidence_sentences_json: str) -> str:
        sents = self._loads_json_list(self._safe_str(evidence_sentences_json))
        sents = [self._safe_str(x).strip() for x in sents if self._safe_str(x).strip()]
        sents = sents[: self.max_evidence_sentences]
        return self._truncate_text("\n\n".join(sents))

    def _build_input_text(
            self,
            case_context: str,
            evidence_text: str,
            claim_text: str,
    ) -> str:
        case_text = self._safe_str(case_context).strip()
        evidence_text = self._safe_str(evidence_text).strip()
        claim_text = self._safe_str(claim_text).strip()

        return (
            "[TASK] support_verification\n\n"
            "[CASE_BEGIN]\n"
            f"{case_text}\n"
            "[CASE_END]\n\n"
            "[EVIDENCE_BEGIN]\n"
            f"{evidence_text}\n"
            "[EVIDENCE_END]\n\n"
            "[CLAIM_BEGIN]\n"
            f"{claim_text}\n"
            "[CLAIM_END]"
        )


    def _get_row_group_columns(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached

        cols = [
            "verifier_row_id",
            "qa_row_id",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        t = self.pf.read_row_group(rg, columns=cols)
        payload = {c: t.column(c).to_pylist() for c in cols}
        self._rg_cache[rg] = payload
        return payload

    def get_label_counts(self) -> Dict[int, int]:
        if self._label_counts_cache is not None:
            return dict(self._label_counts_cache)

        counts = {0: 0, 1: 0}
        for rg, row_in_group in self._rg_row_index:
            payload = self._get_row_group_columns(rg)
            try:
                y = int(payload["label_id"][row_in_group])
            except Exception:
                y = 0
            counts[y] = counts.get(y, 0) + 1

        self._label_counts_cache = counts
        return dict(counts)

    def __getitem__(self, idx: int) -> Dict:
        rg, row_in_group = self._rg_row_index[idx]
        payload = self._get_row_group_columns(rg)

        verifier_row_id = int(payload["verifier_row_id"][row_in_group])
        qa_row_id = int(payload["qa_row_id"][row_in_group])

        case_context = payload["case_context"][row_in_group]
        claim_text = payload["claim_text"][row_in_group]
        evidence_sentences_json = payload["evidence_sentences_json"][row_in_group]
        label_id = int(payload["label_id"][row_in_group])

        evidence_text = self._materialize_evidence(evidence_sentences_json)
        input_text = self._build_input_text(
            case_context=case_context,
            evidence_text=evidence_text,
            claim_text=claim_text,
        )

        return {
            "verifier_row_id": verifier_row_id,
            "qa_row_id": qa_row_id,
            "input_text": input_text,
            "target": label_id,
        }


def verifier_binary_collate_fn(batch: List[Dict]) -> VerifierBatch:
    inputs = [str(x["input_text"]).strip() for x in batch]
    targets = torch.tensor([int(x["target"]) for x in batch], dtype=torch.long)
    qa_row_ids = [int(x["qa_row_id"]) for x in batch]
    verifier_row_ids = [int(x["verifier_row_id"]) for x in batch]

    return VerifierBatch(
        inputs=inputs,
        targets=targets,
        qa_row_ids=qa_row_ids,
        verifier_row_ids=verifier_row_ids,
    )











@dataclass(frozen=True)
class VerifierBatch_randomp:
    inputs: List[str]
    targets: torch.Tensor
    qa_row_ids: List[int]
    verifier_row_ids: List[int]
    batch_top_p: int


class VerifierBinaryDataset_randomp(Dataset):
    """
    Reads binary verifier supervision parquet.

    Input format at collate time:
        CASE + EVIDENCE + CLAIM

    Expected parquet columns:
        verifier_row_id
        qa_row_id
        split
        case_context
        claim_text
        evidence_sentences_json
        label_id
    """

    def __init__(
        self,
        cfg_path: str,
        split_name: str,
        supervision_relpath: str = "verifier_binary_dataset/verifier_binary_trainvaltest.parquet",
        max_evidence_chars: int = 12000,
    ) -> None:
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.base_path = self.params["Preprocessing"]["base_path_evidenceData"]

        self.split_name = str(split_name)
        self.max_evidence_chars = int(max_evidence_chars)

        self.supervision_path = os.path.join(self.base_path, supervision_relpath)
        if not os.path.exists(self.supervision_path):
            raise FileNotFoundError(f"Verifier supervision parquet not found: {self.supervision_path}")

        self.pf = pq.ParquetFile(self.supervision_path)

        required = [
            "verifier_row_id",
            "qa_row_id",
            "split",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        schema_names = set(self.pf.schema.names)
        for c in required:
            if c not in schema_names:
                raise ValueError(f"Missing required column in verifier parquet: {c}")

        self._rg_row_index: List[tuple[int, int]] = []
        self._build_split_index()

        self._rg_cache: Dict[int, Dict[str, List]] = {}
        self._label_counts_cache: Optional[Dict[int, int]] = None

    def _build_split_index(self) -> None:
        for rg in range(self.pf.num_row_groups):
            t = self.pf.read_row_group(rg, columns=["split"])
            split_list = t.column(0).to_pylist()
            for i, s in enumerate(split_list):
                if str(s) == self.split_name:
                    self._rg_row_index.append((rg, i))

        if len(self._rg_row_index) == 0:
            print(f"[WARN] No rows found for split='{self.split_name}' in {self.supervision_path}")

    def __len__(self) -> int:
        return len(self._rg_row_index)

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(self, s: str) -> List:
        try:
            out = json.loads(s)
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _truncate_text(self, s: str) -> str:
        s = self._safe_str(s)
        if self.max_evidence_chars > 0 and len(s) > self.max_evidence_chars:
            return s[: self.max_evidence_chars]
        return s

    def _materialize_evidence(self, evidence_sentences_json: str, max_evidence_sentences: int) -> str:
        sents = self._loads_json_list(self._safe_str(evidence_sentences_json))
        sents = [self._safe_str(x).strip() for x in sents if self._safe_str(x).strip()]
        sents = sents[: int(max_evidence_sentences)]
        return self._truncate_text("\n\n".join(sents))

    def _build_input_text(
        self,
        case_context: str,
        evidence_text: str,
        claim_text: str,
    ) -> str:
        case_text = self._safe_str(case_context).strip()
        evidence_text = self._safe_str(evidence_text).strip()
        claim_text = self._safe_str(claim_text).strip()

        return (
            "[TASK] support_verification\n\n"
            "[CASE_BEGIN]\n"
            f"{case_text}\n"
            "[CASE_END]\n\n"
            "[EVIDENCE_BEGIN]\n"
            f"{evidence_text}\n"
            "[EVIDENCE_END]\n\n"
            "[CLAIM_BEGIN]\n"
            f"{claim_text}\n"
            "[CLAIM_END]"
        )

    def _get_row_group_columns(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached

        cols = [
            "verifier_row_id",
            "qa_row_id",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        t = self.pf.read_row_group(rg, columns=cols)
        payload = {c: t.column(c).to_pylist() for c in cols}
        self._rg_cache[rg] = payload
        return payload

    def get_label_counts(self) -> Dict[int, int]:
        if self._label_counts_cache is not None:
            return dict(self._label_counts_cache)

        counts = {0: 0, 1: 0}
        for rg, row_in_group in self._rg_row_index:
            payload = self._get_row_group_columns(rg)
            try:
                y = int(payload["label_id"][row_in_group])
            except Exception:
                y = 0
            counts[y] = counts.get(y, 0) + 1

        self._label_counts_cache = counts
        return dict(counts)

    def __getitem__(self, idx: int) -> Dict:
        rg, row_in_group = self._rg_row_index[idx]
        payload = self._get_row_group_columns(rg)

        verifier_row_id = int(payload["verifier_row_id"][row_in_group])
        qa_row_id = int(payload["qa_row_id"][row_in_group])

        case_context = payload["case_context"][row_in_group]
        claim_text = payload["claim_text"][row_in_group]
        evidence_sentences_json = payload["evidence_sentences_json"][row_in_group]
        label_id = int(payload["label_id"][row_in_group])

        return {
            "verifier_row_id": verifier_row_id,
            "qa_row_id": qa_row_id,
            "case_context": case_context,
            "claim_text": claim_text,
            "evidence_sentences_json": evidence_sentences_json,
            "target": label_id,
        }


class RandomTopPCollator:
    """
    Samples one top-p value per batch and uses it for all examples in that batch.

    Training use:
        RandomTopPCollator(dataset, train_top_p_choices=[2,3,4,5,6,7,8,9])

    Validation use:
        RandomTopPCollator(dataset, fixed_top_p=2)
    """

    def __init__(
        self,
        dataset: VerifierBinaryDataset_randomp,
        train_top_p_choices: Optional[Sequence[int]] = None,
        fixed_top_p: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.train_top_p_choices = list(train_top_p_choices) if train_top_p_choices is not None else [2, 3, 4, 5, 6, 7, 8, 9]
        self.fixed_top_p = int(fixed_top_p) if fixed_top_p is not None else None

        if self.fixed_top_p is None and len(self.train_top_p_choices) == 0:
            raise ValueError("train_top_p_choices must be non-empty when fixed_top_p is None")

    def _sample_top_p(self) -> int:
        if self.fixed_top_p is not None:
            return self.fixed_top_p
        return int(random.choice(self.train_top_p_choices))

    def __call__(self, batch: List[Dict]) -> VerifierBatch_randomp:
        batch_top_p = self._sample_top_p()

        inputs: List[str] = []
        targets: List[int] = []
        qa_row_ids: List[int] = []
        verifier_row_ids: List[int] = []

        for x in batch:
            evidence_text = self.dataset._materialize_evidence(
                x["evidence_sentences_json"],
                max_evidence_sentences=batch_top_p,
            )

            input_text = self.dataset._build_input_text(
                case_context=x["case_context"],
                evidence_text=evidence_text,
                claim_text=x["claim_text"],
            )

            inputs.append(str(input_text).strip())
            targets.append(int(x["target"]))
            qa_row_ids.append(int(x["qa_row_id"]))
            verifier_row_ids.append(int(x["verifier_row_id"]))

        return VerifierBatch_randomp(
            inputs=inputs,
            targets=torch.tensor(targets, dtype=torch.long),
            qa_row_ids=qa_row_ids,
            verifier_row_ids=verifier_row_ids,
            batch_top_p=batch_top_p,
        )


def verifier_binary_collate_fn_fixed_top_p(
    batch: List[Dict],
    dataset: VerifierBinaryDataset_randomp,
    max_evidence_sentences: int,
) -> VerifierBatch_randomp:
    """
    Optional helper if you want the old fixed-top-p behavior without instantiating the class.
    """
    collator = RandomTopPCollator(
        dataset=dataset,
        fixed_top_p=int(max_evidence_sentences),
    )
    return collator(batch)












class VerifierBinaryTestDataset(Dataset):
    """
    Test dataset for verifier evaluation on a specific scenario parquet.
    """

    def __init__(
            self,
            cfg_path: str,
            test_parquet_path: str,
            split_name: str = "test",
            max_evidence_sentences: int = 2,
            max_evidence_chars: int = 12000,
    ) -> None:
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)

        self.test_parquet_path = str(test_parquet_path)
        self.split_name = str(split_name)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.max_evidence_chars = int(max_evidence_chars)

        if not os.path.exists(self.test_parquet_path):
            raise FileNotFoundError(f"Test parquet not found: {self.test_parquet_path}")

        self.pf = pq.ParquetFile(self.test_parquet_path)

        required = [
            "verifier_row_id",
            "qa_row_id",
            "split",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        schema_names = set(self.pf.schema.names)
        for c in required:
            if c not in schema_names:
                raise ValueError(f"Missing required column in verifier parquet: {c}")

        self._rg_row_index = []
        self._build_index()
        self._rg_cache: Dict[int, Dict[str, List]] = {}


    def _build_index(self) -> None:
        for rg in range(self.pf.num_row_groups):
            t = self.pf.read_row_group(rg, columns=["split"])
            split_list = t.column(0).to_pylist()

            for i, s in enumerate(split_list):
                if str(s) == self.split_name:
                    self._rg_row_index.append((rg, i))

        if len(self._rg_row_index) == 0:
            print(f"[WARN] No rows found for split='{self.split_name}' in {self.test_parquet_path}")


    def __len__(self) -> int:
        return len(self._rg_row_index)

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(self, s: str) -> List:
        try:
            out = json.loads(s)
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _truncate_text(self, s: str) -> str:
        s = self._safe_str(s)
        if self.max_evidence_chars > 0 and len(s) > self.max_evidence_chars:
            return s[: self.max_evidence_chars]
        return s

    def _materialize_evidence(self, evidence_sentences_json: str) -> str:
        sents = self._loads_json_list(self._safe_str(evidence_sentences_json))
        sents = [self._safe_str(x).strip() for x in sents if self._safe_str(x).strip()]
        sents = sents[: self.max_evidence_sentences]
        return self._truncate_text("\n\n".join(sents))

    def _build_input_text(
        self,
        case_context: str,
        evidence_text: str,
        claim_text: str,
    ) -> str:
        case_text = self._safe_str(case_context).strip()
        evidence_text = self._safe_str(evidence_text).strip()
        claim_text = self._safe_str(claim_text).strip()

        return (
            "[TASK] support_verification\n\n"
            "[CASE_BEGIN]\n"
            f"{case_text}\n"
            "[CASE_END]\n\n"
            "[EVIDENCE_BEGIN]\n"
            f"{evidence_text}\n"
            "[EVIDENCE_END]\n\n"
            "[CLAIM_BEGIN]\n"
            f"{claim_text}\n"
            "[CLAIM_END]"
        )

    def _get_row_group_columns(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached

        cols = [
            "verifier_row_id",
            "qa_row_id",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        t = self.pf.read_row_group(rg, columns=cols)
        payload = {c: t.column(c).to_pylist() for c in cols}
        self._rg_cache[rg] = payload
        return payload

    def __getitem__(self, idx: int) -> Dict:
        rg, row_in_group = self._rg_row_index[idx]
        payload = self._get_row_group_columns(rg)

        verifier_row_id = int(payload["verifier_row_id"][row_in_group])
        qa_row_id = int(payload["qa_row_id"][row_in_group])

        case_context = payload["case_context"][row_in_group]
        claim_text = payload["claim_text"][row_in_group]
        evidence_sentences_json = payload["evidence_sentences_json"][row_in_group]
        label_id = int(payload["label_id"][row_in_group])

        evidence_text = self._materialize_evidence(evidence_sentences_json)
        input_text = self._build_input_text(
            case_context=case_context,
            evidence_text=evidence_text,
            claim_text=claim_text,
        )

        return {
            "verifier_row_id": verifier_row_id,
            "qa_row_id": qa_row_id,
            "input_text": input_text,
            "target": label_id,
        }







@dataclass(frozen=True)
class VerifierMCQBatch:
    inputs: List[str]
    targets: torch.Tensor
    qa_row_ids: List[int]
    verifier_row_ids: List[int]
    question_uids: List[str]
    option_ids: List[int]
    correct_option_ids: List[int]
    mcq_labels: List[str]


class VerifierBinaryVignetteTestDataset(Dataset):
    """
    Test dataset for MCQ-style vignette evaluation from a verifier parquet.

    Each row is still a standard verifier row:
        case_context + evidence + claim -> label_id

    Additional MCQ metadata are carried so that question-level metrics
    can be computed later by grouping options belonging to the same question.
    """

    def __init__(
        self,
        cfg_path: str,
        test_parquet_path: str,
        split_name: str = "test",
        max_evidence_sentences: int = 2,
        max_evidence_chars: int = 12000,
    ) -> None:
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)

        self.test_parquet_path = str(test_parquet_path)
        self.split_name = str(split_name)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.max_evidence_chars = int(max_evidence_chars)

        if not os.path.exists(self.test_parquet_path):
            raise FileNotFoundError(f"Test parquet not found: {self.test_parquet_path}")

        self.pf = pq.ParquetFile(self.test_parquet_path)

        required = [
            "verifier_row_id",
            "qa_row_id",
            "split",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
            "question_uid",
            "option_id",
            "correct_option_id",
            "mcq_label",
        ]
        schema_names = set(self.pf.schema.names)
        for c in required:
            if c not in schema_names:
                raise ValueError(f"Missing required column in vignette verifier parquet: {c}")

        self._rg_row_index = []
        self._rg_cache: Dict[int, Dict[str, List]] = {}
        self._build_index()

    def _build_index(self) -> None:
        for rg in range(self.pf.num_row_groups):
            t = self.pf.read_row_group(rg, columns=["split"])
            split_list = t.column(0).to_pylist()

            for i, s in enumerate(split_list):
                if str(s) == self.split_name:
                    self._rg_row_index.append((rg, i))

        if len(self._rg_row_index) == 0:
            print(f"[WARN] No rows found for split='{self.split_name}' in {self.test_parquet_path}")

    def __len__(self) -> int:
        return len(self._rg_row_index)

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _safe_int(self, x, default: int = -1) -> int:
        try:
            if x is None:
                return int(default)
            s = str(x).strip()
            if s == "" or s.lower() == "nan":
                return int(default)
            return int(float(s))
        except Exception:
            return int(default)

    def _loads_json_list(self, s: str) -> List:
        try:
            out = json.loads(s)
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _truncate_text(self, s: str) -> str:
        s = self._safe_str(s)
        if self.max_evidence_chars > 0 and len(s) > self.max_evidence_chars:
            return s[: self.max_evidence_chars]
        return s

    def _materialize_evidence(self, evidence_sentences_json: str) -> str:
        sents = self._loads_json_list(self._safe_str(evidence_sentences_json))
        sents = [self._safe_str(x).strip() for x in sents if self._safe_str(x).strip()]
        sents = sents[: self.max_evidence_sentences]
        return self._truncate_text("\n\n".join(sents))

    def _build_input_text(
        self,
        case_context: str,
        evidence_text: str,
        claim_text: str,
    ) -> str:
        case_text = self._safe_str(case_context).strip()
        evidence_text = self._safe_str(evidence_text).strip()
        claim_text = self._safe_str(claim_text).strip()

        return (
            "[TASK] support_verification\n\n"
            "[CASE_BEGIN]\n"
            f"{case_text}\n"
            "[CASE_END]\n\n"
            "[EVIDENCE_BEGIN]\n"
            f"{evidence_text}\n"
            "[EVIDENCE_END]\n\n"
            "[CLAIM_BEGIN]\n"
            f"{claim_text}\n"
            "[CLAIM_END]"
        )

    def _get_row_group_columns(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached

        cols = [
            "verifier_row_id",
            "qa_row_id",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
            "question_uid",
            "option_id",
            "correct_option_id",
            "mcq_label",
        ]
        t = self.pf.read_row_group(rg, columns=cols)
        payload = {c: t.column(c).to_pylist() for c in cols}
        self._rg_cache[rg] = payload
        return payload

    def __getitem__(self, idx: int) -> Dict:
        rg, row_in_group = self._rg_row_index[idx]
        payload = self._get_row_group_columns(rg)

        verifier_row_id = self._safe_int(payload["verifier_row_id"][row_in_group], default=-1)
        qa_row_id = self._safe_int(payload["qa_row_id"][row_in_group], default=-1)

        case_context = payload["case_context"][row_in_group]
        claim_text = payload["claim_text"][row_in_group]
        evidence_sentences_json = payload["evidence_sentences_json"][row_in_group]
        label_id = self._safe_int(payload["label_id"][row_in_group], default=0)

        question_uid = self._safe_str(payload["question_uid"][row_in_group])
        option_id = self._safe_int(payload["option_id"][row_in_group], default=-1)
        correct_option_id = self._safe_int(payload["correct_option_id"][row_in_group], default=-1)
        mcq_label = self._safe_str(payload["mcq_label"][row_in_group])

        evidence_text = self._materialize_evidence(evidence_sentences_json)
        input_text = self._build_input_text(
            case_context=case_context,
            evidence_text=evidence_text,
            claim_text=claim_text,
        )

        return {
            "verifier_row_id": verifier_row_id,
            "qa_row_id": qa_row_id,
            "input_text": input_text,
            "target": label_id,
            "question_uid": question_uid,
            "option_id": option_id,
            "correct_option_id": correct_option_id,
            "mcq_label": mcq_label,
        }


def verifier_binary_vignette_collate_fn(batch: List[Dict]) -> VerifierMCQBatch:
    inputs = [str(x["input_text"]).strip() for x in batch]
    targets = torch.tensor([int(x["target"]) for x in batch], dtype=torch.long)
    qa_row_ids = [int(x["qa_row_id"]) for x in batch]
    verifier_row_ids = [int(x["verifier_row_id"]) for x in batch]

    question_uids = [str(x["question_uid"]) for x in batch]
    option_ids = [int(x["option_id"]) for x in batch]
    correct_option_ids = [int(x["correct_option_id"]) for x in batch]
    mcq_labels = [str(x["mcq_label"]) for x in batch]

    return VerifierMCQBatch(
        inputs=inputs,
        targets=targets,
        qa_row_ids=qa_row_ids,
        verifier_row_ids=verifier_row_ids,
        question_uids=question_uids,
        option_ids=option_ids,
        correct_option_ids=correct_option_ids,
        mcq_labels=mcq_labels,
    )