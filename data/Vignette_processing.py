"""
Created on March 2, 2026.
Vignette_processing.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import re
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import hashlib
from transformers import AutoTokenizer, AutoModel
from dataclasses import asdict, dataclass
import faiss
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import random
import numpy as np
import os
import yaml
import math
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')






@dataclass
class MCQExample:
    qa_row_id: int
    source_file: str
    question_id: str
    question: str
    options: List[str]          # length K
    correct_index: int          # 0..K-1
    num_options: int            # K
    # fields to match your tier eval schema
    subject_id: str
    context: str                # u_i
    q: str                      # q_i
    a: str                      # a_i (option text)
    claim: str                  # c_i
    label: int                  # correct_index, duplicated for convenience


class RadiologyVignetteProcessor:
    """
    Loads the two CSV format (4-option and 5-option), normalizes them into a
    single canonical MCQ dataset, and expands into per-option rows suitable for tier evaluation.

    Output has:
      - One "question table" (one row per question)
      - One "expanded QA table" (one row per option, with y as {0,1} for that option)
    """

    def __init__(
        self,
        seed: int = 13,
        q_field_prompt: str = "Select the correct answer.",
        claim_template: str = "The correct answer is: {option}.",
        # If True, retrieval query will include the vignette context + claim.
        # This is usually best for these vignette questions.
        retrieval_query_mode: str = "context_plus_claim",  # {"claim_only", "context_plus_claim"}
    ):
        self.seed = int(seed)
        self.q_field_prompt = q_field_prompt
        self.claim_template = claim_template
        if retrieval_query_mode not in {"claim_only", "context_plus_claim"}:
            raise ValueError("retrieval_query_mode must be one of: claim_only, context_plus_claim")
        self.retrieval_query_mode = retrieval_query_mode

    def build_from_csvs(
        self,
        csv_paths: Union[str, Path, List[Union[str, Path]]],
        dataset_name: str = "radiology_vignettes",
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns dict with:
          - questions_df: one row per question (canonical)
          - expanded_df: one row per (question, option) for evaluation
        """
        paths = self._normalize_paths(csv_paths)
        all_questions: List[Dict[str, Any]] = []
        for p in paths:
            df = pd.read_csv(p)
            qrows = self._parse_any_format(df, source_file=str(Path(p).name))
            all_questions.extend(qrows)

        # canonical question table
        questions_df = pd.DataFrame(all_questions)
        questions_df = self._postprocess_questions_df(questions_df, dataset_name)

        # expanded table: one row per option (this is what your tier eval will likely ingest)
        expanded_df = self._expand_questions_to_option_rows(questions_df, dataset_name)

        return {"questions_df": questions_df, "expanded_df": expanded_df}

    def save_artifacts(
        self,
        out_dir: Union[str, Path],
        artifacts: Dict[str, pd.DataFrame],
        prefix: str = "vignettes169",
    ) -> Dict[str, str]:
        """
        Saves questions_df and expanded_df as parquet + jsonl (useful for auditing).
        Returns file paths.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}
        for key, df in artifacts.items():
            pq = out / f"{prefix}.{key}.parquet"
            jl = out / f"{prefix}.{key}.jsonl"
            df.to_parquet(pq, index=False)
            df.to_json(jl, orient="records", lines=True, force_ascii=False)
            paths[key] = str(pq)
        return paths


    def _parse_any_format(self, df: pd.DataFrame, source_file: str) -> List[Dict[str, Any]]:
        """
        Detects which vignette CSV format is being used and parses accordingly.
        """
        cols = set(df.columns)

        # 4-option format: expect option_1..option_4 and answer_idx (1-4)
        if {"question", "option 1", "option 2", "option 3", "option 4"}.issubset(cols):
            # answer column may vary; handle common variants
            answer_col = None
            for c in ["answer_idx", "answer_index", "answer index", "correct_index", "label", "answer"]:
                if c in cols:
                    answer_col = c
                    break
            if answer_col is None:
                raise ValueError(
                    f"[{source_file}] 4-option format detected but no answer column found. "
                    "Expected one of: answer_idx, answer_index, correct_index, label, answer."
                )
            return self._parse_4option(df, source_file, answer_col)

        # 5-option format: expect option_A..option_E and solution_index (A-E or 1-5)
        if {"question", "option_A", "option_B", "option_C", "option_D", "option_E"}.issubset(cols):
            sol_col = None
            for c in ["solution_index", "answer_index", "answer index", "answer", "label"]:
                if c in cols:
                    sol_col = c
                    break
            if sol_col is None:
                raise ValueError(
                    f"[{source_file}] 5-option format detected but no solution column found. "
                    "Expected one of: solution_index, answer_index, answer, label."
                )
            return self._parse_5option(df, source_file, sol_col)

        raise ValueError(
            f"[{source_file}] Unrecognized vignette CSV format. "
            f"Columns: {sorted(list(cols))[:40]}..."
        )

    def _parse_4option(self, df: pd.DataFrame, source_file: str, answer_col: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, r in df.iterrows():
            question = self._clean_text(r["question"])
            options = [self._clean_text(r[f"option {k}"]) for k in [1, 2, 3, 4]]

            ans_raw = r[answer_col]
            correct_index = self._coerce_answer_to_index(
                ans_raw=ans_raw,
                options=options,
                source_file=source_file,
                row_idx=int(idx),
                expect_num_options=4,
            )

            question_id = self._make_question_id(source_file, idx)
            rows.append(
                dict(
                    source_file=source_file,
                    question_id=question_id,
                    question=question,
                    options=options,
                    correct_index=correct_index,
                    num_options=len(options),
                )
            )
        return rows

    def _parse_5option(self, df: pd.DataFrame, source_file: str, sol_col: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, r in df.iterrows():
            question = self._clean_text(r["question"])
            options = [self._clean_text(r[f"option_{ch}"]) for ch in ["A", "B", "C", "D", "E"]]

            sol_raw = r[sol_col]
            correct_index = self._coerce_answer_to_index(
                ans_raw=sol_raw,
                options=options,
                source_file=source_file,
                row_idx=int(idx),
                expect_num_options=5,
            )

            question_id = self._make_question_id(source_file, idx)
            rows.append(
                dict(
                    source_file=source_file,
                    question_id=question_id,
                    question=question,
                    options=options,
                    correct_index=correct_index,
                    num_options=len(options),
                )
            )
        return rows


    def _postprocess_questions_df(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        if df.empty:
            raise ValueError("No questions loaded.")

        # deterministic ordering
        df = df.reset_index(drop=True)

        # basic validations
        bad = df[(df["correct_index"] < 0) | (df["correct_index"] >= df["num_options"])]
        if len(bad) > 0:
            raise ValueError(f"Found {len(bad)} rows with invalid correct_index.")

        # stable qa_row_id
        df.insert(0, "qa_row_id", range(len(df)))

        # provide stable subject_id analogue (not patient, but preserves your eval schema)
        df["subject_id"] = df["question_id"].apply(lambda s: f"{dataset_name}:{s}")

        # make a deterministic "context" (u_i) and standard q-field
        df["context"] = df["question"]
        df["q"] = self.q_field_prompt

        return df[
            [
                "qa_row_id",
                "source_file",
                "question_id",
                "subject_id",
                "question",
                "context",
                "q",
                "options",
                "correct_index",
                "num_options",
            ]
        ]

    def _expand_questions_to_option_rows(self, questions_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Expands each question into K rows, one per option, with:
          - a: option text
          - claim: deterministic claim
          - y: 1 for correct option else 0  (for multiclass scoring later you can regroup)
        """
        out_rows: List[Dict[str, Any]] = []
        for _, r in questions_df.iterrows():
            options: List[str] = r["options"]
            K = int(r["num_options"])
            correct = int(r["correct_index"])

            for k in range(K):
                option_text = options[k]
                claim = self.claim_template.format(option=option_text)

                if self.retrieval_query_mode == "claim_only":
                    retrieval_query = claim
                else:
                    retrieval_query = f"{r['context'].strip()}\n\nClaim: {claim}"

                out_rows.append(
                    dict(
                        qa_row_id=int(r["qa_row_id"]),          # question-level id (used to regroup)
                        qa_option_row_id=f"{int(r['qa_row_id'])}:{k}",  # unique row id per option
                        source_file=r["source_file"],
                        question_id=r["question_id"],
                        subject_id=r["subject_id"],
                        num_options=K,
                        option_index=k,
                        correct_index=correct,
                        is_correct=int(k == correct),
                        # fields matching your pipeline naming
                        context=r["context"],
                        question=r["q"],
                        answer=option_text,
                        claim=claim,
                        retrieval_query=retrieval_query,
                        # store options for debugging / later aggregation
                        options_json=json.dumps(options, ensure_ascii=False),
                    )
                )

        df = pd.DataFrame(out_rows)

        # deterministic sort: question id then option index
        df = df.sort_values(["qa_row_id", "option_index"], kind="mergesort").reset_index(drop=True)
        return df


    def _normalize_paths(self, csv_paths) -> List[str]:
        if isinstance(csv_paths, (str, Path)):
            return [str(csv_paths)]
        return [str(p) for p in csv_paths]

    def _make_question_id(self, source_file: str, row_idx: int) -> str:
        # stable, deterministic id
        base = Path(source_file).stem
        return f"{base}:{row_idx:04d}"

    def _clean_text(self, x: Any) -> str:
        if pd.isna(x):
            return ""
        s = str(x)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _coerce_answer_to_index(
        self,
        ans_raw: Any,
        options: List[str],
        source_file: str,
        row_idx: int,
        expect_num_options: int,
    ) -> int:
        """
        Accepts:
          - 1..K (int-like)
          - 0..K-1 (int-like)
          - 'A'..'E'
          - exact option text match (fallback)
        Returns 0..K-1.
        """
        K = len(options)
        if K != expect_num_options:
            raise ValueError(f"[{source_file}] row {row_idx}: expected {expect_num_options} options, got {K}")

        # numeric
        try:
            if isinstance(ans_raw, str) and ans_raw.strip() == "":
                raise ValueError("empty answer")
            n = int(ans_raw)
            # allow 1-based
            if 1 <= n <= K:
                return n - 1
            # allow 0-based
            if 0 <= n < K:
                return n
        except Exception:
            pass

        # letter
        if isinstance(ans_raw, str):
            s = ans_raw.strip().upper()
            if s in ["A", "B", "C", "D", "E"][:K]:
                return ["A", "B", "C", "D", "E"].index(s)

        # text match (last resort)
        s_ans = self._clean_text(ans_raw)
        if s_ans:
            for k, opt in enumerate(options):
                if self._clean_text(opt) == s_ans:
                    return k

        raise ValueError(f"[{source_file}] row {row_idx}: cannot parse answer '{ans_raw}' for K={K}")









if __name__ == "__main__":

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_mimicData = params["Preprocessing"]["base_path_mimicData"]
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]


    ''' step 1 '''
    # Paths to your CSV files
    csv_4 = os.path.join(base_path_mimicData, "Vignette_4option.csv")
    csv_5 = os.path.join(base_path_mimicData, "Vignette_5option.csv")

    # Initialize processor
    processor = RadiologyVignetteProcessor(retrieval_query_mode="context_plus_claim")

    # Build canonical artifacts
    artifacts = processor.build_from_csvs(csv_paths=[csv_4, csv_5], dataset_name="radiology169" )

    questions_df = artifacts["questions_df"]  # 1 row per question
    expanded_df = artifacts["expanded_df"]  # 1 row per (question, option)

    print("Questions:", len(questions_df))
    print("Expanded rows:", len(expanded_df))

    out_paths = processor.save_artifacts(
        out_dir= base_path_mimicData,
        artifacts=artifacts,
        prefix="radiology169")

    print(out_paths)