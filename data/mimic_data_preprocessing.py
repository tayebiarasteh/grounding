"""
Created on Jan 28, 2026.
mimic_data_preprocessing.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import re
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple
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
from collections import Counter
import math
import pickle
import heapq


import warnings
warnings.filterwarnings('ignore')





class MimicCaseContextBuilder:
    def __init__(
        self,
        master_csv: str,
        reports_root: str | Path,
        out_path: Optional[str] = None,
        encoding: str = "utf-8",
        tqdm_desc: str = "Building case_context",
        min_lines_fallback: int = 2,
    ) -> None:
        self.master_csv = str(master_csv)
        self.reports_root = Path(reports_root)
        self.out_path = out_path
        self.encoding = encoding
        self.tqdm_desc = tqdm_desc
        self.min_lines_fallback = min_lines_fallback

        # Headers + section markers (whole-line)
        self.header_re = re.compile(
            r"^\s*(FINAL REPORT|EXAMINATION|REASON FOR EXAMINATION|INDICATION|HISTORY|TECHNIQUE|COMPARISON|"
            r"CLINICAL INDICATION|CLINICAL HISTORY|DATE|TIME|DICTATED BY|SIGNED|PROVIDER|ATTENDING|EXAM|STUDY)\s*[:\-]?\s*",
            re.IGNORECASE,
        )
        self.section_hdr_re = re.compile(r"^\s*(FINDINGS|IMPRESSION|CONCLUSION|RESULTS?)\s*:\s*$", re.IGNORECASE)

        # inline section tokens that must be stripped wherever they appear
        self.inline_section_token_re = re.compile(
            r"\b(FINDINGS|IMPRESSION|CONCLUSION|RESULTS?)\s*:\s*",
            re.IGNORECASE,
        )

        # Acquisition/view boilerplate words (line-level heuristic; do not drop clinical content)
        self.acq_boilerplate_re = re.compile(
            r"\b(portable|pa\b|ap\b|lateral|upright|semi[-\s]?erect|frontal|"
            r"views?|radiograph|radiographs|cxr|chest\s+x[-\s]?ray|film)\b",
            re.IGNORECASE,
        )

        # Strong clinical signal tokens
        self.clinical_keep_re = re.compile(
            r"\b(no|without|absent|present|seen|shows?|demonstrates?|consistent|cannot|exclude|likely|possible|"
            r"opacity|opacities|effusion|pneumothorax|pneumomediastinum|edema|atelectasis|consolidation|"
            r"airspace|vascular congestion|cardiomediastinal|cardiac|cardiomegaly|hilar|mediastinum|"
            r"tube|line|catheter|picc|central venous|cvc|endotracheal|ett|nasogastric|ng tube|enteric|dobhoff|"
            r"chest tube|pigtail|swan[-\s]?ganz|pa catheter|sv[co]|cavoatrial|junction|right atrium|carina|"
            r"hemidiaphragm|rib|nodule|mass|calcified lymph nodes)\b",
            re.IGNORECASE,
        )

        # Lines to drop when they are purely admin/communication
        self.drop_tail_re = re.compile(
            r"\b(results were communicated|communicated to|discussed with|called to|by phone|phone|paged|notified)\b",
            re.IGNORECASE,
        )

        self.underscore_re = re.compile(r"_+")

    def _read_report_text(self, report_rel_path: str) -> str:
        p = self.reports_root / report_rel_path
        if not p.exists():
            return ""
        return p.read_text(encoding=self.encoding, errors="ignore")

    def clean_report_to_case_context(self, raw_text: str) -> str:
        """
        inline "IMPRESSION:" / "FINDINGS:" tokens leaking into case_context.

        - Always keep clinically meaningful narrative, even when outside FINDINGS/IMPRESSION
        - Prefer FINDINGS+IMPRESSION, but add extra narrative blocks with clinical signal
        - Drop admin headers, timestamps/signatures, and communication lines
        - Strip inline section tokens anywhere in kept lines (IMPRESSION:, FINDINGS:, etc.)
        """
        if not raw_text:
            return ""

        # Normalize
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = self.underscore_re.sub(" ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return ""

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        # Parse sections (FINDINGS/IMPRESSION/...) for the "prefer" logic
        sections: Dict[str, List[str]] = {}
        current: Optional[str] = None
        for ln in lines:
            m = self.section_hdr_re.match(ln)
            if m:
                current = m.group(1).upper()
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(ln)

        def is_admin_line(s: str) -> bool:
            if self.header_re.match(s):
                return True
            if re.search(r"\b(dictated|signed|attending|electronically)\b", s, re.IGNORECASE):
                return True
            if re.search(r"\b\d{1,2}:\d{2}\b", s):
                return True
            return False

        def strip_inline_section_tokens(s: str) -> str:
            # removes "IMPRESSION:" etc wherever it appears, keeps the content after
            s2 = self.inline_section_token_re.sub("", s)
            return s2.strip()

        def keep_line(s: str) -> bool:
            if not s:
                return False
            if is_admin_line(s):
                return False
            if self.drop_tail_re.search(s):
                return False
            if self.acq_boilerplate_re.search(s) and not self.clinical_keep_re.search(s):
                return False
            return True

        findings_lines = sections.get("FINDINGS", [])
        impression_lines = sections.get("IMPRESSION", [])

        # Base selection: FINDINGS + IMPRESSION if present, else all lines
        if findings_lines or impression_lines:
            base = []
            base.extend(findings_lines)
            base.extend(impression_lines)
        else:
            base = lines

        base_kept = [ln for ln in base if keep_line(ln)]

        # Add clinically-rich non-section lines (fixes REPORT 18 / REPORT 23 patterns)
        base_set = set(base)
        extra = []
        for ln in lines:
            if ln in base_set:
                continue
            if not keep_line(ln):
                continue
            if self.clinical_keep_re.search(ln):
                extra.append(ln)

        keep_set = set(base_kept) | set(extra)
        combined = [ln for ln in lines if ln in keep_set]

        if len(combined) < self.min_lines_fallback:
            combined = [ln for ln in lines if ln.strip() and not is_admin_line(ln)]

        # strip inline section tokens at the very end
        cleaned_lines = []
        for ln in combined:
            ln2 = strip_inline_section_tokens(ln)
            if ln2:
                cleaned_lines.append(ln2)

        out = "\n".join(cleaned_lines)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out

    def build_case_context_table(self) -> pd.DataFrame:
        df = pd.read_csv(self.master_csv)

        required = {"report_rel_path", "subject_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in master CSV: {missing}")

        keep_cols = [c for c in ["subject_id", "report_rel_path", "split"] if c in df.columns]
        uniq = df[keep_cols].drop_duplicates(subset=["report_rel_path"]).reset_index(drop=True)

        case_contexts: List[str] = []
        missing_files = 0

        for rel in tqdm(uniq["report_rel_path"].astype(str), desc=self.tqdm_desc, total=len(uniq)):
            raw = self._read_report_text(rel)
            if not raw:
                case_contexts.append("")
                missing_files += 1
                continue
            case_contexts.append(self.clean_report_to_case_context(raw))

        uniq["case_context"] = case_contexts

        if missing_files:
            print(f"[WARN] Missing report files for {missing_files} rows. case_context set to empty string.")

        if self.out_path:
            outp = Path(self.out_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            uniq.to_csv(outp, index=False)

        return uniq






class MimicFactTableBuilder:
    """
    Step 2: Build one row per report with only report-level metadata and case_context.

    Output schema:
      subject_id, split, report_rel_path, case_context
    """

    def __init__(
        self,
        master_csv: str,
        case_context_csv: str,
        out_path: Optional[str] = None,
        split_col: str = "split",
        subject_col: str = "subject_id",
        report_path_col: str = "report_rel_path",
    ) -> None:
        self.master_csv = str(master_csv)
        self.case_context_csv = str(case_context_csv)
        self.out_path = out_path

        self.split_col = split_col
        self.subject_col = subject_col
        self.report_path_col = report_path_col

    def build_fact_table(self) -> pd.DataFrame:
        master = pd.read_csv(self.master_csv)
        ctx = pd.read_csv(self.case_context_csv)

        required_master = {self.subject_col, self.report_path_col}
        if self.split_col in master.columns:
            required_master.add(self.split_col)
        missing_master = required_master - set(master.columns)
        if missing_master:
            raise ValueError(f"Master CSV missing columns: {missing_master}")

        required_ctx = {self.subject_col, self.report_path_col, "case_context"}
        if self.split_col in ctx.columns:
            required_ctx.add(self.split_col)
        missing_ctx = required_ctx - set(ctx.columns)
        if missing_ctx:
            raise ValueError(f"Case-context CSV missing columns: {missing_ctx}")

        master_keep = [self.subject_col, self.report_path_col]
        if self.split_col in master.columns:
            master_keep.append(self.split_col)

        ctx_keep = [self.subject_col, self.report_path_col, "case_context"]
        if self.split_col in ctx.columns:
            ctx_keep.append(self.split_col)

        master_u = (
            master[master_keep]
            .drop_duplicates(subset=[self.report_path_col])
            .reset_index(drop=True)
        )
        ctx_u = (
            ctx[ctx_keep]
            .drop_duplicates(subset=[self.report_path_col])
            .reset_index(drop=True)
        )

        merged = master_u.merge(
            ctx_u,
            on=[self.subject_col, self.report_path_col],
            how="inner",
            suffixes=("", "_ctx"),
        )

        if self.split_col in merged.columns and f"{self.split_col}_ctx" in merged.columns:
            merged[self.split_col] = merged[f"{self.split_col}_ctx"].fillna(merged[self.split_col])
            merged = merged.drop(columns=[f"{self.split_col}_ctx"])
        elif f"{self.split_col}_ctx" in merged.columns:
            merged[self.split_col] = merged[f"{self.split_col}_ctx"]
            merged = merged.drop(columns=[f"{self.split_col}_ctx"])

        facts = merged[[self.subject_col, self.report_path_col, "case_context"] + ([self.split_col] if self.split_col in merged.columns else [])].copy()

        if self.out_path:
            outp = Path(self.out_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            facts.to_csv(outp, index=False)

        return facts






@dataclass(frozen=True)
class QAItem:
    subject_id: int
    report_rel_path: str
    split: Optional[str]
    case_context: str
    qa_type: str
    question: str
    answer_target: str
    claim_text: str
    supporting_span: str


class MimicQATableBuilder:
    """
    Step 3
    Build strict, closed-form, case-grounded QA from case_context.

    Kept qa_types:
      - normality_statement
      - finding_presence

    Kept finding concepts only:
      - pleural effusion
      - pneumothorax
      - cardiomegaly
      - atelectasis
      - consolidation
    """

    def __init__(
        self,
        facts_long_csv: str,
        out_path: str,
        split_col: str = "split",
        subject_col: str = "subject_id",
        report_path_col: str = "report_rel_path",
        case_context_col: str = "case_context",
        add_normality: bool = True,
        add_finding_presence: bool = True,
        tqdm_desc: str = "Building QA table (Step 3)",
    ) -> None:
        self.facts_long_csv = str(facts_long_csv)
        self.out_path = str(out_path)

        self.split_col = split_col
        self.subject_col = subject_col
        self.report_path_col = report_path_col
        self.case_context_col = case_context_col

        self.add_normality = bool(add_normality)
        self.add_finding_presence = bool(add_finding_presence)
        self.tqdm_desc = tqdm_desc

        self.sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")

        self.admin_re = re.compile(
            r"\b("
            r"history|indication|clinical (history|indication)|reason for exam|reason for examination|"
            r"technique|comparison|exam|examination|portable chest|single ap portable view|pa and lateral|"
            r"results were communicated|discussed with|called to|phone|paged|paging|notified|communicated|"
            r"assess for|evaluate for|eval for|check for"
            r")\b",
            re.IGNORECASE,
        )

        self.bad_start_re = re.compile(r"^\s*(/|//|\?|\:)\s*")
        self.malformed_re = re.compile(r"^[^A-Za-z0-9]{0,3}$")

        self.negation_re = re.compile(
            r"\b("
            r"no|without|absent|negative for|free of|neither|"
            r"no evidence of|no definite|no focal|no sign of|no signs of"
            r")\b",
            re.IGNORECASE,
        )

        self.uncertainty_re = re.compile(
            r"\b("
            r"may|might|possible|possibly|probable|probably|likely|unlikely|"
            r"suggesting|suggestive of|cannot exclude|cannot be excluded|"
            r"difficult to exclude|concerning for|suspicious for|question of|"
            r"could represent|may represent|appears to|appearance of|"
            r"rule out|r/o|rather than"
            r")\b",
            re.IGNORECASE,
        )

        self.resolution_re = re.compile(
            r"\b("
            r"resolved|has resolved|have resolved|essentially resolved|near complete resolution|"
            r"cleared|has cleared|have cleared|improved|markedly improved|significantly improved|"
            r"decreased|markedly decreased|significantly decreased|diminished"
            r")\b",
            re.IGNORECASE,
        )

        self.history_re = re.compile(
            r"\b("
            r"previous|prior|previously|old|remote|history of"
            r")\b",
            re.IGNORECASE,
        )

        self.normality_re = re.compile(
            r"\b("
            r"no acute cardiopulmonary (process|abnormality)|"
            r"no acute intrathoracic process|"
            r"no acute chest abnormality"
            r")\b",
            re.IGNORECASE,
        )

        self.finding_patterns: List[Tuple[str, re.Pattern]] = [
            ("pleural effusion", re.compile(r"\bpleural effusions?\b|\beffusions?\b", re.IGNORECASE)),
            ("pneumothorax", re.compile(r"\bpneumothorax\b|\bpneumothoraces\b", re.IGNORECASE)),
            ("cardiomegaly", re.compile(r"\bcardiomegaly\b|\benlarged cardiac silhouette\b|\bcardiac silhouette is enlarged\b|\benlarged heart\b", re.IGNORECASE)),
            ("atelectasis", re.compile(r"\batelecta(?:sis|tic)\b", re.IGNORECASE)),
            ("consolidation", re.compile(r"\bconsolidation\b", re.IGNORECASE)),
        ]

    def _normalize_text(self, t: str) -> str:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            return ""
        x = str(t)
        x = x.replace("\r\n", "\n").replace("\r", "\n")
        x = re.sub(r"\n+", " ", x)
        x = re.sub(r"\s+", " ", x)
        return x.strip()

    def _stable_hash(self, s: str) -> str:
        return hashlib.md5(self._normalize_text(s).encode("utf-8", errors="ignore")).hexdigest()[:10]

    def _split_sentences(self, text: str) -> List[str]:
        t = self._normalize_text(text)
        if not t:
            return []

        parts = [p.strip() for p in self.sent_split_re.split(t) if p and p.strip()]
        clean: List[str] = []

        for p in parts:
            if len(p) < 8:
                continue
            if self.bad_start_re.search(p):
                continue
            if self.malformed_re.search(p):
                continue
            if self.admin_re.search(p):
                continue
            clean.append(p)

        return clean

    def _canonical_answer(self, state: str) -> str:
        if state == "present":
            return "Yes."
        if state == "absent":
            return "No."
        if state == "uncertain":
            return "Uncertain."
        raise ValueError(state)

    def _canonical_claim(self, concept: str, state: str) -> str:
        if state == "present":
            return f"{concept} is present"
        if state == "absent":
            return f"{concept} is absent"
        if state == "uncertain":
            return f"{concept} is uncertain"
        raise ValueError(state)

    def _question_for_finding(self, concept: str) -> str:
        return f"Is there {concept}?"

    def _is_sentence_usable_for_concept(self, sent: str, concept: str) -> bool:
        s = sent.lower()

        if "eval for" in s or "evaluate for" in s or "assess for" in s or "check for" in s:
            return False

        if self.admin_re.search(sent):
            return False

        if self.history_re.search(sent):
            if not any(tok in s for tok in ["unchanged", "stable", "persistent", "remains", "still"]):
                return False

        if concept == "consolidation" and "no focal consolidation" in s:
            return True

        return True

    def _classify_concept_in_sentence(self, sent: str, pat: re.Pattern) -> Optional[str]:
        m = pat.search(sent)
        if not m:
            return None

        lo = max(0, m.start() - 80)
        hi = min(len(sent), m.end() + 80)
        window = sent[lo:hi]

        has_neg = bool(self.negation_re.search(window))
        has_unc = bool(self.uncertainty_re.search(window))
        has_res = bool(self.resolution_re.search(window))

        if has_neg:
            return "absent"

        if has_unc and has_res:
            return "uncertain"

        if has_unc:
            return "uncertain"

        if has_res:
            return "absent"

        return "present"

    def _aggregate_states(self, states: List[str]) -> Optional[str]:
        if not states:
            return None
        s = set(states)
        if "present" in s:
            return "present"
        if "uncertain" in s:
            return "uncertain"
        if "absent" in s:
            return "absent"
        return None

    def _supporting_span_for_state(self, sents: List[str], pat: re.Pattern, state: str) -> str:
        for s in sents:
            if self._classify_concept_in_sentence(s, pat) == state:
                return s
        return ""

    def _detect_normality(self, sents: List[str]) -> Tuple[Optional[str], str]:
        for s in sents:
            if not self.normality_re.search(s):
                continue
            if self.uncertainty_re.search(s):
                return "uncertain", s
            return "absent", s
        return None, ""

    def build_qa_table(self) -> pd.DataFrame:
        facts = pd.read_csv(self.facts_long_csv)

        required = {self.subject_col, self.report_path_col, self.case_context_col}
        missing = required - set(facts.columns)
        if missing:
            raise ValueError(f"facts_long is missing columns: {missing}")

        if self.split_col not in facts.columns:
            facts[self.split_col] = None

        facts[self.report_path_col] = facts[self.report_path_col].astype(str).str.strip()
        facts[self.subject_col] = pd.to_numeric(facts[self.subject_col], errors="coerce")
        facts = facts[~facts[self.subject_col].isna()].copy()
        facts[self.subject_col] = facts[self.subject_col].astype(int)

        facts = (
            facts[[self.subject_col, self.report_path_col, self.case_context_col, self.split_col]]
            .drop_duplicates(subset=[self.report_path_col])
            .reset_index(drop=True)
        )

        out_rows: List[Dict[str, object]] = []

        for _, row in tqdm(facts.iterrows(), desc=self.tqdm_desc, total=len(facts)):
            report_rel_path = str(row[self.report_path_col]).strip()
            subject_id = int(row[self.subject_col])
            split = row[self.split_col] if self.split_col in facts.columns else None
            case_context = self._normalize_text(row[self.case_context_col])

            if not case_context or case_context.lower() == "nan":
                continue

            sents = self._split_sentences(case_context)
            if not sents:
                continue

            seen_keys: Set[Tuple[str, str, str]] = set()

            if self.add_normality:
                state, span = self._detect_normality(sents)
                if state is not None and span:
                    question = "Is the study described as showing an acute cardiopulmonary abnormality?"
                    key = (report_rel_path, "normality_statement", question)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.subject_col: subject_id,
                                self.split_col: split,
                                self.report_path_col: report_rel_path,
                                "qa_type": "normality_statement",
                                self.case_context_col: case_context,
                                "question": question,
                                "answer_target": self._canonical_answer(state),
                                "claim_text": self._canonical_claim("acute cardiopulmonary abnormality", state),
                                "supporting_span": span,
                            }
                        )

            if self.add_finding_presence:
                for concept, pat in self.finding_patterns:
                    states: List[str] = []
                    usable_sents: List[str] = []

                    for s in sents:
                        if not self._is_sentence_usable_for_concept(s, concept):
                            continue
                        st = self._classify_concept_in_sentence(s, pat)
                        if st is not None:
                            states.append(st)
                            usable_sents.append(s)

                    state = self._aggregate_states(states)
                    if state is None:
                        continue

                    span = self._supporting_span_for_state(usable_sents, pat, state)
                    if not span:
                        continue

                    question = self._question_for_finding(concept)
                    key = (report_rel_path, "finding_presence", question)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    out_rows.append(
                        {
                            self.subject_col: subject_id,
                            self.split_col: split,
                            self.report_path_col: report_rel_path,
                            "qa_type": "finding_presence",
                            self.case_context_col: case_context,
                            "question": question,
                            "answer_target": self._canonical_answer(state),
                            "claim_text": self._canonical_claim(concept, state),
                            "supporting_span": span,
                        }
                    )

        qa = pd.DataFrame(out_rows)

        if self.report_path_col in qa.columns:
            qa[self.report_path_col] = qa[self.report_path_col].astype(str).str.strip()

        outp = Path(self.out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        qa.to_csv(outp, index=False)

        return qa






@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: int
    article_id: str
    article_title: str
    section_title: Optional[str]
    chunk_text: str
    chunk_token_count: int
    article_relpath: str
    chunk_hash: str

class RadiopaediaSnapshotBuilder:
    """
    Step 4A
    Build a frozen, offline evidence snapshot from cleaned Radiopaedia .txt files.

    Output (in out_dir):
      - chunks.csv (always)
      - chunks.parquet
      - manifest.json
    """

    def __init__(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
        snapshot_name: str = "RadiopaediaSnapshot_v1",
        tokenizer_name: str = "sentence-transformers/all-mpnet-base-v2",
        target_tokens: int = 200,
        min_tokens: int = 50,
        write_csv: bool = True,
        write_parquet: bool = True,
        tqdm_desc: str = "Building RadiopaediaSnapshot_v1 chunks",
        encoding: str = "utf-8",
    ) -> None:
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.snapshot_name = str(snapshot_name)

        self.tokenizer_name = str(tokenizer_name)
        self.target_tokens = int(target_tokens)
        self.min_tokens = int(min_tokens)

        self.write_csv = bool(write_csv)
        self.write_parquet = bool(write_parquet)
        self.tqdm_desc = str(tqdm_desc)
        self.encoding = str(encoding)

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        # We are NOT running the encoder model, only tokenizing.
        # So set a huge max length to avoid noisy warnings.
        try:
            self._tokenizer.model_max_length = int(1e9)
        except Exception:
            pass

        self._blankline_re = re.compile(r"\n{3,}")
        self._ws_re = re.compile(r"[ \t]+")


    def _list_txt_files(self) -> List[Path]:
        files = sorted(self.in_dir.rglob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found under: {self.in_dir}")
        return files

    def _read_text(self, p: Path) -> str:
        return p.read_text(encoding=self.encoding, errors="ignore")

    def _first_nonempty_line(self, text: str) -> str:
        for ln in text.splitlines():
            s = ln.strip()
            if s:
                return s
        return ""

    def _normalize_whitespace(self, text: str) -> str:
        if not text:
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not t:
            return ""
        lines = [ln.rstrip() for ln in t.split("\n")]
        out_lines: List[str] = []
        blank_run = 0
        for ln in lines:
            if ln.strip() == "":
                blank_run += 1
                if blank_run <= 1:
                    out_lines.append("")
                continue
            blank_run = 0
            out_lines.append(" ".join(ln.split()))
        out = "\n".join(out_lines).strip()
        out = self._blankline_re.sub("\n\n", out).strip()
        return out

    def _make_article_id(self, p: Path) -> str:
        return p.stem

    def _stable_hash(self, s: str) -> str:
        return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


    def _tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids, clean_up_tokenization_spaces=True).strip()

    def _split_into_token_chunks(self, text: str) -> List[Tuple[str, int]]:
        ids = self._tokenize(text)
        if not ids:
            return []

        out: List[Tuple[str, int]] = []
        for i in range(0, len(ids), self.target_tokens):
            block = ids[i : i + self.target_tokens]
            if len(block) < self.min_tokens:
                continue
            chunk_text = self._decode(block)
            if chunk_text:
                out.append((chunk_text, len(block)))
        return out

    # main build

    def build_snapshot(self) -> pd.DataFrame:
        files = self._list_txt_files()

        rows: List[Dict[str, object]] = []
        chunk_id = 0

        for p in tqdm(files, desc=self.tqdm_desc, total=len(files)):
            raw = self._read_text(p).strip()
            if not raw:
                continue

            title = self._first_nonempty_line(raw)
            body = self._normalize_whitespace(raw)
            if not body:
                continue

            article_id = self._make_article_id(p)
            relpath = str(p.relative_to(self.in_dir))

            section_title = None  # keep None for now

            chunks = self._split_into_token_chunks(body)
            for ctext, ntoks in chunks:
                chash = self._stable_hash(ctext)
                rows.append(
                    asdict(
                        EvidenceChunk(
                            chunk_id=chunk_id,
                            article_id=article_id,
                            article_title=title,
                            section_title=section_title,
                            chunk_text=ctext,
                            chunk_token_count=int(ntoks),
                            article_relpath=relpath,
                            chunk_hash=chash,
                        )
                    )
                )
                chunk_id += 1

        return pd.DataFrame(rows)

    def _parquet_available(self) -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except Exception:
            pass
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False

    def write_outputs(self, df: pd.DataFrame) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        chunks_csv = self.out_dir / "chunks.csv"
        if self.write_csv:
            df.to_csv(chunks_csv, index=False)

        chunks_parquet = self.out_dir / "chunks.parquet"
        wrote_parquet = False
        if self.write_parquet:
            if self._parquet_available():
                df.to_parquet(chunks_parquet, index=False)
                wrote_parquet = True
            else:
                print("[WARN] Parquet requested but pyarrow/fastparquet not installed. Skipping parquet.")

        manifest = {
            "snapshot_name": self.snapshot_name,
            "input_dir": str(self.in_dir.resolve()),
            "num_articles": int(df["article_id"].nunique()) if not df.empty else 0,
            "num_chunks": int(len(df)),
            "tokenizer": self.tokenizer_name,
            "chunking": {
                "method": "fixed_token_blocks",
                "target_tokens": self.target_tokens,
                "stride": self.target_tokens,
                "min_tokens": self.min_tokens,
            },
            "files": {
                "chunks_csv": "chunks.csv" if self.write_csv else None,
                "chunks_parquet": "chunks.parquet" if wrote_parquet else None,
            },
        }

        manifest_path = self.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {manifest_path}")
        if self.write_csv:
            print(f"Wrote: {chunks_csv}  (rows={len(df):,}, articles={manifest['num_articles']:,})")
        if wrote_parquet:
            print(f"Wrote: {chunks_parquet}  (rows={len(df):,}, articles={manifest['num_articles']:,})")

    def run(self) -> pd.DataFrame:
        df = self.build_snapshot()
        self.write_outputs(df)
        return df





@dataclass(frozen=True)
class WeakAlignmentRow:
    qa_row_id: int
    report_rel_path: str
    qa_type: str
    question: str
    claim_text: str
    supporting_span: str
    retrieval_query: str
    alignment_status: str
    retrieved_chunk_ids_json: str
    support_snippets_json: str
    support_weights_json: str
    embed_scores_json: str
    lexical_scores_json: str
    local_match_scores_json: str
    combined_scores_json: str


class RadiopaediaWeakEvidenceAligner:
    """
    Step 4B: Weak claim–evidence alignment

    Input:
      - qa_csv: Step 3 output
      - chunks_parquet: Step 4A output

    Output:
      - out_parquet: per-QA row with top-k retrieved chunk_ids + soft weights

    Foundational changes vs old version:
      - retrieval is case-grounded, not claim-only
      - case-specific anchors are extracted from supporting_span and case_context
      - retrieval query uses claim + compressed anchor set, not raw supporting span
      - reranking rewards evidence only when claim-state and case anchors align in the same snippet
      - contradiction inside chunk is explicitly penalized
      - generic concept-only chunks are penalized
      - abstention is allowed when support is not case-specific enough
      - best supporting snippet per chunk is stored for downstream supervision
      - deterministic tie-breaking by combined score, local score, chunk_id
    """

    def __init__(
        self,
        qa_csv: str | Path,
        chunks_parquet: str | Path,
        out_parquet: str | Path,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        max_query_tokens: int = 256,
        max_doc_tokens: int = 256,
        top_k: int = 20,
        dense_pool_k: int = 100,
        lexical_pool_k: int = 100,
        qa_batch_size: int = 256,
        doc_batch_size: int = 128,
        alpha: float = 0.60,
        temperature: float = 0.20,
        local_match_weight: float = 1.75,
        anchor_match_weight: float = 2.50,
        contradiction_weight: float = 2.25,
        generic_penalty_weight: float = 0.75,
        min_local_match_score: float = 0.75,
        min_anchor_match_score: float = 0.20,
        min_combined_score: float = 0.25,
        max_anchor_terms: int = 8,
        max_anchor_phrases: int = 6,
        allow_empty_when_no_match: bool = True,
        use_question_in_query: bool = False,
        use_qa_type_prefix: bool = False,
        cache_dir: Optional[str | Path] = None,
        faiss_index_name: str = "radiopaedia_flatip.index",
        chunk_embeds_name: str = "radiopaedia_chunk_embeds.npy",
        bm25_cache_name: str = "radiopaedia_bm25_cache.pkl",
        qa_row_id_col: str = "qa_row_id",
        question_col: str = "question",
        claim_col: str = "claim_text",
        supporting_span_col: str = "supporting_span",
        case_context_col: str = "case_context",
        report_path_col: str = "report_rel_path",
        qa_type_col: str = "qa_type",
        chunk_id_col: str = "chunk_id",
        chunk_text_col: str = "chunk_text",
        seed: int = 13,
        device: str = "cuda",
        force_cpu_faiss: bool = True,
        tqdm_desc: str = "Step 4B: weak claim–evidence alignment",
        qa_chunksize_rows: int = 50_000,
        parquet_compression: str = "zstd",
    ) -> None:
        self.qa_csv = Path(qa_csv)
        self.chunks_parquet = Path(chunks_parquet)
        self.out_parquet = Path(out_parquet)

        self.model_name = str(model_name)
        self.max_query_tokens = int(max_query_tokens)
        self.max_doc_tokens = int(max_doc_tokens)

        self.top_k = int(top_k)
        self.dense_pool_k = int(max(dense_pool_k, top_k))
        self.lexical_pool_k = int(max(lexical_pool_k, top_k))
        self.qa_batch_size = int(qa_batch_size)
        self.doc_batch_size = int(doc_batch_size)

        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.local_match_weight = float(local_match_weight)
        self.anchor_match_weight = float(anchor_match_weight)
        self.contradiction_weight = float(contradiction_weight)
        self.generic_penalty_weight = float(generic_penalty_weight)
        self.min_local_match_score = float(min_local_match_score)
        self.min_anchor_match_score = float(min_anchor_match_score)
        self.min_combined_score = float(min_combined_score)

        self.max_anchor_terms = int(max_anchor_terms)
        self.max_anchor_phrases = int(max_anchor_phrases)

        self.allow_empty_when_no_match = bool(allow_empty_when_no_match)
        self.use_question_in_query = bool(use_question_in_query)
        self.use_qa_type_prefix = bool(use_qa_type_prefix)

        self.cache_dir = Path(cache_dir) if cache_dir else self.out_parquet.parent
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_index_path = self.cache_dir / faiss_index_name
        self.chunk_embeds_path = self.cache_dir / chunk_embeds_name
        self.bm25_cache_path = self.cache_dir / bm25_cache_name

        self.qa_row_id_col = str(qa_row_id_col)
        self.question_col = str(question_col)
        self.claim_col = str(claim_col)
        self.supporting_span_col = str(supporting_span_col)
        self.case_context_col = str(case_context_col)
        self.report_path_col = str(report_path_col)
        self.qa_type_col = str(qa_type_col)

        self.chunk_id_col = str(chunk_id_col)
        self.chunk_text_col = str(chunk_text_col)

        self.seed = int(seed)
        self.device = str(device)
        self.force_cpu_faiss = bool(force_cpu_faiss)
        self.tqdm_desc = str(tqdm_desc)
        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_compression = str(parquet_compression)

        self._word_re = re.compile(r"[a-z0-9]+")
        self._sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._pa_schema: Optional[pa.Schema] = None

        self._bm25_state: Optional[Dict[str, object]] = None
        self._chunk_text_map: Optional[Dict[int, str]] = None

        self.negation_re = re.compile(
            r"\b("
            r"no|without|absent|negative for|free of|neither|"
            r"no evidence of|no definite|no focal|no sign of|no signs of"
            r")\b",
            re.IGNORECASE,
        )
        self.uncertainty_re = re.compile(
            r"\b("
            r"may|might|possible|possibly|probable|probably|likely|unlikely|"
            r"suggesting|suggestive of|cannot exclude|cannot be excluded|"
            r"difficult to exclude|concerning for|suspicious for|question of|"
            r"could represent|may represent|appears to|appearance of|"
            r"rule out|r/o|rather than|versus|vs\b"
            r")\b",
            re.IGNORECASE,
        )
        self.resolution_re = re.compile(
            r"\b("
            r"resolved|has resolved|have resolved|essentially resolved|near complete resolution|"
            r"cleared|has cleared|have cleared|improved|markedly improved|significantly improved|"
            r"decreased|markedly decreased|significantly decreased|diminished"
            r")\b",
            re.IGNORECASE,
        )
        self.generic_penalty_re = re.compile(
            r"\b("
            r"summary|reference article|more references needed|peer review|"
            r"these were assessed during peer review|paid speaker|shareholder|"
            r"ongoing|micro - x ltd|philips australia|diagnostic criteria|"
            r"epidemiology|pathology|treatment and prognosis|differential diagnosis|"
            r"classification|staging|risk factors|etiology|aetiology"
            r")\b",
            re.IGNORECASE,
        )
        self.author_noise_re = re.compile(
            r"\b("
            r"jeremy jones|stefan tigges|henry knipe|craig hacking|"
            r"behrang amini|andrew murphy|yuranga weerakkody|yahya baba|"
            r"luke danaher|yi - jin kuok"
            r")\b",
            re.IGNORECASE,
        )
        self.normality_re = re.compile(
            r"\b("
            r"no acute cardiopulmonary (process|abnormality)|"
            r"no acute intrathoracic process|"
            r"no acute chest abnormality"
            r")\b",
            re.IGNORECASE,
        )

        self.finding_patterns: Dict[str, re.Pattern] = {
            "pleural effusion": re.compile(r"\bpleural effusions?\b|\beffusions?\b", re.IGNORECASE),
            "pneumothorax": re.compile(r"\bpneumothorax\b|\bpneumothoraces\b", re.IGNORECASE),
            "cardiomegaly": re.compile(
                r"\bcardiomegaly\b|\benlarged cardiac silhouette\b|\bcardiac silhouette is enlarged\b|\benlarged heart\b",
                re.IGNORECASE,
            ),
            "atelectasis": re.compile(r"\batelecta(?:sis|tic)\b", re.IGNORECASE),
            "consolidation": re.compile(r"\bconsolidation\b", re.IGNORECASE),
            "acute cardiopulmonary abnormality": self.normality_re,
        }

        self._stop_terms = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "without",
            "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
            "there", "here", "from", "into", "at", "by", "as", "it", "its", "than", "then",
            "study", "exam", "examination", "report", "shows", "show", "seen", "present",
            "absent", "uncertain", "question", "claim", "finding", "findings", "impression",
            "acute", "cardiopulmonary", "abnormality", "process", "described", "showing",
            "describes", "evidence", "suggest", "suggests", "suggested", "consistent",
            "compatible", "could", "would", "should", "can", "cannot", "not", "likely",
            "possible", "possibly", "probably", "versus", "vs", "mildly", "moderately",
            "severely", "slightly", "somewhat",
        }

        self._anchor_keep_phrases = [
            "left", "right", "bilateral", "bibasilar", "basilar", "apical", "retrocardiac",
            "perihilar", "upper", "lower", "mid", "base", "bases", "lung bases",
            "right base", "left base", "right upper", "left upper", "right lower", "left lower",
            "small", "trace", "moderate", "large", "massive", "focal", "patchy", "multifocal",
            "diffuse", "persistent", "unchanged", "new", "increased", "decreased", "improved",
            "worsened", "worsening", "adjacent", "low lung volumes", "pulmonary edema",
            "vascular congestion", "airspace opacity", "airspace opacities", "pneumonia",
            "infection", "aspiration", "edema", "opacity", "opacities", "effusion", "effusions",
            "atelectatic", "atelectatic change", "cardiac silhouette", "cardiomediastinal silhouette",
            "hyperinflation", "hyperinflated", "pleural", "left pleural", "right pleural",
        ]

        self._set_determinism()

    def _set_determinism(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

        if self.device != "cuda":
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    def _safe_str(self, s) -> str:
        if s is None:
            return ""
        x = str(s)
        return "" if x.lower() == "nan" else x

    def _normalize_text(self, s: str) -> str:
        x = self._safe_str(s).lower()
        x = re.sub(r"\s+", " ", x).strip()
        return x

    def _tokenize_words(self, s: str) -> List[str]:
        s = self._normalize_text(s)
        if not s:
            return []
        return self._word_re.findall(s)

    def _split_sentences(self, text: str) -> List[str]:
        t = self._safe_str(text)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            return []
        sents = [x.strip() for x in self._sent_split_re.split(t) if x and x.strip()]
        return [x for x in sents if len(x) >= 8]

    def _infer_claim_state(self, claim_text: str) -> str:
        c = self._normalize_text(claim_text)
        if c.endswith(" is present"):
            return "present"
        if c.endswith(" is absent"):
            return "absent"
        if c.endswith(" is uncertain"):
            return "uncertain"
        return "unknown"

    def _infer_claim_concept(self, claim_text: str) -> str:
        c = self._safe_str(claim_text).strip()
        for suffix in [" is present", " is absent", " is uncertain"]:
            if c.endswith(suffix):
                return c[: -len(suffix)].strip()
        return c

    def _get_concept_pattern(self, concept: str) -> re.Pattern:
        if concept in self.finding_patterns:
            return self.finding_patterns[concept]
        return re.compile(r"\b" + re.escape(concept) + r"\b", re.IGNORECASE)

    def _concept_token_set(self, concept: str) -> Set[str]:
        toks = set(self._tokenize_words(concept))
        extra = set()
        if concept == "pleural effusion":
            extra.update({"effusion", "effusions", "pleural"})
        elif concept == "pneumothorax":
            extra.update({"pneumothorax", "pneumothoraces"})
        elif concept == "cardiomegaly":
            extra.update({"cardiomegaly", "cardiac", "silhouette", "enlarged", "heart"})
        elif concept == "atelectasis":
            extra.update({"atelectasis", "atelectatic"})
        elif concept == "consolidation":
            extra.update({"consolidation"})
        elif concept == "acute cardiopulmonary abnormality":
            extra.update({"acute", "cardiopulmonary", "abnormality", "process"})
        return toks | extra

    def _state_token_set(self) -> Set[str]:
        return {
            "present", "absent", "uncertain", "no", "without", "negative", "possible",
            "possibly", "likely", "unlikely", "cannot", "exclude", "excluded",
            "resolved", "improved", "decreased", "increased",
        }

    def _extract_anchor_terms(
        self,
        supporting_span: str,
        case_context: str,
        concept: str,
        qa_type: str,
    ) -> Tuple[List[str], List[str]]:
        concept_toks = self._concept_token_set(concept)
        banned = concept_toks | self._state_token_set() | self._stop_terms

        primary = self._normalize_text(supporting_span)
        secondary = self._normalize_text(case_context)

        term_order: List[str] = []
        seen_terms: Set[str] = set()

        def add_term(tok: str) -> None:
            if not tok:
                return
            if tok in banned:
                return
            if len(tok) <= 2:
                return
            if tok.isdigit():
                return
            if tok in seen_terms:
                return
            seen_terms.add(tok)
            term_order.append(tok)

        for tok in self._tokenize_words(primary):
            add_term(tok)

        if len(term_order) < self.max_anchor_terms:
            pat = self._get_concept_pattern(concept)
            for sent in self._split_sentences(secondary):
                if not pat.search(sent):
                    continue
                for tok in self._tokenize_words(sent):
                    add_term(tok)
                    if len(term_order) >= self.max_anchor_terms:
                        break
                if len(term_order) >= self.max_anchor_terms:
                    break

        phrase_order: List[str] = []
        seen_phrases: Set[str] = set()

        raw_text = f"{primary} || {secondary}"
        for phr in self._anchor_keep_phrases:
            phr_norm = self._normalize_text(phr)
            if not phr_norm:
                continue
            if phr_norm in raw_text and phr_norm not in seen_phrases:
                seen_phrases.add(phr_norm)
                phrase_order.append(phr_norm)
                if len(phrase_order) >= self.max_anchor_phrases:
                    break

        tokens_primary = self._tokenize_words(primary)
        candidate_phrases: List[str] = []
        for n in [3, 2]:
            for i in range(0, max(0, len(tokens_primary) - n + 1)):
                gram = tokens_primary[i:i + n]
                if any(t in banned for t in gram):
                    continue
                if sum(len(t) >= 3 for t in gram) < n:
                    continue
                phrase = " ".join(gram)
                candidate_phrases.append(phrase)

        for phr in candidate_phrases:
            if phr in seen_phrases:
                continue
            seen_phrases.add(phr)
            phrase_order.append(phr)
            if len(phrase_order) >= self.max_anchor_phrases:
                break

        return term_order[: self.max_anchor_terms], phrase_order[: self.max_anchor_phrases]

    def _build_retrieval_query(
        self,
        question: str,
        claim_text: str,
        qa_type: str,
        anchor_terms: List[str],
        anchor_phrases: List[str],
    ) -> str:
        parts: List[str] = []

        c = self._safe_str(claim_text).strip()
        q = self._safe_str(question).strip()
        t = self._safe_str(qa_type).strip()

        if self.use_qa_type_prefix and t:
            parts.append(t)

        if c:
            parts.append(c)

        if anchor_phrases:
            parts.append("ANCHORS: " + " ; ".join(anchor_phrases[: self.max_anchor_phrases]))
        elif anchor_terms:
            parts.append("ANCHORS: " + " ".join(anchor_terms[: self.max_anchor_terms]))

        if self.use_question_in_query and q:
            parts.append(q)

        return "\n\n".join(parts).strip()

    def _get_tokenizer_and_model(self) -> Tuple[AutoTokenizer, AutoModel]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model = AutoModel.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
        model.eval()

        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        self._tokenizer = tok
        self._model = model
        return tok, model

    def _mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _encode_texts(
        self,
        texts: List[str],
        max_len: int,
        batch_size: int,
        pbar: Optional["tqdm"] = None,
    ) -> np.ndarray:
        tok, model = self._get_tokenizer_and_model()

        embs: List[np.ndarray] = []
        n = len(texts)

        for i in range(0, n, batch_size):
            batch = texts[i:i + batch_size]
            batch = [self._normalize_text(x) for x in batch]

            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embs.append(pooled.detach().cpu().numpy().astype(np.float32))

            if pbar is not None:
                pbar.update(len(batch))

        if embs:
            return np.vstack(embs)
        hidden = getattr(model.config, "hidden_size", 768)
        return np.zeros((0, hidden), dtype=np.float32)

    def _load_chunks_table(self) -> pd.DataFrame:
        cols = [self.chunk_id_col, self.chunk_text_col]
        df = pd.read_parquet(self.chunks_parquet, columns=cols)
        df[self.chunk_id_col] = pd.to_numeric(df[self.chunk_id_col], errors="coerce")
        df = df[~df[self.chunk_id_col].isna()].copy()
        df[self.chunk_id_col] = df[self.chunk_id_col].astype(int)
        df[self.chunk_text_col] = df[self.chunk_text_col].astype(str)
        df = df.sort_values(self.chunk_id_col).reset_index(drop=True)
        return df

    def _build_or_load_chunk_embeddings(self, chunks_df: pd.DataFrame) -> np.ndarray:
        if self.chunk_embeds_path.exists():
            emb = np.load(self.chunk_embeds_path)
            if emb.shape[0] != len(chunks_df):
                raise ValueError(
                    f"Loaded embeddings rows ({emb.shape[0]}) != num chunks ({len(chunks_df)}). "
                    f"Delete {self.chunk_embeds_path} and rerun."
                )
            return emb.astype(np.float32)

        texts = chunks_df[self.chunk_text_col].tolist()

        chunk_pbar = tqdm(total=len(texts), desc="Embedding Radiopaedia chunks", unit="chunks", miniters=1)
        try:
            emb = self._encode_texts(
                texts,
                max_len=self.max_doc_tokens,
                batch_size=self.doc_batch_size,
                pbar=chunk_pbar,
            )
        finally:
            chunk_pbar.close()

        np.save(self.chunk_embeds_path, emb)
        return emb

    def _build_or_load_faiss_index(self, chunk_embeds: np.ndarray) -> faiss.Index:
        if self.faiss_index_path.exists():
            return faiss.read_index(str(self.faiss_index_path))

        dim = int(chunk_embeds.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(chunk_embeds)
        faiss.write_index(index, str(self.faiss_index_path))
        return index

    def _build_or_load_bm25_state(self, chunks_df: pd.DataFrame) -> Dict[str, object]:
        if self.bm25_cache_path.exists():
            with open(self.bm25_cache_path, "rb") as f:
                return pickle.load(f)

        chunk_ids = chunks_df[self.chunk_id_col].to_numpy(dtype=np.int64)
        chunk_texts = chunks_df[self.chunk_text_col].tolist()

        postings: Dict[str, List[Tuple[int, int]]] = {}
        dfreq: Dict[str, int] = {}
        doc_len = np.zeros(len(chunk_texts), dtype=np.int32)

        for doc_idx, txt in enumerate(tqdm(chunk_texts, desc="Building BM25 cache", unit="docs", miniters=1)):
            toks = self._tokenize_words(txt)
            doc_len[doc_idx] = len(toks)

            if not toks:
                continue

            tf: Dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1

            for w, c in tf.items():
                postings.setdefault(w, []).append((doc_idx, c))
                dfreq[w] = dfreq.get(w, 0) + 1

        n_docs = len(chunk_texts)
        avgdl = float(doc_len.mean()) if n_docs > 0 else 0.0

        idf: Dict[str, float] = {}
        for w, dfw in dfreq.items():
            idf[w] = float(math.log(1.0 + (n_docs - dfw + 0.5) / (dfw + 0.5)))

        state = {
            "chunk_ids": chunk_ids,
            "postings": postings,
            "idf": idf,
            "doc_len": doc_len,
            "avgdl": avgdl,
            "k1": 1.2,
            "b": 0.75,
        }

        with open(self.bm25_cache_path, "wb") as f:
            pickle.dump(state, f)

        return state

    def _bm25_topk(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        assert self._bm25_state is not None

        toks = self._tokenize_words(query)
        if not toks:
            return [], []

        chunk_ids = self._bm25_state["chunk_ids"]
        postings = self._bm25_state["postings"]
        idf = self._bm25_state["idf"]
        doc_len = self._bm25_state["doc_len"]
        avgdl = float(self._bm25_state["avgdl"])
        k1 = float(self._bm25_state["k1"])
        b = float(self._bm25_state["b"])

        scores: Dict[int, float] = {}

        for t in toks:
            plist = postings.get(t, None)
            if not plist:
                continue

            idft = float(idf.get(t, 0.0))
            for doc_idx, tf in plist:
                dl = float(doc_len[doc_idx])
                denom = tf + k1 * (1.0 - b + b * (dl / (avgdl + 1e-12)))
                s = idft * (tf * (k1 + 1.0)) / (denom + 1e-12)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + float(s)

        if not scores:
            return [], []

        best = heapq.nlargest(top_k, scores.items(), key=lambda x: (x[1], -int(chunk_ids[x[0]])))
        doc_idxs = [int(d) for d, _ in best]
        vals = [float(s) for _, s in best]

        ids = [int(chunk_ids[d]) for d in doc_idxs]
        return ids, vals

    def _standardize_scores(self, xs: List[float]) -> List[float]:
        if not xs:
            return []

        arr = np.asarray(xs, dtype=np.float32)
        mu = float(arr.mean())
        sd = float(arr.std())

        if sd <= 1e-8:
            return [0.0 for _ in xs]

        z = (arr - mu) / sd
        return [float(x) for x in z.tolist()]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x)
        ex = np.exp(x)
        s = float(np.sum(ex))
        if s <= 0:
            return np.ones_like(x, dtype=np.float32) / float(len(x))
        return ex / s

    def _weights_from_scores(self, combined: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            w = np.zeros_like(combined, dtype=np.float32)
            if len(w) > 0:
                w[int(np.argmax(combined))] = 1.0
            return w
        return self._softmax(combined / float(self.temperature)).astype(np.float32)

    def _generic_chunk_penalty(self, text: str) -> float:
        t = self._safe_str(text)
        penalty = 0.0

        if self.generic_penalty_re.search(t):
            penalty += 1.0

        if self.author_noise_re.search(t):
            penalty += 1.0

        toks = self._tokenize_words(t)
        if len(toks) > 350:
            penalty += 0.25

        if t.count(":") >= 8:
            penalty += 0.25

        return float(penalty)

    def _classify_sentence_state(
        self,
        sent: str,
        concept: str,
        qa_type: str,
        pat: re.Pattern,
    ) -> Optional[str]:
        s = self._safe_str(sent)

        if qa_type == "normality_statement":
            if self.normality_re.search(s):
                if self.uncertainty_re.search(s):
                    return "uncertain"
                return "absent"
            return None

        m = pat.search(s)
        if not m:
            return None

        lo = max(0, m.start() - 100)
        hi = min(len(s), m.end() + 100)
        window = s[lo:hi]

        has_neg = bool(self.negation_re.search(window))
        has_unc = bool(self.uncertainty_re.search(window))
        has_res = bool(self.resolution_re.search(window))

        if has_neg:
            return "absent"
        if has_unc and has_res:
            return "uncertain"
        if has_unc:
            return "uncertain"
        if has_res:
            return "absent"
        return "present"

    def _score_sentence_state_match(
        self,
        sent: str,
        concept: str,
        qa_type: str,
        target_state: str,
        pat: re.Pattern,
    ) -> Tuple[float, Optional[str]]:
        state = self._classify_sentence_state(sent, concept, qa_type, pat)
        if state is None:
            return -1e9, None

        score = 0.0

        if state == target_state:
            score += 1.0
        elif target_state == "uncertain" and state in {"present", "absent"}:
            score -= 0.25
        else:
            score -= 1.0

        s_norm = self._normalize_text(sent)
        concept_norm = self._normalize_text(concept)
        if concept_norm and concept_norm in s_norm:
            score += 0.10

        if self.generic_penalty_re.search(sent):
            score -= 0.15

        return float(score), state

    def _score_sentence_anchor_match(
        self,
        sent: str,
        anchor_terms: List[str],
        anchor_phrases: List[str],
    ) -> float:
        if not anchor_terms and not anchor_phrases:
            return 0.0

        s_norm = self._normalize_text(sent)
        sent_toks = set(self._tokenize_words(sent))

        term_hits = 0
        for tok in anchor_terms:
            if tok in sent_toks:
                term_hits += 1

        phrase_hits = 0
        for phr in anchor_phrases:
            if phr in s_norm:
                phrase_hits += 1

        term_score = float(term_hits) / float(len(anchor_terms)) if anchor_terms else 0.0
        phrase_score = float(phrase_hits) / float(len(anchor_phrases)) if anchor_phrases else 0.0

        score = 0.60 * term_score + 1.00 * phrase_score
        return float(score)

    def _best_local_support(
        self,
        chunk_text: str,
        concept: str,
        qa_type: str,
        target_state: str,
        anchor_terms: List[str],
        anchor_phrases: List[str],
    ) -> Tuple[float, float, float, float, str]:
        sents = self._split_sentences(chunk_text)
        if not sents:
            return -1e9, -1e9, -1e9, -1e9, ""

        pat = self._get_concept_pattern(concept)

        best_support_score = -1e9
        best_support_claim = -1e9
        best_support_anchor = -1e9
        best_support_sent = ""

        best_contradiction_score = -1e9

        for sent in sents:
            claim_score, _ = self._score_sentence_state_match(
                sent=sent,
                concept=concept,
                qa_type=qa_type,
                target_state=target_state,
                pat=pat,
            )
            if claim_score <= -1e8:
                continue

            anchor_score = self._score_sentence_anchor_match(
                sent=sent,
                anchor_terms=anchor_terms,
                anchor_phrases=anchor_phrases,
            )

            local_score = float(self.local_match_weight * claim_score + self.anchor_match_weight * anchor_score)

            if claim_score > 0:
                if local_score > best_support_score:
                    best_support_score = float(local_score)
                    best_support_claim = float(claim_score)
                    best_support_anchor = float(anchor_score)
                    best_support_sent = sent
            else:
                contradiction_mag = float((-claim_score) + 0.35 * anchor_score)
                if contradiction_mag > best_contradiction_score:
                    best_contradiction_score = contradiction_mag

        if best_support_score <= -1e8:
            return -1e9, -1e9, -1e9, max(best_contradiction_score, -1e9), ""

        return (
            float(best_support_score),
            float(best_support_claim),
            float(best_support_anchor),
            float(max(best_contradiction_score, 0.0)),
            self._safe_str(best_support_sent),
        )

    def _merge_and_rerank_candidates(
        self,
        dense_ids: List[int],
        dense_scores: List[float],
        lexical_ids: List[int],
        lexical_scores: List[float],
        concept: str,
        qa_type: str,
        target_state: str,
        anchor_terms: List[str],
        anchor_phrases: List[str],
    ) -> Tuple[List[int], List[str], List[float], List[float], List[float], List[float], str]:
        assert self._chunk_text_map is not None

        dense_map = {int(i): float(s) for i, s in zip(dense_ids, dense_scores)}
        lexical_map = {int(i): float(s) for i, s in zip(lexical_ids, lexical_scores)}

        cand_ids = sorted(set(dense_map.keys()) | set(lexical_map.keys()))
        if not cand_ids:
            return [], [], [], [], [], [], "empty_candidate_pool"

        dense_raw = [float(dense_map.get(cid, 0.0)) for cid in cand_ids]
        lexical_raw = [float(lexical_map.get(cid, 0.0)) for cid in cand_ids]

        dense_std = self._standardize_scores(dense_raw)
        lexical_std = self._standardize_scores(lexical_raw)

        rows: List[Tuple[int, str, float, float, float, float, float]] = []

        for cid, ds, ls, dr, lr in zip(cand_ids, dense_std, lexical_std, dense_raw, lexical_raw):
            txt = self._chunk_text_map.get(int(cid), "")

            support_local, claim_score, anchor_score, contradiction_score, best_sent = self._best_local_support(
                chunk_text=txt,
                concept=concept,
                qa_type=qa_type,
                target_state=target_state,
                anchor_terms=anchor_terms,
                anchor_phrases=anchor_phrases,
            )

            penalty = self.generic_penalty_weight * self._generic_chunk_penalty(txt)

            local_net = float(support_local - self.contradiction_weight * max(0.0, contradiction_score))
            combined = float(self.alpha * ds + (1.0 - self.alpha) * ls + local_net - penalty)

            rows.append(
                (
                    int(cid),
                    self._safe_str(best_sent),
                    float(dr),
                    float(lr),
                    float(claim_score + anchor_score),
                    float(local_net),
                    float(combined),
                )
            )

        rows.sort(key=lambda x: (-x[6], -x[5], x[0]))

        if self.allow_empty_when_no_match:
            keep = []
            for r in rows:
                snippet = self._safe_str(r[1])
                semantic_local = float(r[4])
                local_net = float(r[5])
                combined = float(r[6])

                local_ok = semantic_local >= self.min_local_match_score
                anchor_ok = True
                if anchor_terms or anchor_phrases:
                    anchor_ok = semantic_local >= (self.min_local_match_score + self.min_anchor_match_score)

                combined_ok = combined >= self.min_combined_score
                snippet_ok = bool(snippet.strip())

                if local_ok and anchor_ok and combined_ok and snippet_ok:
                    keep.append(r)

            if not keep:
                if anchor_terms or anchor_phrases:
                    return [], [], [], [], [], [], "abstained_no_case_specific_support"
                return [], [], [], [], [], [], "abstained_no_decision_relevant_support"

            rows = keep

        dedup_rows: List[Tuple[int, str, float, float, float, float, float]] = []
        seen_snippets: Set[str] = set()

        for r in rows:
            snippet_key = self._normalize_text(r[1])
            if snippet_key and snippet_key in seen_snippets:
                continue
            if snippet_key:
                seen_snippets.add(snippet_key)
            dedup_rows.append(r)
            if len(dedup_rows) >= self.top_k:
                break

        if not dedup_rows:
            return [], [], [], [], [], [], "abstained_after_dedup"

        final_ids = [int(x[0]) for x in dedup_rows]
        final_snippets = [self._safe_str(x[1]) for x in dedup_rows]
        final_dense = [float(x[2]) for x in dedup_rows]
        final_lex = [float(x[3]) for x in dedup_rows]
        final_local = [float(x[5]) for x in dedup_rows]
        final_combined = [float(x[6]) for x in dedup_rows]

        return final_ids, final_snippets, final_dense, final_lex, final_local, final_combined, "ok"

    def _init_writer(self) -> pq.ParquetWriter:
        self.out_parquet.parent.mkdir(parents=True, exist_ok=True)

        self._pa_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("question", pa.string()),
                ("claim_text", pa.string()),
                ("supporting_span", pa.string()),
                ("retrieval_query", pa.string()),
                ("alignment_status", pa.string()),
                ("retrieved_chunk_ids_json", pa.string()),
                ("support_snippets_json", pa.string()),
                ("support_weights_json", pa.string()),
                ("embed_scores_json", pa.string()),
                ("lexical_scores_json", pa.string()),
                ("local_match_scores_json", pa.string()),
                ("combined_scores_json", pa.string()),
            ]
        )

        return pq.ParquetWriter(
            str(self.out_parquet),
            schema=self._pa_schema,
            compression=self.parquet_compression,
        )

    def _coerce_batch_df_for_parquet(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        batch_df["qa_row_id"] = pd.to_numeric(batch_df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)

        for c in [
            "report_rel_path",
            "qa_type",
            "question",
            "claim_text",
            "supporting_span",
            "retrieval_query",
            "alignment_status",
            "retrieved_chunk_ids_json",
            "support_snippets_json",
            "support_weights_json",
            "embed_scores_json",
            "lexical_scores_json",
            "local_match_scores_json",
            "combined_scores_json",
        ]:
            batch_df[c] = batch_df[c].astype(str)

        return batch_df

    def run(self) -> None:
        if not self.qa_csv.exists():
            raise FileNotFoundError(f"qa_csv not found: {self.qa_csv}")
        if not self.chunks_parquet.exists():
            raise FileNotFoundError(f"chunks_parquet not found: {self.chunks_parquet}")

        chunks_df = self._load_chunks_table()
        chunk_ids = chunks_df[self.chunk_id_col].to_numpy(dtype=np.int64)

        self._chunk_text_map = {
            int(cid): self._safe_str(txt)
            for cid, txt in zip(
                chunks_df[self.chunk_id_col].tolist(),
                chunks_df[self.chunk_text_col].tolist(),
            )
        }

        chunk_embeds = self._build_or_load_chunk_embeddings(chunks_df)
        index = self._build_or_load_faiss_index(chunk_embeds)
        self._bm25_state = self._build_or_load_bm25_state(chunks_df)

        writer = self._init_writer()
        assert self._pa_schema is not None

        total_rows = sum(1 for _ in open(self.qa_csv, "r", encoding="utf-8", errors="ignore")) - 1
        pbar = tqdm(total=max(total_rows, 0), desc=self.tqdm_desc, unit="rows", miniters=1)

        next_qa_row_id = 0

        try:
            for df in pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows):
                required = [self.report_path_col, self.qa_type_col, self.question_col, self.claim_col, self.supporting_span_col]
                for c in required:
                    if c not in df.columns:
                        raise ValueError(f"QA CSV missing required column: {c}")

                if self.case_context_col not in df.columns:
                    df[self.case_context_col] = ""

                if self.qa_row_id_col not in df.columns:
                    df[self.qa_row_id_col] = np.arange(next_qa_row_id, next_qa_row_id + len(df), dtype=np.int64)

                df[self.qa_row_id_col] = pd.to_numeric(df[self.qa_row_id_col], errors="coerce")
                bad_mask = df[self.qa_row_id_col].isna()
                if bad_mask.any():
                    n_bad = int(bad_mask.sum())
                    fill_vals = np.arange(next_qa_row_id, next_qa_row_id + n_bad, dtype=np.int64)
                    df.loc[bad_mask, self.qa_row_id_col] = fill_vals

                df[self.qa_row_id_col] = df[self.qa_row_id_col].astype(np.int64)
                next_qa_row_id = int(df[self.qa_row_id_col].max()) + 1

                for c in [
                    self.report_path_col,
                    self.qa_type_col,
                    self.question_col,
                    self.claim_col,
                    self.supporting_span_col,
                    self.case_context_col,
                ]:
                    df[c] = df[c].astype(str)

                anchor_terms_per_row: List[List[str]] = []
                anchor_phrases_per_row: List[List[str]] = []
                retrieval_queries: List[str] = []

                for q, c, t, s, ctx in zip(
                    df[self.question_col].tolist(),
                    df[self.claim_col].tolist(),
                    df[self.qa_type_col].tolist(),
                    df[self.supporting_span_col].tolist(),
                    df[self.case_context_col].tolist(),
                ):
                    concept = self._infer_claim_concept(c)
                    anchor_terms, anchor_phrases = self._extract_anchor_terms(
                        supporting_span=s,
                        case_context=ctx,
                        concept=concept,
                        qa_type=t,
                    )
                    anchor_terms_per_row.append(anchor_terms)
                    anchor_phrases_per_row.append(anchor_phrases)

                    retrieval_queries.append(
                        self._build_retrieval_query(
                            question=q,
                            claim_text=c,
                            qa_type=t,
                            anchor_terms=anchor_terms,
                            anchor_phrases=anchor_phrases,
                        )
                    )

                q_emb = self._encode_texts(
                    retrieval_queries,
                    max_len=self.max_query_tokens,
                    batch_size=self.qa_batch_size,
                    pbar=pbar,
                )

                D_dense, I_dense = index.search(q_emb, self.dense_pool_k)

                out_rows: List[Dict[str, object]] = []

                for i in range(len(df)):
                    qa_row_id = int(df.iloc[i][self.qa_row_id_col])
                    report_rel_path = self._safe_str(df.iloc[i][self.report_path_col])
                    qa_type = self._safe_str(df.iloc[i][self.qa_type_col])
                    question = self._safe_str(df.iloc[i][self.question_col])
                    claim_text = self._safe_str(df.iloc[i][self.claim_col])
                    supporting_span = self._safe_str(df.iloc[i][self.supporting_span_col])
                    retrieval_query = retrieval_queries[i]
                    query_norm = self._normalize_text(retrieval_query)

                    if not query_norm:
                        out_rows.append(
                            asdict(
                                WeakAlignmentRow(
                                    qa_row_id=qa_row_id,
                                    report_rel_path=report_rel_path,
                                    qa_type=qa_type,
                                    question=question,
                                    claim_text=claim_text,
                                    supporting_span=supporting_span,
                                    retrieval_query=retrieval_query,
                                    alignment_status="empty_query",
                                    retrieved_chunk_ids_json="[]",
                                    support_snippets_json="[]",
                                    support_weights_json="[]",
                                    embed_scores_json="[]",
                                    lexical_scores_json="[]",
                                    local_match_scores_json="[]",
                                    combined_scores_json="[]",
                                )
                            )
                        )
                        continue

                    concept = self._infer_claim_concept(claim_text)
                    target_state = self._infer_claim_state(claim_text)
                    anchor_terms = anchor_terms_per_row[i]
                    anchor_phrases = anchor_phrases_per_row[i]

                    dense_ids: List[int] = []
                    dense_scores: List[float] = []
                    seen_dense: Set[int] = set()

                    for j, sc in zip(I_dense[i].tolist(), D_dense[i].astype(np.float32).tolist()):
                        if j < 0:
                            continue
                        cid = int(chunk_ids[j])
                        if cid in seen_dense:
                            continue
                        seen_dense.add(cid)
                        dense_ids.append(cid)
                        dense_scores.append(float(sc))

                    lexical_ids, lexical_scores = self._bm25_topk(query_norm, self.lexical_pool_k)

                    final_ids, final_snippets, final_dense, final_lex, final_local, final_combined, alignment_status = self._merge_and_rerank_candidates(
                        dense_ids=dense_ids,
                        dense_scores=dense_scores,
                        lexical_ids=lexical_ids,
                        lexical_scores=lexical_scores,
                        concept=concept,
                        qa_type=qa_type,
                        target_state=target_state,
                        anchor_terms=anchor_terms,
                        anchor_phrases=anchor_phrases,
                    )

                    if final_ids:
                        combined_arr = np.asarray(final_combined, dtype=np.float32)
                        weights = self._weights_from_scores(combined_arr)
                    else:
                        weights = np.asarray([], dtype=np.float32)

                    out_rows.append(
                        asdict(
                            WeakAlignmentRow(
                                qa_row_id=qa_row_id,
                                report_rel_path=report_rel_path,
                                qa_type=qa_type,
                                question=question,
                                claim_text=claim_text,
                                supporting_span=supporting_span,
                                retrieval_query=retrieval_query,
                                alignment_status=alignment_status,
                                retrieved_chunk_ids_json=json.dumps([int(x) for x in final_ids]),
                                support_snippets_json=json.dumps([self._safe_str(x) for x in final_snippets]),
                                support_weights_json=json.dumps([float(x) for x in weights.tolist()]),
                                embed_scores_json=json.dumps([float(x) for x in final_dense]),
                                lexical_scores_json=json.dumps([float(x) for x in final_lex]),
                                local_match_scores_json=json.dumps([float(x) for x in final_local]),
                                combined_scores_json=json.dumps([float(x) for x in final_combined]),
                            )
                        )
                    )

                batch_df = pd.DataFrame(out_rows)
                batch_df = self._coerce_batch_df_for_parquet(batch_df)

                table = pa.Table.from_pandas(batch_df, schema=self._pa_schema, preserve_index=False)
                writer.write_table(table)

        finally:
            pbar.close()
            writer.close()

        manifest = {
            "step": "4B",
            "qa_csv": str(self.qa_csv.resolve()),
            "chunks_parquet": str(self.chunks_parquet.resolve()),
            "out_parquet": str(self.out_parquet.resolve()),
            "bootstrap_embedder": self.model_name,
            "retrieval_query_format": "CLAIM_PLUS_CASE_ANCHORS" if not self.use_question_in_query else "CLAIM_PLUS_CASE_ANCHORS_PLUS_QUESTION",
            "params": {
                "max_query_tokens": self.max_query_tokens,
                "max_doc_tokens": self.max_doc_tokens,
                "top_k": self.top_k,
                "dense_pool_k": self.dense_pool_k,
                "lexical_pool_k": self.lexical_pool_k,
                "alpha": self.alpha,
                "temperature": self.temperature,
                "local_match_weight": self.local_match_weight,
                "anchor_match_weight": self.anchor_match_weight,
                "contradiction_weight": self.contradiction_weight,
                "generic_penalty_weight": self.generic_penalty_weight,
                "min_local_match_score": self.min_local_match_score,
                "min_anchor_match_score": self.min_anchor_match_score,
                "min_combined_score": self.min_combined_score,
                "max_anchor_terms": self.max_anchor_terms,
                "max_anchor_phrases": self.max_anchor_phrases,
                "allow_empty_when_no_match": self.allow_empty_when_no_match,
                "use_question_in_query": self.use_question_in_query,
                "use_qa_type_prefix": self.use_qa_type_prefix,
                "seed": self.seed,
                "force_cpu_faiss": self.force_cpu_faiss,
                "qa_chunksize_rows": self.qa_chunksize_rows,
                "qa_batch_size": self.qa_batch_size,
                "doc_batch_size": self.doc_batch_size,
                "device": self.device,
            },
            "cache": {
                "chunk_embeds_npy": str(self.chunk_embeds_path),
                "faiss_index": str(self.faiss_index_path),
                "bm25_cache": str(self.bm25_cache_path),
            },
            "notes": {
                "supporting_span_used_raw_in_query": False,
                "supporting_span_used_for_anchor_extraction": True,
                "case_context_used_for_anchor_backfill": True,
                "lexical_signal": "corpus_bm25",
                "candidate_generation": "dense_pool UNION lexical_pool",
                "chunk_reranking": "dense_lexical_plus_claim_state_match_plus_case_anchor_match_minus_contradiction_minus_generic_penalty",
                "stores_best_snippet_per_chunk": True,
                "abstention_enabled": self.allow_empty_when_no_match,
            },
        }
        (self.out_parquet.parent / "manifest_step4b.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {self.out_parquet.parent / 'manifest_step4b.json'}")
        print(f"Cache: {self.chunk_embeds_path}  |  {self.faiss_index_path}  |  {self.bm25_cache_path}")





@dataclass(frozen=True)
class RetrieverSupervisionRow:
    qa_row_id: int
    subject_id: int
    split: str
    report_rel_path: str
    qa_type: str
    claim_text: str
    question: str
    supporting_span: str
    retrieval_query: str
    pos_chunk_ids_json: str
    pos_weights_json: str


@dataclass(frozen=True)
class VerifierSupervisionRow:
    qa_row_id: int
    subject_id: int
    split: str
    report_rel_path: str
    qa_type: str
    case_context: str
    question: str
    answer_target: str
    claim_text: str
    supporting_span: str
    retrieval_query: str
    alignment_status: str
    support_snippets_json: str
    evidence_chunk_ids_json: str
    evidence_weights_json: str
    soft_support_score: float


class RadiopaediaStep4CSupervisionBuilder:
    """
    Step 4C: Convert Step 3 + Step 4B outputs into final supervision artifacts.

    Foundational changes vs old version:
      - explicitly respects Step 4B abstention status
      - uses local sentence match and support snippet presence, not only combined score
      - retriever positives are mined only from decision-relevant aligned chunks
      - verifier target is sharp and piecewise, so abstained / weak rows stay low
      - verifier evidence is smaller and filtered to what the verifier will actually consume
      - does not apply dataset-level target normalization
      - uses existing split column if it is already patient-consistent
      - otherwise builds patient-wise split map
      - writes parquet in batches
    """

    def __init__(
        self,
        qa_csv: str | Path,
        weak_alignment_parquet: str | Path,
        out_dir: str | Path,
        subject_id_col: str = "subject_id",
        qa_row_id_col: str = "qa_row_id",
        report_path_col: str = "report_rel_path",
        qa_type_col: str = "qa_type",
        claim_col: str = "claim_text",
        case_context_col: str = "case_context",
        question_col: str = "question",
        answer_col: str = "answer_target",
        supporting_span_col: str = "supporting_span",
        retrieval_query_col: str = "retrieval_query",
        split_col: str = "split",
        qa_chunksize_rows: int = 50_000,
        parquet_batch_rows: int = 50_000,
        top_k: Optional[int] = 20,
        min_weight: float = 0.0,
        retriever_pos_top_k: int = 2,
        retriever_pos_min_weight: float = 0.15,
        retriever_pos_min_rel_to_top: float = 0.50,
        retriever_pos_min_combined: float = 1.25,
        retriever_pos_min_local: float = 1.25,
        verifier_top_k: int = 3,
        verifier_min_weight: float = 0.05,
        verifier_min_combined: float = 0.75,
        verifier_min_local: float = 0.75,
        soft_top1_threshold: float = 2.50,
        soft_margin_threshold: float = 0.75,
        soft_local_threshold: float = 1.50,
        abstained_score: float = 0.02,
        weak_score_floor: float = 0.08,
        weak_score_ceiling: float = 0.28,
        medium_score_floor: float = 0.42,
        medium_score_ceiling: float = 0.62,
        strong_score_floor: float = 0.78,
        strong_score_ceiling: float = 0.98,
        split_seed: int = 13,
        frac_train: float = 0.75,
        frac_valid: float = 0.10,
        frac_test: float = 0.15,
        normalize_soft_support: bool = False,
        tqdm_desc: str = "Step 4C: building supervision",
        parquet_compression: str = "zstd",
    ) -> None:
        self.qa_csv = Path(qa_csv)
        self.weak_alignment_parquet = Path(weak_alignment_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.subject_id_col = str(subject_id_col)
        self.qa_row_id_col = str(qa_row_id_col)
        self.report_path_col = str(report_path_col)
        self.qa_type_col = str(qa_type_col)
        self.claim_col = str(claim_col)
        self.case_context_col = str(case_context_col)
        self.question_col = str(question_col)
        self.answer_col = str(answer_col)
        self.supporting_span_col = str(supporting_span_col)
        self.retrieval_query_col = str(retrieval_query_col)
        self.split_col = str(split_col)

        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_batch_rows = int(parquet_batch_rows)

        self.top_k = int(top_k) if top_k is not None else None
        self.min_weight = float(min_weight)

        self.retriever_pos_top_k = int(retriever_pos_top_k)
        self.retriever_pos_min_weight = float(retriever_pos_min_weight)
        self.retriever_pos_min_rel_to_top = float(retriever_pos_min_rel_to_top)
        self.retriever_pos_min_combined = float(retriever_pos_min_combined)
        self.retriever_pos_min_local = float(retriever_pos_min_local)

        self.verifier_top_k = int(verifier_top_k)
        self.verifier_min_weight = float(verifier_min_weight)
        self.verifier_min_combined = float(verifier_min_combined)
        self.verifier_min_local = float(verifier_min_local)

        self.soft_top1_threshold = float(soft_top1_threshold)
        self.soft_margin_threshold = float(soft_margin_threshold)
        self.soft_local_threshold = float(soft_local_threshold)

        self.abstained_score = float(abstained_score)
        self.weak_score_floor = float(weak_score_floor)
        self.weak_score_ceiling = float(weak_score_ceiling)
        self.medium_score_floor = float(medium_score_floor)
        self.medium_score_ceiling = float(medium_score_ceiling)
        self.strong_score_floor = float(strong_score_floor)
        self.strong_score_ceiling = float(strong_score_ceiling)

        self.split_seed = int(split_seed)
        self.frac_train = float(frac_train)
        self.frac_valid = float(frac_valid)
        self.frac_test = float(frac_test)
        if not np.isclose(self.frac_train + self.frac_valid + self.frac_test, 1.0):
            raise ValueError("Split fractions must sum to 1.0.")

        self.normalize_soft_support = bool(normalize_soft_support)
        self.tqdm_desc = str(tqdm_desc)
        self.parquet_compression = str(parquet_compression)

        self.retriever_out = self.out_dir / "retriever_supervision.parquet"
        self.verifier_out = self.out_dir / "verifier_supervision.parquet"
        self.manifest_out = self.out_dir / "manifest_step4c.json"
        self.split_map_out = self.out_dir / "patient_split_map.json"

        self._retr_schema: Optional[pa.Schema] = None
        self._ver_schema: Optional[pa.Schema] = None
        self._patient_split_map: Optional[Dict[int, str]] = None
        self._split_method: str = "unknown"

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(self, s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _clip01(self, x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    def _renormalize_weights(self, w: List[float]) -> List[float]:
        if not w:
            return []

        arr = np.asarray(w, dtype=np.float32)
        arr = np.maximum(arr, 0.0)

        s = float(arr.sum())
        if s <= 0.0:
            arr = np.ones_like(arr, dtype=np.float32) / float(len(arr))
        else:
            arr = arr / s

        return [float(x) for x in arr.tolist()]

    def _piecewise_unit(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0 if x >= hi else 0.0
        return self._clip01((float(x) - float(lo)) / float(hi - lo))

    def _entropy_concentration(self, w: List[float]) -> float:
        if not w:
            return 0.0

        arr = np.asarray(self._renormalize_weights(w), dtype=np.float32)
        if len(arr) == 1:
            return 1.0

        ent = float(-(arr * np.log(np.clip(arr, 1e-12, 1.0))).sum())
        max_ent = float(np.log(len(arr)))
        if max_ent <= 1e-12:
            return 1.0

        return self._clip01(1.0 - (ent / max_ent))

    def _filter_and_truncate(
        self,
        ids: List[int],
        w: List[float],
        combined: List[float],
        local_match: List[float],
        snippets: List[str],
    ) -> Tuple[List[int], List[float], List[float], List[float], List[str]]:
        m = min(len(ids), len(w), len(combined), len(local_match), len(snippets))
        ids = ids[:m]
        w = w[:m]
        combined = combined[:m]
        local_match = local_match[:m]
        snippets = snippets[:m]

        if self.top_k is not None:
            ids = ids[: self.top_k]
            w = w[: self.top_k]
            combined = combined[: self.top_k]
            local_match = local_match[: self.top_k]
            snippets = snippets[: self.top_k]

        if self.min_weight > 0:
            keep = [
                (cid, ww, cc, ll, ss)
                for cid, ww, cc, ll, ss in zip(ids, w, combined, local_match, snippets)
                if float(ww) >= self.min_weight
            ]
            ids = [int(x[0]) for x in keep]
            w = [float(x[1]) for x in keep]
            combined = [float(x[2]) for x in keep]
            local_match = [float(x[3]) for x in keep]
            snippets = [self._safe_str(x[4]) for x in keep]

        w = self._renormalize_weights(w)
        m = min(len(ids), len(w), len(combined), len(local_match), len(snippets))
        return ids[:m], w[:m], combined[:m], local_match[:m], snippets[:m]

    def _build_patient_split_map(self, qa_df: pd.DataFrame) -> Dict[int, str]:
        subjects = pd.to_numeric(qa_df[self.subject_id_col], errors="coerce")
        subjects = subjects[~subjects.isna()].astype(np.int64).unique().tolist()
        subjects = sorted(int(x) for x in subjects)

        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(subjects))
        shuffled = [subjects[i] for i in perm]

        n = len(shuffled)
        n_train = int(round(self.frac_train * n))
        n_valid = int(round(self.frac_valid * n))
        n_test = n - n_train - n_valid

        if n > 0 and n_train == 0:
            n_train = 1
        if n_train + n_valid > n:
            n_valid = max(0, n - n_train)
        n_test = max(0, n - n_train - n_valid)

        train_ids = shuffled[:n_train]
        valid_ids = shuffled[n_train:n_train + n_valid]
        test_ids = shuffled[n_train + n_valid:n_train + n_valid + n_test]

        split_map: Dict[int, str] = {}
        for sid in train_ids:
            split_map[int(sid)] = "train"
        for sid in valid_ids:
            split_map[int(sid)] = "valid"
        for sid in test_ids:
            split_map[int(sid)] = "test"

        self.split_map_out.write_text(
            json.dumps({str(k): v for k, v in split_map.items()}, indent=2),
            encoding="utf-8",
        )
        return split_map

    def _try_use_existing_split_map(self, qa_df: pd.DataFrame) -> Optional[Dict[int, str]]:
        if self.split_col not in qa_df.columns:
            return None

        tmp = qa_df[[self.subject_id_col, self.split_col]].copy()
        tmp[self.subject_id_col] = pd.to_numeric(tmp[self.subject_id_col], errors="coerce")
        tmp = tmp[~tmp[self.subject_id_col].isna()].copy()
        tmp[self.subject_id_col] = tmp[self.subject_id_col].astype(np.int64)
        tmp[self.split_col] = tmp[self.split_col].astype(str).str.strip().str.lower()

        allowed = {"train", "valid", "test"}
        tmp = tmp[tmp[self.split_col].isin(allowed)].copy()
        if len(tmp) == 0:
            return None

        nunique = tmp.groupby(self.subject_id_col)[self.split_col].nunique()
        if int((nunique > 1).sum()) > 0:
            return None

        split_map = (
            tmp.drop_duplicates(subset=[self.subject_id_col])
            .set_index(self.subject_id_col)[self.split_col]
            .to_dict()
        )

        self.split_map_out.write_text(
            json.dumps({str(int(k)): str(v) for k, v in split_map.items()}, indent=2),
            encoding="utf-8",
        )
        return {int(k): str(v) for k, v in split_map.items()}

    def _select_retriever_positives(
        self,
        ids: List[int],
        weights: List[float],
        combined: List[float],
        local_match: List[float],
        snippets: List[str],
        alignment_status: str,
    ) -> Tuple[List[int], List[float]]:
        status = self._safe_str(alignment_status)

        if status != "ok":
            return [], []

        if not ids or not weights or not combined or not local_match or not snippets:
            return [], []

        m = min(len(ids), len(weights), len(combined), len(local_match), len(snippets))
        ids = ids[:m]
        weights = self._renormalize_weights(weights[:m])
        combined = combined[:m]
        local_match = local_match[:m]
        snippets = snippets[:m]

        top_w = float(weights[0]) if weights else 0.0
        out_ids: List[int] = []
        out_w: List[float] = []

        for cid, ww, cc, ll, ss in zip(ids, weights, combined, local_match, snippets):
            rel = float(ww / max(top_w, 1e-12)) if top_w > 0 else 0.0

            if not self._safe_str(ss).strip():
                continue
            if ww < self.retriever_pos_min_weight:
                continue
            if rel < self.retriever_pos_min_rel_to_top:
                continue
            if cc < self.retriever_pos_min_combined:
                continue
            if ll < self.retriever_pos_min_local:
                continue

            out_ids.append(int(cid))
            out_w.append(float(ww))

            if self.retriever_pos_top_k > 0 and len(out_ids) >= self.retriever_pos_top_k:
                break

        if not out_ids:
            return [], []

        out_w = self._renormalize_weights(out_w)
        return out_ids, out_w

    def _select_verifier_evidence(
        self,
        ids: List[int],
        weights: List[float],
        combined: List[float],
        local_match: List[float],
        snippets: List[str],
        alignment_status: str,
    ) -> Tuple[List[int], List[float], List[float], List[float], List[str]]:
        status = self._safe_str(alignment_status)

        if status != "ok":
            return [], [], [], [], []

        if not ids or not weights or not combined or not local_match or not snippets:
            return [], [], [], [], []

        m = min(len(ids), len(weights), len(combined), len(local_match), len(snippets))
        rows = [
            (int(ids[i]), float(weights[i]), float(combined[i]), float(local_match[i]), self._safe_str(snippets[i]))
            for i in range(m)
        ]

        rows = [x for x in rows if x[4].strip() != ""]
        rows = [x for x in rows if x[1] >= self.verifier_min_weight]
        rows = [x for x in rows if x[2] >= self.verifier_min_combined]
        rows = [x for x in rows if x[3] >= self.verifier_min_local]

        if self.verifier_top_k > 0:
            rows = rows[: self.verifier_top_k]

        if not rows:
            return [], [], [], [], []

        out_ids = [int(x[0]) for x in rows]
        out_w = self._renormalize_weights([float(x[1]) for x in rows])
        out_c = [float(x[2]) for x in rows]
        out_l = [float(x[3]) for x in rows]
        out_s = [self._safe_str(x[4]) for x in rows]

        m2 = min(len(out_ids), len(out_w), len(out_c), len(out_l), len(out_s))
        return out_ids[:m2], out_w[:m2], out_c[:m2], out_l[:m2], out_s[:m2]

    def _soft_support_from_alignment(
        self,
        weights: List[float],
        combined_scores: List[float],
        local_match_scores: List[float],
        alignment_status: str,
    ) -> float:
        status = self._safe_str(alignment_status)

        if status != "ok":
            return self._clip01(self.abstained_score)

        if not weights or not combined_scores or not local_match_scores:
            return self._clip01(self.abstained_score)

        m = min(len(weights), len(combined_scores), len(local_match_scores))
        w = np.asarray(self._renormalize_weights(weights[:m]), dtype=np.float32)
        c = np.asarray(combined_scores[:m], dtype=np.float32)
        l = np.asarray(local_match_scores[:m], dtype=np.float32)

        top1_c = float(c[0]) if len(c) > 0 else 0.0
        top2_c = float(c[1]) if len(c) > 1 else float(top1_c)
        margin = float(top1_c - top2_c) if len(c) > 1 else float(top1_c)
        top1_l = float(l[0]) if len(l) > 0 else 0.0
        top1_w = float(w[0]) if len(w) > 0 else 0.0
        conc = self._entropy_concentration(w.tolist())

        s_top = self._piecewise_unit(top1_c, self.verifier_min_combined, self.soft_top1_threshold)
        s_margin = self._piecewise_unit(margin, 0.0, self.soft_margin_threshold)
        s_local = self._piecewise_unit(top1_l, self.verifier_min_local, self.soft_local_threshold)
        s_mass = self._clip01(top1_w)
        s_conc = self._clip01(conc)

        strength = 0.45 * s_top + 0.35 * s_local + 0.20 * s_margin
        confidence = 0.55 * s_mass + 0.45 * s_conc

        if s_top < 0.25 or s_local < 0.25:
            return self._clip01(self.weak_score_floor + (self.weak_score_ceiling - self.weak_score_floor) * (0.70 * strength + 0.30 * confidence))

        if s_top < 0.70 or s_local < 0.70:
            return self._clip01(self.medium_score_floor + (self.medium_score_ceiling - self.medium_score_floor) * (0.65 * strength + 0.35 * confidence))

        return self._clip01(self.strong_score_floor + (self.strong_score_ceiling - self.strong_score_floor) * (0.70 * strength + 0.30 * confidence))

    def _init_retriever_writer(self) -> pq.ParquetWriter:
        self._retr_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("subject_id", pa.int64()),
                ("split", pa.string()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("claim_text", pa.string()),
                ("question", pa.string()),
                ("supporting_span", pa.string()),
                ("retrieval_query", pa.string()),
                ("pos_chunk_ids_json", pa.string()),
                ("pos_weights_json", pa.string()),
            ]
        )
        return pq.ParquetWriter(
            str(self.retriever_out),
            schema=self._retr_schema,
            compression=self.parquet_compression,
        )

    def _init_verifier_writer(self) -> pq.ParquetWriter:
        self._ver_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("subject_id", pa.int64()),
                ("split", pa.string()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("case_context", pa.string()),
                ("question", pa.string()),
                ("answer_target", pa.string()),
                ("claim_text", pa.string()),
                ("supporting_span", pa.string()),
                ("retrieval_query", pa.string()),
                ("alignment_status", pa.string()),
                ("support_snippets_json", pa.string()),
                ("evidence_chunk_ids_json", pa.string()),
                ("evidence_weights_json", pa.string()),
                ("soft_support_score", pa.float32()),
            ]
        )
        return pq.ParquetWriter(
            str(self.verifier_out),
            schema=self._ver_schema,
            compression=self.parquet_compression,
        )

    def _to_table(self, df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
        for c in schema.names:
            if c not in df.columns:
                df[c] = None

        df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").fillna(-1).astype(np.int64)

        string_cols = [
            "split",
            "report_rel_path",
            "qa_type",
            "claim_text",
            "question",
            "supporting_span",
            "retrieval_query",
            "alignment_status",
            "support_snippets_json",
            "pos_chunk_ids_json",
            "pos_weights_json",
            "case_context",
            "answer_target",
            "evidence_chunk_ids_json",
            "evidence_weights_json",
        ]
        for c in string_cols:
            if c in df.columns:
                df[c] = df[c].astype(str)

        if "soft_support_score" in df.columns:
            df["soft_support_score"] = pd.to_numeric(
                df["soft_support_score"], errors="coerce"
            ).fillna(0.0).astype(np.float32)

        return pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    def _flush_buffer(
        self,
        writer: pq.ParquetWriter,
        rows: List[Dict[str, object]],
        schema: pa.Schema,
    ) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        writer.write_table(self._to_table(df, schema))
        rows.clear()

    def run(self) -> None:
        if not self.qa_csv.exists():
            raise FileNotFoundError(f"QA CSV not found: {self.qa_csv}")
        if not self.weak_alignment_parquet.exists():
            raise FileNotFoundError(f"Weak alignment parquet not found: {self.weak_alignment_parquet}")

        qa_df = pd.read_csv(self.qa_csv)
        align_df = pd.read_parquet(self.weak_alignment_parquet)

        required_qa = [
            self.subject_id_col,
            self.report_path_col,
            self.qa_type_col,
            self.claim_col,
            self.case_context_col,
            self.question_col,
            self.answer_col,
        ]
        for c in required_qa:
            if c not in qa_df.columns:
                raise ValueError(f"QA CSV missing required column: {c}")

        if self.qa_row_id_col not in qa_df.columns:
            qa_df[self.qa_row_id_col] = np.arange(len(qa_df), dtype=np.int64)

        required_align = [
            self.qa_row_id_col,
            "alignment_status",
            "retrieved_chunk_ids_json",
            "support_snippets_json",
            "support_weights_json",
            "local_match_scores_json",
            "combined_scores_json",
        ]
        for c in required_align:
            if c not in align_df.columns:
                raise ValueError(f"Weak alignment parquet missing required column: {c}")

        if self.supporting_span_col not in qa_df.columns:
            qa_df[self.supporting_span_col] = ""
        if self.retrieval_query_col not in align_df.columns:
            align_df[self.retrieval_query_col] = ""

        qa_df[self.qa_row_id_col] = pd.to_numeric(qa_df[self.qa_row_id_col], errors="coerce")
        qa_df = qa_df[~qa_df[self.qa_row_id_col].isna()].copy()
        qa_df[self.qa_row_id_col] = qa_df[self.qa_row_id_col].astype(np.int64)

        align_df[self.qa_row_id_col] = pd.to_numeric(align_df[self.qa_row_id_col], errors="coerce")
        align_df = align_df[~align_df[self.qa_row_id_col].isna()].copy()
        align_df[self.qa_row_id_col] = align_df[self.qa_row_id_col].astype(np.int64)

        qa_df[self.subject_id_col] = pd.to_numeric(qa_df[self.subject_id_col], errors="coerce")
        qa_df = qa_df[~qa_df[self.subject_id_col].isna()].copy()
        qa_df[self.subject_id_col] = qa_df[self.subject_id_col].astype(np.int64)

        qa_df[self.report_path_col] = qa_df[self.report_path_col].astype(str)
        qa_df[self.qa_type_col] = qa_df[self.qa_type_col].astype(str)
        qa_df[self.claim_col] = qa_df[self.claim_col].astype(str)
        qa_df[self.case_context_col] = qa_df[self.case_context_col].astype(str)
        qa_df[self.question_col] = qa_df[self.question_col].astype(str)
        qa_df[self.answer_col] = qa_df[self.answer_col].astype(str)
        qa_df[self.supporting_span_col] = qa_df[self.supporting_span_col].astype(str)

        if self.split_col in qa_df.columns:
            qa_df[self.split_col] = qa_df[self.split_col].astype(str)

        for c in [
            "alignment_status",
            "retrieved_chunk_ids_json",
            "support_snippets_json",
            "support_weights_json",
            "local_match_scores_json",
            "combined_scores_json",
            self.retrieval_query_col,
        ]:
            align_df[c] = align_df[c].astype(str)

        existing_split_map = self._try_use_existing_split_map(qa_df)
        if existing_split_map is not None:
            self._patient_split_map = existing_split_map
            self._split_method = "existing_patient_consistent_split"
        else:
            self._patient_split_map = self._build_patient_split_map(qa_df)
            self._split_method = "patient_wise_random"

        merged = qa_df.merge(
            align_df[
                [
                    self.qa_row_id_col,
                    "alignment_status",
                    "retrieved_chunk_ids_json",
                    "support_snippets_json",
                    "support_weights_json",
                    "local_match_scores_json",
                    "combined_scores_json",
                    self.retrieval_query_col,
                ]
            ],
            on=self.qa_row_id_col,
            how="inner",
            validate="one_to_one",
        )

        total_rows = len(merged)
        pbar = tqdm(total=total_rows, desc=self.tqdm_desc, unit="rows", miniters=1)

        retr_writer = self._init_retriever_writer()
        ver_writer = self._init_verifier_writer()
        assert self._retr_schema is not None
        assert self._ver_schema is not None
        assert self._patient_split_map is not None

        retr_buffer: List[Dict[str, object]] = []
        ver_buffer: List[Dict[str, object]] = []

        try:
            for i in range(total_rows):
                row = merged.iloc[i]

                qa_row_id = int(row[self.qa_row_id_col])
                subject_id = int(row[self.subject_id_col])
                split = self._patient_split_map.get(subject_id, "train")

                report_rel_path = self._safe_str(row[self.report_path_col])
                qa_type = self._safe_str(row[self.qa_type_col])
                claim_text = self._safe_str(row[self.claim_col])
                case_context = self._safe_str(row[self.case_context_col])
                question = self._safe_str(row[self.question_col])
                answer_target = self._safe_str(row[self.answer_col])
                supporting_span = self._safe_str(row[self.supporting_span_col])
                retrieval_query = self._safe_str(row.get(self.retrieval_query_col, ""))
                alignment_status = self._safe_str(row.get("alignment_status", ""))

                ids = self._loads_json_list(self._safe_str(row.get("retrieved_chunk_ids_json", "[]")))
                snippets = self._loads_json_list(self._safe_str(row.get("support_snippets_json", "[]")))
                w = self._loads_json_list(self._safe_str(row.get("support_weights_json", "[]")))
                local_match = self._loads_json_list(self._safe_str(row.get("local_match_scores_json", "[]")))
                combined = self._loads_json_list(self._safe_str(row.get("combined_scores_json", "[]")))

                ids = [int(x) for x in ids if str(x).strip() != ""]
                snippets = [self._safe_str(x) for x in snippets]
                w = [float(x) for x in w]
                local_match = [float(x) for x in local_match]
                combined = [float(x) for x in combined]

                ids, w, combined, local_match, snippets = self._filter_and_truncate(
                    ids=ids,
                    w=w,
                    combined=combined,
                    local_match=local_match,
                    snippets=snippets,
                )

                retr_ids, retr_w = self._select_retriever_positives(
                    ids=ids,
                    weights=w,
                    combined=combined,
                    local_match=local_match,
                    snippets=snippets,
                    alignment_status=alignment_status,
                )

                ver_ids, ver_w, ver_combined, ver_local, ver_snippets = self._select_verifier_evidence(
                    ids=ids,
                    weights=w,
                    combined=combined,
                    local_match=local_match,
                    snippets=snippets,
                    alignment_status=alignment_status,
                )

                soft_support_score = self._soft_support_from_alignment(
                    weights=ver_w,
                    combined_scores=ver_combined,
                    local_match_scores=ver_local,
                    alignment_status=alignment_status,
                )

                retr_buffer.append(
                    asdict(
                        RetrieverSupervisionRow(
                            qa_row_id=qa_row_id,
                            subject_id=subject_id,
                            split=split,
                            report_rel_path=report_rel_path,
                            qa_type=qa_type,
                            claim_text=claim_text,
                            question=question,
                            supporting_span=supporting_span,
                            retrieval_query=retrieval_query,
                            pos_chunk_ids_json=json.dumps(retr_ids),
                            pos_weights_json=json.dumps([float(x) for x in retr_w]),
                        )
                    )
                )

                ver_buffer.append(
                    asdict(
                        VerifierSupervisionRow(
                            qa_row_id=qa_row_id,
                            subject_id=subject_id,
                            split=split,
                            report_rel_path=report_rel_path,
                            qa_type=qa_type,
                            case_context=case_context,
                            question=question,
                            answer_target=answer_target,
                            claim_text=claim_text,
                            supporting_span=supporting_span,
                            retrieval_query=retrieval_query,
                            alignment_status=alignment_status,
                            support_snippets_json=json.dumps([self._safe_str(x) for x in ver_snippets]),
                            evidence_chunk_ids_json=json.dumps(ver_ids),
                            evidence_weights_json=json.dumps([float(x) for x in ver_w]),
                            soft_support_score=float(soft_support_score),
                        )
                    )
                )

                if len(retr_buffer) >= self.parquet_batch_rows:
                    self._flush_buffer(retr_writer, retr_buffer, self._retr_schema)

                if len(ver_buffer) >= self.parquet_batch_rows:
                    self._flush_buffer(ver_writer, ver_buffer, self._ver_schema)

                pbar.update(1)

            if self.normalize_soft_support:
                raise ValueError(
                    "normalize_soft_support=True is disabled by design in the new Step 4C. "
                    "Global dataset normalization was one of the foundational problems."
                )

            self._flush_buffer(retr_writer, retr_buffer, self._retr_schema)
            self._flush_buffer(ver_writer, ver_buffer, self._ver_schema)

        finally:
            pbar.close()
            retr_writer.close()
            ver_writer.close()

        manifest = {
            "step": "4C",
            "inputs": {
                "qa_csv": str(self.qa_csv.resolve()),
                "weak_alignment_parquet": str(self.weak_alignment_parquet.resolve()),
            },
            "outputs": {
                "retriever_supervision": str(self.retriever_out.resolve()),
                "verifier_supervision": str(self.verifier_out.resolve()),
                "patient_split_map": str(self.split_map_out.resolve()),
            },
            "split": {
                "method": self._split_method,
                "seed": self.split_seed,
                "fractions": {
                    "train": self.frac_train,
                    "valid": self.frac_valid,
                    "test": self.frac_test,
                },
                "labels": ["train", "valid", "test"],
            },
            "params": {
                "top_k": self.top_k,
                "min_weight": self.min_weight,
                "retriever_pos_top_k": self.retriever_pos_top_k,
                "retriever_pos_min_weight": self.retriever_pos_min_weight,
                "retriever_pos_min_rel_to_top": self.retriever_pos_min_rel_to_top,
                "retriever_pos_min_combined": self.retriever_pos_min_combined,
                "retriever_pos_min_local": self.retriever_pos_min_local,
                "verifier_top_k": self.verifier_top_k,
                "verifier_min_weight": self.verifier_min_weight,
                "verifier_min_combined": self.verifier_min_combined,
                "verifier_min_local": self.verifier_min_local,
                "soft_top1_threshold": self.soft_top1_threshold,
                "soft_margin_threshold": self.soft_margin_threshold,
                "soft_local_threshold": self.soft_local_threshold,
                "abstained_score": self.abstained_score,
                "weak_score_floor": self.weak_score_floor,
                "weak_score_ceiling": self.weak_score_ceiling,
                "medium_score_floor": self.medium_score_floor,
                "medium_score_ceiling": self.medium_score_ceiling,
                "strong_score_floor": self.strong_score_floor,
                "strong_score_ceiling": self.strong_score_ceiling,
                "normalize_soft_support": False,
                "join_key": "qa_row_id",
            },
            "notes": {
                "uses_alignment_status_from_step4b": True,
                "uses_support_snippets_for_positive_mining": True,
                "uses_local_match_scores_for_positive_mining": True,
                "global_target_minmax_normalization": False,
                "retriever_positive_mining": "strict_decision_relevant_only",
                "verifier_evidence_filtered_and_renormalized": True,
                "abstained_rows_stay_low": True,
                "soft_support_scalar": "three-regime piecewise target from top1 combined, top1 local, top1-top2 gap, top1 weight, and concentration",
            },
        }
        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.retriever_out}")
        print(f"Wrote: {self.verifier_out}")
        print(f"Wrote: {self.split_map_out}")
        print(f"Wrote: {self.manifest_out}")







class _RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_v = float("inf")
        self.max_v = float("-inf")

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.astype(np.float64, copy=False)
        self.min_v = min(self.min_v, float(np.min(x)))
        self.max_v = max(self.max_v, float(np.max(x)))

        for v in x:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    def as_dict(self) -> Dict[str, float]:
        if self.n <= 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        var = self.M2 / (self.n - 1) if self.n > 1 else 0.0
        return {
            "min": float(self.min_v),
            "max": float(self.max_v),
            "mean": float(self.mean),
            "std": float(np.sqrt(var)),
        }


class EvidenceTextStore:
    """
    Minimal, deterministic chunk-id -> text lookup.

    Assumptions:
      - evidence parquet has explicit id column + text column
      - ids in Step 4C match this table
    """

    def __init__(
        self,
        evidence_parquet: str | Path,
        *,
        id_col: str = "chunk_id",
        text_col: str = "chunk_text",
        preload: bool = False,
        cache_row_groups: int = 2,
        evidence_id_base: int = 0,
    ) -> None:
        self.evidence_path = Path(evidence_parquet)
        if not self.evidence_path.exists():
            raise FileNotFoundError(f"Evidence parquet not found: {self.evidence_path}")

        self.id_col = str(id_col)
        self.text_col = str(text_col)
        self.preload = bool(preload)
        self.cache_row_groups = int(cache_row_groups)
        self.evidence_id_base = int(evidence_id_base)

        self.pf = pq.ParquetFile(str(self.evidence_path))
        schema_names = set(self.pf.schema.names)
        if self.id_col not in schema_names or self.text_col not in schema_names:
            raise ValueError(
                f"Evidence parquet must contain columns '{self.id_col}' and '{self.text_col}'. "
                f"Found: {sorted(schema_names)}"
            )

        self._map: Optional[Dict[int, str]] = None
        self._rg_cache: Dict[int, Dict[str, List]] = {}
        self._rg_cache_order: List[int] = []

        if self.preload:
            self._preload_all()

    def _preload_all(self) -> None:
        m: Dict[int, str] = {}
        for b in self.pf.iter_batches(batch_size=100_000, columns=[self.id_col, self.text_col]):
            t = pa.Table.from_batches([b])
            ids = t[self.id_col].to_pylist()
            txt = t[self.text_col].to_pylist()
            for i, s in zip(ids, txt):
                try:
                    m[int(i)] = "" if s is None else str(s)
                except Exception:
                    continue
        self._map = m

    def _cache_put(self, rg: int, payload: Dict[str, List]) -> None:
        self._rg_cache[rg] = payload
        self._rg_cache_order.append(rg)
        while len(self._rg_cache_order) > self.cache_row_groups:
            old = self._rg_cache_order.pop(0)
            self._rg_cache.pop(old, None)

    def _get_rg_payload(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached
        t = self.pf.read_row_group(rg, columns=[self.id_col, self.text_col])
        payload = {
            self.id_col: t.column(self.id_col).to_pylist(),
            self.text_col: t.column(self.text_col).to_pylist(),
        }
        self._cache_put(rg, payload)
        return payload

    def get(self, chunk_id: int) -> str:
        chunk_id = int(chunk_id)

        if self._map is not None:
            return self._map.get(chunk_id, "")

        for rg in range(self.pf.num_row_groups):
            payload = self._get_rg_payload(rg)
            ids = payload[self.id_col]
            try:
                for i, cid in enumerate(ids):
                    if int(cid) == chunk_id:
                        s = payload[self.text_col][i]
                        return "" if s is None else str(s)
            except Exception:
                continue
        return ""


def create_fixed_mimic_test_eval_parquet(
    verifier_supervision_parquet: str | Path,
    out_parquet: str | Path,
    *,
    evidence_parquet: str | Path,
    evidence_id_col: str = "chunk_id",
    evidence_text_col: str = "chunk_text",
    evidence_id_base: int = 0,
    preload_evidence_texts: bool = False,
    evidence_join_delim: str = "\n\n---\n\n",
    max_evidence_chars: int = 20_000,
    split_value: str = "test",
    parquet_batch_rows: int = 50_000,
    parquet_compression: str = "zstd",
    seed: int = 13,
    write_manifest_json: bool = True,
) -> Dict[str, object]:
    """
    Freeze the evaluation object for S1-S4 using the new Step 4C output.

    Reads:
      step4c_supervision/verifier_supervision.parquet

    Materializes evidence_text from:
      evidence_chunk_ids_json + evidence_weights_json

    Output columns:
      qa_row_id, subject_id, split, qa_type, case_context, question, answer_target,
      claim_text, supporting_span, retrieval_query,
      evidence_chunk_ids_json, evidence_weights_json, evidence_text,
      soft_support_score,
      report_length_bucket, device_present, uncertainty_present, label_vs_extractive
    """
    _ = np.random.RandomState(seed)

    in_path = Path(verifier_supervision_parquet)
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        raise FileNotFoundError(f"verifier_supervision_parquet not found: {in_path}")

    ev_store = EvidenceTextStore(
        evidence_parquet=evidence_parquet,
        id_col=evidence_id_col,
        text_col=evidence_text_col,
        preload=preload_evidence_texts,
        cache_row_groups=2,
        evidence_id_base=evidence_id_base,
    )

    def _safe_str(x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(s: str) -> List:
        try:
            return json.loads(s)
        except Exception:
            return []

    def _truncate(s: str) -> str:
        s = _safe_str(s)
        if max_evidence_chars > 0 and len(s) > max_evidence_chars:
            return s[:max_evidence_chars]
        return s

    def materialize_evidence(ids_json: str, weights_json: str) -> str:
        ids = _loads_json_list(_safe_str(ids_json))
        weights = _loads_json_list(_safe_str(weights_json))

        pairs = []
        m = min(len(ids), len(weights))
        for i in range(m):
            try:
                pairs.append((int(ids[i]), float(weights[i])))
            except Exception:
                continue

        if len(ids) > m:
            for i in range(m, len(ids)):
                try:
                    pairs.append((int(ids[i]), 0.0))
                except Exception:
                    continue

        pairs.sort(key=lambda x: x[1], reverse=True)
        chunks = [ev_store.get(cid) for cid, _ in pairs]
        return _truncate(evidence_join_delim.join([c for c in chunks if c]))

    device_re = re.compile(
        r"\b("
        r"picc|peripherally inserted central catheter|central venous catheter|cvc|"
        r"endotracheal tube|ett|nasogastric|ng tube|enteric|dobhoff|feeding tube|"
        r"chest tube|pigtail|pleural catheter|"
        r"swan[-\s]?ganz|pa catheter|introducer sheath|"
        r"subclavian catheter|internal jugular|ij\b|"
        r"line tip|catheter tip|tip\b|terminates|terminating|"
        r"svc|s\.v\.c\.|cavoatrial junction|right atrium|carina|cavoatrial"
        r")\b",
        re.IGNORECASE,
    )
    uncertainty_re = re.compile(
        r"\b("
        r"may|might|possible|likely|cannot exclude|cannot be excluded|"
        r"limited|limitation|poorly assessed|not well seen|underlying .* cannot be excluded|"
        r"uncertain"
        r")\b",
        re.IGNORECASE,
    )

    def report_length_bucket(case_context: str) -> str:
        n = len(str(case_context).split())
        if n < 80:
            return "short"
        if n < 160:
            return "medium"
        return "long"

    def label_vs_extractive_flag(qa_type: str) -> str:
        t = str(qa_type).strip().lower()
        if t in {"normality_statement", "finding_presence"}:
            return "closed_form"
        return "other"

    required_in_cols = [
        "qa_row_id",
        "subject_id",
        "split",
        "qa_type",
        "case_context",
        "question",
        "answer_target",
        "claim_text",
        "supporting_span",
        "retrieval_query",
        "evidence_chunk_ids_json",
        "evidence_weights_json",
        "soft_support_score",
    ]

    pf = pq.ParquetFile(str(in_path))
    schema_names = set(pf.schema.names)
    for c in required_in_cols:
        if c not in schema_names:
            raise ValueError(f"Missing required column in Step 4C verifier supervision parquet: {c}")

    out_schema = pa.schema(
        [
            ("qa_row_id", pa.int64()),
            ("subject_id", pa.int64()),
            ("split", pa.string()),
            ("qa_type", pa.string()),
            ("case_context", pa.string()),
            ("question", pa.string()),
            ("answer_target", pa.string()),
            ("claim_text", pa.string()),
            ("supporting_span", pa.string()),
            ("retrieval_query", pa.string()),
            ("evidence_chunk_ids_json", pa.string()),
            ("evidence_weights_json", pa.string()),
            ("evidence_text", pa.string()),
            ("soft_support_score", pa.float32()),
            ("report_length_bucket", pa.string()),
            ("device_present", pa.int8()),
            ("uncertainty_present", pa.int8()),
            ("label_vs_extractive", pa.string()),
        ]
    )

    writer = pq.ParquetWriter(str(out_path), schema=out_schema, compression=parquet_compression)

    total_rows = pf.metadata.num_rows if pf.metadata is not None else 0
    pbar = tqdm(total=total_rows, desc="Freezing mimic_test_eval.parquet", unit="rows", miniters=1)

    subj_set: set[int] = set()
    qa_type_ctr = Counter()
    bucket_ctr = Counter()
    device_ctr = Counter()
    uncert_ctr = Counter()
    label_ctr = Counter()
    target_stats = _RunningStats()

    written = 0
    kept = 0

    try:
        for batch in pf.iter_batches(batch_size=parquet_batch_rows, columns=required_in_cols):
            df = batch.to_pandas()

            df["split"] = df["split"].astype(str)
            mask = df["split"] == split_value
            if not mask.any():
                pbar.update(len(df))
                written += len(df)
                continue

            d = df.loc[mask].copy()

            d["qa_row_id"] = pd.to_numeric(d["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
            d["subject_id"] = pd.to_numeric(d["subject_id"], errors="coerce").fillna(-1).astype(np.int64)
            d["soft_support_score"] = pd.to_numeric(d["soft_support_score"], errors="coerce").fillna(0.0).astype(np.float32)

            for c in [
                "split",
                "qa_type",
                "case_context",
                "question",
                "answer_target",
                "claim_text",
                "supporting_span",
                "retrieval_query",
                "evidence_chunk_ids_json",
                "evidence_weights_json",
            ]:
                d[c] = d[c].astype(str)

            d["evidence_text"] = [
                materialize_evidence(ids_json, weights_json)
                for ids_json, weights_json in zip(
                    d["evidence_chunk_ids_json"].tolist(),
                    d["evidence_weights_json"].tolist(),
                )
            ]

            d["report_length_bucket"] = d["case_context"].map(report_length_bucket).astype(str)
            d["device_present"] = d["case_context"].map(lambda x: 1 if device_re.search(str(x)) else 0).astype(np.int8)
            d["uncertainty_present"] = (
                d["case_context"].map(lambda x: 1 if uncertainty_re.search(str(x)) else 0).astype(np.int8)
            )
            d["label_vs_extractive"] = d["qa_type"].map(label_vs_extractive_flag).astype(str)

            subj_set.update(map(int, d["subject_id"].tolist()))
            qa_type_ctr.update(d["qa_type"].tolist())
            bucket_ctr.update(d["report_length_bucket"].tolist())
            device_ctr.update(map(int, d["device_present"].tolist()))
            uncert_ctr.update(map(int, d["uncertainty_present"].tolist()))
            label_ctr.update(d["label_vs_extractive"].tolist())
            target_stats.update(d["soft_support_score"].to_numpy())

            d = d[
                [
                    "qa_row_id",
                    "subject_id",
                    "split",
                    "qa_type",
                    "case_context",
                    "question",
                    "answer_target",
                    "claim_text",
                    "supporting_span",
                    "retrieval_query",
                    "evidence_chunk_ids_json",
                    "evidence_weights_json",
                    "evidence_text",
                    "soft_support_score",
                    "report_length_bucket",
                    "device_present",
                    "uncertainty_present",
                    "label_vs_extractive",
                ]
            ]

            table = pa.Table.from_pandas(d, schema=out_schema, preserve_index=False)
            writer.write_table(table)

            kept += len(d)
            pbar.update(len(df))
            written += len(df)

    finally:
        pbar.close()
        writer.close()

    summary = {
        "basic_info": {
            "rows": int(kept),
            "unique_subjects": int(len(subj_set)),
            "qa_types": int(len(qa_type_ctr)),
        },
        "qa_type_distribution": dict(qa_type_ctr),
        "report_length_bucket": dict(bucket_ctr),
        "device_present": dict(device_ctr),
        "uncertainty_present": dict(uncert_ctr),
        "label_vs_extractive": dict(label_ctr),
        "target_statistics_soft_support_score": target_stats.as_dict(),
    }

    print("==== BASIC INFO ====")
    print("Rows:", summary["basic_info"]["rows"])
    print("Unique subjects:", summary["basic_info"]["unique_subjects"])
    print("QA types:", summary["basic_info"]["qa_types"])
    print()

    print("==== QA TYPE DISTRIBUTION ====")
    print(pd.Series(summary["qa_type_distribution"]).sort_values(ascending=False))
    print()

    print("==== REPORT LENGTH BUCKET ====")
    print(pd.Series(summary["report_length_bucket"]).sort_values(ascending=False))
    print()

    print("==== DEVICE PRESENT ====")
    print(pd.Series(summary["device_present"]).sort_index())
    print()

    print("==== UNCERTAINTY PRESENT ====")
    print(pd.Series(summary["uncertainty_present"]).sort_index())
    print()

    print("==== LABEL vs EXTRACTIVE ====")
    print(pd.Series(summary["label_vs_extractive"]).sort_values(ascending=False))
    print()

    print("==== TARGET STATISTICS (soft_support_score) ====")
    ts = summary["target_statistics_soft_support_score"]
    print("Min:", ts["min"])
    print("Max:", ts["max"])
    print("Mean:", ts["mean"])
    print("Std:", ts["std"])

    manifest = {
        "phase": "1",
        "input": str(in_path.resolve()),
        "output": str(out_path.resolve()),
        "filter": {"split_value": split_value},
        "evidence_materialization": {
            "evidence_parquet": str(Path(evidence_parquet).resolve()),
            "evidence_id_col": evidence_id_col,
            "evidence_text_col": evidence_text_col,
            "evidence_id_base": int(evidence_id_base),
            "preload_evidence_texts": bool(preload_evidence_texts),
            "evidence_join_delim": evidence_join_delim,
            "max_evidence_chars": int(max_evidence_chars),
            "ordering": "descending evidence_weights_json",
        },
        "params": {
            "parquet_batch_rows": int(parquet_batch_rows),
            "parquet_compression": str(parquet_compression),
            "seed": int(seed),
            "derived_tags": [
                "report_length_bucket",
                "device_present",
                "uncertainty_present",
                "label_vs_extractive",
            ],
        },
        "stats": {
            "input_rows_seen": int(written),
            "rows_written": int(kept),
        },
        "prints_saved": summary,
    }

    if write_manifest_json:
        manifest_path = out_path.parent / "manifest_phase1_test_eval.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}  (rows={kept:,})")
    if write_manifest_json:
        print(f"Wrote: {out_path.parent / 'manifest_phase1_test_eval.json'}")

    return manifest










if __name__ == "__main__":

    with open(global_config_path, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    base_path_mimicData = params['Preprocessing']['base_path_mimicData']
    base_path_evidenceData = params['Preprocessing']['base_path_evidenceData']





    ''' step 1 '''
    # builder = MimicCaseContextBuilder(
    #     master_csv=os.path.join(base_path_mimicData, "mimic_master_list_forReports.csv"),
    #     reports_root=base_path_mimicData,
    #     out_path="mimic_case_contexts.csv",
    # )
    # out_df = builder.build_case_context_table()
    # print(out_df[["report_rel_path", "case_context"]].head(3))
    # print("Rows:", len(out_df))

    ''' step 2 '''
    # builder = MimicFactTableBuilder(
    #     master_csv=os.path.join(base_path_mimicData, "mimic_master_list_forReports.csv"),
    #     case_context_csv=os.path.join(base_path_mimicData, "mimic_case_contexts.csv"),
    #     out_path=os.path.join(base_path_mimicData, "mimic_facts_long.csv"),
    # )
    # reports_df = builder.build_fact_table()
    # print(reports_df.head(10))
    # print("n_reports:", len(reports_df))
    # print(reports_df.columns.tolist())


    # ''' step 3 '''
    # builder = MimicQATableBuilder(
    #     facts_long_csv=os.path.join(base_path_mimicData, "mimic_facts_long.csv"),
    #     out_path=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
    #     add_normality=True,
    #     add_finding_presence=True,
    # )
    # qa_df = builder.build_qa_table()
    # print("QA rows:", len(qa_df))
    # print(qa_df["qa_type"].value_counts())
    # print(qa_df[["qa_type", "question", "answer_target", "claim_text", "supporting_span"]].head(10))


    # ''' step 4a: RadiopaediaSnapshot_v1 (chunking only, no embeddings) '''
    # builder = RadiopaediaSnapshotBuilder(
    #     in_dir=os.path.join(base_path_evidenceData, "articles_clean_txt"),  # folder with ~17k *.txt
    #     out_dir=base_path_evidenceData,
    #     snapshot_name=params['Preprocessing']['snapshot_name'],
    #     tokenizer_name=params['Preprocessing']['tokenizer_name'],
    #     target_tokens=200,
    #     min_tokens=50,
    #     write_csv=False,  # parquet is preferred
    # )
    # chunks_df = builder.run()
    # print(chunks_df.head(10))
    # print("n_chunks:", len(chunks_df))
    # print("n_articles:", chunks_df["article_id"].nunique())
    # print(chunks_df[["chunk_id", "article_id", "article_title", "chunk_token_count"]].head(10))



    ''' step 4b '''
    # builder = RadiopaediaWeakEvidenceAligner(
    #     qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
    #     chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
    #     out_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step4b.parquet"),
    #     model_name=params["Preprocessing"]["tokenizer_name"],
    #     max_query_tokens=128,
    #     max_doc_tokens=256,
    #     top_k=20,
    #     dense_pool_k=100,
    #     lexical_pool_k=100,
    #     qa_batch_size=256,
    #     doc_batch_size=128,
    #     alpha=0.60,
    #     temperature=0.20,
    #     local_match_weight=1.75,
    #     anchor_match_weight=2.50,
    #     contradiction_weight=2.25,
    #     generic_penalty_weight=0.75,
    #     min_local_match_score=0.75,
    #     min_anchor_match_score=0.20,
    #     min_combined_score=0.25,
    #     max_anchor_terms=8,
    #     max_anchor_phrases=6,
    #     allow_empty_when_no_match=True,
    #     use_question_in_query=False,
    #     use_qa_type_prefix=False,
    #     seed=13,
    #     cache_dir=os.path.join(base_path_evidenceData, "cache_step4b"),
    #     device="cuda",
    #     force_cpu_faiss=True,
    # )
    # builder.run()




    ''' step 4c '''
    builder = RadiopaediaStep4CSupervisionBuilder(
        qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
        weak_alignment_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step4b.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "step4c_supervision"),
        subject_id_col="subject_id",
        qa_row_id_col="qa_row_id",
        top_k=20,
        min_weight=0.0,
        retriever_pos_top_k=2,
        retriever_pos_min_weight=0.15,
        retriever_pos_min_rel_to_top=0.50,
        retriever_pos_min_combined=1.25,
        retriever_pos_min_local=1.25,
        verifier_top_k=3,
        verifier_min_weight=0.05,
        verifier_min_combined=0.75,
        verifier_min_local=0.75,
        soft_top1_threshold=2.50,
        soft_margin_threshold=0.75,
        soft_local_threshold=1.50,
        abstained_score=0.02,
        weak_score_floor=0.08,
        weak_score_ceiling=0.28,
        medium_score_floor=0.42,
        medium_score_ceiling=0.62,
        strong_score_floor=0.78,
        strong_score_ceiling=0.98,
        qa_chunksize_rows=50_000,
        parquet_batch_rows=50_000,
        split_seed=13,
        frac_train=0.75,
        frac_valid=0.10,
        frac_test=0.15,
        normalize_soft_support=False,
    )
    builder.run()

    # ''' Creating the final mimic eval parquet from Step 4C '''
    # verifier_supervision_parquet = os.path.join(base_path_evidenceData, "step4c_supervision", "verifier_supervision.parquet")
    # radiopaedia_chunks_parquet = os.path.join(base_path_evidenceData, "chunks.parquet")
    # out_eval_parquet = os.path.join(base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet")
    #
    # create_fixed_mimic_test_eval_parquet(verifier_supervision_parquet=verifier_supervision_parquet, out_parquet=out_eval_parquet,
    #     evidence_parquet=radiopaedia_chunks_parquet, evidence_id_col="chunk_id", evidence_text_col="chunk_text",
    #     evidence_id_base=0,  # change to 1 if ids start from 1
    #     preload_evidence_texts=False,
    #     split_value="test",
    #     parquet_batch_rows=50_000,
    #     parquet_compression="zstd",
    #     seed=13, write_manifest_json=True)

