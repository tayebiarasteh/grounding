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
    retrieved_chunk_ids_json: str
    support_weights_json: str
    embed_scores_json: str
    lexical_scores_json: str
    combined_scores_json: str


class RadiopaediaWeakEvidenceAligner:
    """
    Step 4B: Weak claim–evidence alignment

    Input:
      - qa_csv: Step 3 output
      - chunks_parquet: Step 4A output

    Output:
      - out_parquet: per-QA row with top-k retrieved chunk_ids + soft weights
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
        qa_batch_size: int = 256,
        doc_batch_size: int = 128,
        alpha: float = 0.85,
        temperature: float = 0.07,
        cache_dir: Optional[str | Path] = None,
        faiss_index_name: str = "radiopaedia_flatip.index",
        chunk_embeds_name: str = "radiopaedia_chunk_embeds.npy",
        qa_row_id_col: str = "qa_row_id",
        question_col: str = "question",
        claim_col: str = "claim_text",
        supporting_span_col: str = "supporting_span",
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
        self.qa_batch_size = int(qa_batch_size)
        self.doc_batch_size = int(doc_batch_size)

        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.cache_dir = Path(cache_dir) if cache_dir else self.out_parquet.parent
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_index_path = self.cache_dir / faiss_index_name
        self.chunk_embeds_path = self.cache_dir / chunk_embeds_name

        self.qa_row_id_col = str(qa_row_id_col)
        self.question_col = str(question_col)
        self.claim_col = str(claim_col)
        self.supporting_span_col = str(supporting_span_col)
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
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._pa_schema: Optional[pa.Schema] = None

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

    def _lexical_overlap(self, query: str, doc: str) -> float:
        q = self._tokenize_words(query)
        if not q:
            return 0.0
        qset = set(q)
        dset = set(self._tokenize_words(doc))
        if not dset:
            return 0.0
        return float(len(qset.intersection(dset))) / float(len(qset))

    def _build_retrieval_query(self, question: str, claim_text: str, supporting_span: str) -> str:
        parts: List[str] = []
        q = self._safe_str(question).strip()
        c = self._safe_str(claim_text).strip()
        s = self._safe_str(supporting_span).strip()

        if q:
            parts.extend(["QUESTION:", q])
        if c:
            parts.extend(["CLAIM:", c])
        if s:
            parts.extend(["REPORT_EVIDENCE_HINT:", s])

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
                ("retrieved_chunk_ids_json", pa.string()),
                ("support_weights_json", pa.string()),
                ("embed_scores_json", pa.string()),
                ("lexical_scores_json", pa.string()),
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
            "retrieved_chunk_ids_json",
            "support_weights_json",
            "embed_scores_json",
            "lexical_scores_json",
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
        chunk_texts = chunks_df[self.chunk_text_col].tolist()

        chunk_embeds = self._build_or_load_chunk_embeddings(chunks_df)
        index = self._build_or_load_faiss_index(chunk_embeds)

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

                for c in [self.report_path_col, self.qa_type_col, self.question_col, self.claim_col, self.supporting_span_col]:
                    df[c] = df[c].astype(str)

                retrieval_queries = [
                    self._build_retrieval_query(q, c, s)
                    for q, c, s in zip(
                        df[self.question_col].tolist(),
                        df[self.claim_col].tolist(),
                        df[self.supporting_span_col].tolist(),
                    )
                ]

                q_emb = self._encode_texts(
                    retrieval_queries,
                    max_len=self.max_query_tokens,
                    batch_size=self.qa_batch_size,
                    pbar=pbar,
                )

                D, I = index.search(q_emb, self.top_k)

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
                                    retrieved_chunk_ids_json="[]",
                                    support_weights_json="[]",
                                    embed_scores_json="[]",
                                    lexical_scores_json="[]",
                                    combined_scores_json="[]",
                                )
                            )
                        )
                        continue

                    faiss_idxs = I[i].tolist()
                    embed_scores = D[i].astype(np.float32).tolist()

                    retrieved_ids: List[int] = []
                    retrieved_texts: List[str] = []
                    valid_embed_scores: List[float] = []

                    for j, sc in zip(faiss_idxs, embed_scores):
                        if j < 0:
                            continue
                        retrieved_ids.append(int(chunk_ids[j]))
                        retrieved_texts.append(chunk_texts[j])
                        valid_embed_scores.append(float(sc))

                    lex_scores = [float(self._lexical_overlap(query_norm, t)) for t in retrieved_texts]

                    e = np.array(valid_embed_scores, dtype=np.float32)
                    l = np.array(lex_scores, dtype=np.float32)
                    combined = self.alpha * e + (1.0 - self.alpha) * l
                    w = self._weights_from_scores(combined)

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
                                retrieved_chunk_ids_json=json.dumps(retrieved_ids),
                                support_weights_json=json.dumps([float(x) for x in w.tolist()]),
                                embed_scores_json=json.dumps([float(x) for x in e.tolist()]),
                                lexical_scores_json=json.dumps([float(x) for x in l.tolist()]),
                                combined_scores_json=json.dumps([float(x) for x in combined.tolist()]),
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
            "retrieval_query_format": "QUESTION + CLAIM + REPORT_EVIDENCE_HINT",
            "params": {
                "max_query_tokens": self.max_query_tokens,
                "max_doc_tokens": self.max_doc_tokens,
                "top_k": self.top_k,
                "alpha": self.alpha,
                "temperature": self.temperature,
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
            },
        }
        (self.out_parquet.parent / "manifest_step4b.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {self.out_parquet.parent / 'manifest_step4b.json'}")
        print(f"Cache: {self.chunk_embeds_path}  |  {self.faiss_index_path}")






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
    evidence_chunk_ids_json: str
    evidence_weights_json: str
    soft_support_score: float


class RadiopaediaStep4CSupervisionBuilder:
    """
    Step 4C: Convert Step 3 + Step 4B outputs into final supervision artifacts.

    Fixes versus old version:
      - merges by qa_row_id instead of relying on row order
      - actually builds and writes patient-wise split map if missing
      - keeps question / supporting_span / retrieval_query in supervision
      - replaces invalid soft_support_score = max(combined_scores)
      - uses weighted positive retrieval score as weak support target
      - applies optional global min-max normalization to soft support
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
        qa_chunksize_rows: int = 50_000,
        parquet_batch_rows: int = 50_000,
        top_k: Optional[int] = None,
        min_weight: float = 0.0,
        split_seed: int = 13,
        frac_train: float = 0.75,
        frac_valid: float = 0.10,
        frac_test: float = 0.15,
        normalize_soft_support: bool = True,
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

        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_batch_rows = int(parquet_batch_rows)

        self.top_k = int(top_k) if top_k is not None else None
        self.min_weight = float(min_weight)

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

    def _filter_and_truncate(self, ids: List[int], w: List[float], combined: List[float]) -> Tuple[List[int], List[float], List[float]]:
        m = min(len(ids), len(w), len(combined))
        ids = ids[:m]
        w = w[:m]
        combined = combined[:m]

        if self.top_k is not None:
            ids = ids[: self.top_k]
            w = w[: self.top_k]
            combined = combined[: self.top_k]

        if self.min_weight > 0:
            keep = [(i, ww, cc) for i, ww, cc in zip(ids, w, combined) if float(ww) >= self.min_weight]
            ids = [int(i) for i, _, _ in keep]
            w = [float(ww) for _, ww, _ in keep]
            combined = [float(cc) for _, _, cc in keep]

        return ids, w, combined

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

    def _weighted_support_from_scores(self, weights: List[float], combined_scores: List[float]) -> float:
        if not weights or not combined_scores:
            return 0.0

        m = min(len(weights), len(combined_scores))
        w = np.asarray(weights[:m], dtype=np.float32)
        c = np.asarray(combined_scores[:m], dtype=np.float32)

        c = np.maximum(c, 0.0)

        s = float(w.sum())
        if s > 0:
            w = w / s
        elif len(w) > 0:
            w = np.ones_like(w, dtype=np.float32) / float(len(w))
        else:
            return 0.0

        return float(np.sum(w * c))

    def _normalize_soft_support_scores(self, xs: List[float]) -> List[float]:
        if not xs:
            return xs
        arr = np.asarray(xs, dtype=np.float32)
        lo = float(arr.min())
        hi = float(arr.max())
        if hi <= lo + 1e-12:
            return [0.5 for _ in xs]
        arr = (arr - lo) / (hi - lo)
        return [float(x) for x in arr.tolist()]

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
            "retrieved_chunk_ids_json",
            "support_weights_json",
            "combined_scores_json",
        ]
        for c in required_align:
            if c not in align_df.columns:
                raise ValueError(f"Weak alignment parquet missing required column: {c}")

        if self.supporting_span_col not in qa_df.columns:
            qa_df[self.supporting_span_col] = ""
        if self.retrieval_query_col not in align_df.columns:
            align_df[self.retrieval_query_col] = ""
        if self.question_col not in align_df.columns:
            align_df[self.question_col] = ""
        if self.supporting_span_col not in align_df.columns:
            align_df[self.supporting_span_col] = ""

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

        align_df["retrieved_chunk_ids_json"] = align_df["retrieved_chunk_ids_json"].astype(str)
        align_df["support_weights_json"] = align_df["support_weights_json"].astype(str)
        align_df["combined_scores_json"] = align_df["combined_scores_json"].astype(str)
        align_df[self.retrieval_query_col] = align_df[self.retrieval_query_col].astype(str)

        self._patient_split_map = self._build_patient_split_map(qa_df)

        merged = qa_df.merge(
            align_df[
                [
                    self.qa_row_id_col,
                    "retrieved_chunk_ids_json",
                    "support_weights_json",
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
        raw_soft_scores: List[float] = []

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

                ids = self._loads_json_list(self._safe_str(row.get("retrieved_chunk_ids_json", "[]")))
                w = self._loads_json_list(self._safe_str(row.get("support_weights_json", "[]")))
                combined = self._loads_json_list(self._safe_str(row.get("combined_scores_json", "[]")))

                ids = [int(x) for x in ids if str(x).strip() != ""]
                w = [float(x) for x in w]
                combined = [float(x) for x in combined]

                ids, w, combined = self._filter_and_truncate(ids, w, combined)

                if len(ids) != len(w):
                    m = min(len(ids), len(w))
                    ids = ids[:m]
                    w = w[:m]
                if len(ids) != len(combined):
                    m = min(len(ids), len(combined))
                    ids = ids[:m]
                    w = w[:m]
                    combined = combined[:m]

                soft_support_score = self._weighted_support_from_scores(w, combined)
                raw_soft_scores.append(float(soft_support_score))

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
                            pos_chunk_ids_json=json.dumps(ids),
                            pos_weights_json=json.dumps([float(x) for x in w]),
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
                            evidence_chunk_ids_json=json.dumps(ids),
                            evidence_weights_json=json.dumps([float(x) for x in w]),
                            soft_support_score=float(soft_support_score),
                        )
                    )
                )

                pbar.update(1)

            if self.normalize_soft_support and ver_buffer:
                norm_scores = self._normalize_soft_support_scores(raw_soft_scores)
                for j, s in enumerate(norm_scores):
                    ver_buffer[j]["soft_support_score"] = float(s)

            retr_df = pd.DataFrame(retr_buffer)
            ver_df = pd.DataFrame(ver_buffer)

            retr_writer.write_table(self._to_table(retr_df, self._retr_schema))
            ver_writer.write_table(self._to_table(ver_df, self._ver_schema))

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
                "method": "patient-wise",
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
                "normalize_soft_support": self.normalize_soft_support,
                "soft_support_scalar": "weighted_sum(max(combined_scores,0), normalized_by_support_weights); optional dataset min-max normalization",
                "join_key": "qa_row_id",
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





    # ''' step 4b '''
    # builder = RadiopaediaWeakEvidenceAligner(
    #     qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
    #     chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
    #     out_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step4b.parquet"),
    #     model_name=params["Preprocessing"]["tokenizer_name"],
    #     max_query_tokens=128,
    #     max_doc_tokens=256,
    #     top_k=20,
    #     alpha=0.85,
    #     temperature=0.07,
    #     seed=13,
    #     cache_dir=os.path.join(base_path_evidenceData, "cache_step4b"),
    #     device="cuda",
    #     force_cpu_faiss=True,
    # )
    # builder.run()
    #
    #
    #
    #
    # ''' step 4c '''
    # builder = RadiopaediaStep4CSupervisionBuilder(
    #     qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
    #     weak_alignment_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step4b.parquet"),
    #     out_dir=os.path.join(base_path_evidenceData, "step4c_supervision"),
    #     subject_id_col="subject_id",
    #     qa_row_id_col="qa_row_id",
    #     top_k=20,
    #     min_weight=0.0,
    #     qa_chunksize_rows=50_000,
    #     parquet_batch_rows=50_000,
    #     split_seed=13,
    #     frac_train=0.75,
    #     frac_valid=0.10,
    #     frac_test=0.15,
    #     normalize_soft_support=True,
    # )
    # builder.run()



    ''' Creating the final mimic eval parquet from Step 4C '''
    verifier_supervision_parquet = os.path.join(base_path_evidenceData, "step4c_supervision", "verifier_supervision.parquet")
    radiopaedia_chunks_parquet = os.path.join(base_path_evidenceData, "chunks.parquet")
    out_eval_parquet = os.path.join(base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet")

    create_fixed_mimic_test_eval_parquet(verifier_supervision_parquet=verifier_supervision_parquet, out_parquet=out_eval_parquet,
        evidence_parquet=radiopaedia_chunks_parquet, evidence_id_col="chunk_id", evidence_text_col="chunk_text",
        evidence_id_base=0,  # change to 1 if ids start from 1
        preload_evidence_texts=False,
        split_value="test",
        parquet_batch_rows=50_000,
        parquet_compression="zstd",
        seed=13, write_manifest_json=True)

