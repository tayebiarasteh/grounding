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
    Step 2: Build per-report 'facts' from the 14 CheXpert labels.

    Output schema (one row per fact):
      subject_id, split, report_rel_path, case_context, label_name, label_value
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

        # CheXpert label semantics:
        # 0 = negative mention, 1 = positive mention, 2 = no mention, 3 = uncertain/ambiguous mention
        self.valid_fact_values = {0, 1, 3}

    def _detect_label_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Finds label columns as all integer-like columns excluding known meta columns.
        If your master CSV has explicit label column names, you can replace this with a fixed list.
        """
        meta = {self.subject_col, self.report_path_col, self.split_col}
        # keep numeric columns that aren't meta
        label_cols = []
        for c in df.columns:
            if c in meta:
                continue
            # label cols are usually small ints 0/1/2/3
            if pd.api.types.is_numeric_dtype(df[c]):
                label_cols.append(c)
        if not label_cols:
            raise ValueError("No numeric label columns found. Provide explicit label column names.")
        return label_cols

    def _aggregate_labels_per_report(self, df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
        """
        Aggregate labels per report_rel_path across multiple image-rows.

        Priority rule:
          if any 1 -> 1
          elif any 3 -> 3
          elif any 0 -> 0
          else -> 2
        """
        # Make sure label cols are numeric (coerce safely)
        work = df[[self.report_path_col, self.subject_col] + (
            [self.split_col] if self.split_col in df.columns else []) + label_cols].copy()
        for c in label_cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        # priority encoding: 2 lowest, then 0, then 3, then 1 highest
        # map label -> priority score
        pri_map = {2: 0, 0: 1, 3: 2, 1: 3}

        # convert each label col to priority scores (NaN stays NaN -> treat as 0)
        for c in label_cols:
            work[c] = work[c].map(pri_map).fillna(0).astype("int8")

        # group by report only (subject_id is constant per report in MIMIC)
        g = work.groupby(self.report_path_col, as_index=False)

        out = g[[self.subject_col]].first()
        if self.split_col in work.columns:
            out[self.split_col] = g[self.split_col].first()[self.split_col]

        # max priority per label per report
        max_pri = g[label_cols].max()

        # map priority back to label
        inv_pri = {0: 2, 1: 0, 2: 3, 3: 1}
        for c in label_cols:
            out[c] = max_pri[c].map(inv_pri).astype("int8")

        return out



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
        missing_ctx = required_ctx - set(ctx.columns)
        if missing_ctx:
            raise ValueError(f"Case-context CSV missing columns: {missing_ctx}")

        label_cols = self._detect_label_columns(master)

        # aggregate labels to one row per report
        per_report = self._aggregate_labels_per_report(master, label_cols)

        # merge with case_context
        merged = per_report.merge(
            ctx[[self.subject_col, self.report_path_col, "case_context"] + ([self.split_col] if self.split_col in ctx.columns else [])],
            on=[self.subject_col, self.report_path_col],
            how="inner",
            suffixes=("", "_ctx"),
        )

        # choose split from ctx if present, else from master
        if self.split_col in merged.columns and f"{self.split_col}_ctx" in merged.columns:
            merged[self.split_col] = merged[f"{self.split_col}_ctx"].fillna(merged[self.split_col])
            merged = merged.drop(columns=[f"{self.split_col}_ctx"])

        # make long "facts" table
        long_rows = []
        for _, row in merged.iterrows():
            for lab in label_cols:
                v = int(row[lab])
                if v in self.valid_fact_values:
                    long_rows.append({
                        self.subject_col: row[self.subject_col],
                        self.split_col: row[self.split_col] if self.split_col in merged.columns else None,
                        self.report_path_col: row[self.report_path_col],
                        "case_context": row["case_context"],
                        "label_name": lab,
                        "label_value": v,
                    })

        facts = pd.DataFrame(long_rows)

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
    Build case-conditioned QA from facts_long.

    Guarantees:
      1) Sentence splitting is newline-safe: join wrapped lines before splitting.
      2) Extractive spans must be substrings of case_context after normalization, else skipped.
      3) No label leakage: label_name/label_value ONLY for:
         - label_presence
         - multi_label_conjunction
         - device_presence (label_name=device, label_value=1)
         All other qa_types: label_name=None, label_value=None.
      4) Dedup per-report using robust keys.
         For device_tip_location we allow multiple rows by including a stable span hash in the key.

    Hard fixes:
      A) device_presence is negation/removal-aware (prevents false Yes).
      B) compositional_finding excludes negated/normality sentences (prevents "abnormal finding" mismatch).
    """

    def __init__(
        self,
        facts_long_csv: str,
        out_path: str,
        split_col: str = "split",
        subject_col: str = "subject_id",
        report_path_col: str = "report_rel_path",
        case_context_col: str = "case_context",
        label_name_col: str = "label_name",
        label_value_col: str = "label_value",
        max_label_qas_per_report: int = 14,
        max_conjunction_qas_per_report: int = 3,
        add_key_findings: bool = True,
        add_normality: bool = True,
        add_multi_label_conjunction: bool = True,
        add_device_presence: bool = True,
        add_device_tip_location: bool = True,
        add_devices_locations: bool = False,
        add_temporal_change: bool = True,
        add_compositional_findings: bool = True,
        add_limitation_uncertainty: bool = True,
        max_sentences_key_findings: int = 3,
        max_sentences_temporal: int = 2,
        max_sentences_compositional: int = 2,
        max_sentences_uncertainty: int = 2,
        max_device_presence_qas_per_report: int = 8,
        max_device_tip_qas_per_report: int = 6,
        tqdm_desc: str = "Building QA table (Step 3)",
        skip_labels: Optional[Set[str]] = None,
        include_negative_labels: bool = True,
        # NEW knobs (safe defaults)
        device_negation_window_chars: int = 45,
    ) -> None:
        self.facts_long_csv = str(facts_long_csv)
        self.out_path = str(out_path)

        self.split_col = split_col
        self.subject_col = subject_col
        self.report_path_col = report_path_col
        self.case_context_col = case_context_col
        self.label_name_col = label_name_col
        self.label_value_col = label_value_col

        self.max_label_qas_per_report = max_label_qas_per_report
        self.max_conjunction_qas_per_report = max_conjunction_qas_per_report

        self.add_key_findings = add_key_findings
        self.add_normality = add_normality
        self.add_multi_label_conjunction = add_multi_label_conjunction
        self.add_device_presence = add_device_presence
        self.add_device_tip_location = add_device_tip_location
        self.add_devices_locations = add_devices_locations
        self.add_temporal_change = add_temporal_change
        self.add_compositional_findings = add_compositional_findings
        self.add_limitation_uncertainty = add_limitation_uncertainty

        self.max_sentences_key_findings = max_sentences_key_findings
        self.max_sentences_temporal = max_sentences_temporal
        self.max_sentences_compositional = max_sentences_compositional
        self.max_sentences_uncertainty = max_sentences_uncertainty
        self.max_device_presence_qas_per_report = max_device_presence_qas_per_report
        self.max_device_tip_qas_per_report = max_device_tip_qas_per_report

        self.tqdm_desc = tqdm_desc

        base_skip = skip_labels or {"no_finding"}
        self.skip_labels = {self._normalize_label_token(s) for s in base_skip}
        self.include_negative_labels = include_negative_labels

        self.device_negation_window_chars = int(device_negation_window_chars)

        # Sentence splitting only on punctuation boundaries (no newline splitting).
        self.sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")

        # Drop admin/history/communication sentences
        self.drop_sentence_re = re.compile(
            r"\b("
            r"history|indication|clinical (history|indication)|reason for examination|technique|comparison|"
            r"results were communicated|discussed with|called to|phone|paged|paging|notified|communicated"
            r")\b",
            re.IGNORECASE,
        )

        # Core clinical signal scoring
        self.clinical_signal_re = re.compile(
            r"\b(no|without|absent|present|seen|demonstrates?|shows?|consistent|cannot|exclude|likely|possible|"
            r"opacity|opacities|effusion|pneumothorax|pneumomediastinum|edema|atelectasis|consolidation|"
            r"airspace|cardiomediastinal|cardiac|cardiomegaly|hilar|mediastinum|nodule|mass|infiltrate|"
            r"vascular|congestion|volume loss|collapse|pleural)\b",
            re.IGNORECASE,
        )
        self.negation_re = re.compile(r"\b(no|without|absent|neither|never)\b", re.IGNORECASE)

        self.removal_re = re.compile(
            r"\b("
            r"removed|removal of|interval removal of|discontinued|extubated|repositioned out of|withdrawn|"
            r"not seen|no longer seen|not visualized|not present"
            r")\b",
            re.IGNORECASE,
        )

        # Normality endpoints (report-level)
        self.normality_re = re.compile(
            r"\b("
            r"no acute cardiopulmonary (process|abnormality)|"
            r"no acute intrathoracic process|"
            r"no acute chest abnormality|"
            r"lungs are clear|"
            r"clear lungs|"
            r"no focal consolidation|"
            r"no (pleural )?effusion( is seen)?|"
            r"no pneumothorax( is seen)?"
            r")\b",
            re.IGNORECASE,
        )
        self.hedge_re = re.compile(
            r"\b(may|might|cannot exclude|cannot be excluded|limited|poorly assessed)\b",
            re.IGNORECASE,
        )

        # Devices
        self.device_sentence_re = re.compile(
            r"\b("
            r"picc|peripherally inserted central catheter|central venous catheter|cvc|"
            r"endotracheal tube|ett|nasogastric|ng tube|enteric|dobhoff|feeding tube|"
            r"chest tube|pigtail|pleural catheter|"
            r"swan[-\s]?ganz|pa catheter|introducer sheath|"
            r"subclavian catheter|internal jugular|ij\b|"
            r"line tip|catheter tip|tip\b|terminates|terminating|"
            r"svc|s\.v\.c\.|cavoatrial junction|right atrium|carina"
            r")\b",
            re.IGNORECASE,
        )

        self.device_presence_vocab: List[Tuple[str, re.Pattern]] = [
            ("picc", re.compile(r"\bpicc\b|\bperipherally inserted central catheter\b", re.IGNORECASE)),
            ("central venous catheter", re.compile(r"\bcentral venous catheter\b|\bcvc\b", re.IGNORECASE)),
            ("endotracheal tube", re.compile(r"\bendotracheal tube\b|\bett\b", re.IGNORECASE)),
            ("nasogastric tube", re.compile(r"\bnasogastric\b|\bng tube\b", re.IGNORECASE)),
            ("enteric tube", re.compile(r"\benteric catheter\b|\bdobhoff\b|\bfeeding tube\b", re.IGNORECASE)),
            ("chest tube", re.compile(r"\bchest tube\b", re.IGNORECASE)),
            ("pigtail catheter", re.compile(r"\bpigtail\b", re.IGNORECASE)),
            ("pleural catheter", re.compile(r"\bpleural catheter\b", re.IGNORECASE)),
            ("swan-ganz catheter", re.compile(r"\bswan[-\s]?ganz\b|\bpa catheter\b", re.IGNORECASE)),
            ("subclavian catheter", re.compile(r"\bsubclavian\b.*\bcatheter\b", re.IGNORECASE)),
            ("internal jugular catheter", re.compile(r"\binternal jugular\b|\bij\b.*\bcatheter\b", re.IGNORECASE)),
        ]

        self.tip_loc_re = re.compile(
            r"\b("
            r"tip is at|tip (lies|projects|terminates|terminating) (in|at)|"
            r"terminates (in|at)|terminating (in|at)|at the level of|"
            r"cavoatrial junction|mid (to[-\s]?lower )?portion of the svc|mid svc|"
            r"right brachiocephalic vein|junction with svc|right atrium|carina|cavoatrial"
            r")\b",
            re.IGNORECASE,
        )

        # Temporal change
        self.temporal_re = re.compile(
            r"\b("
            r"interval|compared to|as compared to|since prior|prior study|previous study|"
            r"improved|worsened|increased|decreased|unchanged|stable|new|resolved|persistent|"
            r"removal|removed|placement|placed|re[-\s]?accumulation|progression"
            r")\b",
            re.IGNORECASE,
        )

        # Compositional findings
        self.pathology_token_re = re.compile(
            r"\b("
            r"effusion|pneumothorax|edema|atelectasis|consolidation|opacity|opacities|airspace|"
            r"nodule|mass|infiltrate|vascular congestion|volume loss|collapse|pneumomediastinum"
            r")\b",
            re.IGNORECASE,
        )
        self.modifier_token_re = re.compile(
            r"\b("
            r"small|mild|moderate|large|trace|tiny|minimal|"
            r"right|left|bilateral|diffuse|focal|perihilar|retrocardiac|lower lobe|upper lobe|mid lung|"
            r"layering|with|consistent with|likely|may represent|cannot exclude"
            r")\b",
            re.IGNORECASE,
        )

        # Limitation / uncertainty
        self.uncertainty_re = re.compile(
            r"\b("
            r"within the limitations of|limitation(s)?|limited evaluation|poorly assessed|"
            r"cannot exclude|cannot be excluded|may represent|might reflect|possible|likely|"
            r"cannot be evaluated|not well seen|underlying .* cannot be excluded"
            r")\b",
            re.IGNORECASE,
        )

    # normalization utilities

    def _normalize_label_token(self, s: str) -> str:
        s = str(s).strip().lower().replace("_", " ")
        s = re.sub(r"\s+", " ", s)
        return s

    def _normalize_text(self, t: str) -> str:
        """
        Used for substring checks and stable sentence ops.
        Join wrapped lines before sentence splitting.
        """
        if t is None or (isinstance(t, float) and pd.isna(t)):
            return ""
        x = str(t)
        x = x.replace("\r\n", "\n").replace("\r", "\n")
        x = re.sub(r"\n+", " ", x)
        x = re.sub(r"\s+", " ", x)
        return x.strip()

    def _is_extractive(self, span: str, case_context: str) -> bool:
        span_n = self._normalize_text(span)
        ctx_n = self._normalize_text(case_context)
        if not span_n or not ctx_n:
            return False
        return span_n in ctx_n

    def _stable_hash(self, s: str) -> str:
        return hashlib.md5(self._normalize_text(s).encode("utf-8", errors="ignore")).hexdigest()[:10]


    # label QA utilities

    def _label_value_to_answer(self, v: int) -> str:
        if v == 1:
            return "Yes."
        if v == 0:
            return "No."
        if v == 3:
            return "Uncertain."
        raise ValueError(f"Unexpected label_value: {v}")

    def _label_value_to_claim(self, label_phrase: str, v: int) -> str:
        if v == 1:
            return f"{label_phrase} is present"
        if v == 0:
            return f"{label_phrase} is absent"
        if v == 3:
            return f"{label_phrase} is uncertain"
        raise ValueError(f"Unexpected label_value: {v}")

    def _safe_int_label_value(self, x) -> Optional[int]:
        try:
            if pd.isna(x):
                return None
            return int(float(x))
        except Exception:
            return None

    def _aggregate_label_values(self, values: List[int]) -> Optional[int]:
        if not values:
            return None
        s = set(values)
        if 1 in s:
            return 1
        if 3 in s:
            return 3
        if 0 in s:
            return 0
        return None


    # sentence utilities

    def _split_sentences(self, text: str) -> List[str]:
        t = self._normalize_text(text)
        if not t:
            return []
        parts = [p.strip() for p in self.sent_split_re.split(t) if p and p.strip()]
        parts = [p for p in parts if len(p) >= 10]
        parts = [p for p in parts if not self.drop_sentence_re.search(p)]
        return parts

    def _score_sentence_for_findings(self, sent: str) -> int:
        score = 0
        if self.clinical_signal_re.search(sent):
            score += 2
        if self.negation_re.search(sent):
            score += 1
        if len(sent) >= 80:
            score += 1
        return score

    def _extract_key_findings(self, case_context: str) -> str:
        sents = self._split_sentences(case_context)
        scored = [(self._score_sentence_for_findings(s), s) for s in sents]
        scored = [x for x in scored if x[0] > 0]
        if not scored:
            return ""
        scored.sort(key=lambda x: (-x[0], -len(x[1])))
        top = [s for _, s in scored[: self.max_sentences_key_findings]]
        span = " ".join(top).strip()
        return span if self._is_extractive(span, case_context) else ""

    def _extract_temporal_change(self, case_context: str) -> str:
        sents = self._split_sentences(case_context)
        tmp = [s for s in sents if self.temporal_re.search(s)]
        if not tmp:
            return ""
        span = " ".join(tmp[: self.max_sentences_temporal]).strip()
        return span if self._is_extractive(span, case_context) else ""

    def _extract_uncertainty_spans(self, case_context: str) -> str:
        sents = self._split_sentences(case_context)
        hits = [s for s in sents if self.uncertainty_re.search(s)]
        if not hits:
            return ""
        span = " ".join(hits[: self.max_sentences_uncertainty]).strip()
        return span if self._is_extractive(span, case_context) else ""

    def _is_sentence_negated_global(self, sent: str) -> bool:
        # For things like "no pneumothorax", "without effusion" etc.
        return bool(self.negation_re.search(sent))

    def _is_sentence_normality(self, sent: str) -> bool:
        return bool(self.normality_re.search(sent))

    def _extract_compositional_spans(self, case_context: str) -> str:
        """
        Only return *abnormal* descriptive sentences:
          - must contain pathology + modifier tokens
          - must NOT be a global negation sentence ("no effusion")
          - must NOT be a normality boilerplate sentence ("lungs are clear")
        """
        sents = self._split_sentences(case_context)
        hits: List[str] = []
        for s in sents:
            if not (self.pathology_token_re.search(s) and self.modifier_token_re.search(s)):
                continue
            if self._is_sentence_normality(s):
                continue
            if self._is_sentence_negated_global(s):
                continue
            hits.append(s)

        if not hits:
            return ""

        span = " ".join(hits[: self.max_sentences_compositional]).strip()
        return span if self._is_extractive(span, case_context) else ""

    def _extract_device_sentences(self, case_context: str) -> List[str]:
        sents = self._split_sentences(case_context)
        return [s for s in sents if self.device_sentence_re.search(s)]

    def _extract_tip_location_spans(self, device_sents: List[str], case_context: str) -> List[str]:
        out: List[str] = []
        for s in device_sents:
            if not self.tip_loc_re.search(s):
                continue
            if self._is_extractive(s, case_context):
                out.append(s)
        return out

    def _find_normality_spans(self, case_context: str) -> str:
        sents = self._split_sentences(case_context)
        hits = [s for s in sents if self.normality_re.search(s)]
        if not hits:
            return ""
        span = " ".join(hits[:2]).strip()
        return span if self._is_extractive(span, case_context) else ""


    # device presence (negation-aware)

    def _is_negated_mention(self, sent: str, match_start: int, match_end: int) -> bool:
        """
        Conservative window-based check around a device mention.
        Reject if negation/removal cues occur near the mention.
        """
        s = sent
        w = self.device_negation_window_chars
        lo = max(0, match_start - w)
        hi = min(len(s), match_end + w)
        window = s[lo:hi]
        if self.removal_re.search(window):
            return True
        if self.negation_re.search(window):
            return True
        return False

    def _detect_present_devices(self, device_sents: List[str]) -> List[str]:
        """
        Mark device present if there exists at least one sentence where:
          - the device pattern matches
          - the match is NOT negated/removed in a local window
        """
        present: List[str] = []
        for name, pat in self.device_presence_vocab:
            ok = False
            for s in device_sents:
                m = pat.search(s)
                if not m:
                    continue
                if self._is_negated_mention(s, m.start(), m.end()):
                    continue
                ok = True
                break
            if ok:
                present.append(name)
        return present


    # main build

    def build_qa_table(self) -> pd.DataFrame:
        facts = pd.read_csv(self.facts_long_csv)
        # facts = facts.head(10_000)

        required = {
            self.subject_col,
            self.report_path_col,
            self.case_context_col,
            self.label_name_col,
            self.label_value_col,
        }
        missing = required - set(facts.columns)
        if missing:
            raise ValueError(f"facts_long is missing columns: {missing}")

        if self.split_col not in facts.columns:
            facts[self.split_col] = None

        # Harden types
        facts[self.report_path_col] = facts[self.report_path_col].astype(str).str.strip()

        facts[self.subject_col] = pd.to_numeric(facts[self.subject_col], errors="coerce")
        facts = facts[~facts[self.subject_col].isna()]
        facts[self.subject_col] = facts[self.subject_col].astype(int)

        groups = facts.groupby(self.report_path_col, sort=False)

        out_rows: List[Dict[str, object]] = []

        for report_rel_path, g in tqdm(groups, desc=self.tqdm_desc, total=groups.ngroups):
            report_rel_path = str(report_rel_path).strip()
            seen_keys: Set[Tuple[str, str, str]] = set()

            g0 = g.iloc[0]
            subject_id = int(g0[self.subject_col])
            split = g0[self.split_col] if self.split_col in g.columns else None

            case_context_raw = g0[self.case_context_col]
            case_context = self._normalize_text(case_context_raw)
            if case_context.lower() == "nan":
                case_context = ""

            # 1) label_presence (robust aggregation)
            label_qas_added = 0
            agg_map: Dict[str, Dict[str, object]] = {}

            gg = g[[self.label_name_col, self.label_value_col]].copy()
            gg = gg[~gg[self.label_name_col].isna()]
            gg[self.label_name_col] = gg[self.label_name_col].astype(str).str.strip()
            gg = gg[gg[self.label_name_col] != ""]

            if not gg.empty:
                gg["__lab_norm"] = gg[self.label_name_col].map(self._normalize_label_token)
                gg["__v"] = gg[self.label_value_col].map(self._safe_int_label_value)
                gg = gg[~gg["__v"].isna()]
                gg["__v"] = gg["__v"].astype(int)
                gg = gg[gg["__v"].isin([0, 1, 3])]

                for _, r in gg.iterrows():
                    lab_raw = str(r[self.label_name_col]).strip()
                    lab_norm = str(r["__lab_norm"]).strip()
                    v = int(r["__v"])

                    if not lab_norm:
                        continue
                    if lab_norm in self.skip_labels:
                        continue
                    if (v == 0) and (not self.include_negative_labels):
                        continue

                    if lab_norm not in agg_map:
                        agg_map[lab_norm] = {"lab_raw": lab_raw, "values": [v]}
                    else:
                        agg_map[lab_norm]["values"].append(v)

                for lab_norm in sorted(agg_map.keys()):
                    v_agg = self._aggregate_label_values([int(x) for x in agg_map[lab_norm]["values"]])
                    if v_agg is None:
                        continue

                    question = f"Is there {lab_norm}?"
                    answer_target = self._label_value_to_answer(int(v_agg))
                    claim_text = self._label_value_to_claim(lab_norm, int(v_agg))

                    key = (report_rel_path, "label_presence", question)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    out_rows.append(
                        {
                            self.subject_col: subject_id,
                            self.split_col: split,
                            self.report_path_col: report_rel_path,
                            "qa_type": "label_presence",
                            self.case_context_col: case_context,
                            "question": question,
                            "answer_target": answer_target,
                            "claim_text": claim_text,
                            "supporting_span": "",
                            "label_name": str(agg_map[lab_norm]["lab_raw"]),
                            "label_value": int(v_agg),
                        }
                    )

                    label_qas_added += 1
                    if self.max_label_qas_per_report > 0 and label_qas_added >= self.max_label_qas_per_report:
                        break

            # 2) normality_statement (extractive)
            if self.add_normality and case_context:
                norm_span = self._find_normality_spans(case_context)
                if norm_span:
                    question = "Is the study described as showing an acute cardiopulmonary abnormality?"
                    if re.search(r"\bno acute\b", norm_span, re.IGNORECASE) and not self.hedge_re.search(norm_span):
                        answer_target = "No."
                        claim_text = "no acute cardiopulmonary abnormality"
                    else:
                        answer_target = norm_span
                        claim_text = norm_span

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
                                "answer_target": answer_target,
                                "claim_text": claim_text,
                                "supporting_span": norm_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

            # 3) multi_label_conjunction (from aggregated labels)
            if self.add_multi_label_conjunction and agg_map:
                lab_to_v: Dict[str, int] = {}
                for lab_norm, payload in agg_map.items():
                    v_agg = self._aggregate_label_values([int(x) for x in payload["values"]])
                    if v_agg is None:
                        continue
                    lab_to_v[lab_norm] = int(v_agg)

                neg = sorted([l for l, v in lab_to_v.items() if v == 0])
                pos = sorted([l for l, v in lab_to_v.items() if v == 1])

                conj_added = 0

                for i in range(len(neg)):
                    for j in range(i + 1, len(neg)):
                        a, b = neg[i], neg[j]
                        question = f"Are {a} and {b} present?"
                        answer_target = "No, neither is present."
                        claim_text = f"{a} is absent; {b} is absent"

                        key = (report_rel_path, "multi_label_conjunction", f"{question}|{a}|{b}")
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        out_rows.append(
                            {
                                self.subject_col: subject_id,
                                self.split_col: split,
                                self.report_path_col: report_rel_path,
                                "qa_type": "multi_label_conjunction",
                                self.case_context_col: case_context,
                                "question": question,
                                "answer_target": answer_target,
                                "claim_text": claim_text,
                                "supporting_span": "",
                                "label_name": f"{a}|{b}",
                                "label_value": 0,
                            }
                        )
                        conj_added += 1
                        if conj_added >= self.max_conjunction_qas_per_report:
                            break
                    if conj_added >= self.max_conjunction_qas_per_report:
                        break

                if conj_added < self.max_conjunction_qas_per_report:
                    for i in range(len(pos)):
                        for j in range(i + 1, len(pos)):
                            a, b = pos[i], pos[j]
                            question = f"Are {a} and {b} present?"
                            answer_target = "Yes, both are present."
                            claim_text = f"{a} is present; {b} is present"

                            key = (report_rel_path, "multi_label_conjunction", f"{question}|{a}|{b}")
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)

                            out_rows.append(
                                {
                                    self.subject_col: subject_id,
                                    self.split_col: split,
                                    self.report_path_col: report_rel_path,
                                    "qa_type": "multi_label_conjunction",
                                    self.case_context_col: case_context,
                                    "question": question,
                                    "answer_target": answer_target,
                                    "claim_text": claim_text,
                                    "supporting_span": "",
                                    "label_name": f"{a}|{b}",
                                    "label_value": 1,
                                }
                            )
                            conj_added += 1
                            if conj_added >= self.max_conjunction_qas_per_report:
                                break
                        if conj_added >= self.max_conjunction_qas_per_report:
                            break

            # 4) key_findings (extractive)
            if self.add_key_findings and case_context:
                findings_span = self._extract_key_findings(case_context)
                if findings_span:
                    question = "What are the key imaging findings?"
                    key = (report_rel_path, "key_findings", question)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.subject_col: subject_id,
                                self.split_col: split,
                                self.report_path_col: report_rel_path,
                                "qa_type": "key_findings",
                                self.case_context_col: case_context,
                                "question": question,
                                "answer_target": findings_span,
                                "claim_text": findings_span,
                                "supporting_span": findings_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

            # 5) devices
            if (self.add_device_presence or self.add_device_tip_location or self.add_devices_locations) and case_context:
                dev_sents = self._extract_device_sentences(case_context)

                # legacy devices_locations
                if self.add_devices_locations and dev_sents:
                    dev_span = " ".join(dev_sents[:2]).strip()
                    if dev_span and self._is_extractive(dev_span, case_context):
                        question = "What devices are present and where are they located?"
                        key = (report_rel_path, "devices_locations", question)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            out_rows.append(
                                {
                                    self.subject_col: subject_id,
                                    self.split_col: split,
                                    self.report_path_col: report_rel_path,
                                    "qa_type": "devices_locations",
                                    self.case_context_col: case_context,
                                    "question": question,
                                    "answer_target": dev_span,
                                    "claim_text": dev_span,
                                    "supporting_span": dev_span,
                                    "label_name": None,
                                    "label_value": None,
                                }
                            )

                # device_presence (negation-aware)
                if self.add_device_presence and dev_sents:
                    present_devices = self._detect_present_devices(dev_sents)
                    dev_added = 0
                    for dev_name in present_devices:
                        question = f"Is a {dev_name} present?"
                        key = (report_rel_path, "device_presence", question)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        out_rows.append(
                            {
                                self.subject_col: subject_id,
                                self.split_col: split,
                                self.report_path_col: report_rel_path,
                                "qa_type": "device_presence",
                                self.case_context_col: case_context,
                                "question": question,
                                "answer_target": "Yes.",
                                "claim_text": f"{dev_name} is present",
                                "supporting_span": "",
                                "label_name": dev_name,
                                "label_value": 1,
                            }
                        )
                        dev_added += 1
                        if dev_added >= self.max_device_presence_qas_per_report:
                            break

                # device_tip_location (allow multiple, dedup by span hash)
                if self.add_device_tip_location and dev_sents:
                    tip_sents = self._extract_tip_location_spans(dev_sents, case_context)
                    tip_added = 0
                    for s in tip_sents:
                        dev_label = None
                        for name, pat in self.device_presence_vocab:
                            if pat.search(s):
                                dev_label = name
                                break

                        question = (
                            f"Where does the {dev_label} terminate?"
                            if dev_label
                            else "Where does the line or tube terminate?"
                        )

                        span_hash = self._stable_hash(s)
                        key = (report_rel_path, "device_tip_location", f"{question}|{span_hash}")
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        out_rows.append(
                            {
                                self.subject_col: subject_id,
                                self.split_col: split,
                                self.report_path_col: report_rel_path,
                                "qa_type": "device_tip_location",
                                self.case_context_col: case_context,
                                "question": question,
                                "answer_target": s,
                                "claim_text": s,
                                "supporting_span": s,
                                "label_name": None,
                                "label_value": None,
                            }
                        )
                        tip_added += 1
                        if tip_added >= self.max_device_tip_qas_per_report:
                            break

            # 6) temporal_change (extractive)
            if self.add_temporal_change and case_context:
                tmp_span = self._extract_temporal_change(case_context)
                if tmp_span:
                    question = "What interval changes are described?"
                    key = (report_rel_path, "temporal_change", question)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.subject_col: subject_id,
                                self.split_col: split,
                                self.report_path_col: report_rel_path,
                                "qa_type": "temporal_change",
                                self.case_context_col: case_context,
                                "question": question,
                                "answer_target": tmp_span,
                                "claim_text": tmp_span,
                                "supporting_span": tmp_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

            # 7) compositional_finding (extractive)
            if self.add_compositional_findings and case_context:
                comp_span = self._extract_compositional_spans(case_context)
                if comp_span:
                    question = "Describe the abnormal finding."
                    key = (report_rel_path, "compositional_finding", f"{question}|{self._stable_hash(comp_span)}")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.subject_col: subject_id,
                                self.split_col: split,
                                self.report_path_col: report_rel_path,
                                "qa_type": "compositional_finding",
                                self.case_context_col: case_context,
                                "question": question,
                                "answer_target": comp_span,
                                "claim_text": comp_span,
                                "supporting_span": comp_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

            # 8) limitation_uncertainty (extractive)
            if self.add_limitation_uncertainty and case_context:
                u_span = self._extract_uncertainty_spans(case_context)
                if u_span:
                    question = "What limitation or uncertainty is described?"
                    key = (report_rel_path, "limitation_uncertainty", f"{question}|{self._stable_hash(u_span)}")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.subject_col: subject_id,
                                self.split_col: split,
                                self.report_path_col: report_rel_path,
                                "qa_type": "limitation_uncertainty",
                                self.case_context_col: case_context,
                                "question": question,
                                "answer_target": u_span,
                                "claim_text": u_span,
                                "supporting_span": u_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

        qa = pd.DataFrame(out_rows)

        # final safety
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
    claim_text: str
    retrieved_chunk_ids_json: str
    support_weights_json: str
    embed_scores_json: str
    lexical_scores_json: str
    combined_scores_json: str

class RadiopaediaWeakEvidenceAligner:
    """
    Step 4B: Weak claim–evidence alignment (no humans)

    Input:
      - qa_csv: Step 3 output (mimic_qa_step3.csv)
      - chunks_parquet: Step 4A output (chunks.parquet)

    Output:
      - out_parquet: per-QA row: top-k retrieved chunk_ids + soft weights (JSON arrays)
    """

    def __init__(
        self,
        qa_csv: str | Path,
        chunks_parquet: str | Path,
        out_parquet: str | Path,
        # HF embedder (fixed bootstrap)
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        max_query_tokens: int = 128,
        max_doc_tokens: int = 256,
        # retrieval
        top_k: int = 20,
        qa_batch_size: int = 256,
        doc_batch_size: int = 128,
        # scoring
        alpha: float = 0.85,
        temperature: float = 0.07,
        # caching
        cache_dir: Optional[str | Path] = None,
        faiss_index_name: str = "radiopaedia_flatip.index",
        chunk_embeds_name: str = "radiopaedia_chunk_embeds.npy",
        # columns
        claim_col: str = "claim_text",
        report_path_col: str = "report_rel_path",
        qa_type_col: str = "qa_type",
        chunk_id_col: str = "chunk_id",
        chunk_text_col: str = "chunk_text",
        # determinism
        seed: int = 13,
        device: str = "cuda",
        force_cpu_faiss: bool = True,
        tqdm_desc: str = "Step 4B: weak claim–evidence alignment",
        qa_chunksize_rows: int = 50_000,
        # writer
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

        self.claim_col = str(claim_col)
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

        # will be set in _init_writer()
        self._pa_schema: Optional[pa.Schema] = None

        self._set_determinism()

    # determinism

    def _set_determinism(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Disable TF32 (important for reproducibility on Ampere+)
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


    def _normalize_text(self, s: str) -> str:
        if s is None:
            return ""
        x = str(s).lower()
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
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B,T,1)
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
            batch = texts[i : i + batch_size]
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

        return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)


    def _load_chunks_table(self) -> pd.DataFrame:
        cols = [self.chunk_id_col, self.chunk_text_col]
        df = pd.read_parquet(self.chunks_parquet, columns=cols)
        df[self.chunk_id_col] = pd.to_numeric(df[self.chunk_id_col], errors="coerce")
        df = df[~df[self.chunk_id_col].isna()].copy()
        df[self.chunk_id_col] = df[self.chunk_id_col].astype(int)
        df[self.chunk_text_col] = df[self.chunk_text_col].astype(str)

        # stable order
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
            idx = faiss.read_index(str(self.faiss_index_path))
            return idx

        dim = int(chunk_embeds.shape[1])
        index = faiss.IndexFlatIP(dim)  # cosine via normalized embeddings
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

        # store schema so every batch uses identical schema
        self._pa_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("claim_text", pa.string()),
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
        # Force consistent dtypes to avoid string vs large_string mismatch
        batch_df["qa_row_id"] = pd.to_numeric(batch_df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)

        for c in [
            "report_rel_path",
            "qa_type",
            "claim_text",
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

        # load chunks + index
        chunks_df = self._load_chunks_table()
        chunk_ids = chunks_df[self.chunk_id_col].to_numpy(dtype=np.int64)
        chunk_texts = chunks_df[self.chunk_text_col].tolist()

        chunk_embeds = self._build_or_load_chunk_embeddings(chunks_df)
        index = self._build_or_load_faiss_index(chunk_embeds)

        # stream QA CSV
        writer = self._init_writer()
        assert self._pa_schema is not None

        total_rows = sum(1 for _ in open(self.qa_csv, "r", encoding="utf-8", errors="ignore")) - 1
        pbar = tqdm(
            total=max(total_rows, 0),
            desc=self.tqdm_desc,
            unit="rows",
            miniters=1,
        )

        qa_row_id = 0
        try:
            for df in pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows):
                for c in [self.report_path_col, self.qa_type_col, self.claim_col]:
                    if c not in df.columns:
                        raise ValueError(f"QA CSV missing required column: {c}")

                df[self.report_path_col] = df[self.report_path_col].astype(str)
                df[self.qa_type_col] = df[self.qa_type_col].astype(str)
                df[self.claim_col] = df[self.claim_col].astype(str)

                claims = df[self.claim_col].tolist()
                q_emb = self._encode_texts(
                    claims,
                    max_len=self.max_query_tokens,
                    batch_size=self.qa_batch_size,
                    pbar=pbar,  # updates per encoder batch
                )

                D, I = index.search(q_emb, self.top_k)  # (B,k)

                out_rows: List[Dict[str, object]] = []

                for i in range(len(df)):
                    report_rel_path = df.iloc[i][self.report_path_col]
                    qa_type = df.iloc[i][self.qa_type_col]
                    claim_text = df.iloc[i][self.claim_col]
                    claim_norm = self._normalize_text(claim_text)

                    if not claim_norm:
                        out_rows.append(
                            asdict(
                                WeakAlignmentRow(
                                    qa_row_id=qa_row_id,
                                    report_rel_path=report_rel_path,
                                    qa_type=qa_type,
                                    claim_text=claim_text,
                                    retrieved_chunk_ids_json="[]",
                                    support_weights_json="[]",
                                    embed_scores_json="[]",
                                    lexical_scores_json="[]",
                                    combined_scores_json="[]",
                                )
                            )
                        )
                        qa_row_id += 1
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

                    lex_scores = [float(self._lexical_overlap(claim_norm, t)) for t in retrieved_texts]

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
                                claim_text=claim_text,
                                retrieved_chunk_ids_json=json.dumps(retrieved_ids),
                                support_weights_json=json.dumps([float(x) for x in w.tolist()]),
                                embed_scores_json=json.dumps([float(x) for x in e.tolist()]),
                                lexical_scores_json=json.dumps([float(x) for x in l.tolist()]),
                                combined_scores_json=json.dumps([float(x) for x in combined.tolist()]),
                            )
                        )
                    )
                    qa_row_id += 1

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
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {self.out_parquet.parent / 'manifest_step4b.json'}")
        print(f"Cache: {self.chunk_embeds_path}  |  {self.faiss_index_path}")








@dataclass(frozen=True)
class RetrieverSupervisionRow:
    qa_row_id: int
    subject_id: int
    split: str                    # "train" / "valid" / "test"
    report_rel_path: str
    qa_type: str
    claim_text: str
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
    evidence_chunk_ids_json: str
    evidence_weights_json: str
    soft_support_score: float


class RadiopaediaStep4CSupervisionBuilder:
    """
    Step 4C: Convert Step 3 + Step 4B outputs into final supervision artifacts

      - Create a fresh patient-wise split (train=75%, valid=10%, test=15%)
      - Write subject_id + split into both output parquets

    Split is computed ONLY from subject_id (no leakage across patients) and is deterministic.
    """

    def __init__(
        self,
        qa_csv: str | Path,
        weak_alignment_parquet: str | Path,
        out_dir: str | Path,
        # columns in qa_csv
        subject_id_col: str = "subject_id",          # NEW
        report_path_col: str = "report_rel_path",
        qa_type_col: str = "qa_type",
        claim_col: str = "claim_text",
        case_context_col: str = "case_context",
        question_col: str = "question",
        answer_col: str = "answer_target",
        # streaming
        qa_chunksize_rows: int = 50_000,
        parquet_batch_rows: int = 50_000,
        # supervision shaping
        top_k: Optional[int] = None,
        min_weight: float = 0.0,
        # split config (NEW)
        split_seed: int = 13,
        frac_train: float = 0.75,
        frac_valid: float = 0.10,
        frac_test: float = 0.15,
        # progress
        tqdm_desc: str = "Step 4C: building supervision",
        parquet_compression: str = "zstd",
    ) -> None:
        self.qa_csv = Path(qa_csv)
        self.weak_alignment_parquet = Path(weak_alignment_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.subject_id_col = str(subject_id_col)     # NEW
        self.report_path_col = str(report_path_col)
        self.qa_type_col = str(qa_type_col)
        self.claim_col = str(claim_col)
        self.case_context_col = str(case_context_col)
        self.question_col = str(question_col)
        self.answer_col = str(answer_col)

        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_batch_rows = int(parquet_batch_rows)

        self.top_k = int(top_k) if top_k is not None else None
        self.min_weight = float(min_weight)

        self.split_seed = int(split_seed)
        self.frac_train = float(frac_train)
        self.frac_valid = float(frac_valid)
        self.frac_test = float(frac_test)
        if not np.isclose(self.frac_train + self.frac_valid + self.frac_test, 1.0):
            raise ValueError("Split fractions must sum to 1.0 (train+valid+test).")

        self.tqdm_desc = str(tqdm_desc)
        self.parquet_compression = str(parquet_compression)

        self.retriever_out = self.out_dir / "retriever_supervision.parquet"
        self.verifier_out = self.out_dir / "verifier_supervision.parquet"
        self.manifest_out = self.out_dir / "manifest_step4c.json"
        self.split_map_out = self.out_dir / "patient_split_map.json"  # NEW

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

    def _filter_and_truncate(self, ids: List[int], w: List[float]) -> Tuple[List[int], List[float]]:
        if self.top_k is not None:
            ids = ids[: self.top_k]
            w = w[: self.top_k]
        if self.min_weight > 0:
            keep = [(i, ww) for i, ww in zip(ids, w) if float(ww) >= self.min_weight]
            ids = [int(i) for i, _ in keep]
            w = [float(ww) for _, ww in keep]
        return ids, w

    def _soft_support_from_combined(self, combined_scores: List[float]) -> float:
        if not combined_scores:
            return 0.0
        try:
            return float(np.max(np.asarray(combined_scores, dtype=np.float32)))
        except Exception:
            return 0.0

    # patient-wise split map (deterministic)
    def _build_patient_split_map(self) -> Dict[int, str]:
        # Read only subject_id column (fast)
        if not self.qa_csv.exists():
            raise FileNotFoundError(f"QA CSV not found: {self.qa_csv}")

        subj = pd.read_csv(self.qa_csv, usecols=[self.subject_id_col])
        subj[self.subject_id_col] = pd.to_numeric(subj[self.subject_id_col], errors="coerce")
        subj = subj[~subj[self.subject_id_col].isna()]
        subject_ids = np.array(sorted(subj[self.subject_id_col].astype(int).unique().tolist()), dtype=np.int64)

        rng = np.random.RandomState(self.split_seed)
        perm = rng.permutation(subject_ids)

        n = len(perm)
        n_train = int(round(self.frac_train * n))
        n_valid = int(round(self.frac_valid * n))
        # ensure exact total
        n_test = n - n_train - n_valid
        if n_test < 0:
            n_test = 0
            n_valid = n - n_train

        split_map: Dict[int, str] = {}
        for sid in perm[:n_train]:
            split_map[int(sid)] = "train"
        for sid in perm[n_train : n_train + n_valid]:
            split_map[int(sid)] = "valid"
        for sid in perm[n_train + n_valid :]:
            split_map[int(sid)] = "test"

        # write map for reuse
        self.split_map_out.write_text(json.dumps(split_map, indent=2), encoding="utf-8")
        return split_map

    def _init_retriever_writer(self) -> pq.ParquetWriter:
        self._retr_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("subject_id", pa.int64()),     # NEW
                ("split", pa.string()),         # NEW
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("claim_text", pa.string()),
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
                ("subject_id", pa.int64()),     # NEW
                ("split", pa.string()),         # NEW
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("case_context", pa.string()),
                ("question", pa.string()),
                ("answer_target", pa.string()),
                ("claim_text", pa.string()),
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

        for c in [
            "split",
            "report_rel_path",
            "qa_type",
            "claim_text",
            "pos_chunk_ids_json",
            "pos_weights_json",
            "case_context",
            "question",
            "answer_target",
            "evidence_chunk_ids_json",
            "evidence_weights_json",
        ]:
            if c in df.columns:
                df[c] = df[c].astype(str)

        if "soft_support_score" in df.columns:
            df["soft_support_score"] = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).astype(
                np.float32
            )

        return pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    def run(self) -> None:
        if not self.qa_csv.exists():
            raise FileNotFoundError(f"QA CSV not found: {self.qa_csv}")
        if not self.weak_alignment_parquet.exists():
            raise FileNotFoundError(f"Weak alignment parquet not found: {self.weak_alignment_parquet}")

        self._patient_split_map = self._build_patient_split_map()

        total_rows = sum(1 for _ in open(self.qa_csv, "r", encoding="utf-8", errors="ignore")) - 1
        pbar = tqdm(total=max(total_rows, 0), desc=self.tqdm_desc, unit="rows", miniters=1)

        retr_writer = self._init_retriever_writer()
        ver_writer = self._init_verifier_writer()
        assert self._retr_schema is not None
        assert self._ver_schema is not None

        pf = pq.ParquetFile(str(self.weak_alignment_parquet))

        align_cols = [
            "qa_row_id",
            "report_rel_path",
            "qa_type",
            "claim_text",
            "retrieved_chunk_ids_json",
            "support_weights_json",
            "combined_scores_json",
        ]

        needed_qa_cols = [
            self.subject_id_col,
            self.report_path_col,
            self.qa_type_col,
            self.claim_col,
            self.case_context_col,
            self.question_col,
            self.answer_col,
        ]

        qa_reader = pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows)

        qa_row_id_expected = 0
        align_row_id_expected = 0

        try:
            align_iter = pf.iter_batches(batch_size=self.parquet_batch_rows, columns=align_cols)

            for qa_chunk in qa_reader:
                for c in needed_qa_cols:
                    if c not in qa_chunk.columns:
                        raise ValueError(f"QA CSV missing required column: {c}")

                qa_chunk = qa_chunk.copy()

                qa_chunk[self.subject_id_col] = pd.to_numeric(qa_chunk[self.subject_id_col], errors="coerce")
                qa_chunk = qa_chunk[~qa_chunk[self.subject_id_col].isna()]
                qa_chunk[self.subject_id_col] = qa_chunk[self.subject_id_col].astype(np.int64)

                qa_chunk[self.report_path_col] = qa_chunk[self.report_path_col].astype(str)
                qa_chunk[self.qa_type_col] = qa_chunk[self.qa_type_col].astype(str)
                qa_chunk[self.claim_col] = qa_chunk[self.claim_col].astype(str)
                qa_chunk[self.case_context_col] = qa_chunk[self.case_context_col].astype(str)
                qa_chunk[self.question_col] = qa_chunk[self.question_col].astype(str)
                qa_chunk[self.answer_col] = qa_chunk[self.answer_col].astype(str)

                n = len(qa_chunk)
                if n == 0:
                    continue

                align_rows: List[pd.DataFrame] = []
                need = n
                while need > 0:
                    batch = next(align_iter, None)
                    if batch is None:
                        raise RuntimeError(f"Alignment parquet ended early. Still needed {need} rows.")
                    bdf = batch.to_pandas()
                    align_rows.append(bdf)
                    need -= len(bdf)

                align_df = pd.concat(align_rows, ignore_index=True)
                if len(align_df) > n:
                    align_df = align_df.iloc[:n].copy()

                if "qa_row_id" not in align_df.columns:
                    raise ValueError("Alignment parquet missing qa_row_id column.")

                first_id = int(align_df.iloc[0]["qa_row_id"])
                if first_id != align_row_id_expected:
                    raise RuntimeError(
                        f"Alignment qa_row_id mismatch: expected {align_row_id_expected}, got {first_id}."
                    )

                retr_rows: List[Dict[str, object]] = []
                ver_rows: List[Dict[str, object]] = []

                assert self._patient_split_map is not None

                for i in range(n):
                    qa_row_id = qa_row_id_expected
                    al_row = align_df.iloc[i]

                    ids = self._loads_json_list(self._safe_str(al_row.get("retrieved_chunk_ids_json", "[]")))
                    w = self._loads_json_list(self._safe_str(al_row.get("support_weights_json", "[]")))
                    combined = self._loads_json_list(self._safe_str(al_row.get("combined_scores_json", "[]")))

                    ids = [int(x) for x in ids if str(x).strip() != ""]
                    w = [float(x) for x in w]
                    combined = [float(x) for x in combined]

                    ids, w = self._filter_and_truncate(ids, w)
                    if len(w) != len(ids):
                        m = min(len(w), len(ids))
                        ids, w = ids[:m], w[:m]

                    subject_id = int(qa_chunk.iloc[i][self.subject_id_col])
                    split = self._patient_split_map.get(subject_id, "train")  # should always exist

                    report_rel_path = self._safe_str(qa_chunk.iloc[i][self.report_path_col])
                    qa_type = self._safe_str(qa_chunk.iloc[i][self.qa_type_col])
                    claim_text = self._safe_str(qa_chunk.iloc[i][self.claim_col])
                    case_context = self._safe_str(qa_chunk.iloc[i][self.case_context_col])
                    question = self._safe_str(qa_chunk.iloc[i][self.question_col])
                    answer_target = self._safe_str(qa_chunk.iloc[i][self.answer_col])

                    retr_rows.append(
                        asdict(
                            RetrieverSupervisionRow(
                                qa_row_id=qa_row_id,
                                subject_id=subject_id,
                                split=split,
                                report_rel_path=report_rel_path,
                                qa_type=qa_type,
                                claim_text=claim_text,
                                pos_chunk_ids_json=json.dumps(ids),
                                pos_weights_json=json.dumps([float(x) for x in w]),
                            )
                        )
                    )

                    ver_rows.append(
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
                                evidence_chunk_ids_json=json.dumps(ids),
                                evidence_weights_json=json.dumps([float(x) for x in w]),
                                soft_support_score=self._soft_support_from_combined(combined),
                            )
                        )
                    )

                    qa_row_id_expected += 1
                    align_row_id_expected += 1

                retr_df = pd.DataFrame(retr_rows)
                ver_df = pd.DataFrame(ver_rows)

                retr_writer.write_table(self._to_table(retr_df, self._retr_schema))
                ver_writer.write_table(self._to_table(ver_df, self._ver_schema))

                pbar.update(n)

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
                "retriever_supervision": str(self.retriever_out),
                "verifier_supervision": str(self.verifier_out),
                "patient_split_map": str(self.split_map_out),
            },
            "split": {
                "method": "patient-wise",
                "seed": self.split_seed,
                "fractions": {"train": self.frac_train, "valid": self.frac_valid, "test": self.frac_test},
                "labels": ["train", "valid", "test"],
            },
            "params": {
                "qa_chunksize_rows": self.qa_chunksize_rows,
                "parquet_batch_rows": self.parquet_batch_rows,
                "top_k": self.top_k,
                "min_weight": self.min_weight,
                "soft_support_scalar": "max(combined_scores)",
            },
        }
        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.retriever_out}")
        print(f"Wrote: {self.verifier_out}")
        print(f"Wrote: {self.split_map_out}")
        print(f"Wrote: {self.manifest_out}")





if __name__ == "__main__":
    # # step 1
    # builder = MimicCaseContextBuilder(
    #     master_csv="//MIMIC/mimic_master_list_forReports.csv",
    #     reports_root="//MIMIC",
    #     out_path="mimic_case_contexts.csv",
    # )
    # out_df = builder.build_case_context_table()
    # print(out_df[["report_rel_path", "case_context"]].head(3))
    # print("Rows:", len(out_df))
    #
    #
    # # step 2
    # builder = MimicFactTableBuilder(
    #     master_csv="//MIMIC/mimic_master_list_forReports.csv",
    #     case_context_csv="/MIMIC/mimic_case_contexts.csv",
    #     out_path="mimic_facts_long.csv",
    # )
    # facts_df = builder.build_fact_table()
    # print(facts_df.head(10))
    # print("n_facts:", len(facts_df))
    #
    #
    #
    # # step 3
    # builder = MimicQATableBuilder(
    #     facts_long_csv="//MIMIC/mimic_facts_long.csv",
    #     out_path="mimic_qa_step3.csv",
    # )
    # qa_df = builder.build_qa_table()
    # print("QA rows:", len(qa_df))
    # print(qa_df[["qa_type", "question", "answer_target"]].head(10))
    #
    #
    #
    # # # step 4a
    # builder = RadiopaediaSnapshotBuilder(
    #     in_dir="/PATH/Radiopaedia/Text_only/Articles_Processed_Jan_2026/articles_clean_txt",
    #     out_dir="/PATH/Text_only/Articles_Processed_Jan_2026/RadiopaediaSnapshot_29_01_2026",
    #     snapshot_name="RadiopaediaSnapshot_29_01_2026",
    #     tokenizer_name="sentence-transformers/all-mpnet-base-v2",
    #     target_tokens=200,
    #     min_tokens=50,
    #     write_csv=False,
    # )
    # chunks_df = builder.run()
    #
    #
    #
    # # # step 4b
    # builder = RadiopaediaWeakEvidenceAligner(
    #     qa_csv="/PATH/MIMIC/mimic_qa_step3.csv",
    #     chunks_parquet="/PATH/Text_only/Articles_Processed_Jan_2026/RadiopaediaSnapshot_29_01_2026/chunks.parquet",
    #     out_parquet="/PATH/Text_only/Articles_Processed_Jan_2026/RadiopaediaSnapshot_29_01_2026/weak_alignment_step4b.parquet",
    #     model_name="sentence-transformers/all-mpnet-base-v2",
    #     top_k=20,
    #     alpha=0.85,
    #     temperature=0.07,  # keep >0; deterministic with fixed seeds + eval()
    #     seed=13,
    #     cache_dir="/PATH/Text_only/Articles_Processed_Jan_2026/RadiopaediaSnapshot_29_01_2026/cache_step4b",
    #     device="cuda",
    #     # device="cpu",
    #     force_cpu_faiss=True,
    # )
    # builder.run()
    #
    #
    #
    # # # step 4c
    # builder = RadiopaediaStep4CSupervisionBuilder(
    #     qa_csv="/PATH/MIMIC/mimic_qa_step3.csv",
    #     weak_alignment_parquet="/PATH/Text_only/Articles_Processed_Jan_2026/RadiopaediaSnapshot_29_01_2026/weak_alignment_step4b.parquet",
    #     out_dir="/PATH/Text_only/Articles_Processed_Jan_2026/RadiopaediaSnapshot_29_01_2026/step4c_supervision",
    #     top_k=20,
    #     min_weight=0.0,
    #     qa_chunksize_rows=50_000,
    #     parquet_batch_rows=50_000,
    #     subject_id_col="subject_id",  # must exist in mimic_qa_step3.csv
    #     split_seed=13,
    #     frac_train=0.75,
    #     frac_valid=0.10,
    #     frac_test=0.15,
    # )
    # builder.run()
