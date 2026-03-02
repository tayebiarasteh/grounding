"""
Created on March 1, 2026.
chexpertplus_data_preprocessing.py

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
import math


import warnings
warnings.filterwarnings('ignore')



class ChexpertPlusCaseContextBuilder:
    """
    Step 1 (CheXpert-Plus):
    Build report-level case_context from a single master CSV that already contains the report text.

    Output:
      One row per unique report unit, with columns:
        deid_patient_id, report_id, report, case_context, n_images

    Notes:
      - If multiple rows share the same (patient, report text), we collapse them into one report unit.
      - If the CSV has "section_findings"/"section_impression" columns, we can prefer them; otherwise we use "report".
    """

    def __init__(
        self,
        master_csv: str,
        out_path: Optional[str] = None,
        encoding: str = "utf-8",
        tqdm_desc: str = "Building CheXpert-Plus case_context",
        min_lines_fallback: int = 2,
        # columns
        report_col: str = "report",
        patient_col: str = "deid_patient_id",
        image_col: str = "path_to_image",
        split_col: str = "split",
        # optional section columns (if present in the CSV)
        findings_col: str = "section_findings",
        impression_col: str = "section_impression",
        narrative_col: str = "section_narrative",
    ) -> None:
        self.master_csv = str(master_csv)
        self.out_path = out_path
        self.encoding = encoding
        self.tqdm_desc = tqdm_desc
        self.min_lines_fallback = int(min_lines_fallback)

        self.report_col = str(report_col)
        self.patient_col = str(patient_col)
        self.image_col = str(image_col)
        self.split_col = str(split_col)

        self.findings_col = str(findings_col)
        self.impression_col = str(impression_col)
        self.narrative_col = str(narrative_col)

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

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _compose_raw_report_text(self, row: pd.Series, has_findings: bool, has_impression: bool, has_narrative: bool) -> str:
        """
        Prefer structured sections if present and non-empty, else fall back to report.
        """
        parts: List[str] = []

        if has_findings:
            f = self._safe_str(row.get(self.findings_col, "")).strip()
            if f:
                parts.append(f)

        if has_impression:
            imp = self._safe_str(row.get(self.impression_col, "")).strip()
            if imp:
                parts.append(imp)

        # If we got nothing from findings/impression, try narrative
        if not parts and has_narrative:
            nar = self._safe_str(row.get(self.narrative_col, "")).strip()
            if nar:
                parts.append(nar)

        # Final fallback
        if not parts:
            rep = self._safe_str(row.get(self.report_col, "")).strip()
            if rep:
                parts.append(rep)

        return "\n".join([p for p in parts if p]).strip()

    def clean_report_to_case_context(self, raw_text: str) -> str:
        """
        Same cleaning philosophy as MIMIC:
          - Prefer FINDINGS+IMPRESSION when present in the text
          - Keep clinically meaningful narrative
          - Drop admin headers, timestamps/signatures, and communication lines
          - Strip inline section tokens anywhere in kept lines
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
            return self.inline_section_token_re.sub("", s).strip()

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

        # Add clinically-rich non-section lines
        base_set = set(base)
        extra: List[str] = []
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

        cleaned_lines: List[str] = []
        for ln in combined:
            ln2 = strip_inline_section_tokens(ln)
            if ln2:
                cleaned_lines.append(ln2)

        out = "\n".join(cleaned_lines)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out

    def build_case_context_table(self) -> pd.DataFrame:
        df = pd.read_csv(self.master_csv, encoding=self.encoding)

        required = {self.report_col, self.patient_col, self.image_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in master CSV: {missing}")

        has_findings = self.findings_col in df.columns
        has_impression = self.impression_col in df.columns
        has_narrative = self.narrative_col in df.columns
        has_split = self.split_col in df.columns

        # report_id: stable within CSV.
        # We use (patient_id + normalized report text) as the unit. This avoids needing an explicit report id column.
        # If CheXpert-Plus guarantees a dedicated report id column later, swap this to that column.
        work = df[[c for c in [self.patient_col, self.image_col, self.split_col] if c in df.columns] + [self.report_col] +
                  ([self.findings_col] if has_findings else []) +
                  ([self.impression_col] if has_impression else []) +
                  ([self.narrative_col] if has_narrative else [])].copy()

        # normalize report text lightly for grouping
        def norm_group_text(x: str) -> str:
            x = self._safe_str(x)
            x = x.replace("\r\n", "\n").replace("\r", "\n")
            x = re.sub(r"\s+", " ", x).strip().lower()
            return x

        # build a per-row raw text (prefer sections)
        raw_texts: List[str] = []
        for _, r in tqdm(work.iterrows(), total=len(work), desc="Composing raw report text", miniters=1):
            raw_texts.append(self._compose_raw_report_text(r, has_findings, has_impression, has_narrative))
        work["__raw_text"] = raw_texts
        work["__raw_text_norm"] = work["__raw_text"].map(norm_group_text)

        # group key
        work["__patient_id_str"] = work[self.patient_col].astype(str).str.strip()
        work["report_id"] = work["__patient_id_str"] + "::" + work["__raw_text_norm"].map(
            lambda s: str(abs(hash(s)))  # deterministic within one Python run, not across runs
        )

        # IMPORTANT: Python's built-in hash is not stable across processes by default.
        # Replace with md5 for cross-run stability.
        import hashlib
        work["report_id"] = work["__patient_id_str"] + "::" + work["__raw_text_norm"].map(
            lambda s: hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:16]
        )

        # collapse to one row per report_id
        agg_cols = {
            self.patient_col: "first",
            "__raw_text": "first",
            self.image_col: "count",
        }
        if has_split:
            agg_cols[self.split_col] = "first"

        uniq = work.groupby("report_id", as_index=False).agg(agg_cols)
        uniq = uniq.rename(columns={"__raw_text": self.report_col, self.image_col: "n_images"})

        # build case_context
        case_contexts: List[str] = []
        for raw in tqdm(uniq[self.report_col].astype(str).tolist(), desc=self.tqdm_desc, total=len(uniq), miniters=1):
            case_contexts.append(self.clean_report_to_case_context(self._safe_str(raw)))

        uniq["case_context"] = case_contexts

        # final column order
        out_cols = ["report_id", self.patient_col]
        if has_split:
            out_cols.append(self.split_col)
        out_cols += ["n_images", self.report_col, "case_context"]
        uniq = uniq[out_cols].copy()

        if self.out_path:
            outp = Path(self.out_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            uniq.to_csv(outp, index=False)

        return uniq






class ChexpertPlusFactsAndQABuilder:
    """
    CheXpert-Plus external dataset: combined Step 2 + Step 3 in one class.

    Key fixes vs your current version:
      1) deid_patient_id is treated as STRING everywhere (never coerced to numeric).
      2) QA builder is robust to empty outputs (no KeyError on report_id).
      3) facts_long stores deid_patient_id as string consistently (no mixed dtypes).

    Assumes:
      - master_csv has columns: deid_patient_id, path_to_image, report
      - case_context_csv has columns: report_id, deid_patient_id, case_context
    """

    def __init__(
        self,
        master_csv: str,
        case_context_csv: str,
        out_facts_path: Optional[str] = None,
        out_qa_path: Optional[str] = None,
        # columns
        split_col: str = "split",
        patient_col: str = "deid_patient_id",
        image_col: str = "path_to_image",
        report_col: str = "report",
        report_id_col: str = "report_id",
        # optional section columns (if present)
        findings_col: str = "section_findings",
        impression_col: str = "section_impression",
        narrative_col: str = "section_narrative",
        # Step 2 behavior
        include_negative_labels: bool = True,
        # Step 3 knobs
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
        skip_labels: Optional[Set[str]] = None,
        device_negation_window_chars: int = 45,
        # progress
        tqdm_desc_facts: str = "Building facts_long (CheXpert-Plus Step 2)",
        tqdm_desc_qa: str = "Building QA table (CheXpert-Plus Step 3)",
    ) -> None:
        self.master_csv = str(master_csv)
        self.case_context_csv = str(case_context_csv)
        self.out_facts_path = out_facts_path
        self.out_qa_path = out_qa_path

        self.split_col = str(split_col)
        self.patient_col = str(patient_col)
        self.image_col = str(image_col)
        self.report_col = str(report_col)
        self.report_id_col = str(report_id_col)

        self.findings_col = str(findings_col)
        self.impression_col = str(impression_col)
        self.narrative_col = str(narrative_col)

        self.include_negative_labels = bool(include_negative_labels)

        self.max_label_qas_per_report = int(max_label_qas_per_report)
        self.max_conjunction_qas_per_report = int(max_conjunction_qas_per_report)

        self.add_key_findings = bool(add_key_findings)
        self.add_normality = bool(add_normality)
        self.add_multi_label_conjunction = bool(add_multi_label_conjunction)
        self.add_device_presence = bool(add_device_presence)
        self.add_device_tip_location = bool(add_device_tip_location)
        self.add_devices_locations = bool(add_devices_locations)
        self.add_temporal_change = bool(add_temporal_change)
        self.add_compositional_findings = bool(add_compositional_findings)
        self.add_limitation_uncertainty = bool(add_limitation_uncertainty)

        self.max_sentences_key_findings = int(max_sentences_key_findings)
        self.max_sentences_temporal = int(max_sentences_temporal)
        self.max_sentences_compositional = int(max_sentences_compositional)
        self.max_sentences_uncertainty = int(max_sentences_uncertainty)
        self.max_device_presence_qas_per_report = int(max_device_presence_qas_per_report)
        self.max_device_tip_qas_per_report = int(max_device_tip_qas_per_report)

        base_skip = skip_labels or {"no_finding"}
        self.skip_labels = {self._normalize_label_token(s) for s in base_skip}

        self.device_negation_window_chars = int(device_negation_window_chars)

        self.tqdm_desc_facts = str(tqdm_desc_facts)
        self.tqdm_desc_qa = str(tqdm_desc_qa)

        # Step 2 label semantics
        self.valid_fact_values = {0, 1, 3}

        # Step 3 sentence splitting only on punctuation boundaries (no newline splitting)
        self.sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")

        self.drop_sentence_re = re.compile(
            r"\b("
            r"history|indication|clinical (history|indication)|reason for examination|technique|comparison|"
            r"results were communicated|discussed with|called to|phone|paged|paging|notified|communicated"
            r")\b",
            re.IGNORECASE,
        )

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

        self.temporal_re = re.compile(
            r"\b("
            r"interval|compared to|as compared to|since prior|prior study|previous study|"
            r"improved|worsened|increased|decreased|unchanged|stable|new|resolved|persistent|"
            r"removal|removed|placement|placed|re[-\s]?accumulation|progression"
            r")\b",
            re.IGNORECASE,
        )

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

        self.uncertainty_re = re.compile(
            r"\b("
            r"within the limitations of|limitation(s)?|limited evaluation|poorly assessed|"
            r"cannot exclude|cannot be excluded|may represent|might reflect|possible|likely|"
            r"cannot be evaluated|not well seen|underlying .* cannot be excluded"
            r")\b",
            re.IGNORECASE,
        )


    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _normalize_label_token(self, s: str) -> str:
        s = str(s).strip().lower().replace("_", " ")
        s = re.sub(r"\s+", " ", s)
        return s

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

    def _is_extractive(self, span: str, case_context: str) -> bool:
        span_n = self._normalize_text(span)
        ctx_n = self._normalize_text(case_context)
        if not span_n or not ctx_n:
            return False
        return span_n in ctx_n



    # Step 1 compatibility: report_id

    def _normalize_group_text(self, x: str) -> str:
        x = self._safe_str(x)
        x = x.replace("\r\n", "\n").replace("\r", "\n")
        x = re.sub(r"\s+", " ", x).strip().lower()
        return x

    def _compose_raw_report_text(
        self, row: pd.Series, has_findings: bool, has_impression: bool, has_narrative: bool
    ) -> str:
        parts: List[str] = []

        if has_findings:
            f = self._safe_str(row.get(self.findings_col, "")).strip()
            if f:
                parts.append(f)

        if has_impression:
            imp = self._safe_str(row.get(self.impression_col, "")).strip()
            if imp:
                parts.append(imp)

        if not parts and has_narrative:
            nar = self._safe_str(row.get(self.narrative_col, "")).strip()
            if nar:
                parts.append(nar)

        if not parts:
            rep = self._safe_str(row.get(self.report_col, "")).strip()
            if rep:
                parts.append(rep)

        return "\n".join([p for p in parts if p]).strip()

    def _compute_report_id(self, patient_id: str, raw_text: str) -> str:
        pid = str(patient_id).strip()
        norm = self._normalize_group_text(raw_text)
        h = hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"{pid}::{h}"

    # Step 2: facts_long

    def _detect_label_columns(self, df: pd.DataFrame) -> List[str]:
        meta = {
            self.patient_col,
            self.image_col,
            self.split_col,
            self.report_col,
            self.report_id_col,
            self.findings_col,
            self.impression_col,
            self.narrative_col,
        }
        label_cols: List[str] = []
        for c in df.columns:
            if c in meta:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                label_cols.append(c)
        if not label_cols:
            raise ValueError("No numeric label columns found. Provide explicit label column names.")
        return label_cols

    def _aggregate_labels_per_report(self, df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
        use_cols = [self.report_id_col, self.patient_col] + (
            [self.split_col] if self.split_col in df.columns else []
        ) + label_cols

        work = df[use_cols].copy()

        # patient_id always as string
        work[self.patient_col] = work[self.patient_col].astype(str).str.strip()
        work[self.report_id_col] = work[self.report_id_col].astype(str).str.strip()

        for c in label_cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        pri_map = {2: 0, 0: 1, 3: 2, 1: 3}
        for c in label_cols:
            work[c] = work[c].map(pri_map).fillna(0).astype("int8")

        g = work.groupby(self.report_id_col, as_index=False)

        out = g[[self.patient_col]].first()
        if self.split_col in work.columns:
            out[self.split_col] = g[self.split_col].first()[self.split_col]

        max_pri = g[label_cols].max()
        inv_pri = {0: 2, 1: 0, 2: 3, 3: 1}
        for c in label_cols:
            out[c] = max_pri[c].map(inv_pri).astype("int8")

        out[self.report_id_col] = g[self.report_id_col].first()[self.report_id_col]
        return out

    def _build_facts_long(self) -> pd.DataFrame:
        master = pd.read_csv(self.master_csv)
        ctx = pd.read_csv(self.case_context_csv)

        required_master = {self.patient_col, self.image_col, self.report_col}
        missing_master = required_master - set(master.columns)
        if missing_master:
            raise ValueError(f"Master CSV missing required columns: {missing_master}")

        required_ctx = {self.report_id_col, self.patient_col, "case_context"}
        missing_ctx = required_ctx - set(ctx.columns)
        if missing_ctx:
            raise ValueError(f"Case-context CSV missing columns: {missing_ctx}")

        # normalize ids in ctx for merge stability
        ctx = ctx.copy()
        ctx[self.patient_col] = ctx[self.patient_col].astype(str).str.strip()
        ctx[self.report_id_col] = ctx[self.report_id_col].astype(str).str.strip()

        has_findings = self.findings_col in master.columns
        has_impression = self.impression_col in master.columns
        has_narrative = self.narrative_col in master.columns

        if self.report_id_col not in master.columns:
            raw_texts: List[str] = []
            for _, r in tqdm(
                master.iterrows(),
                total=len(master),
                desc="Composing raw report text for report_id",
                miniters=1,
            ):
                raw_texts.append(self._compose_raw_report_text(r, has_findings, has_impression, has_narrative))
            master["__raw_text_for_id"] = raw_texts

            master[self.patient_col] = master[self.patient_col].astype(str).str.strip()
            master[self.report_id_col] = [
                self._compute_report_id(pid, txt)
                for pid, txt in zip(
                    master[self.patient_col].tolist(),
                    master["__raw_text_for_id"].tolist(),
                )
            ]
            master = master.drop(columns=["__raw_text_for_id"])

        # normalize ids in master
        master = master.copy()
        master[self.patient_col] = master[self.patient_col].astype(str).str.strip()
        master[self.report_id_col] = master[self.report_id_col].astype(str).str.strip()

        label_cols = self._detect_label_columns(master)
        per_report = self._aggregate_labels_per_report(master, label_cols)

        merged = per_report.merge(
            ctx[[self.report_id_col, self.patient_col, "case_context"] + ([self.split_col] if self.split_col in ctx.columns else [])],
            on=[self.report_id_col, self.patient_col],
            how="inner",
            suffixes=("", "_ctx"),
        )

        if self.split_col in merged.columns and f"{self.split_col}_ctx" in merged.columns:
            merged[self.split_col] = merged[f"{self.split_col}_ctx"].fillna(merged[self.split_col])
            merged = merged.drop(columns=[f"{self.split_col}_ctx"])

        long_rows: List[Dict[str, object]] = []
        for _, row in tqdm(merged.iterrows(), total=len(merged), desc=self.tqdm_desc_facts, miniters=1):
            for lab in label_cols:
                v = int(row[lab])
                if v not in self.valid_fact_values:
                    continue
                if (v == 0) and (not self.include_negative_labels):
                    continue
                long_rows.append(
                    {
                        self.patient_col: str(row[self.patient_col]).strip(),
                        self.split_col: row[self.split_col] if self.split_col in merged.columns else None,
                        self.report_id_col: str(row[self.report_id_col]).strip(),
                        "case_context": row["case_context"],
                        "label_name": lab,
                        "label_value": v,
                    }
                )

        facts = pd.DataFrame(long_rows)

        if self.out_facts_path:
            outp = Path(self.out_facts_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            facts.to_csv(outp, index=False)

        return facts


    # Step 3: QA

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
        return bool(self.negation_re.search(sent))

    def _is_sentence_normality(self, sent: str) -> bool:
        return bool(self.normality_re.search(sent))

    def _extract_compositional_spans(self, case_context: str) -> str:
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

    def _is_negated_mention(self, sent: str, match_start: int, match_end: int) -> bool:
        w = self.device_negation_window_chars
        lo = max(0, match_start - w)
        hi = min(len(sent), match_end + w)
        window = sent[lo:hi]
        if self.removal_re.search(window):
            return True
        if self.negation_re.search(window):
            return True
        return False

    def _detect_present_devices(self, device_sents: List[str]) -> List[str]:
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

    def _build_qa_from_facts_long(self, facts: pd.DataFrame) -> pd.DataFrame:
        required = {
            self.patient_col,
            self.report_id_col,
            "case_context",
            "label_name",
            "label_value",
        }
        missing = required - set(facts.columns)
        if missing:
            raise ValueError(f"facts_long is missing columns: {missing}")

        facts = facts.copy()

        if self.split_col not in facts.columns:
            facts[self.split_col] = None

        # Harden types WITHOUT dropping non-numeric patient ids
        facts[self.report_id_col] = facts[self.report_id_col].astype(str).str.strip()
        facts[self.patient_col] = facts[self.patient_col].astype(str).str.strip()
        facts = facts[facts[self.report_id_col] != ""].copy()

        groups = facts.groupby(self.report_id_col, sort=False)

        out_rows: List[Dict[str, object]] = []

        for report_id, g in tqdm(groups, desc=self.tqdm_desc_qa, total=groups.ngroups, miniters=1):
            report_id = str(report_id).strip()
            seen_keys: Set[Tuple[str, str, str]] = set()

            g0 = g.iloc[0]
            patient_id = str(g0[self.patient_col]).strip()
            split = g0[self.split_col] if self.split_col in g.columns else None

            case_context = self._normalize_text(g0["case_context"])
            if case_context.lower() == "nan":
                case_context = ""

            # 1) label_presence (robust aggregation)
            label_qas_added = 0
            agg_map: Dict[str, Dict[str, object]] = {}

            gg = g[["label_name", "label_value"]].copy()
            gg = gg[~gg["label_name"].isna()]
            gg["label_name"] = gg["label_name"].astype(str).str.strip()
            gg = gg[gg["label_name"] != ""]

            if not gg.empty:
                gg["__lab_norm"] = gg["label_name"].map(self._normalize_label_token)
                gg["__v"] = gg["label_value"].map(self._safe_int_label_value)
                gg = gg[~gg["__v"].isna()]
                gg["__v"] = gg["__v"].astype(int)
                gg = gg[gg["__v"].isin([0, 1, 3])]

                for _, r in gg.iterrows():
                    lab_raw = str(r["label_name"]).strip()
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

                    key = (report_id, "label_presence", question)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    out_rows.append(
                        {
                            self.patient_col: patient_id,
                            self.split_col: split,
                            self.report_id_col: report_id,
                            "qa_type": "label_presence",
                            "case_context": case_context,
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

            # 2) normality_statement
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

                    key = (report_id, "normality_statement", question)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.patient_col: patient_id,
                                self.split_col: split,
                                self.report_id_col: report_id,
                                "qa_type": "normality_statement",
                                "case_context": case_context,
                                "question": question,
                                "answer_target": answer_target,
                                "claim_text": claim_text,
                                "supporting_span": norm_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

            # 3) multi_label_conjunction
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

                        key = (report_id, "multi_label_conjunction", f"{question}|{a}|{b}")
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        out_rows.append(
                            {
                                self.patient_col: patient_id,
                                self.split_col: split,
                                self.report_id_col: report_id,
                                "qa_type": "multi_label_conjunction",
                                "case_context": case_context,
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

                            key = (report_id, "multi_label_conjunction", f"{question}|{a}|{b}")
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)

                            out_rows.append(
                                {
                                    self.patient_col: patient_id,
                                    self.split_col: split,
                                    self.report_id_col: report_id,
                                    "qa_type": "multi_label_conjunction",
                                    "case_context": case_context,
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

            # 4) key_findings
            if self.add_key_findings and case_context:
                findings_span = self._extract_key_findings(case_context)
                if findings_span:
                    question = "What are the key imaging findings?"
                    key = (report_id, "key_findings", question)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.patient_col: patient_id,
                                self.split_col: split,
                                self.report_id_col: report_id,
                                "qa_type": "key_findings",
                                "case_context": case_context,
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

                if self.add_devices_locations and dev_sents:
                    dev_span = " ".join(dev_sents[:2]).strip()
                    if dev_span and self._is_extractive(dev_span, case_context):
                        question = "What devices are present and where are they located?"
                        key = (report_id, "devices_locations", question)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            out_rows.append(
                                {
                                    self.patient_col: patient_id,
                                    self.split_col: split,
                                    self.report_id_col: report_id,
                                    "qa_type": "devices_locations",
                                    "case_context": case_context,
                                    "question": question,
                                    "answer_target": dev_span,
                                    "claim_text": dev_span,
                                    "supporting_span": dev_span,
                                    "label_name": None,
                                    "label_value": None,
                                }
                            )

                if self.add_device_presence and dev_sents:
                    present_devices = self._detect_present_devices(dev_sents)
                    dev_added = 0
                    for dev_name in present_devices:
                        question = f"Is a {dev_name} present?"
                        key = (report_id, "device_presence", question)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        out_rows.append(
                            {
                                self.patient_col: patient_id,
                                self.split_col: split,
                                self.report_id_col: report_id,
                                "qa_type": "device_presence",
                                "case_context": case_context,
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
                            f"Where does the {dev_label} terminate?" if dev_label else "Where does the line or tube terminate?"
                        )
                        span_hash = self._stable_hash(s)
                        key = (report_id, "device_tip_location", f"{question}|{span_hash}")
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        out_rows.append(
                            {
                                self.patient_col: patient_id,
                                self.split_col: split,
                                self.report_id_col: report_id,
                                "qa_type": "device_tip_location",
                                "case_context": case_context,
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

            # 6) temporal_change
            if self.add_temporal_change and case_context:
                tmp_span = self._extract_temporal_change(case_context)
                if tmp_span:
                    question = "What interval changes are described?"
                    key = (report_id, "temporal_change", question)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.patient_col: patient_id,
                                self.split_col: split,
                                self.report_id_col: report_id,
                                "qa_type": "temporal_change",
                                "case_context": case_context,
                                "question": question,
                                "answer_target": tmp_span,
                                "claim_text": tmp_span,
                                "supporting_span": tmp_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

            # 7) compositional_finding
            if self.add_compositional_findings and case_context:
                comp_span = self._extract_compositional_spans(case_context)
                if comp_span:
                    question = "Describe the abnormal finding."
                    key = (report_id, "compositional_finding", f"{question}|{self._stable_hash(comp_span)}")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.patient_col: patient_id,
                                self.split_col: split,
                                self.report_id_col: report_id,
                                "qa_type": "compositional_finding",
                                "case_context": case_context,
                                "question": question,
                                "answer_target": comp_span,
                                "claim_text": comp_span,
                                "supporting_span": comp_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

            # 8) limitation_uncertainty
            if self.add_limitation_uncertainty and case_context:
                u_span = self._extract_uncertainty_spans(case_context)
                if u_span:
                    question = "What limitation or uncertainty is described?"
                    key = (report_id, "limitation_uncertainty", f"{question}|{self._stable_hash(u_span)}")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_rows.append(
                            {
                                self.patient_col: patient_id,
                                self.split_col: split,
                                self.report_id_col: report_id,
                                "qa_type": "limitation_uncertainty",
                                "case_context": case_context,
                                "question": question,
                                "answer_target": u_span,
                                "claim_text": u_span,
                                "supporting_span": u_span,
                                "label_name": None,
                                "label_value": None,
                            }
                        )

        qa = pd.DataFrame(out_rows)

        # robust empty return (prevents KeyError and makes downstream predictable)
        if qa.empty:
            qa = pd.DataFrame(
                columns=[
                    self.patient_col,
                    self.split_col,
                    self.report_id_col,
                    "qa_type",
                    "case_context",
                    "question",
                    "answer_target",
                    "claim_text",
                    "supporting_span",
                    "label_name",
                    "label_value",
                ]
            )
        else:
            qa[self.report_id_col] = qa[self.report_id_col].astype(str).str.strip()
            qa[self.patient_col] = qa[self.patient_col].astype(str).str.strip()

        if self.out_qa_path:
            outp = Path(self.out_qa_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            qa.to_csv(outp, index=False)

        return qa



    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        facts = self._build_facts_long()
        qa = self._build_qa_from_facts_long(facts)
        return facts, qa







@dataclass(frozen=True)
class WeakAlignmentRowChexpert:
    qa_row_id: int
    report_id: str
    deid_patient_id: str          # IMPORTANT: keep raw string, do NOT coerce to int
    deid_patient_id_int64: int    # stable hash -> int64 (helper for downstream if needed)
    split: str
    qa_type: str
    claim_text: str
    retrieved_chunk_ids_json: str
    support_weights_json: str
    embed_scores_json: str
    lexical_scores_json: str
    combined_scores_json: str


class ChexpertPlusRStarWeakEvidenceAligner:
    """
    CheXpert-Plus external dataset: Step 6B-equivalent

    Fix for your Step 6C mismatch:
      - DO NOT coerce deid_patient_id to numeric. Many CheXpert ids are strings like "patient00001".
      - Splits are created patient-wise on the RAW deid_patient_id string.
      - Parquet writes deid_patient_id as STRING (and also writes a stable int64 hash column).

    Inputs:
      - qa_csv: chexpertplus_qa_step3.csv
          required: report_id, deid_patient_id, qa_type, claim_text
      - chunks_parquet: chunks.parquet (for lexical scoring)
      - faiss_index: faiss_Rstar.index
      - chunk_ids_npy: chunk_ids.npy aligned with FAISS rows
      - rstar_query_encoder_ckpt: query encoder weights

    Output:
      - out_parquet: one row per QA row with retrieved chunk ids + weights
      - patient_split_map_chexpertplus.json next to out_parquet (string patient_id -> split)
      - manifest_step6b_chexpertplus.json next to out_parquet
    """

    def __init__(
        self,
        qa_csv: str | Path,
        chunks_parquet: str | Path,
        faiss_index: str | Path,
        chunk_ids_npy: str | Path,
        out_parquet: str | Path,
        # R* query encoder
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        rstar_query_encoder_ckpt: Optional[str | Path] = None,
        # tokenization/encoding
        max_query_tokens: int = 128,
        qa_batch_size: int = 256,
        # retrieval
        top_k: int = 20,
        # scoring
        alpha: float = 0.85,
        temperature: float = 0.07,
        # columns (CheXpert-Plus)
        claim_col: str = "claim_text",
        report_id_col: str = "report_id",
        patient_col: str = "deid_patient_id",
        qa_type_col: str = "qa_type",
        split_col: str = "split",
        chunk_id_col: str = "chunk_id",
        chunk_text_col: str = "chunk_text",
        # determinism/runtime
        seed: int = 13,
        device: str = "cuda",
        tqdm_desc: str = "CheXpert Step 6B: R* weak evidence alignment",
        qa_chunksize_rows: int = 50_000,
        parquet_compression: str = "zstd",
        use_lexical: bool = True,
        # split params
        split_seed: int = 13,
        frac_train: float = 0.75,
        frac_valid: float = 0.10,
        frac_test: float = 0.15,
        # IO knobs
        write_patient_split_map_json: bool = True,
    ) -> None:
        self.qa_csv = Path(qa_csv)
        self.chunks_parquet = Path(chunks_parquet)
        self.faiss_index_path = Path(faiss_index)
        self.chunk_ids_npy = Path(chunk_ids_npy)
        self.out_parquet = Path(out_parquet)

        self.model_name = str(model_name)
        self.rstar_query_encoder_ckpt = Path(rstar_query_encoder_ckpt) if rstar_query_encoder_ckpt else None

        self.max_query_tokens = int(max_query_tokens)
        self.qa_batch_size = int(qa_batch_size)

        self.top_k = int(top_k)

        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.claim_col = str(claim_col)
        self.report_id_col = str(report_id_col)
        self.patient_col = str(patient_col)
        self.qa_type_col = str(qa_type_col)
        self.split_col = str(split_col)

        self.chunk_id_col = str(chunk_id_col)
        self.chunk_text_col = str(chunk_text_col)

        self.seed = int(seed)
        self.device = str(device)
        self.tqdm_desc = str(tqdm_desc)
        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_compression = str(parquet_compression)
        self.use_lexical = bool(use_lexical)

        self.split_seed = int(split_seed)
        self.frac_train = float(frac_train)
        self.frac_valid = float(frac_valid)
        self.frac_test = float(frac_test)

        self.write_patient_split_map_json = bool(write_patient_split_map_json)

        if not (0.0 < self.frac_train < 1.0 and 0.0 < self.frac_valid < 1.0 and 0.0 < self.frac_test < 1.0):
            raise ValueError("Split fractions must be in (0,1).")
        s = self.frac_train + self.frac_valid + self.frac_test
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1. Got {s}")

        self._word_re = re.compile(r"[a-z0-9]+")
        self._pa_schema: Optional[pa.Schema] = None

        self._tokenizer: Optional[AutoTokenizer] = None
        self._query_model: Optional[AutoModel] = None

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

    # ---------- patient id helpers ----------

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _patient_key(self, pid) -> str:
        # canonical raw patient id string used everywhere (split map + parquet)
        return self._safe_str(pid).strip()

    def _patient_to_int64(self, pid_raw: str) -> int:
        # stable hash -> signed positive int64 range (fits parquet int64)
        if not pid_raw:
            return -1
        h = hashlib.md5(pid_raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
        u = int(h, 16)
        return int(u % (2**63 - 1))



    def _make_patient_splits(self, patient_ids_raw: List[str]) -> Dict[str, str]:
        """
        Deterministic patient-wise split on RAW patient_id strings.
        Uses floor-based boundaries to keep counts stable.
        """
        uniq = sorted({p for p in patient_ids_raw if p})
        rng = np.random.RandomState(self.split_seed)
        perm = rng.permutation(len(uniq)).tolist()

        n = len(uniq)
        n_train = int(math.floor(self.frac_train * n))
        n_valid = int(math.floor(self.frac_valid * n))
        # remainder to test
        _ = n - n_train - n_valid  # n_test implicit

        idx_train = set(perm[:n_train])
        idx_valid = set(perm[n_train : n_train + n_valid])

        out: Dict[str, str] = {}
        for i, pid in enumerate(uniq):
            if i in idx_train:
                out[pid] = "train"
            elif i in idx_valid:
                out[pid] = "valid"
            else:
                out[pid] = "test"
        return out


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

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x) if len(x) else x
        ex = np.exp(x)
        s = float(np.sum(ex))
        if s <= 0 or len(x) == 0:
            return np.ones_like(x, dtype=np.float32) / float(max(len(x), 1))
        return ex / s

    def _weights_from_scores(self, combined: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            w = np.zeros_like(combined, dtype=np.float32)
            if len(w) > 0:
                w[int(np.argmax(combined))] = 1.0
            return w
        return self._softmax(combined / float(self.temperature)).astype(np.float32)



    def _get_tokenizer_and_query_model(self) -> Tuple[AutoTokenizer, AutoModel]:
        if self._tokenizer is not None and self._query_model is not None:
            return self._tokenizer, self._query_model

        tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model = AutoModel.from_pretrained(self.model_name, ignore_mismatched_sizes=True)

        if self.rstar_query_encoder_ckpt is not None:
            ckpt = torch.load(str(self.rstar_query_encoder_ckpt), map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing:
                print(f"[WARN] Missing keys when loading R* query encoder: {len(missing)}")
            if unexpected:
                print(f"[WARN] Unexpected keys when loading R* query encoder: {len(unexpected)}")

        model.eval()
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        self._tokenizer = tok
        self._query_model = model
        return tok, model

    def _mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _encode_queries(self, texts: List[str], pbar: Optional["tqdm"] = None) -> np.ndarray:
        tok, model = self._get_tokenizer_and_query_model()

        embs: List[np.ndarray] = []
        for i in range(0, len(texts), self.qa_batch_size):
            batch = texts[i : i + self.qa_batch_size]
            batch = [self._normalize_text(x) for x in batch]

            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_query_tokens,
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

        if not embs:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack(embs)


    # evidence loading

    def _load_chunks_table(self) -> pd.DataFrame:
        cols = [self.chunk_id_col, self.chunk_text_col]
        df = pd.read_parquet(self.chunks_parquet, columns=cols)
        df[self.chunk_id_col] = pd.to_numeric(df[self.chunk_id_col], errors="coerce")
        df = df[~df[self.chunk_id_col].isna()].copy()
        df[self.chunk_id_col] = df[self.chunk_id_col].astype(np.int64)
        df[self.chunk_text_col] = df[self.chunk_text_col].astype(str)
        df = df.sort_values(self.chunk_id_col).reset_index(drop=True)
        return df

    def _load_faiss_and_ids(self) -> Tuple[faiss.Index, np.ndarray]:
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")
        if not self.chunk_ids_npy.exists():
            raise FileNotFoundError(f"chunk_ids.npy not found: {self.chunk_ids_npy}")

        index = faiss.read_index(str(self.faiss_index_path))
        faiss_chunk_ids = np.load(str(self.chunk_ids_npy)).astype(np.int64)

        if index.ntotal != len(faiss_chunk_ids):
            raise RuntimeError(
                f"FAISS index ntotal ({index.ntotal}) != len(chunk_ids.npy) ({len(faiss_chunk_ids)})."
            )
        return index, faiss_chunk_ids



    def _init_writer(self) -> pq.ParquetWriter:
        self.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        self._pa_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("report_id", pa.string()),
                ("deid_patient_id", pa.string()),
                ("deid_patient_id_int64", pa.int64()),
                ("split", pa.string()),
                ("qa_type", pa.string()),
                ("claim_text", pa.string()),
                ("retrieved_chunk_ids_json", pa.string()),
                ("support_weights_json", pa.string()),
                ("embed_scores_json", pa.string()),
                ("lexical_scores_json", pa.string()),
                ("combined_scores_json", pa.string()),
            ]
        )
        return pq.ParquetWriter(str(self.out_parquet), schema=self._pa_schema, compression=self.parquet_compression)

    def _coerce_batch_df_for_parquet(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        assert self._pa_schema is not None

        batch_df["qa_row_id"] = pd.to_numeric(batch_df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        batch_df["deid_patient_id"] = batch_df["deid_patient_id"].astype(str)
        batch_df["deid_patient_id_int64"] = (
            pd.to_numeric(batch_df["deid_patient_id_int64"], errors="coerce").fillna(-1).astype(np.int64)
        )

        for c in self._pa_schema.names:
            if c in {"qa_row_id", "deid_patient_id_int64"}:
                continue
            batch_df[c] = batch_df[c].astype(str)

        return batch_df


    def run(self) -> None:
        if not self.qa_csv.exists():
            raise FileNotFoundError(f"QA CSV not found: {self.qa_csv}")
        if not self.chunks_parquet.exists():
            raise FileNotFoundError(f"chunks.parquet not found: {self.chunks_parquet}")

        # Load lexical texts
        chunks_df = self._load_chunks_table()
        chunk_text_map: Dict[int, str] = {
            int(cid): str(txt)
            for cid, txt in zip(chunks_df[self.chunk_id_col].tolist(), chunks_df[self.chunk_text_col].tolist())
        }

        # Load FAISS + aligned chunk ids
        index, faiss_chunk_ids = self._load_faiss_and_ids()

        # Count csv rows for progress
        total_rows = sum(1 for _ in open(self.qa_csv, "r", encoding="utf-8", errors="ignore")) - 1
        pbar = tqdm(total=max(total_rows, 0), desc=self.tqdm_desc, unit="rows", miniters=1)

        writer = self._init_writer()
        assert self._pa_schema is not None

        # Pass 1: build patient split map (RAW strings)
        patient_set: set[str] = set()
        for df0 in pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows):
            if self.patient_col not in df0.columns:
                raise ValueError(f"QA CSV missing required column: {self.patient_col}")
            for x in df0[self.patient_col].tolist():
                k = self._patient_key(x)
                if k:
                    patient_set.add(k)

        patient_to_split = self._make_patient_splits(sorted(patient_set))

        if self.write_patient_split_map_json:
            split_map_path = self.out_parquet.parent / "patient_split_map_chexpertplus.json"
            split_map_path.write_text(json.dumps(patient_to_split, indent=2), encoding="utf-8")
            print(f"Wrote: {split_map_path}")

        # Pass 2: streaming alignment
        pbar.reset(total=max(total_rows, 0))
        qa_row_id = 0

        try:
            for df in pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows):
                for c in [self.report_id_col, self.patient_col, self.qa_type_col, self.claim_col]:
                    if c not in df.columns:
                        raise ValueError(f"QA CSV missing required column: {c}")

                df = df.copy()
                df[self.report_id_col] = df[self.report_id_col].astype(str).str.strip()
                df[self.qa_type_col] = df[self.qa_type_col].astype(str).str.strip()
                df[self.claim_col] = df[self.claim_col].astype(str)

                # patient raw string + split
                pid_raw_list = [self._patient_key(x) for x in df[self.patient_col].tolist()]
                df[self.patient_col] = pid_raw_list
                df[self.split_col] = [patient_to_split.get(pid, "train") for pid in pid_raw_list]
                df["__pid_int64"] = [self._patient_to_int64(pid) for pid in pid_raw_list]

                claims = df[self.claim_col].tolist()
                q_emb = self._encode_queries(claims, pbar=pbar)

                D, I = index.search(q_emb, self.top_k)  # (B,k)

                out_rows: List[Dict[str, object]] = []
                for i in range(len(df)):
                    report_id = str(df.iloc[i][self.report_id_col]).strip()
                    patient_id_raw = str(df.iloc[i][self.patient_col]).strip()
                    patient_id_int64 = int(df.iloc[i]["__pid_int64"])
                    split = str(df.iloc[i][self.split_col]).strip() or "train"
                    qa_type = str(df.iloc[i][self.qa_type_col]).strip()
                    claim_text = str(df.iloc[i][self.claim_col])
                    claim_norm = self._normalize_text(claim_text)

                    if not claim_norm:
                        out_rows.append(
                            asdict(
                                WeakAlignmentRowChexpert(
                                    qa_row_id=qa_row_id,
                                    report_id=report_id,
                                    deid_patient_id=patient_id_raw,
                                    deid_patient_id_int64=patient_id_int64,
                                    split=split,
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

                    faiss_rows = I[i].tolist()
                    embed_scores = D[i].astype(np.float32).tolist()

                    retrieved_ids: List[int] = []
                    retrieved_texts: List[str] = []
                    valid_embed_scores: List[float] = []

                    for row_idx, sc in zip(faiss_rows, embed_scores):
                        if int(row_idx) < 0:
                            continue
                        cid = int(faiss_chunk_ids[int(row_idx)])
                        retrieved_ids.append(cid)
                        retrieved_texts.append(chunk_text_map.get(cid, ""))
                        valid_embed_scores.append(float(sc))

                    e = np.asarray(valid_embed_scores, dtype=np.float32)
                    if self.use_lexical:
                        l = np.asarray([float(self._lexical_overlap(claim_norm, t)) for t in retrieved_texts], dtype=np.float32)
                    else:
                        l = np.zeros_like(e, dtype=np.float32)

                    combined = self.alpha * e + (1.0 - self.alpha) * l
                    w = self._weights_from_scores(combined)

                    out_rows.append(
                        asdict(
                            WeakAlignmentRowChexpert(
                                qa_row_id=qa_row_id,
                                report_id=report_id,
                                deid_patient_id=patient_id_raw,
                                deid_patient_id_int64=patient_id_int64,
                                split=split,
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
            "step": "6B_chexpertplus",
            "qa_csv": str(self.qa_csv.resolve()),
            "chunks_parquet": str(self.chunks_parquet.resolve()),
            "faiss_index": str(self.faiss_index_path.resolve()),
            "chunk_ids_npy": str(self.chunk_ids_npy.resolve()),
            "out_parquet": str(self.out_parquet.resolve()),
            "query_encoder": {
                "model_name": self.model_name,
                "checkpoint": str(self.rstar_query_encoder_ckpt) if self.rstar_query_encoder_ckpt else None,
            },
            "split": {
                "patient_wise": True,
                "seed": self.split_seed,
                "frac_train": self.frac_train,
                "frac_valid": self.frac_valid,
                "frac_test": self.frac_test,
                "patient_split_map_json": str((self.out_parquet.parent / "patient_split_map_chexpertplus.json").resolve())
                if self.write_patient_split_map_json
                else None,
            },
            "outputs": {
                "patient_id_columns": ["deid_patient_id (string)", "deid_patient_id_int64 (stable hash int64)"]
            },
            "params": {
                "max_query_tokens": self.max_query_tokens,
                "top_k": self.top_k,
                "alpha": self.alpha,
                "temperature": self.temperature,
                "seed": self.seed,
                "device": self.device,
                "qa_chunksize_rows": self.qa_chunksize_rows,
                "qa_batch_size": self.qa_batch_size,
                "use_lexical": self.use_lexical,
            },
        }

        manifest_path = self.out_parquet.parent / "manifest_step6b_chexpertplus.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {manifest_path}")





@dataclass(frozen=True)
class RetrieverSupervisionRowChexpert:
    qa_row_id: int
    deid_patient_id: int
    split: str
    report_id: str
    qa_type: str
    claim_text: str
    pos_chunk_ids_json: str
    pos_weights_json: str


@dataclass(frozen=True)
class VerifierSupervisionRowChexpert:
    qa_row_id: int
    deid_patient_id: int
    split: str
    report_id: str
    qa_type: str
    case_context: str
    question: str
    answer_target: str
    claim_text: str
    evidence_chunk_ids_json: str
    evidence_weights_json: str
    soft_support_score: float


class ChexpertPlusStep6CSupervisionBuilder:
    """
    CheXpert-Plus Step 6C-equivalent.

    Reads:
      - QA CSV (Step 3 output)
      - weak_alignment parquet (Step 6B output)

    Joins ONLY by qa_row_id order (streaming), and optionally checks (report_id, patient_id) consistency.
    """

    def __init__(
        self,
        qa_csv: str | Path,
        weak_alignment_parquet: str | Path,
        out_dir: str | Path,
        # columns in qa_csv
        patient_col: str = "deid_patient_id",
        report_id_col: str = "report_id",
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
        # optional filtering
        allowed_splits: Optional[List[str]] = None,
        # progress
        tqdm_desc: str = "CheXpert Step 6C: building supervision from R* alignments",
        parquet_compression: str = "zstd",
        # safety
        strict_key_check: bool = True,
    ) -> None:
        self.qa_csv = Path(qa_csv)
        self.weak_alignment_parquet = Path(weak_alignment_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.patient_col = str(patient_col)
        self.report_id_col = str(report_id_col)
        self.qa_type_col = str(qa_type_col)
        self.claim_col = str(claim_col)
        self.case_context_col = str(case_context_col)
        self.question_col = str(question_col)
        self.answer_col = str(answer_col)

        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_batch_rows = int(parquet_batch_rows)

        self.top_k = int(top_k) if top_k is not None else None
        self.min_weight = float(min_weight)

        self.allowed_splits = allowed_splits[:] if allowed_splits is not None else None

        self.tqdm_desc = str(tqdm_desc)
        self.parquet_compression = str(parquet_compression)
        self.strict_key_check = bool(strict_key_check)

        self.retriever_out = self.out_dir / "retriever_supervision.parquet"
        self.verifier_out = self.out_dir / "verifier_supervision.parquet"
        self.manifest_out = self.out_dir / "manifest_step6c_chexpertplus.json"

        self._retr_schema: Optional[pa.Schema] = None
        self._ver_schema: Optional[pa.Schema] = None


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

    def _patient_to_int64(self, pid) -> int:
        """
        Deterministic int64 for patient id:
          - numeric -> int
          - otherwise -> md5 hash -> int64 in [0, 2^63-2]
        """
        s = self._safe_str(pid).strip()
        if not s:
            return -1
        try:
            v = int(float(s))
            return int(v)
        except Exception:
            pass
        h = hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:16]
        u = int(h, 16)
        return int(u % (2**63 - 1))

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


    def _init_retriever_writer(self) -> pq.ParquetWriter:
        self._retr_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("deid_patient_id", pa.int64()),
                ("split", pa.string()),
                ("report_id", pa.string()),
                ("qa_type", pa.string()),
                ("claim_text", pa.string()),
                ("pos_chunk_ids_json", pa.string()),
                ("pos_weights_json", pa.string()),
            ]
        )
        return pq.ParquetWriter(str(self.retriever_out), schema=self._retr_schema, compression=self.parquet_compression)

    def _init_verifier_writer(self) -> pq.ParquetWriter:
        self._ver_schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("deid_patient_id", pa.int64()),
                ("split", pa.string()),
                ("report_id", pa.string()),
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
        return pq.ParquetWriter(str(self.verifier_out), schema=self._ver_schema, compression=self.parquet_compression)

    def _to_table(self, df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
        for c in schema.names:
            if c not in df.columns:
                df[c] = None

        df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        df["deid_patient_id"] = pd.to_numeric(df["deid_patient_id"], errors="coerce").fillna(-1).astype(np.int64)

        for c in schema.names:
            if c in {"qa_row_id", "deid_patient_id", "soft_support_score"}:
                continue
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

        total_rows = sum(1 for _ in open(self.qa_csv, "r", encoding="utf-8", errors="ignore")) - 1
        pbar = tqdm(total=max(total_rows, 0), desc=self.tqdm_desc, unit="rows", miniters=1)

        retr_writer = self._init_retriever_writer()
        ver_writer = self._init_verifier_writer()
        assert self._retr_schema is not None
        assert self._ver_schema is not None

        pf = pq.ParquetFile(str(self.weak_alignment_parquet))

        align_cols = [
            "qa_row_id",
            "report_id",
            "deid_patient_id",
            "split",
            "qa_type",
            "claim_text",
            "retrieved_chunk_ids_json",
            "support_weights_json",
            "combined_scores_json",
        ]

        needed_qa_cols = [
            self.patient_col,
            self.report_id_col,
            self.qa_type_col,
            self.claim_col,
            self.case_context_col,
            self.question_col,
            self.answer_col,
        ]

        qa_reader = pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows)

        qa_row_id_expected = 0
        align_row_id_expected = 0

        align_iter = pf.iter_batches(batch_size=self.parquet_batch_rows, columns=align_cols)

        written_rows = 0
        dropped_by_split = 0
        empty_chunks = 0

        try:
            for qa_chunk in qa_reader:
                for c in needed_qa_cols:
                    if c not in qa_chunk.columns:
                        raise ValueError(f"QA CSV missing required column: {c}")

                qa_chunk = qa_chunk.copy()

                qa_chunk["__pid_raw"] = qa_chunk[self.patient_col].astype(str)
                qa_chunk["__pid_int"] = qa_chunk["__pid_raw"].map(self._patient_to_int64).astype(np.int64)

                # keep rows that have non-empty patient id
                mask_ok = qa_chunk["__pid_raw"].map(lambda x: bool(self._safe_str(x).strip()))
                qa_chunk = qa_chunk[mask_ok].copy()

                qa_chunk[self.report_id_col] = qa_chunk[self.report_id_col].astype(str).str.strip()
                qa_chunk[self.qa_type_col] = qa_chunk[self.qa_type_col].astype(str)
                qa_chunk[self.claim_col] = qa_chunk[self.claim_col].astype(str)
                qa_chunk[self.case_context_col] = qa_chunk[self.case_context_col].astype(str)
                qa_chunk[self.question_col] = qa_chunk[self.question_col].astype(str)
                qa_chunk[self.answer_col] = qa_chunk[self.answer_col].astype(str)

                n = len(qa_chunk)
                if n == 0:
                    empty_chunks += 1
                    continue

                # Pull exactly n alignment rows
                align_rows: List[pd.DataFrame] = []
                need = n
                while need > 0:
                    batch = next(align_iter, None)
                    if batch is None:
                        raise RuntimeError(f"Alignment parquet ended early. Still needed {need} rows.")
                    align_rows.append(batch.to_pandas())
                    need -= len(align_rows[-1])

                align_df = pd.concat(align_rows, ignore_index=True)
                if len(align_df) > n:
                    align_df = align_df.iloc[:n].copy()

                first_id = int(align_df.iloc[0]["qa_row_id"])
                if first_id != align_row_id_expected:
                    raise RuntimeError(
                        f"Alignment qa_row_id mismatch: expected {align_row_id_expected}, got {first_id}."
                    )

                retr_rows: List[Dict[str, object]] = []
                ver_rows: List[Dict[str, object]] = []

                for i in range(n):
                    qa_row_id = qa_row_id_expected
                    al_row = align_df.iloc[i]

                    split = self._safe_str(al_row.get("split", "train")).strip() or "train"
                    if self.allowed_splits is not None and split not in self.allowed_splits:
                        dropped_by_split += 1
                        qa_row_id_expected += 1
                        align_row_id_expected += 1
                        continue

                    report_id_qa = self._safe_str(qa_chunk.iloc[i][self.report_id_col]).strip()
                    pid_int_qa = int(qa_chunk.iloc[i]["__pid_int"])

                    report_id_align = self._safe_str(al_row.get("report_id", "")).strip()
                    pid_raw_align = self._safe_str(al_row.get("deid_patient_id", "")).strip()
                    pid_int_align = self._patient_to_int64(pid_raw_align)

                    if self.strict_key_check:
                        if report_id_align and report_id_qa and (report_id_align != report_id_qa):
                            raise RuntimeError(
                                f"report_id mismatch at qa_row_id={qa_row_id}: align='{report_id_align}' vs qa='{report_id_qa}'"
                            )
                        if pid_int_align != pid_int_qa:
                            raise RuntimeError(
                                f"patient_id mismatch at qa_row_id={qa_row_id}: "
                                f"align_raw='{pid_raw_align}' align_int={pid_int_align} vs qa_int={pid_int_qa}"
                            )

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

                    qa_type = self._safe_str(qa_chunk.iloc[i][self.qa_type_col])
                    claim_text = self._safe_str(qa_chunk.iloc[i][self.claim_col])
                    case_context = self._safe_str(qa_chunk.iloc[i][self.case_context_col])
                    question = self._safe_str(qa_chunk.iloc[i][self.question_col])
                    answer_target = self._safe_str(qa_chunk.iloc[i][self.answer_col])

                    retr_rows.append(
                        asdict(
                            RetrieverSupervisionRowChexpert(
                                qa_row_id=qa_row_id,
                                deid_patient_id=pid_int_qa,
                                split=split,
                                report_id=report_id_qa,
                                qa_type=qa_type,
                                claim_text=claim_text,
                                pos_chunk_ids_json=json.dumps(ids),
                                pos_weights_json=json.dumps([float(x) for x in w]),
                            )
                        )
                    )

                    ver_rows.append(
                        asdict(
                            VerifierSupervisionRowChexpert(
                                qa_row_id=qa_row_id,
                                deid_patient_id=pid_int_qa,
                                split=split,
                                report_id=report_id_qa,
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

                if retr_rows:
                    retr_writer.write_table(self._to_table(pd.DataFrame(retr_rows), self._retr_schema))
                    ver_writer.write_table(self._to_table(pd.DataFrame(ver_rows), self._ver_schema))
                    written_rows += len(retr_rows)

                pbar.update(n)

        finally:
            pbar.close()
            retr_writer.close()
            ver_writer.close()

        manifest = {
            "step": "6C_chexpertplus",
            "inputs": {
                "qa_csv": str(self.qa_csv.resolve()),
                "weak_alignment_parquet": str(self.weak_alignment_parquet.resolve()),
            },
            "outputs": {
                "retriever_supervision": str(self.retriever_out.resolve()),
                "verifier_supervision": str(self.verifier_out.resolve()),
            },
            "split": {
                "source": "weak_alignment_parquet.split",
                "allowed_splits": self.allowed_splits if self.allowed_splits is not None else ["train", "valid", "test"],
            },
            "patient_id": {
                "qa_source_col": self.patient_col,
                "align_source_col": "deid_patient_id",
                "output_col": "deid_patient_id (int64)",
                "mapping": "numeric->int, else md5 hash -> int64",
                "strict_key_check": self.strict_key_check,
            },
            "params": {
                "qa_chunksize_rows": self.qa_chunksize_rows,
                "parquet_batch_rows": self.parquet_batch_rows,
                "top_k": self.top_k,
                "min_weight": self.min_weight,
                "soft_support_scalar": "max(combined_scores)",
                "parquet_compression": self.parquet_compression,
            },
            "stats": {
                "written_rows": int(written_rows),
                "dropped_by_split": int(dropped_by_split),
                "empty_qa_chunks": int(empty_chunks),
            },
        }
        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.retriever_out}")
        print(f"Wrote: {self.verifier_out}")
        print(f"Wrote: {self.manifest_out}")
        print(f"Rows written: {written_rows:,} | Dropped by split filter: {dropped_by_split:,} | Empty QA chunks: {empty_chunks:,}")







@dataclass(frozen=True)
class VerifierMaterializedRowChexpert:
    qa_row_id: int
    deid_patient_id: int
    split: str
    report_id: str
    qa_type: str
    case_context: str
    question: str
    answer_target: str
    claim_text: str

    evidence_chunk_ids_json: str
    evidence_weights_json: str

    evidence_texts_json: str
    evidence_text_concat: str
    evidence_text: str  # alias of evidence_text_concat

    soft_support_score: float


class ChexpertPlusStep7VerifierMaterializer:
    """
    CheXpert-Plus: Step 7 — materialize verifier training/eval examples (materialize evidence text)

    Inputs:
      - step6c_supervision_rstar_chexpertplus/verifier_supervision.parquet (from CheXpert Step 6C)
      - chunks.parquet (frozen evidence snapshot)

    Output (in out_dir):
      - verifier_train_materialized.parquet
      - manifest_step7_chexpertplus.json

    Notes:
      - Does NOT change split. Uses split already present in verifier_supervision.
      - Builds chunk_id -> chunk_text lookup from chunks.parquet (in memory).
      - Streams verifier_supervision parquet (stable memory).
      - Writes evidence_text (alias of evidence_text_concat) for dataset compatibility.
    """

    def __init__(
        self,
        verifier_supervision_parquet: str | Path,
        chunks_parquet: str | Path,
        out_dir: str | Path,
        # column names in chunks.parquet
        chunk_id_col: str = "chunk_id",
        chunk_text_col: str = "chunk_text",
        # how to build evidence string
        max_evidence_chunks: Optional[int] = None,
        min_evidence_weight: float = 0.0,
        evidence_sep: str = "\n\n---\n\n",
        # streaming
        parquet_batch_rows: int = 50_000,
        # progress
        tqdm_desc: str = "CheXpert Step 7: materializing verifier evidence text",
        parquet_compression: str = "zstd",
        # column names in verifier_supervision.parquet (CheXpert Step 6C)
        qa_row_id_col: str = "qa_row_id",
        patient_col: str = "deid_patient_id",
        split_col: str = "split",
        report_id_col: str = "report_id",
        qa_type_col: str = "qa_type",
        case_context_col: str = "case_context",
        question_col: str = "question",
        answer_col: str = "answer_target",
        claim_col: str = "claim_text",
        evidence_ids_col: str = "evidence_chunk_ids_json",
        evidence_w_col: str = "evidence_weights_json",
        soft_support_col: str = "soft_support_score",
    ) -> None:
        self.verifier_supervision_parquet = Path(verifier_supervision_parquet)
        self.chunks_parquet = Path(chunks_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_id_col = str(chunk_id_col)
        self.chunk_text_col = str(chunk_text_col)

        self.max_evidence_chunks = int(max_evidence_chunks) if max_evidence_chunks is not None else None
        self.min_evidence_weight = float(min_evidence_weight)
        self.evidence_sep = str(evidence_sep)

        self.parquet_batch_rows = int(parquet_batch_rows)
        self.tqdm_desc = str(tqdm_desc)
        self.parquet_compression = str(parquet_compression)

        self.qa_row_id_col = str(qa_row_id_col)
        self.patient_col = str(patient_col)
        self.split_col = str(split_col)
        self.report_id_col = str(report_id_col)
        self.qa_type_col = str(qa_type_col)
        self.case_context_col = str(case_context_col)
        self.question_col = str(question_col)
        self.answer_col = str(answer_col)
        self.claim_col = str(claim_col)
        self.evidence_ids_col = str(evidence_ids_col)
        self.evidence_w_col = str(evidence_w_col)
        self.soft_support_col = str(soft_support_col)

        self.out_parquet = self.out_dir / "verifier_train_materialized.parquet"
        self.manifest_out = self.out_dir / "manifest_step7_chexpertplus.json"

        self._schema: Optional[pa.Schema] = None
        self._chunk_map: Optional[Dict[int, str]] = None

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

    def _normalize_text(self, s: str) -> str:
        if s is None:
            return ""
        x = str(s)
        if x.lower() == "nan":
            return ""
        return " ".join(x.split()).strip()

    def _filter_and_truncate(self, ids: List[int], w: List[float]) -> Tuple[List[int], List[float]]:
        if self.max_evidence_chunks is not None:
            ids = ids[: self.max_evidence_chunks]
            w = w[: self.max_evidence_chunks]

        if self.min_evidence_weight > 0:
            keep = [(i, ww) for i, ww in zip(ids, w) if float(ww) >= self.min_evidence_weight]
            ids = [int(i) for i, _ in keep]
            w = [float(ww) for _, ww in keep]

        return ids, w

    def _load_chunk_map(self) -> Dict[int, str]:
        if not self.chunks_parquet.exists():
            raise FileNotFoundError(f"chunks.parquet not found: {self.chunks_parquet}")

        cols = [self.chunk_id_col, self.chunk_text_col]
        df = pd.read_parquet(self.chunks_parquet, columns=cols)

        df[self.chunk_id_col] = pd.to_numeric(df[self.chunk_id_col], errors="coerce")
        df = df[~df[self.chunk_id_col].isna()].copy()
        df[self.chunk_id_col] = df[self.chunk_id_col].astype(np.int64)

        df[self.chunk_text_col] = df[self.chunk_text_col].astype(str).map(self._normalize_text)

        chunk_map: Dict[int, str] = {}
        for cid, txt in zip(df[self.chunk_id_col].tolist(), df[self.chunk_text_col].tolist()):
            chunk_map[int(cid)] = txt

        if not chunk_map:
            raise RuntimeError("chunk_map is empty after loading chunks.parquet.")
        return chunk_map

    def _init_writer(self) -> pq.ParquetWriter:
        self._schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("deid_patient_id", pa.int64()),
                ("split", pa.string()),
                ("report_id", pa.string()),
                ("qa_type", pa.string()),
                ("case_context", pa.string()),
                ("question", pa.string()),
                ("answer_target", pa.string()),
                ("claim_text", pa.string()),
                ("evidence_chunk_ids_json", pa.string()),
                ("evidence_weights_json", pa.string()),
                ("evidence_texts_json", pa.string()),
                ("evidence_text_concat", pa.string()),
                ("evidence_text", pa.string()),
                ("soft_support_score", pa.float32()),
            ]
        )
        return pq.ParquetWriter(
            str(self.out_parquet),
            schema=self._schema,
            compression=self.parquet_compression,
        )

    def _coerce_df(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._schema is not None

        for c in self._schema.names:
            if c not in df.columns:
                df[c] = None

        df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)
        df["deid_patient_id"] = pd.to_numeric(df["deid_patient_id"], errors="coerce").fillna(-1).astype(np.int64)

        for c in [
            "split",
            "report_id",
            "qa_type",
            "case_context",
            "question",
            "answer_target",
            "claim_text",
            "evidence_chunk_ids_json",
            "evidence_weights_json",
            "evidence_texts_json",
            "evidence_text_concat",
            "evidence_text",
        ]:
            df[c] = df[c].astype(str)

        df["soft_support_score"] = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).astype(
            np.float32
        )
        return df

    def run(self) -> None:
        if not self.verifier_supervision_parquet.exists():
            raise FileNotFoundError(f"verifier_supervision.parquet not found: {self.verifier_supervision_parquet}")

        self._chunk_map = self._load_chunk_map()
        assert self._chunk_map is not None

        pf = pq.ParquetFile(str(self.verifier_supervision_parquet))

        in_cols = [
            self.qa_row_id_col,
            self.patient_col,
            self.split_col,
            self.report_id_col,
            self.qa_type_col,
            self.case_context_col,
            self.question_col,
            self.answer_col,
            self.claim_col,
            self.evidence_ids_col,
            self.evidence_w_col,
            self.soft_support_col,
        ]

        writer = self._init_writer()
        assert self._schema is not None

        total_rows = pf.metadata.num_rows if pf.metadata is not None else 0
        pbar = tqdm(total=total_rows, desc=self.tqdm_desc, unit="rows", miniters=1)

        written = 0
        try:
            for batch in pf.iter_batches(batch_size=self.parquet_batch_rows, columns=in_cols):
                b = batch

                qa_row_ids = b.column(b.schema.get_field_index(self.qa_row_id_col)).to_pylist()
                patient_ids = b.column(b.schema.get_field_index(self.patient_col)).to_pylist()
                splits = b.column(b.schema.get_field_index(self.split_col)).to_pylist()
                report_ids = b.column(b.schema.get_field_index(self.report_id_col)).to_pylist()
                qa_types = b.column(b.schema.get_field_index(self.qa_type_col)).to_pylist()
                case_contexts = b.column(b.schema.get_field_index(self.case_context_col)).to_pylist()
                questions = b.column(b.schema.get_field_index(self.question_col)).to_pylist()
                answer_targets = b.column(b.schema.get_field_index(self.answer_col)).to_pylist()
                claim_texts = b.column(b.schema.get_field_index(self.claim_col)).to_pylist()
                ids_jsons = b.column(b.schema.get_field_index(self.evidence_ids_col)).to_pylist()
                w_jsons = b.column(b.schema.get_field_index(self.evidence_w_col)).to_pylist()
                scores = b.column(b.schema.get_field_index(self.soft_support_col)).to_pylist()

                out_rows: List[Dict[str, object]] = []
                n = len(qa_row_ids)

                for i in range(n):
                    qa_row_id = int(qa_row_ids[i]) if qa_row_ids[i] is not None else -1
                    patient_id = int(patient_ids[i]) if patient_ids[i] is not None else -1
                    split = self._safe_str(splits[i])

                    report_id = self._safe_str(report_ids[i])
                    qa_type = self._safe_str(qa_types[i])
                    case_context = self._safe_str(case_contexts[i])
                    question = self._safe_str(questions[i])
                    answer_target = self._safe_str(answer_targets[i])
                    claim_text = self._safe_str(claim_texts[i])

                    ids = self._loads_json_list(self._safe_str(ids_jsons[i]))
                    w = self._loads_json_list(self._safe_str(w_jsons[i]))

                    ids = [int(x) for x in ids if str(x).strip() != ""]
                    w = [float(x) for x in w]

                    if len(ids) != len(w):
                        m = min(len(ids), len(w))
                        ids, w = ids[:m], w[:m]

                    ids, w = self._filter_and_truncate(ids, w)
                    if len(ids) != len(w):
                        m = min(len(ids), len(w))
                        ids, w = ids[:m], w[:m]

                    texts: List[str] = []
                    for cid in ids:
                        txt = self._chunk_map.get(int(cid), "")
                        texts.append(self._normalize_text(txt))

                    evidence_concat = self.evidence_sep.join([t for t in texts if t.strip() != ""])
                    evidence_text = evidence_concat

                    try:
                        soft_support_score = float(scores[i]) if scores[i] is not None else 0.0
                    except Exception:
                        soft_support_score = 0.0

                    out_rows.append(
                        asdict(
                            VerifierMaterializedRowChexpert(
                                qa_row_id=qa_row_id,
                                deid_patient_id=patient_id,
                                split=split,
                                report_id=report_id,
                                qa_type=qa_type,
                                case_context=case_context,
                                question=question,
                                answer_target=answer_target,
                                claim_text=claim_text,
                                evidence_chunk_ids_json=json.dumps(ids),
                                evidence_weights_json=json.dumps([float(x) for x in w]),
                                evidence_texts_json=json.dumps(texts),
                                evidence_text_concat=evidence_concat,
                                evidence_text=evidence_text,
                                soft_support_score=soft_support_score,
                            )
                        )
                    )

                out_df = pd.DataFrame(out_rows)
                out_df = self._coerce_df(out_df)
                table = pa.Table.from_pandas(out_df, schema=self._schema, preserve_index=False)
                writer.write_table(table)

                written += len(out_df)
                pbar.update(len(out_df))

        finally:
            pbar.close()
            writer.close()

        manifest = {
            "step": "7_chexpertplus",
            "inputs": {
                "verifier_supervision_parquet": str(self.verifier_supervision_parquet.resolve()),
                "chunks_parquet": str(self.chunks_parquet.resolve()),
            },
            "outputs": {
                "verifier_train_materialized": str(self.out_parquet.resolve()),
                "manifest_step7": str(self.manifest_out.resolve()),
            },
            "params": {
                "max_evidence_chunks": self.max_evidence_chunks,
                "min_evidence_weight": self.min_evidence_weight,
                "evidence_sep": self.evidence_sep,
                "parquet_batch_rows": self.parquet_batch_rows,
                "parquet_compression": self.parquet_compression,
                "evidence_text_column": "evidence_text (alias of evidence_text_concat)",
            },
            "stats": {
                "rows_written": int(written),
                "num_chunks_in_map": int(len(self._chunk_map)),
            },
        }
        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.out_parquet}  (rows={written:,})")
        print(f"Wrote: {self.manifest_out}")








def create_fixed_chexpertplus_test_eval_parquet(verifier_materialized_parquet: str | Path, out_parquet: str | Path, *,
    split_value: str = "test", parquet_batch_rows: int = 50_000, parquet_compression: str = "zstd",
    seed: int = 13, write_manifest_json: bool = True) -> Dict[str, object]:
    """
    Phase 1 — Freeze the CheXpert-Plus evaluation object.

    Reads CheXpert Step 7 output:
      step7_verifier_materialized_rstar_chexpertplus/verifier_train_materialized.parquet

    Filters to split == "test" (split fixed upstream in CheXpert Step 6B and carried through 6C/7),
    writes ONE immutable test artifact:
      chexpertplus_test_eval.parquet

    Output columns:
      qa_row_id, deid_patient_id, split, qa_type, case_context, question, answer_target, claim_text,
      evidence_chunk_ids_json, evidence_text, soft_support_score,
      report_length_bucket, device_present, uncertainty_present, label_vs_extractive

    Notes:
      - Does NOT change any split.
      - Does NOT touch retrieval; just freezes the test set.
      - Streaming read/write (keeps memory stable).
    """
    _ = np.random.RandomState(seed)  # deterministic if you ever add sampling later (currently unused)

    in_path = Path(verifier_materialized_parquet)
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"verifier_materialized_parquet not found: {in_path}")

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
        r"limited|limitation|poorly assessed|not well seen|underlying .* cannot be excluded"
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
        if t in {"label_presence", "multi_label_conjunction", "device_presence"}:
            return "label"
        return "extractive"

    required_in_cols = [
        "qa_row_id",
        "deid_patient_id",
        "split",
        "qa_type",
        "case_context",
        "question",
        "answer_target",
        "claim_text",
        "evidence_chunk_ids_json",
        "evidence_text",
        "soft_support_score",
    ]

    pf = pq.ParquetFile(str(in_path))

    out_schema = pa.schema(
        [
            ("qa_row_id", pa.int64()),
            ("deid_patient_id", pa.int64()),
            ("split", pa.string()),
            ("qa_type", pa.string()),
            ("case_context", pa.string()),
            ("question", pa.string()),
            ("answer_target", pa.string()),
            ("claim_text", pa.string()),
            ("evidence_chunk_ids_json", pa.string()),
            ("evidence_text", pa.string()),
            ("soft_support_score", pa.float32()),
            ("report_length_bucket", pa.string()),
            ("device_present", pa.int8()),
            ("uncertainty_present", pa.int8()),
            ("label_vs_extractive", pa.string()),
        ]
    )

    writer = pq.ParquetWriter(
        str(out_path),
        schema=out_schema,
        compression=parquet_compression,
    )

    total_rows = pf.metadata.num_rows if pf.metadata is not None else 0
    pbar = tqdm(total=total_rows, desc="Phase 1: freezing chexpertplus_test_eval.parquet", unit="rows", miniters=1)

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
            d["deid_patient_id"] = pd.to_numeric(d["deid_patient_id"], errors="coerce").fillna(-1).astype(np.int64)
            d["soft_support_score"] = pd.to_numeric(d["soft_support_score"], errors="coerce").fillna(0.0).astype(
                np.float32
            )

            for c in [
                "split",
                "qa_type",
                "case_context",
                "question",
                "answer_target",
                "claim_text",
                "evidence_chunk_ids_json",
                "evidence_text",
            ]:
                d[c] = d[c].astype(str)

            d["report_length_bucket"] = d["case_context"].map(report_length_bucket).astype(str)
            d["device_present"] = d["case_context"].map(lambda x: 1 if device_re.search(str(x)) else 0).astype(np.int8)
            d["uncertainty_present"] = d["case_context"].map(lambda x: 1 if uncertainty_re.search(str(x)) else 0).astype(
                np.int8
            )
            d["label_vs_extractive"] = d["qa_type"].map(label_vs_extractive_flag).astype(str)

            d = d[
                [
                    "qa_row_id",
                    "deid_patient_id",
                    "split",
                    "qa_type",
                    "case_context",
                    "question",
                    "answer_target",
                    "claim_text",
                    "evidence_chunk_ids_json",
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

    manifest = {
        "phase": "1",
        "dataset": "chexpertplus",
        "input": str(in_path.resolve()),
        "output": str(out_path.resolve()),
        "filter": {"split_value": split_value},
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
    }

    if write_manifest_json:
        manifest_path = out_path.parent / "manifest_phase1_test_eval_chexpertplus.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}  (rows={kept:,})")
    if write_manifest_json:
        print(f"Wrote: {out_path.parent / 'manifest_phase1_test_eval_chexpertplus.json'}")

    return manifest






def print_test_eval_stats(test_eval_parquet: str):
    df = pd.read_parquet(test_eval_parquet)

    # pick the id column that exists
    if "subject_id" in df.columns:
        id_col = "subject_id"
        id_name = "subjects"
    elif "deid_patient_id" in df.columns:
        id_col = "deid_patient_id"
        id_name = "patients"
    else:
        id_col = None
        id_name = "ids"

    print("==== BASIC INFO ====")
    print("Rows:", len(df))
    if id_col is not None:
        print(f"Unique {id_name}:", df[id_col].nunique())
    else:
        print("Unique ids: N/A (no subject_id or deid_patient_id column found)")
    print("QA types:", df["qa_type"].nunique() if "qa_type" in df.columns else "N/A")
    print()

    def vc(col):
        if col not in df.columns:
            print(f"(missing column: {col})\n")
            return
        print(df[col].value_counts(dropna=False))
        print()

    print("==== QA TYPE DISTRIBUTION ====")
    vc("qa_type")

    print("==== REPORT LENGTH BUCKET ====")
    vc("report_length_bucket")

    print("==== DEVICE PRESENT ====")
    vc("device_present")

    print("==== UNCERTAINTY PRESENT ====")
    vc("uncertainty_present")

    print("==== LABEL vs EXTRACTIVE ====")
    vc("label_vs_extractive")

    print("==== TARGET STATISTICS (soft_support_score) ====")
    if "soft_support_score" in df.columns:
        s = pd.to_numeric(df["soft_support_score"], errors="coerce")
        print("Min:", float(s.min()))
        print("Max:", float(s.max()))
        print("Mean:", float(s.mean()))
        print("Std:", float(s.std()))
    else:
        print("(missing column: soft_support_score)")






if __name__ == "__main__":

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_chexpertData = params["Preprocessing"]["base_path_chexpertData"]
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]




    ''' step 1 '''
    # builder = ChexpertPlusCaseContextBuilder(
    #     master_csv=os.path.join(base_path_chexpertData, "df_chexpert_plus_240401.csv"),
    #     out_path=os.path.join(base_path_chexpertData, "chexpertplus_case_contexts.csv"),
    # )
    # out_df = builder.build_case_context_table()
    # print(out_df[["report_id", "deid_patient_id", "case_context"]].head(3))
    # print("Rows:", len(out_df))



    ''' steps 2 & 3 '''
    # builder = ChexpertPlusFactsAndQABuilder(
    #     master_csv=os.path.join(base_path_chexpertData, "df_chexpert_plus_240401.csv"),
    #     case_context_csv=os.path.join(base_path_chexpertData, "chexpertplus_case_contexts.csv"),
    #     out_facts_path=os.path.join(base_path_chexpertData, "chexpertplus_facts_long.csv"),
    #     out_qa_path=os.path.join(base_path_chexpertData, "chexpertplus_qa_step3.csv"),
    # )
    # facts_df, qa_df = builder.run()
    # print("facts:", len(facts_df), "qa:", len(qa_df))




    ''' step 6B '''
    # builder = ChexpertPlusRStarWeakEvidenceAligner(
    #     qa_csv=os.path.join(base_path_chexpertData, "chexpertplus_qa_step3.csv"),
    #     chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
    #     faiss_index=os.path.join(base_path_evidenceData, "step6_faiss_rstar", "faiss_Rstar.index"),
    #     chunk_ids_npy=os.path.join(base_path_evidenceData, "step6_faiss_rstar", "chunk_ids.npy"),
    #     out_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step6b_Rstar_chexpertplus.parquet"),
    #     model_name=params["Preprocessing"]["tokenizer_name"],
    #     rstar_query_encoder_ckpt=model_weights_path,
    #     top_k=20,
    #     alpha=0.85,
    #     temperature=0.07,
    #     seed=13,
    #     device="cuda",
    #     use_lexical=True,
    #     # split
    #     split_seed=13,
    #     frac_train=0.75,
    #     frac_valid=0.10,
    #     frac_test=0.15,
    # )
    # builder.run()




    ''' step 6C '''
    # builder = ChexpertPlusStep6CSupervisionBuilder(
    #     qa_csv=os.path.join(base_path_chexpertData, "chexpertplus_qa_step3.csv"),
    #     weak_alignment_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step6b_Rstar_chexpertplus.parquet"),
    #     out_dir=os.path.join(base_path_evidenceData, "step6c_supervision_rstar_chexpertplus"),
    #     top_k=20,
    #     min_weight=0.0,
    #     qa_chunksize_rows=50_000,
    #     parquet_batch_rows=50_000,
    #     # allowed_splits=["train", "valid"],
    # )
    # builder.run()





    ''' step 7 '''
    # builder = ChexpertPlusStep7VerifierMaterializer(
    #     verifier_supervision_parquet=os.path.join(
    #         base_path_evidenceData,
    #         "step6c_supervision_rstar_chexpertplus/verifier_supervision.parquet",
    #     ),
    #     chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
    #     out_dir=os.path.join(base_path_evidenceData, "step7_verifier_materialized_rstar_chexpertplus"),
    #     max_evidence_chunks=20,
    #     min_evidence_weight=0.0,
    #     parquet_batch_rows=50_000,
    #     parquet_compression="zstd",
    # )
    # builder.run()





    ''' Creating the final mimic test parquet '''
    # in_parquet = os.path.join(base_path_evidenceData, "step7_verifier_materialized_rstar_chexpertplus",
    #     "verifier_train_materialized.parquet")
    #
    # out_parquet = os.path.join(base_path_evidenceData, "phase1_fixed_test_eval_chexpertplus",
    #     "chexpertplus_test_eval.parquet")
    #
    # create_fixed_chexpertplus_test_eval_parquet(verifier_materialized_parquet=in_parquet,
    #     out_parquet=out_parquet, split_value="test", parquet_batch_rows=50_000,
    #     parquet_compression="zstd", seed=13, write_manifest_json=True)

    print_test_eval_stats(test_eval_parquet=os.path.join(base_path_evidenceData, "phase1_fixed_test_eval_chexpertplus", "chexpertplus_test_eval.parquet"))
