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
from sklearn.model_selection import train_test_split
from collections import Counter

import warnings
warnings.filterwarnings('ignore')




def patient_split(csv_path, outpath, patient_col="subject_id",
                  train_frac=0.75, valid_frac=0.10, test_frac=0.15, seed=42):
    assert abs((train_frac + valid_frac + test_frac) - 1.0) < 1e-9

    df = pd.read_csv(csv_path)

    # Get unique patients as a plain NumPy array (avoid pyarrow-backed arrays)
    patients = (
        df[patient_col]
        .dropna()
        .astype(str)          # optional but helps if IDs are mixed types
        .unique()
    )
    patients = np.asarray(patients)

    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(n * train_frac))
    n_valid = int(round(n * valid_frac))
    # ensure exact total
    n_test = n - n_train - n_valid

    train_patients = set(patients[:n_train])
    valid_patients = set(patients[n_train:n_train + n_valid])
    test_patients  = set(patients[n_train + n_valid:])

    # Overwrite split column (ignore existing values)
    df["split"] = "unassigned"
    pid = df[patient_col].astype(str)

    df.loc[pid.isin(train_patients), "split"] = "train"
    df.loc[pid.isin(valid_patients), "split"] = "valid"
    df.loc[pid.isin(test_patients),  "split"] = "test"

    df.to_csv(outpath, index=False)

    print("Patient counts:", n, "| train/valid/test:", n_train, n_valid, n_test)
    print(df["split"].value_counts(dropna=False, normalize=True))




class ChexpertPlusCaseContextBuilder:
    """
    Step 1 for CheXpert-Plus

    Goal:
      Build one stable report-level case_context per report/study unit, using
      MIMIC-like cleaning and section-aware reconstruction.

    Output columns:
      - subject_id
      - report_id
      - split (if available)
      - n_images
      - report
      - case_context

    Main fixes vs older version:
      1) Reconstructs a real report-like raw text with explicit section headers.
      2) Uses stable study/report identity:
           preferred: subject_id + patient_report_date_order
           fallback : subject_id + md5(normalized_report_text)
      3) Avoids flattening sections before section parsing.
      4) Removes CheXpert-Plus anonymization/footer artifacts.
      5) Keeps cleaning behavior aligned with the MIMIC pipeline.
    """

    def __init__(
        self,
        master_csv: str,
        out_path: Optional[str] = None,
        encoding: str = "utf-8",
        tqdm_desc: str = "Building CheXpert-Plus case_context",
        min_lines_fallback: int = 2,
        # core columns
        report_col: str = "report",
        patient_col: str = "subject_id",
        image_col: str = "path_to_image",
        split_col: str = "split",
        report_order_col: str = "patient_report_date_order",
        # optional section columns
        findings_col: str = "section_findings",
        impression_col: str = "section_impression",
        narrative_col: str = "section_narrative",
        history_col: str = "section_history",
        clinical_history_col: str = "section_clinical_history",
        comparison_col: str = "section_comparison",
        technique_col: str = "section_technique",
        procedure_comments_col: str = "section_procedure_comments",
        summary_col: str = "section_summary",
        accession_col: str = "section_accession_number",
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
        self.report_order_col = str(report_order_col)

        self.findings_col = str(findings_col)
        self.impression_col = str(impression_col)
        self.narrative_col = str(narrative_col)
        self.history_col = str(history_col)
        self.clinical_history_col = str(clinical_history_col)
        self.comparison_col = str(comparison_col)
        self.technique_col = str(technique_col)
        self.procedure_comments_col = str(procedure_comments_col)
        self.summary_col = str(summary_col)
        self.accession_col = str(accession_col)

        # Whole-line administrative headers
        self.header_re = re.compile(
            r"^\s*(FINAL REPORT|EXAMINATION|REASON FOR EXAMINATION|INDICATION|HISTORY|TECHNIQUE|COMPARISON|"
            r"CLINICAL INDICATION|CLINICAL HISTORY|DATE|TIME|DICTATED BY|SIGNED|PROVIDER|ATTENDING|EXAM|STUDY|"
            r"NARRATIVE|ACCESSION NUMBER|SUMMARY)\s*[:\-]?\s*$",
            re.IGNORECASE,
        )

        # True section headers when they appear as their own line
        self.section_hdr_re = re.compile(
            r"^\s*(FINDINGS|IMPRESSION|CONCLUSION|RESULTS?)\s*:\s*$",
            re.IGNORECASE,
        )

        # Inline section tokens that must be stripped from final kept text
        self.inline_section_token_re = re.compile(
            r"\b(FINDINGS|IMPRESSION|CONCLUSION|RESULTS?|NARRATIVE|SUMMARY|ACCESSION NUMBER|HISTORY|COMPARISON|TECHNIQUE)\s*:\s*",
            re.IGNORECASE,
        )

        # Acquisition/view boilerplate
        self.acq_boilerplate_re = re.compile(
            r"\b(portable|pa\b|ap\b|lateral|upright|semi[-\s]?erect|frontal|"
            r"views?|radiograph|radiographs|cxr|chest\s+x[-\s]?ray|film|1\s*view|2\s*views?)\b",
            re.IGNORECASE,
        )

        # Strong clinical signal tokens
        self.clinical_keep_re = re.compile(
            r"\b(no|without|absent|present|seen|shows?|demonstrates?|consistent|cannot|exclude|likely|possible|"
            r"opacity|opacities|effusion|pneumothorax|pneumomediastinum|edema|atelectasis|consolidation|"
            r"airspace|vascular congestion|cardiomediastinal|cardiac|cardiomegaly|hilar|mediastinum|"
            r"tube|line|catheter|picc|central venous|cvc|endotracheal|ett|nasogastric|ng tube|enteric|dobhoff|"
            r"chest tube|pigtail|swan[-\s]?ganz|pa catheter|sv[co]|cavoatrial|junction|right atrium|carina|"
            r"hemidiaphragm|rib|nodule|mass|calcified lymph nodes|tracheostomy|aeration|atelectatic)\b",
            re.IGNORECASE,
        )

        # Pure communication/admin tail lines
        self.drop_tail_re = re.compile(
            r"\b(results were communicated|communicated to|discussed with|called to|by phone|phone|paged|notified)\b",
            re.IGNORECASE,
        )

        # CheXpert-Plus specific footer noise
        self.footer_noise_re = re.compile(
            r"(this report has been anonymized\..*?$|"
            r"i have personally reviewed the images.*?$)",
            re.IGNORECASE | re.DOTALL,
        )

        self.underscore_re = re.compile(r"_+")

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _normalize_for_hash(self, x: str) -> str:
        x = self._safe_str(x)
        x = x.replace("\r\n", "\n").replace("\r", "\n")
        x = re.sub(r"\s+", " ", x).strip().lower()
        return x

    def _stable_text_hash(self, x: str) -> str:
        return hashlib.md5(self._normalize_for_hash(x).encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _build_report_id(self, row: pd.Series) -> str:
        """
        Prefer a real study-level identifier:
          subject_id + patient_report_date_order

        Fallback:
          subject_id + md5(normalized raw report text)
        """
        patient_id = self._safe_str(row.get(self.patient_col, "")).strip()

        if self.report_order_col in row.index:
            order_val = self._safe_str(row.get(self.report_order_col, "")).strip()
            if order_val:
                return f"{patient_id}::ord::{order_val}"

        raw_report = self._safe_str(row.get(self.report_col, ""))
        return f"{patient_id}::hash::{self._stable_text_hash(raw_report)}"

    def _add_section(self, parts: List[str], header: str, text: str) -> None:
        text = self._safe_str(text).strip()
        if not text:
            return
        parts.append(f"{header}:")
        parts.append(text)

    def _compose_raw_report_text(self, row: pd.Series) -> str:
        """
        Reconstruct a report-like text object with explicit headers.
        This is crucial so that downstream cleaning behaves like the MIMIC pipeline.
        """
        parts: List[str] = []

        # Prefer explicit sections if they exist
        self._add_section(parts, "NARRATIVE", row.get(self.narrative_col, ""))
        self._add_section(parts, "CLINICAL HISTORY", row.get(self.clinical_history_col, ""))
        self._add_section(parts, "HISTORY", row.get(self.history_col, ""))
        self._add_section(parts, "COMPARISON", row.get(self.comparison_col, ""))
        self._add_section(parts, "TECHNIQUE", row.get(self.technique_col, ""))
        self._add_section(parts, "PROCEDURE COMMENTS", row.get(self.procedure_comments_col, ""))
        self._add_section(parts, "FINDINGS", row.get(self.findings_col, ""))
        self._add_section(parts, "IMPRESSION", row.get(self.impression_col, ""))
        self._add_section(parts, "SUMMARY", row.get(self.summary_col, ""))
        self._add_section(parts, "ACCESSION NUMBER", row.get(self.accession_col, ""))

        # If section fields are empty, fall back to full raw report
        if not parts:
            rep = self._safe_str(row.get(self.report_col, "")).strip()
            if rep:
                return rep

        return "\n\n".join(parts).strip()

    def _preclean_raw_report_text(self, raw_text: str) -> str:
        """
        Remove known dataset footer noise before line-based cleaning.
        """
        if not raw_text:
            return ""

        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = self.footer_noise_re.sub("", text)
        text = self.underscore_re.sub(" ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean_report_to_case_context(self, raw_text: str) -> str:
        """
        Same philosophy as MIMIC:

        - Prefer FINDINGS + IMPRESSION when present
        - Keep clinically meaningful narrative outside sections
        - Drop admin headers, timestamps, signatures, communication lines
        - Strip inline section tokens at the end
        """
        raw_text = self._preclean_raw_report_text(raw_text)
        if not raw_text:
            return ""

        lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
        if not lines:
            return ""

        # Parse true sections
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
            if re.search(r"\baccession number\b", s, re.IGNORECASE):
                return True
            if re.search(r"\banonymized\b", s, re.IGNORECASE):
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
            if re.search(r"\bi have personally reviewed the images\b", s, re.IGNORECASE):
                return False
            if re.search(r"\bthis report has been anonymized\b", s, re.IGNORECASE):
                return False
            if self.acq_boilerplate_re.search(s) and not self.clinical_keep_re.search(s):
                return False
            return True

        findings_lines = sections.get("FINDINGS", [])
        impression_lines = sections.get("IMPRESSION", [])

        # Base selection: prefer findings+impression if present, else all lines
        if findings_lines or impression_lines:
            base = []
            base.extend(findings_lines)
            base.extend(impression_lines)
        else:
            base = lines

        base_kept = [ln for ln in base if keep_line(ln)]

        # Add clinically rich non-section lines
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

        required = {self.patient_col, self.report_col, self.image_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in master CSV: {missing}")

        has_split = self.split_col in df.columns

        # Keep only needed columns
        keep_cols = [self.patient_col, self.report_col, self.image_col]
        if has_split:
            keep_cols.append(self.split_col)
        if self.report_order_col in df.columns:
            keep_cols.append(self.report_order_col)

        optional_cols = [
            self.narrative_col,
            self.clinical_history_col,
            self.history_col,
            self.comparison_col,
            self.technique_col,
            self.procedure_comments_col,
            self.findings_col,
            self.impression_col,
            self.summary_col,
            self.accession_col,
        ]
        for c in optional_cols:
            if c in df.columns:
                keep_cols.append(c)

        work = df[keep_cols].copy()
        work[self.patient_col] = work[self.patient_col].astype(str).str.strip()

        # Stable report_id
        report_ids: List[str] = []
        raw_texts: List[str] = []

        for _, row in tqdm(
            work.iterrows(),
            total=len(work),
            desc="Composing report_id and raw report text",
            miniters=1,
        ):
            report_ids.append(self._build_report_id(row))
            raw_texts.append(self._compose_raw_report_text(row))

        work["report_id"] = report_ids
        work["__raw_text"] = raw_texts

        # Collapse to one row per report_id
        agg_cols = {
            self.patient_col: "first",
            "__raw_text": "first",
            self.image_col: "count",
        }
        if has_split:
            agg_cols[self.split_col] = "first"

        uniq = work.groupby("report_id", as_index=False).agg(agg_cols)
        uniq = uniq.rename(columns={"__raw_text": self.report_col, self.image_col: "n_images"})

        # Build case_context
        case_contexts: List[str] = []
        for raw in tqdm(
            uniq[self.report_col].astype(str).tolist(),
            desc=self.tqdm_desc,
            total=len(uniq),
            miniters=1,
        ):
            case_contexts.append(self.clean_report_to_case_context(self._safe_str(raw)))

        uniq["case_context"] = case_contexts

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
    CheXpert-Plus external dataset: Step 2 + Step 3

    IMPORTANT:
      - CheXpert-Plus does NOT use MIMIC-style CheXpert label columns here.
      - Therefore Step 2 is NOT "aggregate 14 labels into facts_long".
      - Instead, Step 2 builds lightweight text-derived facts from case_context.
      - Step 3 builds QA rows from those report-level text facts.

    Inputs:
      - master_csv: chexpert_plus_reports.csv
      - case_context_csv: chexpertplus_case_contexts.csv

    Outputs:
      - facts_long: report-level text facts
      - qa_step3: QA rows derived from case_context / text facts

    subject_id is always treated as string.
    """

    def __init__(
        self,
        master_csv: str,
        case_context_csv: str,
        out_facts_path: Optional[str] = None,
        out_qa_path: Optional[str] = None,
        # columns
        split_col: str = "split",
        patient_col: str = "subject_id",
        image_col: str = "path_to_image",
        report_col: str = "report",
        report_id_col: str = "report_id",
        report_order_col: str = "patient_report_date_order",
        # step 3 knobs
        add_key_findings: bool = True,
        add_normality: bool = True,
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
        device_negation_window_chars: int = 45,
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
        self.report_order_col = str(report_order_col)

        self.add_key_findings = bool(add_key_findings)
        self.add_normality = bool(add_normality)
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
        self.device_negation_window_chars = int(device_negation_window_chars)

        self.tqdm_desc_facts = str(tqdm_desc_facts)
        self.tqdm_desc_qa = str(tqdm_desc_qa)

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

    def _compute_report_id(self, patient_id: str, report_order) -> str:
        pid = str(patient_id).strip()
        order = self._safe_str(report_order).strip()
        return f"{pid}::{order}"

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

    def _build_facts_long(self) -> pd.DataFrame:
        """
        Step 2 for CheXpert-Plus:
        build lightweight report-level text facts from case_context.
        No label columns are used.
        """
        ctx = pd.read_csv(self.case_context_csv)

        required_ctx = {self.report_id_col, self.patient_col, "case_context"}
        missing_ctx = required_ctx - set(ctx.columns)
        if missing_ctx:
            raise ValueError(f"Case-context CSV missing columns: {missing_ctx}")

        if self.split_col not in ctx.columns:
            ctx[self.split_col] = None

        ctx = ctx.copy()
        ctx[self.patient_col] = ctx[self.patient_col].astype(str).str.strip()
        ctx[self.report_id_col] = ctx[self.report_id_col].astype(str).str.strip()
        ctx["case_context"] = ctx["case_context"].map(self._safe_str)

        rows: List[Dict[str, object]] = []

        for _, row in tqdm(ctx.iterrows(), total=len(ctx), desc=self.tqdm_desc_facts, miniters=1):
            subject_id = str(row[self.patient_col]).strip()
            report_id = str(row[self.report_id_col]).strip()
            split = row[self.split_col]
            case_context = self._normalize_text(row["case_context"])

            if not case_context:
                continue

            norm_span = self._find_normality_spans(case_context)
            key_span = self._extract_key_findings(case_context)
            tmp_span = self._extract_temporal_change(case_context)
            comp_span = self._extract_compositional_spans(case_context)
            unc_span = self._extract_uncertainty_spans(case_context)

            device_sents = self._extract_device_sentences(case_context)
            tip_sents = self._extract_tip_location_spans(device_sents, case_context)
            present_devices = self._detect_present_devices(device_sents)

            if norm_span:
                rows.append(
                    {
                        self.patient_col: subject_id,
                        self.split_col: split,
                        self.report_id_col: report_id,
                        "case_context": case_context,
                        "fact_type": "normality_statement",
                        "fact_text": norm_span,
                    }
                )

            if key_span:
                rows.append(
                    {
                        self.patient_col: subject_id,
                        self.split_col: split,
                        self.report_id_col: report_id,
                        "case_context": case_context,
                        "fact_type": "key_findings",
                        "fact_text": key_span,
                    }
                )

            if tmp_span:
                rows.append(
                    {
                        self.patient_col: subject_id,
                        self.split_col: split,
                        self.report_id_col: report_id,
                        "case_context": case_context,
                        "fact_type": "temporal_change",
                        "fact_text": tmp_span,
                    }
                )

            if comp_span:
                rows.append(
                    {
                        self.patient_col: subject_id,
                        self.split_col: split,
                        self.report_id_col: report_id,
                        "case_context": case_context,
                        "fact_type": "compositional_finding",
                        "fact_text": comp_span,
                    }
                )

            if unc_span:
                rows.append(
                    {
                        self.patient_col: subject_id,
                        self.split_col: split,
                        self.report_id_col: report_id,
                        "case_context": case_context,
                        "fact_type": "limitation_uncertainty",
                        "fact_text": unc_span,
                    }
                )

            for dev_name in present_devices:
                rows.append(
                    {
                        self.patient_col: subject_id,
                        self.split_col: split,
                        self.report_id_col: report_id,
                        "case_context": case_context,
                        "fact_type": "device_presence",
                        "fact_text": dev_name,
                    }
                )

            for s in tip_sents:
                rows.append(
                    {
                        self.patient_col: subject_id,
                        self.split_col: split,
                        self.report_id_col: report_id,
                        "case_context": case_context,
                        "fact_type": "device_tip_location",
                        "fact_text": s,
                    }
                )

        facts = pd.DataFrame(rows)
        if facts.empty:
            facts = pd.DataFrame(
                columns=[
                    self.patient_col,
                    self.split_col,
                    self.report_id_col,
                    "case_context",
                    "fact_type",
                    "fact_text",
                ]
            )

        if self.out_facts_path:
            outp = Path(self.out_facts_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            facts.to_csv(outp, index=False)

        return facts

    def _build_qa_from_facts_long(self, facts: pd.DataFrame) -> pd.DataFrame:
        required = {
            self.patient_col,
            self.report_id_col,
            "case_context",
            "fact_type",
            "fact_text",
        }
        missing = required - set(facts.columns)
        if missing:
            raise ValueError(f"facts_long is missing columns: {missing}")

        facts = facts.copy()
        if self.split_col not in facts.columns:
            facts[self.split_col] = None

        facts[self.report_id_col] = facts[self.report_id_col].astype(str).str.strip()
        facts[self.patient_col] = facts[self.patient_col].astype(str).str.strip()
        facts["fact_type"] = facts["fact_type"].astype(str).str.strip()
        facts["fact_text"] = facts["fact_text"].map(self._safe_str)
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

            # normality_statement
            if self.add_normality:
                rows = g[g["fact_type"] == "normality_statement"]
                if len(rows) > 0:
                    norm_span = self._safe_str(rows.iloc[0]["fact_text"])
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

            # key_findings
            if self.add_key_findings:
                rows = g[g["fact_type"] == "key_findings"]
                if len(rows) > 0:
                    findings_span = self._safe_str(rows.iloc[0]["fact_text"])
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

            # device_presence
            if self.add_device_presence:
                rows = g[g["fact_type"] == "device_presence"]
                dev_added = 0
                for _, r in rows.iterrows():
                    dev_name = self._safe_str(r["fact_text"])
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

            # device_tip_location
            if self.add_device_tip_location:
                rows = g[g["fact_type"] == "device_tip_location"]
                tip_added = 0
                for _, r in rows.iterrows():
                    s = self._safe_str(r["fact_text"])
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

            # devices_locations
            if self.add_devices_locations:
                dev_rows = g[g["fact_type"].isin(["device_presence", "device_tip_location"])]
                if len(dev_rows) > 0:
                    dev_texts = dev_rows["fact_text"].astype(str).tolist()[:2]
                    dev_span = " ".join(dev_texts).strip()
                    if dev_span:
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

            # temporal_change
            if self.add_temporal_change:
                rows = g[g["fact_type"] == "temporal_change"]
                if len(rows) > 0:
                    tmp_span = self._safe_str(rows.iloc[0]["fact_text"])
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

            # compositional_finding
            if self.add_compositional_findings:
                rows = g[g["fact_type"] == "compositional_finding"]
                if len(rows) > 0:
                    comp_span = self._safe_str(rows.iloc[0]["fact_text"])
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

            # limitation_uncertainty
            if self.add_limitation_uncertainty:
                rows = g[g["fact_type"] == "limitation_uncertainty"]
                if len(rows) > 0:
                    u_span = self._safe_str(rows.iloc[0]["fact_text"])
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
class RetrieverSupervisionRow:
    qa_row_id: int
    subject_id: str
    split: str
    report_rel_path: str
    qa_type: str
    claim_text: str
    pos_chunk_ids_json: str
    pos_weights_json: str


@dataclass(frozen=True)
class VerifierSupervisionRow:
    qa_row_id: int
    subject_id: str
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


class CheXpertRadiopaediaStep4ExternalTestBuilder:
    """
    Single Step-4 class for CheXpert external test evaluation.

    Input:
      - chexpertplus_qa_step3.csv (already produced by Step 3)
      - Radiopaedia chunks.parquet

    Output (same schema intent as MIMIC Step 4C):
      - out_dir/retriever_supervision.parquet
      - out_dir/verifier_supervision.parquet
      - out_dir/manifest_step4_external_test.json

    Key rules:
      - Drops train/valid immediately. Only keeps split == test_split_value.
      - Assigns qa_row_id sequentially over ONLY test rows (0..N_test-1).
      - Writes split="test" for all outputs.
      - Reads CheXpert Step3 `report_id` but writes it into output `report_rel_path`.
      - subject_id stays STRING everywhere.
    """

    def __init__(
        self,
        qa_csv: str | Path,
        chunks_parquet: str | Path,
        out_dir: str | Path,
        *,
        split_col: str = "split",
        test_split_value: str = "test",
        # columns in qa_csv (CheXpert Step3)
        subject_id_col: str = "subject_id",
        report_path_col: str = "report_id",
        qa_type_col: str = "qa_type",
        claim_col: str = "claim_text",
        case_context_col: str = "case_context",
        question_col: str = "question",
        answer_col: str = "answer_target",
        # chunks parquet columns
        chunk_id_col: str = "chunk_id",
        chunk_text_col: str = "chunk_text",
        # embedder
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
        min_weight: float = 0.0,
        # caching
        cache_dir: Optional[str | Path] = None,
        faiss_index_name: str = "radiopaedia_flatip.index",
        chunk_embeds_name: str = "radiopaedia_chunk_embeds.npy",
        # streaming + writer
        qa_chunksize_rows: int = 50_000,
        parquet_compression: str = "zstd",
        seed: int = 13,
        device: str = "cuda",
        force_cpu_faiss: bool = True,
        # tqdm
        tqdm_desc: str = "CheXpert Step 4 (external test only)",
        prescan_total_test_rows: bool = True,   # set False to skip pre-scan
    ) -> None:
        self.qa_csv = Path(qa_csv)
        self.chunks_parquet = Path(chunks_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.split_col = str(split_col)
        self.test_split_value = str(test_split_value)

        self.subject_id_col = str(subject_id_col)
        self.report_path_col = str(report_path_col)
        self.qa_type_col = str(qa_type_col)
        self.claim_col = str(claim_col)
        self.case_context_col = str(case_context_col)
        self.question_col = str(question_col)
        self.answer_col = str(answer_col)

        self.chunk_id_col = str(chunk_id_col)
        self.chunk_text_col = str(chunk_text_col)

        self.model_name = str(model_name)
        self.max_query_tokens = int(max_query_tokens)
        self.max_doc_tokens = int(max_doc_tokens)

        self.top_k = int(top_k)
        self.qa_batch_size = int(qa_batch_size)
        self.doc_batch_size = int(doc_batch_size)

        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.min_weight = float(min_weight)

        self.cache_dir = Path(cache_dir) if cache_dir else (self.out_dir / "cache_step4_external")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path = self.cache_dir / faiss_index_name
        self.chunk_embeds_path = self.cache_dir / chunk_embeds_name

        self.qa_chunksize_rows = int(qa_chunksize_rows)
        self.parquet_compression = str(parquet_compression)

        self.seed = int(seed)
        self.device = str(device)
        self.force_cpu_faiss = bool(force_cpu_faiss)

        self.tqdm_desc = str(tqdm_desc)
        self.prescan_total_test_rows = bool(prescan_total_test_rows)

        self.retriever_out = self.out_dir / "retriever_supervision.parquet"
        self.verifier_out = self.out_dir / "verifier_supervision.parquet"
        self.manifest_out = self.out_dir / "manifest_step4_external_test.json"

        self._word_re = re.compile(r"[a-z0-9]+")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None

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

    def _filter_and_truncate(self, ids: List[int], w: List[float]) -> Tuple[List[int], List[float]]:
        ids = ids[: self.top_k]
        w = w[: self.top_k]
        if self.min_weight > 0:
            keep = [(i, ww) for i, ww in zip(ids, w) if float(ww) >= self.min_weight]
            ids = [int(i) for i, _ in keep]
            w = [float(ww) for _, ww in keep]
        return ids, w

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
            batch = [self._normalize_text(x) for x in texts[i : i + batch_size]]
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
        index = faiss.IndexFlatIP(dim)
        index.add(chunk_embeds)
        faiss.write_index(index, str(self.faiss_index_path))
        return index

    def _init_retriever_writer(self) -> Tuple[pq.ParquetWriter, pa.Schema]:
        schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("subject_id", pa.string()),   # FIX: string
                ("split", pa.string()),
                ("report_rel_path", pa.string()),
                ("qa_type", pa.string()),
                ("claim_text", pa.string()),
                ("pos_chunk_ids_json", pa.string()),
                ("pos_weights_json", pa.string()),
            ]
        )
        w = pq.ParquetWriter(str(self.retriever_out), schema=schema, compression=self.parquet_compression)
        return w, schema

    def _init_verifier_writer(self) -> Tuple[pq.ParquetWriter, pa.Schema]:
        schema = pa.schema(
            [
                ("qa_row_id", pa.int64()),
                ("subject_id", pa.string()),   # FIX: string
                ("split", pa.string()),
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
        w = pq.ParquetWriter(str(self.verifier_out), schema=schema, compression=self.parquet_compression)
        return w, schema

    def _to_table(self, df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
        for c in schema.names:
            if c not in df.columns:
                df[c] = None

        df["qa_row_id"] = pd.to_numeric(df["qa_row_id"], errors="coerce").fillna(-1).astype(np.int64)

        # subject_id MUST remain string
        if "subject_id" in df.columns:
            df["subject_id"] = df["subject_id"].astype(str)

        if "soft_support_score" in df.columns:
            df["soft_support_score"] = pd.to_numeric(df["soft_support_score"], errors="coerce").fillna(0.0).astype(
                np.float32
            )

        for c in schema.names:
            if schema.field(c).type == pa.string() and c in df.columns:
                df[c] = df[c].astype(str)

        return pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    def _estimate_total_test_rows(self) -> int:
        if not self.prescan_total_test_rows:
            return 0
        total = 0
        usecols = [self.split_col]
        try:
            for df in pd.read_csv(self.qa_csv, usecols=usecols, chunksize=200_000):
                df[self.split_col] = df[self.split_col].astype(str)
                total += int((df[self.split_col] == self.test_split_value).sum())
        except Exception:
            return 0
        return int(total)

    def run(self) -> None:
        if not self.qa_csv.exists():
            raise FileNotFoundError(f"QA CSV not found: {self.qa_csv}")
        if not self.chunks_parquet.exists():
            raise FileNotFoundError(f"chunks.parquet not found: {self.chunks_parquet}")

        # Load chunks + index once
        chunks_df = self._load_chunks_table()
        chunk_ids = chunks_df[self.chunk_id_col].to_numpy(dtype=np.int64)
        chunk_texts = chunks_df[self.chunk_text_col].tolist()

        chunk_embeds = self._build_or_load_chunk_embeddings(chunks_df)
        index = self._build_or_load_faiss_index(chunk_embeds)

        retr_writer, retr_schema = self._init_retriever_writer()
        ver_writer, ver_schema = self._init_verifier_writer()

        needed_cols = [
            self.split_col,
            self.subject_id_col,
            self.report_path_col,
            self.qa_type_col,
            self.claim_col,
            self.case_context_col,
            self.question_col,
            self.answer_col,
        ]

        qa_row_id = 0
        kept_total = 0

        total_test = self._estimate_total_test_rows()
        main_pbar = tqdm(
            total=total_test if total_test > 0 else None,
            desc=self.tqdm_desc,
            unit="rows",
            miniters=1,
        )

        # A second pbar that shows embedding throughput (helps you see GPU progress)
        embed_pbar = tqdm(
            total=total_test if total_test > 0 else None,
            desc="Embedding claims (test only)",
            unit="claims",
            miniters=1,
            leave=False,
        )

        try:
            for df in pd.read_csv(self.qa_csv, chunksize=self.qa_chunksize_rows):
                for c in needed_cols:
                    if c not in df.columns:
                        raise ValueError(f"CheXpert Step3 CSV missing required column: {c}")

                # keep test only
                df[self.split_col] = df[self.split_col].astype(str)
                df = df[df[self.split_col] == self.test_split_value]
                if len(df) == 0:
                    continue

                # keep subject_id as string (no dropping by numeric coercion)
                df[self.subject_id_col] = df[self.subject_id_col].astype(str)
                df = df[df[self.subject_id_col].str.strip() != ""].copy()
                if len(df) == 0:
                    continue

                # other cols as string
                for c in [
                    self.report_path_col,
                    self.qa_type_col,
                    self.claim_col,
                    self.case_context_col,
                    self.question_col,
                    self.answer_col,
                ]:
                    df[c] = df[c].astype(str)

                claims = df[self.claim_col].tolist()

                # embed with progress
                q_emb = self._encode_texts(
                    claims,
                    max_len=self.max_query_tokens,
                    batch_size=self.qa_batch_size,
                    pbar=embed_pbar,
                )
                D, I = index.search(q_emb, self.top_k)

                retr_rows: List[Dict[str, object]] = []
                ver_rows: List[Dict[str, object]] = []

                for i in range(len(df)):
                    claim_text = df.iloc[i][self.claim_col]
                    claim_norm = self._normalize_text(claim_text)

                    subject_id = str(df.iloc[i][self.subject_id_col])  # FIX: keep string
                    report_rel_path = str(df.iloc[i][self.report_path_col])  # report_id -> report_rel_path
                    qa_type = str(df.iloc[i][self.qa_type_col])
                    case_context = str(df.iloc[i][self.case_context_col])
                    question = str(df.iloc[i][self.question_col])
                    answer_target = str(df.iloc[i][self.answer_col])

                    if not claim_norm:
                        ids: List[int] = []
                        w: List[float] = []
                        combined_list: List[float] = []
                    else:
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

                        w_arr = self._weights_from_scores(combined)

                        ids = retrieved_ids
                        w = [float(x) for x in w_arr.tolist()]
                        combined_list = [float(x) for x in combined.tolist()]

                        ids, w = self._filter_and_truncate(ids, w)
                        if len(ids) != len(w):
                            m = min(len(ids), len(w))
                            ids, w = ids[:m], w[:m]
                            combined_list = combined_list[:m]

                    soft_support = float(np.max(np.asarray(combined_list, dtype=np.float32))) if combined_list else 0.0

                    retr_rows.append(
                        asdict(
                            RetrieverSupervisionRow(
                                qa_row_id=qa_row_id,
                                subject_id=subject_id,
                                split="test",
                                report_rel_path=report_rel_path,
                                qa_type=qa_type,
                                claim_text=claim_text,
                                pos_chunk_ids_json=json.dumps(ids),
                                pos_weights_json=json.dumps(w),
                            )
                        )
                    )

                    ver_rows.append(
                        asdict(
                            VerifierSupervisionRow(
                                qa_row_id=qa_row_id,
                                subject_id=subject_id,
                                split="test",
                                report_rel_path=report_rel_path,
                                qa_type=qa_type,
                                case_context=case_context,
                                question=question,
                                answer_target=answer_target,
                                claim_text=claim_text,
                                evidence_chunk_ids_json=json.dumps(ids),
                                evidence_weights_json=json.dumps(w),
                                soft_support_score=float(soft_support),
                            )
                        )
                    )

                    qa_row_id += 1
                    kept_total += 1
                    main_pbar.update(1)

                retr_df = pd.DataFrame(retr_rows)
                ver_df = pd.DataFrame(ver_rows)

                retr_writer.write_table(self._to_table(retr_df, retr_schema))
                ver_writer.write_table(self._to_table(ver_df, ver_schema))

        finally:
            embed_pbar.close()
            main_pbar.close()
            retr_writer.close()
            ver_writer.close()

        manifest = {
            "step": "4_external_test",
            "inputs": {
                "qa_csv": str(self.qa_csv.resolve()),
                "chunks_parquet": str(self.chunks_parquet.resolve()),
            },
            "outputs": {
                "retriever_supervision": str(self.retriever_out.resolve()),
                "verifier_supervision": str(self.verifier_out.resolve()),
            },
            "filter": {"split_col": self.split_col, "test_split_value": self.test_split_value},
            "params": {
                "bootstrap_embedder": self.model_name,
                "max_query_tokens": self.max_query_tokens,
                "max_doc_tokens": self.max_doc_tokens,
                "top_k": self.top_k,
                "alpha": self.alpha,
                "temperature": self.temperature,
                "min_weight": self.min_weight,
                "qa_chunksize_rows": self.qa_chunksize_rows,
                "qa_batch_size": self.qa_batch_size,
                "doc_batch_size": self.doc_batch_size,
                "seed": self.seed,
                "device": self.device,
                "force_cpu_faiss": self.force_cpu_faiss,
            },
            "cache": {
                "chunk_embeds_npy": str(self.chunk_embeds_path.resolve()),
                "faiss_index": str(self.faiss_index_path.resolve()),
            },
            "stats": {
                "rows_written_test_only": int(kept_total),
                "estimated_total_test_rows": int(total_test),
            },
        }
        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.retriever_out}  (rows={kept_total:,})")
        print(f"Wrote: {self.verifier_out}   (rows={kept_total:,})")
        print(f"Wrote: {self.manifest_out}")
        print(f"Cache: {self.chunk_embeds_path}  |  {self.faiss_index_path}")





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
        return {"min": float(self.min_v), "max": float(self.max_v), "mean": float(self.mean), "std": float(np.sqrt(var))}


class EvidenceTextStore:
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


def create_fixed_chexpert_test_eval_parquet(
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
    Same as create_fixed_mimic_test_eval_parquet, but for CheXpert Step4 outputs.

    Reads:
      chexpert_step4c/verifier_supervision.parquet

    Writes:
      chexpert_test_eval.parquet

    Output columns (same as MIMIC):
      qa_row_id, subject_id, split, qa_type, case_context, question, answer_target, claim_text,
      evidence_chunk_ids_json, evidence_text, soft_support_score,
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

    def _loads_json_list(s: str) -> List[int]:
        try:
            xs = json.loads(s)
            return [int(x) for x in xs]
        except Exception:
            return []

    def _truncate(s: str) -> str:
        s = "" if s is None else str(s)
        if max_evidence_chars > 0 and len(s) > max_evidence_chars:
            return s[:max_evidence_chars]
        return s

    def materialize_evidence(ids_json: str) -> str:
        ids = _loads_json_list(str(ids_json))
        if not ids:
            return ""
        chunks = [ev_store.get(cid) for cid in ids]
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
        "subject_id",
        "split",
        "qa_type",
        "case_context",
        "question",
        "answer_target",
        "claim_text",
        "evidence_chunk_ids_json",
        "soft_support_score",
    ]

    pf = pq.ParquetFile(str(in_path))
    schema_names = set(pf.schema.names)
    for c in required_in_cols:
        if c not in schema_names:
            raise ValueError(f"Missing required column in CheXpert Step4 verifier supervision parquet: {c}")

    # IMPORTANT: keep SAME column set as mimic; subject_id type here is string because CheXpert uses string ids.
    out_schema = pa.schema(
        [
            ("qa_row_id", pa.int64()),
            ("subject_id", pa.string()),  # CheXpert: string ids
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

    writer = pq.ParquetWriter(str(out_path), schema=out_schema, compression=parquet_compression)

    total_rows = pf.metadata.num_rows if pf.metadata is not None else 0
    pbar = tqdm(total=total_rows, desc="Phase 1: freezing chexpert_test_eval.parquet", unit="rows", miniters=1)

    subj_set: set[str] = set()
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
            d["subject_id"] = d["subject_id"].astype(str)
            d["soft_support_score"] = pd.to_numeric(d["soft_support_score"], errors="coerce").fillna(0.0).astype(np.float32)

            for c in ["split", "qa_type", "case_context", "question", "answer_target", "claim_text", "evidence_chunk_ids_json"]:
                d[c] = d[c].astype(str)

            d["evidence_text"] = d["evidence_chunk_ids_json"].map(materialize_evidence).astype(str)

            d["report_length_bucket"] = d["case_context"].map(report_length_bucket).astype(str)
            d["device_present"] = d["case_context"].map(lambda x: 1 if device_re.search(str(x)) else 0).astype(np.int8)
            d["uncertainty_present"] = d["case_context"].map(lambda x: 1 if uncertainty_re.search(str(x)) else 0).astype(np.int8)
            d["label_vs_extractive"] = d["qa_type"].map(label_vs_extractive_flag).astype(str)

            subj_set.update(d["subject_id"].tolist())
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
    # patient_split(csv_path=os.path.join(base_path_chexpertData, "df_chexpert_plus_240401.csv"),
    #               outpath=os.path.join(base_path_chexpertData, "chexpert_plus_reports.csv"))

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_chexpertData = params["Preprocessing"]["base_path_mimicData"]
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]

    ''' step 1 '''
    # builder = ChexpertPlusCaseContextBuilder(
    #     master_csv=os.path.join(base_path_chexpertData, "chexpert_plus_reports.csv"),
    #     out_path=os.path.join(base_path_chexpertData, "chexpertplus_case_contexts.csv"),
    #     report_col="report",
    #     patient_col="subject_id",
    #     image_col="path_to_image",
    #     split_col="split",
    #     report_order_col="patient_report_date_order",
    #     findings_col="section_findings",
    #     impression_col="section_impression",
    #     narrative_col="section_narrative",
    #     history_col="section_history",
    #     clinical_history_col="section_clinical_history",
    #     comparison_col="section_comparison",
    #     technique_col="section_technique",
    #     procedure_comments_col="section_procedure_comments",
    #     summary_col="section_summary",
    #     accession_col="section_accession_number",
    # )
    # out_df = builder.build_case_context_table()
    # print(out_df[["report_id", "subject_id", "case_context"]].head(3))
    # print("Rows:", len(out_df))

    ''' steps 2 & 3 '''
    # builder = ChexpertPlusFactsAndQABuilder(
    #     master_csv=os.path.join(base_path_chexpertData, "chexpert_plus_reports.csv"),
    #     case_context_csv=os.path.join(base_path_chexpertData, "chexpertplus_case_contexts.csv"),
    #     out_facts_path=os.path.join(base_path_chexpertData, "chexpertplus_facts_long.csv"),
    #     out_qa_path=os.path.join(base_path_chexpertData, "chexpertplus_qa_step3.csv"),
    #     patient_col="subject_id",
    #     image_col="path_to_image",
    #     report_col="report",
    #     report_id_col="report_id",
    #     report_order_col="patient_report_date_order",
    #     split_col="split",
    # )
    # facts_df, qa_df = builder.run()
    # print("facts:", len(facts_df), "qa:", len(qa_df))
    # print(facts_df.head(3))
    # print(qa_df[["report_id", "qa_type", "question", "answer_target"]].head(5))



    ''' step 4C '''
    builder = CheXpertRadiopaediaStep4ExternalTestBuilder(
        qa_csv=os.path.join(base_path_chexpertData, "chexpertplus_qa_step3.csv"),
        chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "chexpert_step4c"),
        top_k=20,
        alpha=0.85,
        temperature=0.07,
        seed=13,
    )
    builder.run()



    ''' Creating the final mimic eval parquet from Step 4C '''
    # verifier_supervision_parquet = os.path.join(base_path_evidenceData, "chexpert_step4c", "verifier_supervision.parquet")
    # radiopaedia_chunks_parquet = os.path.join(base_path_evidenceData, "chunks.parquet")
    # out_eval_parquet = os.path.join(base_path_evidenceData, "phase1_fixed_test_eval_chexpertplus", "chexpertplus_test_eval.parquet")
    #
    #
    # create_fixed_chexpert_test_eval_parquet(
    #     verifier_supervision_parquet=verifier_supervision_parquet,
    #     out_parquet=out_eval_parquet,
    #     evidence_parquet=radiopaedia_chunks_parquet,
    #     evidence_id_col="chunk_id",
    #     evidence_text_col="chunk_text",
    #     evidence_id_base=0,  # change to 1 if ids start from 1
    #     preload_evidence_texts=False,
    #     split_value="test",
    #     parquet_batch_rows=50_000,
    #     parquet_compression="zstd",
    #     seed=13,
    #     write_manifest_json=True,
    # )