"""
Created on Jan 28, 2026.
mimic_data_preprocessing.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import hashlib



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







if __name__ == "__main__":
    builder = MimicCaseContextBuilder(
        master_csv="//MIMIC/mimic_master_list_forReports.csv",
        reports_root="//MIMIC",
        out_path="mimic_case_contexts.csv",
    )
    out_df = builder.build_case_context_table()
    print(out_df[["report_rel_path", "case_context"]].head(3))
    print("Rows:", len(out_df))


    # step 2
    builder = MimicFactTableBuilder(
        master_csv="//MIMIC/mimic_master_list_forReports.csv",
        case_context_csv="//MIMIC/mimic_case_contexts.csv",
        out_path="mimic_facts_long.csv",
    )
    facts_df = builder.build_fact_table()
    print(facts_df.head(10))
    print("n_facts:", len(facts_df))



    # step 3
    builder = MimicQATableBuilder(
        facts_long_csv="//MIMIC/mimic_facts_long.csv",
        out_path="mimic_qa_step3.csv",
    )
    qa_df = builder.build_qa_table()
    print("QA rows:", len(qa_df))
    print(qa_df[["qa_type", "question", "answer_target"]].head(10))

