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
import pickle

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





class CheXpertPlusGroundingDatasetBuilder:
    """
    Steps 0-5 for CheXpert Plus grounding benchmark.

    Outputs:
      reports_master.parquet
      reports_sections.parquet
      reports_impression_labels.parquet
      qa_grounding_raw.parquet
      qa_grounding_main.parquet
      qa_grounding_easy_control.parquet
      qa_grounding_main_with_anchors.parquet
      benchmark_manifest.json

    Notes:
      - uses existing split column from CheXpert Plus
      - uses provided section_* columns first
      - extracts study_id from path_to_image
      - deduplicates repeated frontal/lateral image rows into report-level rows
      - keeps output schema aligned with MimicGroundingDatasetBuilder
    """

    def __init__(self, master_csv: str, out_dir: str, seed: int = 13):
        self.master_csv = Path(master_csv)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)

        self.sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")
        self.word_re = re.compile(r"[a-z0-9]+")
        self.study_re = re.compile(r"/study(\d+)/", re.I)
        self.patient_re = re.compile(r"patient(\d+)", re.I)

        self.communication_re = re.compile(
            r"\b(discussed with|communicated to|called to|paged|notified|by phone|phone at|results were communicated)\b",
            re.I,
        )
        self.admin_inline_re = re.compile(
            r"\b(recommend short[- ]term follow[- ]up|these findings were discovered|covering md|immediately paged)\b",
            re.I,
        )
        self.negation_re = re.compile(
            r"\b(no|without|absent|negative for|free of|no evidence of|no focal|neither|none)\b",
            re.I,
        )
        self.uncertainty_re = re.compile(
            r"\b(possible|possibly|may|might|likely|unlikely|cannot exclude|could represent|"
            r"suggestive of|question of|suspicious for|concerning for|may represent|"
            r"could have similar appearance|not excluded|versus)\b",
            re.I,
        )
        self.resolution_re = re.compile(
            r"\b(resolved|improved|decreased|cleared|near complete resolution|interval resolution)\b",
            re.I,
        )

        self.imaging_anchor_re = re.compile(
            r"\b("
            r"opacity|opacities|airspace|interstitial|vascular|congestion|edema|"
            r"effusion|pneumothorax|atelectatic|atelectasis|consolidation|"
            r"blunting|costophrenic|retrocardiac|bibasilar|basilar|perihilar|"
            r"lower lobe|upper lobe|right base|left base|bilateral|diffuse|focal|"
            r"cardiac silhouette|cardiomediastinal silhouette|hilar vasculature|"
            r"rib|fracture|fractures|osseous|streaky|markings"
            r")\b",
            re.I,
        )

        self.concepts = {
            "pleural effusion": re.compile(
                r"\bpleural effusions?\b|\beffusions?\b", re.I
            ),
            "pneumothorax": re.compile(
                r"\bpneumothorax\b|\bpneumothoraces\b|\bptx\b", re.I
            ),
            "cardiomegaly": re.compile(
                r"\bcardiomegaly\b|"
                r"\benlarged cardiac silhouette\b|"
                r"\bcardiac silhouette is enlarged\b|"
                r"\bcardiac silhouette enlarged\b|"
                r"\benlarged heart\b|"
                r"\bheart is enlarged\b|"
                r"\bheart remains enlarged\b|"
                r"\bcardiac size is within normal limits\b|"
                r"\bheart size is enlarged\b|"
                r"\bmild cardiomegaly\b|"
                r"\bmoderate(?:ly)? enlarged\b",
                re.I,
            ),
            "atelectasis": re.compile(
                r"\batelectasis\b|"
                r"\batelectatic\b|"
                r"\bsubsegmental atelectatic\b|"
                r"\bbibasilar atelectatic\b|"
                r"\bbasal atelectatic\b|"
                r"\bplate(?:-| )like atelectasis\b|"
                r"\bdiscoid atelectasis\b|"
                r"\blower lobe collapse\b",
                re.I,
            ),
            "consolidation": re.compile(
                r"\bconsolidation\b|"
                r"\blobar consolidation\b|"
                r"\bfocal consolidation\b|"
                r"\bairspace consolidation\b|"
                r"\bairspace disease\b|"
                r"\bopacification\b|"
                r"\bfocal air space opacit(?:y|ies)\b|"
                r"\bretrocardiac opacity\b",
                re.I,
            ),
            "pulmonary edema": re.compile(
                r"\bpulmonary edema\b|"
                r"\bedema\b",
                re.I,
            ),
            "vascular congestion": re.compile(
                r"\bvascular congestion\b|"
                r"\bpulmonary vascular congestion\b|"
                r"\bcentral pulmonary vascular congestion\b|"
                r"\bvascular indistinctness\b|"
                r"\bpulmonary vascularity\b|"
                r"\bprominent pulmonary vasculature\b|"
                r"\bhilar vasculature\b",
                re.I,
            ),
            "interstitial edema": re.compile(
                r"\binterstitial edema\b|"
                r"\binterstitial pulmonary edema\b|"
                r"\bprominent interstitial markings\b",
                re.I,
            ),
            "rib fracture": re.compile(
                r"\brib fractures?\b|"
                r"\bfractured ribs?\b|"
                r"\bninth rib fracture\b|"
                r"\bright ninth rib fracture\b",
                re.I,
            ),
            "fracture": re.compile(
                r"\bfracture\b|\bfractures\b|\bold fracture\b|\bhealed .* rib fracture\b",
                re.I,
            ),
        }

        self.shortcut_patterns = {
            "pleural effusion": re.compile(
                r"\b(no|without|trace|small|mild|moderate|large|tiny)?\s*pleural effusions?\b|"
                r"\bno effusion\b|\bno effusions\b",
                re.I,
            ),
            "pneumothorax": re.compile(
                r"\b(no|without|small|tiny|trace|residual)?\s*(evidence of )?pneumothorax\b|"
                r"\bpneumothoraces\b|\bno ptx\b",
                re.I,
            ),
            "cardiomegaly": self.concepts["cardiomegaly"],
            "atelectasis": self.concepts["atelectasis"],
            "consolidation": self.concepts["consolidation"],
            "pulmonary edema": self.concepts["pulmonary edema"],
            "vascular congestion": self.concepts["vascular congestion"],
            "interstitial edema": self.concepts["interstitial edema"],
            "rib fracture": self.concepts["rib fracture"],
            "fracture": self.concepts["fracture"],
        }

        self.anchor_stop = {
            "again", "noted", "seen", "demonstrated", "identified", "there", "here", "this", "that",
            "these", "those", "study", "exam", "report", "finding", "findings", "impression",
            "present", "absent", "uncertain", "possible", "likely", "question", "described",
            "showing", "shows", "show", "patient", "view", "views", "portable", "upright", "ap", "pa",
            "lateral", "frontal", "radiograph", "radiographs", "chest", "single", "obtained",
            "comparison", "history", "indication", "technique", "year", "old", "female", "male",
            "woman", "man", "normal", "limits", "within", "stable", "unchanged", "little", "change",
            "acute", "cardiopulmonary", "process", "abnormality", "clinical", "data",
            "is", "are", "was", "were", "be", "been", "the", "a", "an", "and", "or", "of", "to",
            "in", "on", "for", "with", "without", "as", "at", "by", "from", "into",
            "mild", "moderate", "large", "small", "trace", "tiny",
        }

        self.anchor_phrases = [
            "costophrenic angle", "right costophrenic angle", "left costophrenic angle",
            "right lower lobe", "left lower lobe", "right upper lobe", "left upper lobe",
            "right base", "left base", "lung bases", "bibasal", "bibasilar", "basilar",
            "retrocardiac", "perihilar", "hilar vasculature", "cardiac silhouette",
            "cardiomediastinal silhouette", "airspace opacity", "airspace opacities",
            "vascular congestion", "interstitial edema", "interstitial markings",
            "linear opacity", "patchy opacity", "patchy opacities", "blunting",
            "right", "left", "bilateral", "diffuse", "focal", "multifocal", "apical", "lower",
            "upper", "central", "peribronchovascular", "interstitial", "markings", "streaky",
            "scoliosis", "rib", "ribs", "osseous", "fracture", "fractures",
        ]

        self.question_templates = {
            "pleural effusion": {
                "present": [
                    "Is there pleural effusion?",
                    "Is pleural effusion present?",
                    "Does the study show pleural effusion?",
                    "Are pleural effusions seen?",
                ],
                "absent": [
                    "Is pleural effusion absent?",
                    "Is there no pleural effusion?",
                    "Does the study show no pleural effusion?",
                    "Are pleural effusions absent?",
                ],
                "uncertain": [
                    "Is pleural effusion possible?",
                    "Is pleural effusion uncertain?",
                    "Could there be pleural effusion?",
                    "Is pleural effusion suspected?",
                ],
            },
            "pneumothorax": {
                "present": [
                    "Is there pneumothorax?",
                    "Is pneumothorax present?",
                    "Does the study show pneumothorax?",
                    "Is a pneumothorax seen?",
                ],
                "absent": [
                    "Is pneumothorax absent?",
                    "Is there no pneumothorax?",
                    "Does the study show no pneumothorax?",
                    "Is pneumothorax not seen?",
                ],
                "uncertain": [
                    "Is pneumothorax possible?",
                    "Is pneumothorax uncertain?",
                    "Could there be pneumothorax?",
                    "Is pneumothorax suspected?",
                ],
            },
            "cardiomegaly": {
                "present": [
                    "Is there cardiomegaly?",
                    "Is cardiomegaly present?",
                    "Does the study show cardiomegaly?",
                    "Is the heart enlarged?",
                ],
                "absent": [
                    "Is cardiomegaly absent?",
                    "Is there no cardiomegaly?",
                    "Does the study show no cardiomegaly?",
                    "Is the heart not enlarged?",
                ],
                "uncertain": [
                    "Is cardiomegaly possible?",
                    "Is cardiomegaly uncertain?",
                    "Could there be cardiomegaly?",
                    "Is cardiomegaly suspected?",
                ],
            },
            "atelectasis": {
                "present": [
                    "Is there atelectasis?",
                    "Is atelectasis present?",
                    "Does the study show atelectasis?",
                    "Are atelectatic changes seen?",
                ],
                "absent": [
                    "Is atelectasis absent?",
                    "Is there no atelectasis?",
                    "Does the study show no atelectasis?",
                    "Are atelectatic changes absent?",
                ],
                "uncertain": [
                    "Is atelectasis possible?",
                    "Is atelectasis uncertain?",
                    "Could there be atelectasis?",
                    "Is atelectasis suspected?",
                ],
            },
            "consolidation": {
                "present": [
                    "Is there consolidation?",
                    "Is consolidation present?",
                    "Does the study show consolidation?",
                    "Is airspace consolidation seen?",
                ],
                "absent": [
                    "Is consolidation absent?",
                    "Is there no consolidation?",
                    "Does the study show no consolidation?",
                    "Is airspace consolidation absent?",
                ],
                "uncertain": [
                    "Is consolidation possible?",
                    "Is consolidation uncertain?",
                    "Could there be consolidation?",
                    "Is consolidation suspected?",
                ],
            },
            "pulmonary edema": {
                "present": [
                    "Is there pulmonary edema?",
                    "Is pulmonary edema present?",
                    "Does the study show pulmonary edema?",
                    "Are findings consistent with pulmonary edema?",
                ],
                "absent": [
                    "Is pulmonary edema absent?",
                    "Is there no pulmonary edema?",
                    "Does the study show no pulmonary edema?",
                    "Are findings not consistent with pulmonary edema?",
                ],
                "uncertain": [
                    "Is pulmonary edema possible?",
                    "Is pulmonary edema uncertain?",
                    "Could there be pulmonary edema?",
                    "Is pulmonary edema suspected?",
                ],
            },
            "vascular congestion": {
                "present": [
                    "Is there vascular congestion?",
                    "Is vascular congestion present?",
                    "Does the study show vascular congestion?",
                    "Is pulmonary vascular congestion seen?",
                ],
                "absent": [
                    "Is vascular congestion absent?",
                    "Is there no vascular congestion?",
                    "Does the study show no vascular congestion?",
                    "Is pulmonary vascular congestion absent?",
                ],
                "uncertain": [
                    "Is vascular congestion possible?",
                    "Is vascular congestion uncertain?",
                    "Could there be vascular congestion?",
                    "Is vascular congestion suspected?",
                ],
            },
            "interstitial edema": {
                "present": [
                    "Is there interstitial edema?",
                    "Is interstitial edema present?",
                    "Does the study show interstitial edema?",
                    "Are findings consistent with interstitial edema?",
                ],
                "absent": [
                    "Is interstitial edema absent?",
                    "Is there no interstitial edema?",
                    "Does the study show no interstitial edema?",
                    "Are findings not consistent with interstitial edema?",
                ],
                "uncertain": [
                    "Is interstitial edema possible?",
                    "Is interstitial edema uncertain?",
                    "Could there be interstitial edema?",
                    "Is interstitial edema suspected?",
                ],
            },
            "rib fracture": {
                "present": [
                    "Is there a rib fracture?",
                    "Is rib fracture present?",
                    "Does the study show rib fracture?",
                    "Are rib fractures seen?",
                ],
                "absent": [
                    "Is rib fracture absent?",
                    "Is there no rib fracture?",
                    "Does the study show no rib fracture?",
                    "Are rib fractures absent?",
                ],
                "uncertain": [
                    "Is rib fracture possible?",
                    "Is rib fracture uncertain?",
                    "Could there be a rib fracture?",
                    "Is rib fracture suspected?",
                ],
            },
            "fracture": {
                "present": [
                    "Is there a fracture?",
                    "Is fracture present?",
                    "Does the study show fracture?",
                    "Are fractures seen?",
                ],
                "absent": [
                    "Is fracture absent?",
                    "Is there no fracture?",
                    "Does the study show no fracture?",
                    "Are fractures absent?",
                ],
                "uncertain": [
                    "Is fracture possible?",
                    "Is fracture uncertain?",
                    "Could there be a fracture?",
                    "Is fracture suspected?",
                ],
            },
        }

    # ---------- helpers ----------

    def _norm(self, s: str) -> str:
        if s is None:
            return ""
        s = str(s).replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"_+", " ", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _tokens(self, s: str) -> List[str]:
        return self.word_re.findall(self._norm(s).lower())

    def _stable_hash(self, s: str) -> str:
        return hashlib.md5(self._norm(s).encode("utf-8", errors="ignore")).hexdigest()

    def _clean_section_text(self, text: str) -> str:
        if not text:
            return ""
        t = self._norm(text)

        # remove anonymization / admin tails
        t = re.split(r"\bACCESSION NUMBER\b", t, maxsplit=1, flags=re.I)[0]
        t = re.split(r"\bThis report has been anonymized\b", t, maxsplit=1, flags=re.I)[0]
        t = re.split(r"\bI have personally reviewed the images\b", t, maxsplit=1, flags=re.I)[0]
        t = re.split(r"\bEND OF IMPRESSION\b", t, maxsplit=1, flags=re.I)[0]
        t = re.split(r"\bSUMMARY\b", t, maxsplit=1, flags=re.I)[0]

        lines = [x.strip() for x in t.split("\n") if x.strip()]
        kept = []
        for ln in lines:
            if self.communication_re.search(ln):
                continue
            if self.admin_inline_re.search(ln):
                continue
            kept.append(ln)

        return re.sub(r"\s+", " ", " ".join(kept)).strip()

    def _clinical_text_ok(self, text: str) -> bool:
        toks = set(self._tokens(text))
        if len(toks) < 4:
            return False
        clinical = {
            "lung", "lungs", "pleural", "effusion", "effusions", "pneumothorax", "ptx",
            "cardiac", "cardiomediastinal", "opacity", "opacities", "atelectasis",
            "atelectatic", "consolidation", "edema", "hilar", "mediastinal",
            "silhouette", "vascular", "retrocardiac", "basilar", "bibasilar",
            "interstitial", "fracture", "fractures", "rib", "ribs", "osseous",
            "airspace", "aeration",
        }
        return len(toks & clinical) > 0

    def _extract_subject_id_int(self, x: str) -> Optional[int]:
        m = self.patient_re.search(str(x))
        if m:
            return int(m.group(1))
        try:
            return int(x)
        except Exception:
            return None

    def _extract_study_id(self, path_to_image: str, fallback_order) -> int:
        m = self.study_re.search(str(path_to_image))
        if m:
            return int(m.group(1))
        try:
            return int(fallback_order)
        except Exception:
            return -1

    def _make_report_rel_path(self, path_to_image: str) -> str:
        p = str(path_to_image).replace("\\", "/")
        parts = p.split("/")
        if len(parts) >= 3:
            return "/".join(parts[:3])   # train/patientXXXXX/studyY
        return p

    def _choose_question(self, concept: str, state: str, qa_row_id: int) -> str:
        candidates = self.question_templates[concept][state]
        idx = (int(qa_row_id) + self.seed) % len(candidates)
        return candidates[idx]

    def _concept_in_text(self, text: str, concept: str) -> bool:
        return bool(self.concepts[concept].search(self._norm(text)))

    def _fallback_findings_from_report(self, raw_report: str) -> str:
        raw = self._norm(raw_report)
        if not raw:
            return ""

        # if explicit FINDINGS exists, use only that part
        m_find = re.search(r"\bFINDINGS\s*:\s*", raw, flags=re.I)
        if m_find:
            tail = raw[m_find.end():]
            tail = re.split(r"\bIMPRESSION\s*:\s*|\bSUMMARY\s*:\s*|\bACCESSION NUMBER\s*:\s*", tail, maxsplit=1, flags=re.I)[0]
            tail = self._clean_section_text(tail)
            if self._clinical_text_ok(tail):
                return tail

        # otherwise use pre-impression body
        m_imp = re.search(r"\bIMPRESSION\s*:\s*", raw, flags=re.I)
        if m_imp:
            head = raw[:m_imp.start()]
            head = self._clean_section_text(head)
            if self._clinical_text_ok(head):
                return head

        return ""

    def _fallback_impression_from_report(self, raw_report: str) -> str:
        raw = self._norm(raw_report)
        if not raw:
            return ""

        m_imp = re.search(r"\bIMPRESSION\s*:\s*", raw, flags=re.I)
        if m_imp:
            tail = raw[m_imp.end():]
            tail = re.split(r"\bSUMMARY\s*:\s*|\bACCESSION NUMBER\s*:\s*|\bThis report has been anonymized\b",
                            tail, maxsplit=1, flags=re.I)[0]
            tail = self._clean_section_text(tail)
            if self._clinical_text_ok(tail):
                return tail

        return ""

    # ---------- Step 0 ----------

    def _step0_master(self) -> pd.DataFrame:
        df = pd.read_csv(self.master_csv)

        required = {"path_to_image", "subject_id", "patient_report_date_order", "report", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CheXpert Plus CSV missing required columns: {missing}")

        out = df.copy()

        out["subject_id_int"] = out["subject_id"].apply(self._extract_subject_id_int)
        out["study_id"] = [
            self._extract_study_id(p, o)
            for p, o in zip(out["path_to_image"], out["patient_report_date_order"])
        ]
        out["report_rel_path"] = out["path_to_image"].apply(self._make_report_rel_path)
        out["raw_report_text"] = out["report"].fillna("").astype(str).apply(self._norm)
        out["split"] = out["split"].astype(str).str.strip().str.lower()

        out = out[
            out["subject_id_int"].notna()
            & out["raw_report_text"].str.len().gt(0)
            & out["split"].isin(["train", "val", "valid", "test"])
        ].copy()

        out["subject_id_int"] = out["subject_id_int"].astype(int)
        out["study_id"] = out["study_id"].astype(int)
        out["split"] = out["split"].replace({"valid": "val"})

        # deduplicate repeated image rows with same study/report
        out["report_hash"] = out["raw_report_text"].apply(self._stable_hash)
        out = (
            out.sort_values(["subject_id_int", "study_id", "path_to_image"])
               .drop_duplicates(subset=["subject_id_int", "study_id", "report_hash"])
               .reset_index(drop=True)
        )

        master = pd.DataFrame({
            "subject_id": out["subject_id_int"].astype(int),
            "study_id": out["study_id"].astype(int),
            "report_rel_path": out["report_rel_path"].astype(str),
            "split": out["split"].astype(str),
            "raw_report_text": out["raw_report_text"].astype(str),
            "report_hash": out["report_hash"].astype(str),
        })

        master.to_parquet(self.out_dir / "reports_master.parquet", index=False)
        return master

    # ---------- Step 1 ----------

    def _step1_sections(self, reports_master: pd.DataFrame) -> pd.DataFrame:
        # re-read source CSV so we can use provided section_* columns aligned with report_hash
        src = pd.read_csv(self.master_csv).copy()
        src["subject_id_int"] = src["subject_id"].apply(self._extract_subject_id_int)
        src["study_id"] = [
            self._extract_study_id(p, o)
            for p, o in zip(src["path_to_image"], src["patient_report_date_order"])
        ]
        src["raw_report_text"] = src["report"].fillna("").astype(str).apply(self._norm)
        src["report_hash"] = src["raw_report_text"].apply(self._stable_hash)

        section_cols = [
            "section_narrative",
            "section_clinical_history",
            "section_history",
            "section_comparison",
            "section_technique",
            "section_procedure_comments",
            "section_findings",
            "section_impression",
            "section_end_of_impression",
            "section_summary",
            "section_accession_number",
        ]
        for c in section_cols:
            if c not in src.columns:
                src[c] = ""

        src = (
            src.sort_values(["subject_id_int", "study_id"])
               .drop_duplicates(subset=["subject_id_int", "study_id", "report_hash"])
               .reset_index(drop=True)
        )

        merged = reports_master.merge(
            src[
                ["subject_id_int", "study_id", "report_hash"] + section_cols
            ],
            left_on=["subject_id", "study_id", "report_hash"],
            right_on=["subject_id_int", "study_id", "report_hash"],
            how="left",
        )

        rows = []
        for _, r in merged.iterrows():
            raw = self._norm(r["raw_report_text"])

            findings = self._clean_section_text(r.get("section_findings", ""))
            impression = self._clean_section_text(r.get("section_impression", ""))

            # use report-derived fallback when section columns are missing/empty
            if not findings:
                findings = self._fallback_findings_from_report(raw)
            if not impression:
                impression = self._fallback_impression_from_report(raw)

            findings_ok = self._clinical_text_ok(findings)
            impression_ok = self._clinical_text_ok(impression)

            if findings_ok and impression_ok:
                report_type = "both_findings_and_impression"
                conf = "high"
                notes = "section_columns_or_report_fallback"
                eligible = True
            elif findings_ok:
                report_type = "findings_only"
                conf = "low"
                notes = "missing_impression"
                eligible = False
            elif impression_ok:
                report_type = "impression_only"
                conf = "low"
                notes = "missing_findings"
                eligible = False
            else:
                report_type = "unusable"
                conf = "low"
                notes = "no_reliable_sections"
                eligible = False

            rows.append({
                "subject_id": int(r["subject_id"]),
                "study_id": int(r["study_id"]),
                "report_rel_path": r["report_rel_path"],
                "split": r["split"],
                "raw_report_text": raw,
                "report_format_type": report_type,
                "parse_confidence": conf,
                "findings_text": findings,
                "impression_text": impression,
                "parse_notes": notes,
                "is_main_benchmark_eligible": eligible,
            })

        out = pd.DataFrame(rows)
        out.to_parquet(self.out_dir / "reports_sections.parquet", index=False)
        return out

    # ---------- Step 2 ----------

    def _classify_concept_state(self, text: str, concept_pat: re.Pattern) -> Tuple[str, str]:
        sents = [x.strip() for x in self.sent_split_re.split(text) if x.strip()]
        for s in sents:
            if not concept_pat.search(s):
                continue
            if self.negation_re.search(s):
                return "absent", s
            if self.uncertainty_re.search(s):
                return "uncertain", s
            if self.resolution_re.search(s):
                return "absent", s
            return "present", s
        return "", ""

    def _step2_impression_labels(self, reports_sections: pd.DataFrame) -> pd.DataFrame:
        rows = []
        df = reports_sections[reports_sections["is_main_benchmark_eligible"] == True].copy()

        for _, r in df.iterrows():
            imp = r["impression_text"]
            for concept, pat in self.concepts.items():
                state, source = self._classify_concept_state(imp, pat)
                if not state:
                    continue
                rows.append({
                    "subject_id": int(r["subject_id"]),
                    "study_id": int(r["study_id"]),
                    "report_rel_path": r["report_rel_path"],
                    "split": r["split"],
                    "findings_text": r["findings_text"],
                    "impression_text": r["impression_text"],
                    "concept": concept,
                    "label_state": state,
                    "label_source_text": source,
                })

        out = pd.DataFrame(rows)
        out.to_parquet(self.out_dir / "reports_impression_labels.parquet", index=False)
        return out

    # ---------- Step 3 ----------

    def _step3_build_qa(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for i, r in labels_df.reset_index(drop=True).iterrows():
            concept = r["concept"]
            state = r["label_state"]
            findings_text = self._norm(r["findings_text"])
            impression_text = self._norm(r["impression_text"])

            rows.append({
                "qa_row_id": int(i),
                "subject_id": int(r["subject_id"]),
                "study_id": int(r["study_id"]),
                "report_rel_path": r["report_rel_path"],
                "split": r["split"],
                "findings_text": findings_text,
                "impression_text": impression_text,
                "concept": concept,
                "question": self._choose_question(concept, state, i),
                "claim_text": f"{concept} is {state}",
                "gold_state": state,
                "label_state": state,
                "label_source_text": r["label_source_text"],
                "concept_present_in_findings": bool(self._concept_in_text(findings_text, concept)),
                "concept_present_in_impression": bool(self._concept_in_text(impression_text, concept)),
            })

        out = pd.DataFrame(rows)
        out.to_parquet(self.out_dir / "qa_grounding_raw.parquet", index=False)
        return out

    # ---------- Step 4 ----------

    def _is_direct_shortcut(self, findings_text: str, concept: str) -> bool:
        sents = [x.strip() for x in self.sent_split_re.split(findings_text) if x.strip()]
        pat = self.shortcut_patterns[concept]
        for s in sents:
            if pat.search(s):
                return True
        return False

    def _step4_hardness_filter(self, qa_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        main_rows, easy_rows = [], []

        for _, r in qa_raw.iterrows():
            direct = self._is_direct_shortcut(r["findings_text"], r["concept"])
            row = dict(r)
            row["is_direct_mention"] = bool(direct)
            row["is_main_hard_example"] = not direct
            row["hardness_reason"] = "direct_concept_mention" if direct else "indirect_findings"

            if direct:
                easy_rows.append(row)
            else:
                main_rows.append(row)

        main_df = pd.DataFrame(main_rows)
        easy_df = pd.DataFrame(easy_rows)

        main_df.to_parquet(self.out_dir / "qa_grounding_main.parquet", index=False)
        easy_df.to_parquet(self.out_dir / "qa_grounding_easy_control.parquet", index=False)

        return main_df, easy_df

    # ---------- Step 5 ----------

    def _best_anchor_sentence(self, findings_text: str, concept: str) -> Tuple[str, bool]:
        sents = [x.strip() for x in self.sent_split_re.split(findings_text) if x.strip()]
        concept_pat = self.concepts[concept]

        best_sent = ""
        best_score = -10**9
        best_has_concept = False

        for s in sents:
            s_norm = s.lower()
            toks = self._tokens(s)

            if len(toks) < 5:
                continue

            phrase_hits = sum(1 for p in self.anchor_phrases if p in s_norm)
            imaging_hits = len(self.imaging_anchor_re.findall(s))
            has_concept = bool(concept_pat.search(s))
            concept_penalty = 4 if has_concept else 0

            length_bonus = 0.0
            if 7 <= len(toks) <= 28:
                length_bonus = 0.5
            elif len(toks) <= 40:
                length_bonus = 0.2

            score = float(
                2.0 * phrase_hits
                + 1.0 * imaging_hits
                + length_bonus
                - concept_penalty
            )

            if score > best_score:
                best_score = score
                best_sent = s
                best_has_concept = has_concept

        return best_sent, best_has_concept

    def _extract_anchor_terms(self, sent: str, concept: str) -> List[str]:
        concept_toks = set(self._tokens(concept))
        toks = self._tokens(sent)

        kept = []
        for t in toks:
            if t in concept_toks:
                continue
            if t in self.anchor_stop:
                continue
            if len(t) < 3:
                continue
            kept.append(t)

        kept = list(dict.fromkeys(kept))

        sent_norm = sent.lower()
        phrase_terms = []
        for p in self.anchor_phrases:
            if p in sent_norm:
                for t in self._tokens(p):
                    if t not in concept_toks and t not in self.anchor_stop and t not in phrase_terms:
                        phrase_terms.append(t)

        final = []
        for t in phrase_terms + kept:
            if t not in final:
                final.append(t)

        return final[:8]

    def _step5_build_anchors(self, qa_main: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in qa_main.iterrows():
            anchor_sent, anchor_has_concept = self._best_anchor_sentence(r["findings_text"], r["concept"])
            terms = self._extract_anchor_terms(anchor_sent, r["concept"])

            if len(terms) >= 4:
                conf = "high"
            elif len(terms) >= 2:
                conf = "medium"
            else:
                conf = "low"

            rows.append({
                "qa_row_id": int(r["qa_row_id"]),
                "anchor_sentence_text": anchor_sent,
                "anchor_terms_json": json.dumps(terms),
                "anchor_confidence": conf,
                "anchor_terms_source": "findings",
                "anchor_sentence_has_concept": bool(anchor_has_concept),
                "anchor_terms_count": int(len(terms)),
            })

        anchors = pd.DataFrame(rows)
        out = qa_main.merge(anchors, on="qa_row_id", how="left")
        out.to_parquet(self.out_dir / "qa_grounding_main_with_anchors.parquet", index=False)
        return out

    # ---------- Run all ----------

    def run_all(self):
        reports_master = self._step0_master()
        reports_sections = self._step1_sections(reports_master)
        impression_labels = self._step2_impression_labels(reports_sections)
        qa_raw = self._step3_build_qa(impression_labels)
        qa_main, qa_easy = self._step4_hardness_filter(qa_raw)
        qa_main_with_anchors = self._step5_build_anchors(qa_main)

        manifest = {
            "seed": self.seed,
            "split_policy": "use_existing_chexpertplus_split_column",
            "main_benchmark_source": "both_findings_and_impression_high_confidence",
            "label_source": "impression_only",
            "future_verifier_input": "findings_only",
            "concepts": list(self.concepts.keys()),
            "notes": {
                "gold_state_column_added": True,
                "concept_presence_flags_added": True,
                "easy_control_rows_retained": True,
                "anchor_terms_interpreted_as_case_cues": True,
                "reports_deduplicated_by_subject_study_reporthash": True,
                "section_columns_used_first": True,
                "report_fallback_used_when_sections_missing": True,
            },
            "outputs": {
                "reports_master": str((self.out_dir / "reports_master.parquet").resolve()),
                "reports_sections": str((self.out_dir / "reports_sections.parquet").resolve()),
                "reports_impression_labels": str((self.out_dir / "reports_impression_labels.parquet").resolve()),
                "qa_grounding_raw": str((self.out_dir / "qa_grounding_raw.parquet").resolve()),
                "qa_grounding_main": str((self.out_dir / "qa_grounding_main.parquet").resolve()),
                "qa_grounding_easy_control": str((self.out_dir / "qa_grounding_easy_control.parquet").resolve()),
                "qa_grounding_main_with_anchors": str((self.out_dir / "qa_grounding_main_with_anchors.parquet").resolve()),
            },
        }
        (self.out_dir / "benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        return {
            "reports_master": reports_master,
            "reports_sections": reports_sections,
            "reports_impression_labels": impression_labels,
            "qa_grounding_raw": qa_raw,
            "qa_grounding_main": qa_main,
            "qa_grounding_easy_control": qa_easy,
            "qa_grounding_main_with_anchors": qa_main_with_anchors,
        }






@dataclass(frozen=True)
class VerifierBinaryRow:
    verifier_row_id: int
    qa_row_id: int
    split: str
    subject_id: int
    report_rel_path: str

    concept: str
    gold_state: str
    claim_state: str
    claim_text: str

    case_context: str

    evidence_sentence_ids_json: str
    evidence_sentences_json: str
    evidence_article_titles_json: str

    row_type: str
    evidence_source_type: str

    label: str
    label_id: int


class CheXpertPlusVerifierBinaryDatasetBuilder:
    """
    Build binary verifier supervision for CheXpert Plus:
        input:  case + evidence + claim
        target: supported / not_supported

    Row types:
        C = claim matches gold state + supporting evidence -> 1
        D = opposite claim + opposite-support evidence -> 0
        A = positive claim + non-support evidence -> 0
        B = negative claim + non-support evidence -> 0

    Important design choice:
        after row creation, we balance within each split x concept group
        so that A/B/C/D have equal counts there.
    """

    def __init__(
        self,
        qa_parquet: str,
        evidence_sentences_parquet: str,
        out_dir: str,
        seed: int = 13,
        max_evidence_sentences: int = 2,
        max_rows_per_type_per_concept_split: Optional[int] = None,
        balance_within_split_concept: bool = True,
        min_rows_per_type_required: int = 1,
    ):
        self.qa_parquet = Path(qa_parquet)
        self.evidence_sentences_parquet = Path(evidence_sentences_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.max_rows_per_type_per_concept_split = max_rows_per_type_per_concept_split
        self.balance_within_split_concept = bool(balance_within_split_concept)
        self.min_rows_per_type_required = int(min_rows_per_type_required)

        self.out_parquet = self.out_dir / "verifier_binary_trainvaltest.parquet"
        self.manifest_out = self.out_dir / "manifest_verifier_binary.json"

        self.word_re = re.compile(r"[a-z0-9]+")

        self.core_concepts = {
            "pleural effusion",
            "pneumothorax",
            "pulmonary edema",
            "consolidation",
            "atelectasis",
            "fracture",
            "cardiomegaly",
        }

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------

    def _norm(self, s) -> str:
        if s is None:
            return ""
        s = str(s).replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _loads_json_list(self, s) -> List:
        try:
            out = json.loads(self._norm(s))
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _loads_json_dict(self, s) -> Dict:
        try:
            out = json.loads(self._norm(s))
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}

    def _claim_text(self, concept: str, claim_state: str) -> str:
        return f"{concept} is {claim_state}"

    def _opposite_state(self, state: str) -> Optional[str]:
        if state == "present":
            return "absent"
        if state == "absent":
            return "present"
        return None

    def _sample_n(self, rows: List[Dict], n: int) -> List[Dict]:
        if len(rows) <= n:
            return list(rows)
        return self.rng.sample(rows, n)

    def _dedup_pool(self, rows: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for r in rows:
            key = (int(r["sentence_id"]), self._norm(r["sentence_text"]).lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _take_evidence(
        self,
        candidates: List[Dict],
        source_type: str,
    ) -> Optional[Tuple[List[int], List[str], List[str], str]]:
        if not candidates:
            return None

        chosen = self._sample_n(candidates, self.max_evidence_sentences)
        chosen = sorted(chosen, key=lambda x: int(x["sentence_id"]))

        sids = [int(x["sentence_id"]) for x in chosen]
        sents = [self._norm(x["sentence_text"]) for x in chosen]
        arts = [self._norm(x["article_title"]) for x in chosen]

        return sids, sents, arts, source_type

    def _is_imaging_like_support_sentence(self, s: str) -> bool:
        s = self._norm(s).lower()
        keywords = [
            "on radiograph",
            "on chest radiograph",
            "on x-ray",
            "on xray",
            "on ct",
            "on imaging",
            "radiographic",
            "imaging shows",
            "appears as",
            "seen as",
            "findings include",
            "demonstrates",
            "demonstrated",
            "visible on",
            "characterized by",
        ]
        return any(k in s for k in keywords)

    # --------------------------------------------------
    # load inputs
    # --------------------------------------------------

    def _load_qa(self) -> pd.DataFrame:
        qa = pd.read_parquet(self.qa_parquet).copy()

        required = {
            "qa_row_id",
            "subject_id",
            "report_rel_path",
            "split",
            "concept",
            "findings_text",
            "claim_text",
        }
        missing = required - set(qa.columns)
        if missing:
            raise ValueError(f"QA parquet missing columns: {missing}")

        if "gold_state" not in qa.columns:
            qa["gold_state"] = qa["claim_text"].astype(str).str.extract(
                r"\s+is\s+(present|absent|uncertain)\s*$",
                expand=False,
            )

        qa["qa_row_id"] = pd.to_numeric(qa["qa_row_id"], errors="coerce")
        qa["subject_id"] = pd.to_numeric(qa["subject_id"], errors="coerce")
        qa = qa[~qa["qa_row_id"].isna() & ~qa["subject_id"].isna()].copy()
        qa["qa_row_id"] = qa["qa_row_id"].astype(int)
        qa["subject_id"] = qa["subject_id"].astype(int)

        qa["concept"] = qa["concept"].astype(str).str.strip()
        qa["gold_state"] = qa["gold_state"].astype(str).str.strip()
        qa["case_context"] = qa["findings_text"].astype(str)
        qa["split"] = qa["split"].astype(str).str.strip().str.lower()

        qa = qa[qa["concept"].isin(self.core_concepts)].copy()
        qa = qa[qa["gold_state"].isin(["present", "absent"])].copy()
        qa = qa[qa["split"].isin(["train", "val", "test"])].copy()

        qa = qa.reset_index(drop=True)
        return qa

    def _load_evidence(self) -> pd.DataFrame:
        ev = pd.read_parquet(self.evidence_sentences_parquet).copy()

        required = {
            "sentence_id",
            "sentence_text",
            "article_title",
            "is_chest_relevant",
            "is_definition_like",
            "is_metadata_like",
            "is_fragment_like",
            "matched_concepts_json",
            "concept_states_json",
            "num_matched_concepts",
        }
        missing = required - set(ev.columns)
        if missing:
            raise ValueError(f"Evidence parquet missing columns: {missing}")

        ev["sentence_id"] = pd.to_numeric(ev["sentence_id"], errors="coerce")
        ev = ev[~ev["sentence_id"].isna()].copy()
        ev["sentence_id"] = ev["sentence_id"].astype(int)

        ev["sentence_text"] = ev["sentence_text"].astype(str)
        ev["article_title"] = ev["article_title"].astype(str)

        ev["matched_concepts"] = ev["matched_concepts_json"].apply(self._loads_json_list)
        ev["concept_states"] = ev["concept_states_json"].apply(self._loads_json_dict)

        ev = ev[ev["is_metadata_like"] == False].copy()
        ev = ev[ev["is_fragment_like"] == False].copy()
        ev = ev[ev["is_definition_like"] == False].copy()

        ev = ev.reset_index(drop=True)
        return ev

    # --------------------------------------------------
    # evidence pools
    # --------------------------------------------------

    def _build_evidence_pools(self, ev: pd.DataFrame) -> Dict[str, Dict[str, List[Dict]]]:
        pools: Dict[str, Dict[str, List[Dict]]] = {}
        chest_ev = ev[ev["is_chest_relevant"] == True].copy()

        for concept in sorted(self.core_concepts):
            support_present: List[Dict] = []
            support_absent: List[Dict] = []
            nonsupport_hard: List[Dict] = []
            nonsupport_easy: List[Dict] = []

            for _, r in chest_ev.iterrows():
                concept_states: Dict[str, str] = r["concept_states"]
                matched_concepts: List[str] = r["matched_concepts"]

                row = {
                    "sentence_id": int(r["sentence_id"]),
                    "sentence_text": self._norm(r["sentence_text"]),
                    "article_title": self._norm(r["article_title"]),
                }

                if concept in concept_states:
                    st = str(concept_states[concept]).strip()
                    sent_text = row["sentence_text"]

                    if st == "present":
                        if (concept in sent_text.lower()) and self._is_imaging_like_support_sentence(sent_text):
                            support_present.append(row)
                        else:
                            nonsupport_hard.append(row)

                    elif st == "absent":
                        if (concept in sent_text.lower()) and self._is_imaging_like_support_sentence(sent_text):
                            support_absent.append(row)
                        else:
                            nonsupport_hard.append(row)

                    else:
                        nonsupport_hard.append(row)

                else:
                    if len(matched_concepts) > 0:
                        nonsupport_hard.append(row)
                    else:
                        nonsupport_easy.append(row)

            pools[concept] = {
                "support_present": self._dedup_pool(support_present),
                "support_absent": self._dedup_pool(support_absent),
                "nonsupport_hard": self._dedup_pool(nonsupport_hard),
                "nonsupport_easy": self._dedup_pool(nonsupport_easy),
            }

        return pools

    # --------------------------------------------------
    # row creation
    # --------------------------------------------------

    def _make_one_row(
        self,
        verifier_row_id: int,
        qa_row: pd.Series,
        claim_state: str,
        evidence_pack: Tuple[List[int], List[str], List[str], str],
        row_type: str,
        label_id: int,
    ) -> Dict:
        sids, sents, arts, evidence_source_type = evidence_pack
        label = "supported" if int(label_id) == 1 else "not_supported"

        return asdict(
            VerifierBinaryRow(
                verifier_row_id=int(verifier_row_id),
                qa_row_id=int(qa_row["qa_row_id"]),
                split=str(qa_row["split"]),
                subject_id=int(qa_row["subject_id"]),
                report_rel_path=str(qa_row["report_rel_path"]),
                concept=str(qa_row["concept"]),
                gold_state=str(qa_row["gold_state"]),
                claim_state=str(claim_state),
                claim_text=self._claim_text(str(qa_row["concept"]), str(claim_state)),
                case_context=self._norm(qa_row["case_context"]),
                evidence_sentence_ids_json=json.dumps([int(x) for x in sids]),
                evidence_sentences_json=json.dumps([self._norm(x) for x in sents], ensure_ascii=False),
                evidence_article_titles_json=json.dumps([self._norm(x) for x in arts], ensure_ascii=False),
                row_type=str(row_type),
                evidence_source_type=str(evidence_source_type),
                label=label,
                label_id=int(label_id),
            )
        )

    def _select_nonsupport_evidence(
        self,
        concept_pools: Dict[str, List[Dict]],
        hard_first: bool = True,
    ) -> Optional[Tuple[List[int], List[str], List[str], str]]:
        hard_pool = concept_pools["nonsupport_hard"]
        easy_pool = concept_pools["nonsupport_easy"]

        mixed: List[Dict] = []

        if hard_first:
            mixed.extend(
                self._sample_n(hard_pool, min(len(hard_pool), max(1, self.max_evidence_sentences)))
            )
            if len(mixed) < self.max_evidence_sentences:
                need = self.max_evidence_sentences - len(mixed)
                mixed.extend(self._sample_n(easy_pool, min(len(easy_pool), need)))
        else:
            mixed.extend(
                self._sample_n(easy_pool, min(len(easy_pool), max(1, self.max_evidence_sentences)))
            )
            if len(mixed) < self.max_evidence_sentences:
                need = self.max_evidence_sentences - len(mixed)
                mixed.extend(self._sample_n(hard_pool, min(len(hard_pool), need)))

        if not mixed:
            return None

        mixed = self._dedup_pool(mixed)
        if not mixed:
            return None

        source_type = "nonsupport_hard" if hard_first else "nonsupport_easy"
        return self._take_evidence(mixed, source_type=source_type)

    def _balance_rows(self, out: pd.DataFrame) -> pd.DataFrame:
        """
        Balance within each split x concept group.
        Keep only groups that have all four row types.
        Downsample all row types there to the same size.
        """
        required_row_types = ["A", "B", "C", "D"]
        kept_parts = []

        for (split, concept), g in out.groupby(["split", "concept"], sort=False):
            counts = g["row_type"].value_counts().to_dict()

            if any(rt not in counts for rt in required_row_types):
                continue

            min_count = min(int(counts[rt]) for rt in required_row_types)
            if min_count < self.min_rows_per_type_required:
                continue

            if self.max_rows_per_type_per_concept_split is not None:
                min_count = min(min_count, int(self.max_rows_per_type_per_concept_split))

            for rt in required_row_types:
                gg = g[g["row_type"] == rt]
                if len(gg) > min_count:
                    gg = gg.sample(n=min_count, random_state=self.seed)
                kept_parts.append(gg)

        if not kept_parts:
            raise ValueError("Balancing removed all rows. Check concept coverage and min_rows_per_type_required.")

        out_bal = pd.concat(kept_parts, axis=0).reset_index(drop=True)
        out_bal = out_bal.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        out_bal["verifier_row_id"] = range(len(out_bal))
        return out_bal

    # --------------------------------------------------
    # main
    # --------------------------------------------------

    def run(self) -> pd.DataFrame:
        qa = self._load_qa()
        ev = self._load_evidence()
        pools = self._build_evidence_pools(ev)

        rows: List[Dict] = []
        verifier_row_id = 0

        for _, q in qa.iterrows():
            concept = str(q["concept"])
            gold_state = str(q["gold_state"])
            opp_state = self._opposite_state(gold_state)
            if opp_state is None:
                continue

            concept_pools = pools.get(concept, None)
            if concept_pools is None:
                continue

            support_gold_pool = concept_pools["support_present"] if gold_state == "present" else concept_pools["support_absent"]
            support_opp_pool = concept_pools["support_present"] if opp_state == "present" else concept_pools["support_absent"]

            # C
            c_pack = self._take_evidence(support_gold_pool, source_type=f"support_{gold_state}")
            if c_pack is not None:
                rows.append(
                    self._make_one_row(
                        verifier_row_id=verifier_row_id,
                        qa_row=q,
                        claim_state=gold_state,
                        evidence_pack=c_pack,
                        row_type="C",
                        label_id=1,
                    )
                )
                verifier_row_id += 1

            # D
            d_pack = self._take_evidence(support_opp_pool, source_type=f"support_{opp_state}")
            if d_pack is not None:
                rows.append(
                    self._make_one_row(
                        verifier_row_id=verifier_row_id,
                        qa_row=q,
                        claim_state=opp_state,
                        evidence_pack=d_pack,
                        row_type="D",
                        label_id=0,
                    )
                )
                verifier_row_id += 1

            # A
            a_pack = self._select_nonsupport_evidence(concept_pools, hard_first=True)
            if a_pack is not None:
                rows.append(
                    self._make_one_row(
                        verifier_row_id=verifier_row_id,
                        qa_row=q,
                        claim_state="present",
                        evidence_pack=a_pack,
                        row_type="A",
                        label_id=0,
                    )
                )
                verifier_row_id += 1

            # B
            b_pack = self._select_nonsupport_evidence(concept_pools, hard_first=False)
            if b_pack is not None:
                rows.append(
                    self._make_one_row(
                        verifier_row_id=verifier_row_id,
                        qa_row=q,
                        claim_state="absent",
                        evidence_pack=b_pack,
                        row_type="B",
                        label_id=0,
                    )
                )
                verifier_row_id += 1

        out = pd.DataFrame(rows)

        if out.empty:
            raise ValueError("No verifier rows were created. Check evidence pools and concept coverage.")

        if self.balance_within_split_concept:
            out = self._balance_rows(out)
        elif self.max_rows_per_type_per_concept_split is not None:
            kept_parts = []
            for (split, concept, row_type), g in out.groupby(["split", "concept", "row_type"], sort=False):
                if len(g) <= self.max_rows_per_type_per_concept_split:
                    kept_parts.append(g)
                else:
                    kept_parts.append(g.sample(self.max_rows_per_type_per_concept_split, random_state=self.seed))
            out = pd.concat(kept_parts, axis=0).reset_index(drop=True)
            out["verifier_row_id"] = range(len(out))

        out.to_parquet(self.out_parquet, index=False)

        manifest = {
            "task": "binary_verifier_support",
            "dataset_name": "chexpert_plus",
            "input_definition": "case_context + evidence + claim -> supported / not_supported",
            "labels": {
                "not_supported": 0,
                "supported": 1,
            },
            "row_types": {
                "A": "positive claim + non-support evidence",
                "B": "negative claim + non-support evidence",
                "C": "claim matches gold state + supporting evidence",
                "D": "opposite claim + opposite-support evidence",
            },
            "inputs": {
                "qa_parquet": str(self.qa_parquet.resolve()),
                "evidence_sentences_parquet": str(self.evidence_sentences_parquet.resolve()),
            },
            "outputs": {
                "verifier_binary_parquet": str(self.out_parquet.resolve()),
            },
            "params": {
                "seed": self.seed,
                "max_evidence_sentences": self.max_evidence_sentences,
                "max_rows_per_type_per_concept_split": self.max_rows_per_type_per_concept_split,
                "balance_within_split_concept": self.balance_within_split_concept,
                "min_rows_per_type_required": self.min_rows_per_type_required,
                "core_concepts": sorted(self.core_concepts),
            },
            "counts": {
                "num_rows": int(len(out)),
                "num_supported": int((out["label_id"] == 1).sum()),
                "num_not_supported": int((out["label_id"] == 0).sum()),
                "num_unique_qa_rows": int(out["qa_row_id"].nunique()),
            },
            "counts_by_row_type": {k: int(v) for k, v in out["row_type"].value_counts().to_dict().items()},
            "counts_by_concept": {k: int(v) for k, v in out["concept"].value_counts().to_dict().items()},
            "counts_by_split": {k: int(v) for k, v in out["split"].value_counts().to_dict().items()},
        }

        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {self.manifest_out}")
        print("\nLabel counts:")
        print(out["label"].value_counts(dropna=False).to_string())
        print("\nRow-type counts:")
        print(out["row_type"].value_counts(dropna=False).to_string())
        print("\nSplit counts:")
        print(out["split"].value_counts(dropna=False).to_string())
        print("\nConcept counts:")
        print(out["concept"].value_counts(dropna=False).to_string())

        return out








class OLDRadiologyVignetteMCQRetrieverBuilder:
    """
    Build vignette MCQ retrieval rows in a MIMIC-style verifier schema.

    Expected input CSV columns for your vignette file:
        q_number
        question
        option 1
        option 2
        option 3
        option 4
        answer index

    Also supports variable number of options, e.g. option 5.

    Output label semantics:
        label_id = 1 only for the correct option of a question
        label_id = 0 for all other options
    """

    def __init__(
        self,
        vignette_csv: str,
        evidence_sentences_parquet: str,
        out_dir: str,
        seed: int = 13,
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        retrieval_top_k_full: int = 30,
        max_evidence_sentences: int = 2,
        bm25_cache_name: str = "vignette_bm25.pkl",
        split_name: str = "test",
        default_source_name: str = "vignette",
    ):
        self.vignette_csv = Path(vignette_csv)
        self.evidence_sentences_parquet = Path(evidence_sentences_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        self.tokenizer_name = tokenizer_name
        self.retrieval_top_k_full = int(retrieval_top_k_full)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.split_name = str(split_name)
        self.default_source_name = str(default_source_name)

        self.out_parquet = self.out_dir / "vignette_verifier_retrieval.parquet"
        self.out_csv = self.out_dir / "vignette_verifier_retrieval.csv"
        self.manifest_out = self.out_dir / "manifest_vignette_verifier_retrieval.json"
        self.bm25_cache_path = self.out_dir / bm25_cache_name

        self.word_re = re.compile(r"[a-z0-9]+")
        self.ws_re = re.compile(r"\s+")
        self.sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")
        self.imaging_anchor_re = re.compile(
            r"\b("
            r"opacity|opacities|airspace|interstitial|vascular|congestion|edema|oedema|"
            r"effusion|pneumothorax|atelectatic|atelectasis|consolidation|blunting|"
            r"costophrenic|retrocardiac|bibasilar|basilar|perihilar|hilar|mediastinal|"
            r"right|left|bilateral|diffuse|focal|multifocal|apical|lower|upper|central|"
            r"nodule|mass|calcification|fracture|fractures|rib|ribs|osseous|breast|"
            r"silhouette|cardiomediastinal|cardiac|hemithorax|pleural|parenchymal"
            r")\b",
            re.I,
        )

        self.anchor_stop = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "without",
            "as", "at", "by", "from", "into", "is", "are", "was", "were", "be", "been",
            "this", "that", "these", "those", "what", "which", "who", "where", "when",
            "most", "likely", "diagnosis", "best", "next", "appropriate", "finding",
            "following", "patient", "study", "image", "images", "radiograph", "radiographs",
            "ct", "mri", "xray", "x-ray", "exam", "view", "views",
        }

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        try:
            self._tokenizer.model_max_length = int(1e9)
        except Exception:
            pass

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------

    def _norm(self, s) -> str:
        if s is None:
            return ""
        s = str(s).replace("\r\n", "\n").replace("\r", "\n")
        s = self.ws_re.sub(" ", s)
        return s.strip()

    def _tokens(self, s: str) -> List[str]:
        return self.word_re.findall(self._norm(s).lower())

    def _stable_hash(self, s: str) -> str:
        return hashlib.md5(str(s).encode("utf-8", errors="ignore")).hexdigest()

    def _loads_json_list(self, s) -> List:
        try:
            out = json.loads(self._norm(s))
            return out if isinstance(out, list) else []
        except Exception:
            return []

    # --------------------------------------------------
    # vignette parsing
    # --------------------------------------------------

    def _split_case_and_question(self, question_full_text: str) -> Tuple[str, str]:
        full = self._norm(question_full_text)
        if not full:
            return "", ""

        sents = [x.strip() for x in self.sent_split_re.split(full) if x.strip()]
        if not sents:
            return full, ""

        q_idx = None
        for i in range(len(sents) - 1, -1, -1):
            if sents[i].endswith("?"):
                q_idx = i
                break

        if q_idx is None:
            return full, ""

        question_text = sents[q_idx]
        case_context = self._norm(" ".join(sents[:q_idx]))
        return case_context, question_text

    def _best_anchor_sentence(self, case_context: str) -> str:
        sents = [x.strip() for x in self.sent_split_re.split(self._norm(case_context)) if x.strip()]
        if not sents:
            return ""

        best_sent = ""
        best_score = -10**9

        for s in sents:
            toks = self._tokens(s)
            if len(toks) < 5:
                continue

            imaging_hits = len(self.imaging_anchor_re.findall(s))
            length_bonus = 0.0
            if 7 <= len(toks) <= 30:
                length_bonus = 0.5
            elif len(toks) <= 45:
                length_bonus = 0.2

            score = float(imaging_hits + length_bonus)
            if score > best_score:
                best_score = score
                best_sent = s

        if not best_sent:
            best_sent = sents[0]
        return best_sent

    def _extract_anchor_terms(self, sent: str) -> List[str]:
        toks = self._tokens(sent)
        kept = []
        for t in toks:
            if t in self.anchor_stop:
                continue
            if len(t) < 3:
                continue
            if t not in kept:
                kept.append(t)
        return kept[:10]

    def _load_and_expand_vignettes(self) -> pd.DataFrame:
        df = pd.read_csv(self.vignette_csv).copy()
        cols_lower = {c.lower().strip(): c for c in df.columns}

        required_core = ["q_number", "question", "answer index"]
        missing_core = [c for c in required_core if c not in cols_lower]
        if missing_core:
            raise ValueError(f"Missing required vignette columns: {missing_core}")

        qid_col = cols_lower["q_number"]
        question_col = cols_lower["question"]
        answer_col = cols_lower["answer index"]

        # detect option columns like: option 1, option 2, ..., option 5
        option_cols = []
        for c in df.columns:
            m = re.match(r"^option\s*([0-9]+)$", c.strip().lower())
            if m:
                option_cols.append((c, str(int(m.group(1)))))

        option_cols = sorted(option_cols, key=lambda x: int(x[1]))

        if len(option_cols) < 2:
            raise ValueError("Could not detect at least two option columns like 'option 1', 'option 2', ...")

        rows = []
        qa_row_id = 0

        for _, r in df.iterrows():
            original_question_id = r[qid_col]
            question_full_text = self._norm(r[question_col])
            if not question_full_text:
                continue

            try:
                correct_option_id = str(int(float(r[answer_col])))
            except Exception:
                correct_option_id = self._norm(r[answer_col])

            option_map = {}
            for c, oid in option_cols:
                txt = self._norm(r[c])
                if txt:
                    option_map[oid] = txt

            if len(option_map) < 2:
                continue

            if correct_option_id not in option_map:
                raise ValueError(
                    f"Question {original_question_id}: answer index '{correct_option_id}' "
                    f"not found among detected option ids {sorted(option_map.keys())}"
                )

            case_context, question_text = self._split_case_and_question(question_full_text)
            if not case_context:
                case_context = question_full_text

            anchor_sentence_text = self._best_anchor_sentence(case_context)
            anchor_terms = self._extract_anchor_terms(anchor_sentence_text)

            question_uid = f"{self.default_source_name}_{self._norm(original_question_id)}_{self._stable_hash(question_full_text)[:12]}"
            cur_qa_row_id = qa_row_id
            qa_row_id += 1

            subject_id = int(cur_qa_row_id)

            for option_id, option_text in sorted(option_map.items(), key=lambda x: int(x[0])):
                is_correct = (str(option_id) == str(correct_option_id))

                rows.append({
                    "qa_row_id": int(cur_qa_row_id),
                    "split": self.split_name,
                    "subject_id": subject_id,
                    "report_rel_path": question_uid,
                    "question_uid": question_uid,
                    "dataset_source": self.default_source_name,
                    "original_question_id": original_question_id,
                    "question_full_text": question_full_text,
                    "case_context": case_context,
                    "question_text": question_text,
                    "question_type": "mcq_vignette",
                    "anchor_sentence_text": anchor_sentence_text,
                    "anchor_terms_json": json.dumps(anchor_terms, ensure_ascii=False),
                    "option_id": str(option_id),
                    "claim_text": option_text,
                    "correct_option_id": str(correct_option_id),
                    "is_correct_option": bool(is_correct),
                    "mcq_label": str(correct_option_id),
                    "num_options": int(len(option_map)),
                    # MIMIC-style compatibility fields
                    "concept": "",
                    "gold_state": str(correct_option_id),
                    "claim_state": str(option_id),
                    "label": "supported" if is_correct else "not_supported",
                    "label_id": int(1 if is_correct else 0),
                    "row_type": "mcq_retrieved",
                })

        out = pd.DataFrame(rows)

        if out.empty:
            raise ValueError("No vignette-option rows were created.")

        pos_counts = out.groupby("qa_row_id")["label_id"].sum()
        bad = pos_counts[(pos_counts != 1)]
        if len(bad) > 0:
            raise ValueError(
                f"Invalid label construction: {len(bad)} question(s) do not have exactly one positive option."
            )

        return out.reset_index(drop=True)

    # --------------------------------------------------
    # evidence loading
    # --------------------------------------------------

    def _load_evidence(self) -> pd.DataFrame:
        ev = pd.read_parquet(self.evidence_sentences_parquet).copy()

        required = {
            "sentence_id",
            "sentence_text",
            "article_title",
            "is_chest_relevant",
            "is_definition_like",
            "is_metadata_like",
            "is_fragment_like",
        }
        missing = required - set(ev.columns)
        if missing:
            raise ValueError(f"Evidence parquet missing columns: {missing}")

        ev["sentence_id"] = pd.to_numeric(ev["sentence_id"], errors="coerce")
        ev = ev[~ev["sentence_id"].isna()].copy()
        ev["sentence_id"] = ev["sentence_id"].astype(int)

        ev = ev[ev["is_metadata_like"] == False].copy()
        ev = ev[ev["is_definition_like"] == False].copy()
        ev = ev[ev["is_fragment_like"] == False].copy()
        ev = ev[ev["is_chest_relevant"] == True].copy()

        ev["sentence_text"] = ev["sentence_text"].astype(str)
        ev["article_title"] = ev["article_title"].astype(str)

        return ev.reset_index(drop=True)

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    def _build_bm25(self, ev: pd.DataFrame):
        if self.bm25_cache_path.exists():
            with open(self.bm25_cache_path, "rb") as f:
                return pickle.load(f)

        postings: Dict[str, List[Tuple[int, int]]] = {}
        dfreq: Dict[str, int] = {}
        sent_ids = ev["sentence_id"].tolist()
        doc_len = np.zeros(len(ev), dtype=np.int32)

        for i, text in enumerate(tqdm(ev["sentence_text"].tolist(), desc="Building vignette BM25", unit="sent")):
            toks = self._tokens(text)
            doc_len[i] = len(toks)

            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            for t, c in tf.items():
                postings.setdefault(t, []).append((i, c))
                dfreq[t] = dfreq.get(t, 0) + 1

        n = len(ev)
        avgdl = float(doc_len.mean()) if n else 0.0
        idf = {t: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for t, df in dfreq.items()}

        state = {
            "sent_ids": sent_ids,
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

    def _bm25_search(self, query: str, bm25_state, topk: int) -> List[int]:
        toks = self._tokens(query)
        if not toks:
            return []

        sent_ids = bm25_state["sent_ids"]
        postings = bm25_state["postings"]
        idf = bm25_state["idf"]
        doc_len = bm25_state["doc_len"]
        avgdl = bm25_state["avgdl"]
        k1 = bm25_state["k1"]
        b = bm25_state["b"]

        scores = {}
        for t in toks:
            plist = postings.get(t, [])
            idft = idf.get(t, 0.0)
            for idx, tf in plist:
                dl = float(doc_len[idx])
                denom = tf + k1 * (1.0 - b + b * dl / (avgdl + 1e-12))
                s = idft * (tf * (k1 + 1.0)) / (denom + 1e-12)
                scores[idx] = scores.get(idx, 0.0) + float(s)

        if not scores:
            return []

        best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [int(sent_ids[i]) for i, _ in best]

    # --------------------------------------------------
    # retrieval
    # --------------------------------------------------

    def _build_query(self, claim_text: str, anchor_terms: List[str], question_text: str, case_context: str) -> str:
        parts = [self._norm(claim_text)]

        if anchor_terms:
            parts.append(" ".join(anchor_terms[:10]))
        if self._norm(question_text):
            parts.append(self._norm(question_text))
        if self._norm(case_context):
            parts.append(self._norm(case_context))

        return "\n\n".join([p for p in parts if p])

    def _attach_retrieval(self, qa_df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
        bm25_state = self._build_bm25(ev)

        sent_meta = {
            int(r["sentence_id"]): {
                "sentence_text": str(r["sentence_text"]),
                "article_title": str(r["article_title"]),
            }
            for _, r in ev.iterrows()
        }

        rows = []
        verifier_row_id = 0

        for _, r in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Retrieving vignette evidence", unit="row"):
            anchor_terms = self._loads_json_list(r["anchor_terms_json"])
            retrieval_query = self._build_query(
                claim_text=r["claim_text"],
                anchor_terms=anchor_terms,
                question_text=r["question_text"],
                case_context=r["case_context"],
            )

            retrieved_ids_full = self._bm25_search(retrieval_query, bm25_state, self.retrieval_top_k_full)
            retrieved_ids_full = [sid for sid in retrieved_ids_full if sid in sent_meta]
            retrieved_ids_shown = retrieved_ids_full[: self.max_evidence_sentences]

            retrieved_sents = [self._norm(sent_meta[sid]["sentence_text"]) for sid in retrieved_ids_shown]
            retrieved_titles = [self._norm(sent_meta[sid]["article_title"]) for sid in retrieved_ids_shown]

            row = dict(r)
            row["verifier_row_id"] = int(verifier_row_id)
            row["evidence_sentence_ids_json"] = json.dumps([int(x) for x in retrieved_ids_shown])
            row["evidence_sentences_json"] = json.dumps(retrieved_sents, ensure_ascii=False)
            row["evidence_article_titles_json"] = json.dumps(retrieved_titles, ensure_ascii=False)
            row["retrieved_full_sentence_ids_json"] = json.dumps([int(x) for x in retrieved_ids_full])
            row["retrieval_query"] = retrieval_query
            row["retrieval_top_k_full"] = int(self.retrieval_top_k_full)
            row["max_evidence_sentences"] = int(self.max_evidence_sentences)
            row["evidence_source_type"] = f"bm25_top{self.retrieval_top_k_full}"

            rows.append(row)
            verifier_row_id += 1

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # schema + write
    # --------------------------------------------------

    def _finalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        wanted_order = [
            "verifier_row_id",
            "qa_row_id",
            "split",
            "subject_id",
            "report_rel_path",
            "concept",
            "gold_state",
            "claim_state",
            "claim_text",
            "case_context",
            "evidence_sentence_ids_json",
            "evidence_sentences_json",
            "evidence_article_titles_json",
            "row_type",
            "evidence_source_type",
            "label",
            "label_id",
            "dataset_source",
            "original_question_id",
            "question_uid",
            "question_full_text",
            "question_text",
            "question_type",
            "anchor_sentence_text",
            "anchor_terms_json",
            "option_id",
            "correct_option_id",
            "is_correct_option",
            "mcq_label",
            "num_options",
            "retrieval_query",
            "retrieval_top_k_full",
            "max_evidence_sentences",
            "retrieved_full_sentence_ids_json",
        ]

        for c in wanted_order:
            if c not in df.columns:
                if c in {"label_id", "verifier_row_id", "qa_row_id", "subject_id", "num_options", "retrieval_top_k_full", "max_evidence_sentences"}:
                    df[c] = 0
                elif c == "is_correct_option":
                    df[c] = False
                else:
                    df[c] = ""

        df = df[wanted_order].copy()

        int_cols = ["verifier_row_id", "qa_row_id", "subject_id", "label_id", "num_options", "retrieval_top_k_full", "max_evidence_sentences"]
        for c in int_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        df["is_correct_option"] = df["is_correct_option"].astype(bool)

        return df

    def run(self) -> pd.DataFrame:
        qa_df = self._load_and_expand_vignettes()
        ev = self._load_evidence()
        out = self._attach_retrieval(qa_df, ev)
        out = self._finalize_schema(out)

        pos_counts = out.groupby("qa_row_id")["label_id"].sum()
        bad = pos_counts[(pos_counts != 1)]
        if len(bad) > 0:
            raise ValueError(
                f"Final dataset invalid: {len(bad)} question(s) do not have exactly one positive option."
            )

        out.to_parquet(self.out_parquet, index=False)
        out.to_csv(self.out_csv, index=False)

        manifest = {
            "task": "vignette_mcq_retrieval_for_binary_verifier",
            "input_definition": "case_context + retrieved_evidence + option_claim -> supported / not_supported",
            "labels": {"not_supported": 0, "supported": 1},
            "positive_rule": "Only the correct option for each question has label_id=1.",
            "inputs": {
                "vignette_csv": str(self.vignette_csv.resolve()),
                "evidence_sentences_parquet": str(self.evidence_sentences_parquet.resolve()),
            },
            "outputs": {
                "parquet": str(self.out_parquet.resolve()),
                "csv": str(self.out_csv.resolve()),
            },
            "params": {
                "seed": self.seed,
                "tokenizer_name": self.tokenizer_name,
                "retrieval_top_k_full": self.retrieval_top_k_full,
                "max_evidence_sentences": self.max_evidence_sentences,
                "split_name": self.split_name,
            },
            "counts": {
                "num_rows": int(len(out)),
                "num_questions": int(out["qa_row_id"].nunique()),
                "num_supported": int((out["label_id"] == 1).sum()),
                "num_not_supported": int((out["label_id"] == 0).sum()),
            },
        }

        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {self.out_csv}")
        print(f"Wrote: {self.manifest_out}")
        print("\nLabel counts:")
        print(out["label"].value_counts(dropna=False).to_string())

        return out






class RadiologyVignetteMCQRetrieverBuilder:
    """
    Build vignette MCQ retrieval rows for downstream MCQ scoring.

    Expected input CSV columns:
        q_number
        question
        option 1
        option 2
        ...
        option N
        answer index

    Output schema is aligned with the vignette test dataloader and prediction code.
    """

    def __init__(
        self,
        vignette_csv: str,
        evidence_sentences_parquet: str,
        out_dir: str,
        seed: int = 13,
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        retrieval_top_k_full: int = 30,
        max_evidence_sentences: int = 2,
        bm25_cache_name: str = "vignette_bm25.pkl",
        split_name: str = "test",
        default_source_name: str = "vignette",
    ):
        self.vignette_csv = Path(vignette_csv)
        self.evidence_sentences_parquet = Path(evidence_sentences_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        self.tokenizer_name = str(tokenizer_name)
        self.retrieval_top_k_full = int(retrieval_top_k_full)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.split_name = str(split_name)
        self.default_source_name = str(default_source_name)

        self.out_parquet = self.out_dir / "vignette_verifier_retrieval.parquet"
        self.out_csv = self.out_dir / "vignette_verifier_retrieval.csv"
        self.manifest_out = self.out_dir / "manifest_vignette_verifier_retrieval.json"
        self.bm25_cache_path = self.out_dir / bm25_cache_name

        self.word_re = re.compile(r"[a-z0-9]+")
        self.ws_re = re.compile(r"\s+")
        self.sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")
        self.imaging_anchor_re = re.compile(
            r"\b("
            r"opacity|opacities|airspace|interstitial|vascular|congestion|edema|oedema|"
            r"effusion|pneumothorax|atelectatic|atelectasis|consolidation|blunting|"
            r"costophrenic|retrocardiac|bibasilar|basilar|perihilar|hilar|mediastinal|"
            r"right|left|bilateral|diffuse|focal|multifocal|apical|lower|upper|central|"
            r"nodule|mass|calcification|fracture|fractures|rib|ribs|osseous|breast|"
            r"silhouette|cardiomediastinal|cardiac|hemithorax|pleural|parenchymal"
            r")\b",
            re.I,
        )

        self.anchor_stop = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "without",
            "as", "at", "by", "from", "into", "is", "are", "was", "were", "be", "been",
            "this", "that", "these", "those", "what", "which", "who", "where", "when",
            "most", "likely", "diagnosis", "best", "next", "appropriate", "finding",
            "following", "patient", "study", "image", "images", "radiograph", "radiographs",
            "ct", "mri", "xray", "x-ray", "exam", "view", "views",
        }

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        try:
            self._tokenizer.model_max_length = int(1e9)
        except Exception:
            pass

    # --------------------------------------------------
    # basic helpers
    # --------------------------------------------------

    def _norm(self, s) -> str:
        if s is None:
            return ""
        s = str(s).replace("\r\n", "\n").replace("\r", "\n")
        s = self.ws_re.sub(" ", s)
        return s.strip()

    def _tokens(self, s: str) -> List[str]:
        return self.word_re.findall(self._norm(s).lower())

    def _stable_hash(self, s: str) -> str:
        return hashlib.md5(str(s).encode("utf-8", errors="ignore")).hexdigest()

    def _loads_json_list(self, s) -> List:
        try:
            out = json.loads(self._norm(s))
            return out if isinstance(out, list) else []
        except Exception:
            return []

    # --------------------------------------------------
    # vignette parsing
    # --------------------------------------------------

    def _split_case_and_question(self, question_full_text: str) -> Tuple[str, str]:
        full = self._norm(question_full_text)
        if not full:
            return "", ""

        sents = [x.strip() for x in self.sent_split_re.split(full) if x.strip()]
        if not sents:
            return full, ""

        q_idx = None
        for i in range(len(sents) - 1, -1, -1):
            if sents[i].endswith("?"):
                q_idx = i
                break

        if q_idx is None:
            return full, ""

        question_text = sents[q_idx]
        case_context = self._norm(" ".join(sents[:q_idx]))
        return case_context, question_text

    def _best_anchor_sentence(self, case_context: str) -> str:
        sents = [x.strip() for x in self.sent_split_re.split(self._norm(case_context)) if x.strip()]
        if not sents:
            return ""

        best_sent = ""
        best_score = -10**9

        for s in sents:
            toks = self._tokens(s)
            if len(toks) < 5:
                continue

            imaging_hits = len(self.imaging_anchor_re.findall(s))
            length_bonus = 0.0
            if 7 <= len(toks) <= 30:
                length_bonus = 0.5
            elif len(toks) <= 45:
                length_bonus = 0.2

            score = float(imaging_hits + length_bonus)
            if score > best_score:
                best_score = score
                best_sent = s

        return best_sent if best_sent else sents[0]

    def _extract_anchor_terms(self, sent: str) -> List[str]:
        toks = self._tokens(sent)
        kept = []
        for t in toks:
            if t in self.anchor_stop:
                continue
            if len(t) < 3:
                continue
            if t not in kept:
                kept.append(t)
        return kept[:10]

    def _load_and_expand_vignettes(self) -> pd.DataFrame:
        df = pd.read_csv(self.vignette_csv).copy()
        cols_lower = {c.lower().strip(): c for c in df.columns}

        required_core = ["q_number", "question", "answer index"]
        missing_core = [c for c in required_core if c not in cols_lower]
        if missing_core:
            raise ValueError(f"Missing required vignette columns: {missing_core}")

        qid_col = cols_lower["q_number"]
        question_col = cols_lower["question"]
        answer_col = cols_lower["answer index"]

        option_cols = []
        for c in df.columns:
            m = re.match(r"^option\s*([0-9]+)$", c.strip().lower())
            if m:
                option_cols.append((c, int(m.group(1))))
        option_cols = sorted(option_cols, key=lambda x: x[1])

        if len(option_cols) < 2:
            raise ValueError("Could not detect at least two option columns like 'option 1', 'option 2', ...")

        rows = []
        qa_row_id = 0

        for _, r in df.iterrows():
            original_question_id = r[qid_col]
            question_full_text = self._norm(r[question_col])
            if not question_full_text:
                continue

            try:
                correct_option_id = int(float(r[answer_col]))
            except Exception:
                raise ValueError(f"Invalid answer index for question {original_question_id}: {r[answer_col]}")

            option_map: Dict[int, str] = {}
            for c, oid in option_cols:
                txt = self._norm(r[c])
                if txt:
                    option_map[int(oid)] = txt

            if len(option_map) < 2:
                continue

            if correct_option_id not in option_map:
                raise ValueError(
                    f"Question {original_question_id}: answer index '{correct_option_id}' "
                    f"not found among option ids {sorted(option_map.keys())}"
                )

            case_context, question_text = self._split_case_and_question(question_full_text)
            if not case_context:
                case_context = question_full_text

            anchor_sentence_text = self._best_anchor_sentence(case_context)
            anchor_terms = self._extract_anchor_terms(anchor_sentence_text)

            question_uid = (
                f"{self.default_source_name}_"
                f"{self._norm(original_question_id)}_"
                f"{self._stable_hash(question_full_text)[:12]}"
            )

            cur_qa_row_id = int(qa_row_id)
            qa_row_id += 1

            for option_id, option_text in sorted(option_map.items(), key=lambda x: x[0]):
                is_correct = bool(int(option_id) == int(correct_option_id))

                rows.append({
                    "qa_row_id": cur_qa_row_id,
                    "split": self.split_name,
                    "case_context": case_context,
                    "claim_text": option_text,
                    "label_id": int(1 if is_correct else 0),
                    "label": "supported" if is_correct else "not_supported",
                    "question_uid": question_uid,
                    "option_id": int(option_id),
                    "correct_option_id": int(correct_option_id),
                    "mcq_label": str(correct_option_id),
                    "verifier_row_id": -1,
                    "question_text": question_text,
                    "question_full_text": question_full_text,
                    "anchor_sentence_text": anchor_sentence_text,
                    "anchor_terms_json": json.dumps(anchor_terms, ensure_ascii=False),
                    "original_question_id": original_question_id,
                    "num_options": int(len(option_map)),
                    "dataset_source": self.default_source_name,
                    "question_type": "mcq_vignette",
                    "subject_id": int(cur_qa_row_id),
                    "report_rel_path": question_uid,
                })

        out = pd.DataFrame(rows)
        if out.empty:
            raise ValueError("No vignette-option rows were created.")

        pos_counts = out.groupby("qa_row_id")["label_id"].sum()
        bad = pos_counts[pos_counts != 1]
        if len(bad) > 0:
            raise ValueError(
                f"Invalid MCQ construction: {len(bad)} question(s) do not have exactly one positive option."
            )

        return out.reset_index(drop=True)

    # --------------------------------------------------
    # evidence loading
    # --------------------------------------------------

    def _load_evidence(self) -> pd.DataFrame:
        ev = pd.read_parquet(self.evidence_sentences_parquet).copy()

        required = {
            "sentence_id",
            "sentence_text",
            "article_title",
            "is_chest_relevant",
            "is_definition_like",
            "is_metadata_like",
            "is_fragment_like",
        }
        missing = required - set(ev.columns)
        if missing:
            raise ValueError(f"Evidence parquet missing columns: {missing}")

        ev["sentence_id"] = pd.to_numeric(ev["sentence_id"], errors="coerce")
        ev = ev[~ev["sentence_id"].isna()].copy()
        ev["sentence_id"] = ev["sentence_id"].astype(int)

        ev = ev[ev["is_metadata_like"] == False].copy()
        ev = ev[ev["is_definition_like"] == False].copy()
        ev = ev[ev["is_fragment_like"] == False].copy()
        ev = ev[ev["is_chest_relevant"] == True].copy()

        ev["sentence_text"] = ev["sentence_text"].astype(str)
        ev["article_title"] = ev["article_title"].astype(str)

        return ev.reset_index(drop=True)

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    def _build_bm25(self, ev: pd.DataFrame):
        if self.bm25_cache_path.exists():
            with open(self.bm25_cache_path, "rb") as f:
                return pickle.load(f)

        postings: Dict[str, List[Tuple[int, int]]] = {}
        dfreq: Dict[str, int] = {}
        sent_ids = ev["sentence_id"].tolist()
        doc_len = np.zeros(len(ev), dtype=np.int32)

        for i, text in enumerate(tqdm(ev["sentence_text"].tolist(), desc="Building vignette BM25", unit="sent")):
            toks = self._tokens(text)
            doc_len[i] = len(toks)

            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            for t, c in tf.items():
                postings.setdefault(t, []).append((i, c))
                dfreq[t] = dfreq.get(t, 0) + 1

        n = len(ev)
        avgdl = float(doc_len.mean()) if n else 0.0
        idf = {t: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for t, df in dfreq.items()}

        state = {
            "sent_ids": sent_ids,
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

    def _bm25_search(self, query: str, bm25_state, topk: int) -> List[int]:
        toks = self._tokens(query)
        if not toks:
            return []

        sent_ids = bm25_state["sent_ids"]
        postings = bm25_state["postings"]
        idf = bm25_state["idf"]
        doc_len = bm25_state["doc_len"]
        avgdl = bm25_state["avgdl"]
        k1 = bm25_state["k1"]
        b = bm25_state["b"]

        scores = {}
        for t in toks:
            plist = postings.get(t, [])
            idft = idf.get(t, 0.0)
            for idx, tf in plist:
                dl = float(doc_len[idx])
                denom = tf + k1 * (1.0 - b + b * dl / (avgdl + 1e-12))
                s = idft * (tf * (k1 + 1.0)) / (denom + 1e-12)
                scores[idx] = scores.get(idx, 0.0) + float(s)

        if not scores:
            return []

        best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [int(sent_ids[i]) for i, _ in best]

    # --------------------------------------------------
    # retrieval
    # --------------------------------------------------

    def _build_query(
        self,
        claim_text: str,
        anchor_terms: List[str],
        question_text: str,
        case_context: str,
    ) -> str:
        parts = [self._norm(claim_text)]

        if anchor_terms:
            parts.append(" ".join(anchor_terms[:10]))
        if self._norm(question_text):
            parts.append(self._norm(question_text))
        if self._norm(case_context):
            parts.append(self._norm(case_context))

        return "\n\n".join([p for p in parts if p])

    def _attach_retrieval(self, qa_df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
        bm25_state = self._build_bm25(ev)

        sent_meta = {
            int(r["sentence_id"]): {
                "sentence_text": str(r["sentence_text"]),
                "article_title": str(r["article_title"]),
            }
            for _, r in ev.iterrows()
        }

        rows = []
        verifier_row_id = 0

        iterator = qa_df.itertuples(index=False)
        for r in tqdm(iterator, total=len(qa_df), desc="Retrieving vignette evidence", unit="row"):
            anchor_terms = self._loads_json_list(r.anchor_terms_json)

            retrieval_query = self._build_query(
                claim_text=r.claim_text,
                anchor_terms=anchor_terms,
                question_text=r.question_text,
                case_context=r.case_context,
            )

            retrieved_ids_full = self._bm25_search(retrieval_query, bm25_state, self.retrieval_top_k_full)
            retrieved_ids_full = [sid for sid in retrieved_ids_full if sid in sent_meta]
            retrieved_ids_shown = retrieved_ids_full[: self.max_evidence_sentences]

            retrieved_sents = [self._norm(sent_meta[sid]["sentence_text"]) for sid in retrieved_ids_shown]
            retrieved_titles = [self._norm(sent_meta[sid]["article_title"]) for sid in retrieved_ids_shown]

            row = {
                "verifier_row_id": int(verifier_row_id),
                "qa_row_id": int(r.qa_row_id),
                "split": str(r.split),
                "subject_id": int(r.subject_id),
                "report_rel_path": str(r.report_rel_path),
                "case_context": self._norm(r.case_context),
                "claim_text": self._norm(r.claim_text),
                "evidence_sentences_json": json.dumps(retrieved_sents, ensure_ascii=False),
                "label_id": int(r.label_id),
                "question_uid": str(r.question_uid),
                "option_id": int(r.option_id),
                "correct_option_id": int(r.correct_option_id),
                "mcq_label": str(r.mcq_label),
                "dataset_source": str(r.dataset_source),
                "original_question_id": r.original_question_id,
                "question_full_text": self._norm(r.question_full_text),
                "question_text": self._norm(r.question_text),
                "question_type": str(r.question_type),
                "anchor_sentence_text": self._norm(r.anchor_sentence_text),
                "anchor_terms_json": str(r.anchor_terms_json),
                "num_options": int(r.num_options),
                "evidence_sentence_ids_json": json.dumps([int(x) for x in retrieved_ids_shown]),
                "evidence_article_titles_json": json.dumps(retrieved_titles, ensure_ascii=False),
                "evidence_source_type": f"bm25_top{self.retrieval_top_k_full}",
                "retrieval_query": retrieval_query,
                "retrieval_top_k_full": int(self.retrieval_top_k_full),
                "max_evidence_sentences": int(self.max_evidence_sentences),
                "retrieved_full_sentence_ids_json": json.dumps([int(x) for x in retrieved_ids_full]),
            }
            rows.append(row)
            verifier_row_id += 1

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # final schema
    # --------------------------------------------------

    def _finalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        wanted_order = [
            "verifier_row_id",
            "qa_row_id",
            "split",
            "subject_id",
            "report_rel_path",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
            "question_uid",
            "option_id",
            "correct_option_id",
            "mcq_label",
            "evidence_sentence_ids_json",
            "evidence_article_titles_json",
            "dataset_source",
            "original_question_id",
            "question_full_text",
            "question_text",
            "question_type",
            "anchor_sentence_text",
            "anchor_terms_json",
            "num_options",
            "evidence_source_type",
            "retrieval_query",
            "retrieval_top_k_full",
            "max_evidence_sentences",
            "retrieved_full_sentence_ids_json",
        ]

        for c in wanted_order:
            if c not in df.columns:
                if c in {
                    "verifier_row_id", "qa_row_id", "subject_id", "label_id",
                    "option_id", "correct_option_id", "num_options",
                    "retrieval_top_k_full", "max_evidence_sentences"
                }:
                    df[c] = 0
                else:
                    df[c] = ""

        df = df[wanted_order].copy()

        int_cols = [
            "verifier_row_id",
            "qa_row_id",
            "subject_id",
            "label_id",
            "option_id",
            "correct_option_id",
            "num_options",
            "retrieval_top_k_full",
            "max_evidence_sentences",
        ]
        for c in int_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        return df

    # --------------------------------------------------
    # run
    # --------------------------------------------------

    def run(self) -> pd.DataFrame:
        qa_df = self._load_and_expand_vignettes()
        ev = self._load_evidence()
        out = self._attach_retrieval(qa_df, ev)
        out = self._finalize_schema(out)

        pos_counts = out.groupby("qa_row_id")["label_id"].sum()
        bad = pos_counts[pos_counts != 1]
        if len(bad) > 0:
            raise ValueError(
                f"Final dataset invalid: {len(bad)} question(s) do not have exactly one positive option."
            )

        out.to_parquet(self.out_parquet, index=False)
        out.to_csv(self.out_csv, index=False)

        manifest = {
            "task": "vignette_mcq_option_scoring",
            "input_definition": "case_context + retrieved_evidence + option_claim -> score_supported",
            "target_definition": "label_id=1 only for the correct option of each question",
            "inputs": {
                "vignette_csv": str(self.vignette_csv.resolve()),
                "evidence_sentences_parquet": str(self.evidence_sentences_parquet.resolve()),
            },
            "outputs": {
                "parquet": str(self.out_parquet.resolve()),
                "csv": str(self.out_csv.resolve()),
            },
            "params": {
                "seed": self.seed,
                "tokenizer_name": self.tokenizer_name,
                "retrieval_top_k_full": self.retrieval_top_k_full,
                "max_evidence_sentences": self.max_evidence_sentences,
                "split_name": self.split_name,
            },
            "counts": {
                "num_rows": int(len(out)),
                "num_questions": int(out["qa_row_id"].nunique()),
                "num_positive_rows": int((out["label_id"] == 1).sum()),
                "num_negative_rows": int((out["label_id"] == 0).sum()),
            },
        }

        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {self.out_csv}")
        print(f"Wrote: {self.manifest_out}")
        print("\nLabel counts:")
        print(out["label_id"].value_counts(dropna=False).to_string())

        return out







if __name__ == "__main__":
    # patient_split(csv_path=os.path.join(base_path_chexpertData, "df_chexpert_plus_240401.csv"),
    #               outpath=os.path.join(base_path_chexpertData, "chexpert_plus_reports.csv"))

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_chexpertData = params["Preprocessing"]["base_path_mimicData"]
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]


    ''' steps 0 to 5 '''
    # builder = CheXpertPlusGroundingDatasetBuilder(
    #     master_csv=os.path.join(base_path_chexpertData, "chexpert_plus_reports.csv"),
    #     out_dir=os.path.join(base_path_chexpertData, "grounding_dataset_chexpert_plus"),
    #     seed=13,
    # )
    # builder.run_all()



    ''' step 6 '''
    # verifier_builder = CheXpertPlusVerifierBinaryDatasetBuilder(
    #     qa_parquet=os.path.join(base_path_chexpertData, "grounding_dataset_chexpert_plus", "qa_grounding_main_with_anchors.parquet"),
    #     evidence_sentences_parquet=os.path.join(base_path_evidenceData, "evidence_sentences_trainval.parquet"),
    #     # evidence_sentences_parquet=os.path.join(base_path_evidenceData, "evidence_sentences_test.parquet"),
    #     out_dir=os.path.join(base_path_evidenceData, "verifier_binary_dataset_chexpert_plus"),
    #     seed=13,
    #     max_evidence_sentences=2,
    #     max_rows_per_type_per_concept_split=None,
    #     balance_within_split_concept=True,
    #     min_rows_per_type_required=1,
    # )
    # verifier_df = verifier_builder.run()





    ''' Vignette '''
    # builder = OLDRadiologyVignetteMCQRetrieverBuilder(
    #     vignette_csv=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
    #     evidence_sentences_parquet=os.path.join(base_path_evidenceData, "evidence_sentences_trainval.parquet"),
    #     # evidence_sentences_parquet=os.path.join(base_path_evidenceData, "evidence_sentences_test.parquet"),
    #     out_dir=os.path.join(base_path_evidenceData, "vignette_option_retrieval_4"),
    #     seed=13,
    #     tokenizer_name=params["Preprocessing"]["tokenizer_name"],
    #     retrieval_top_k_full=30,
    #     max_evidence_sentences=2,
    #     bm25_cache_name="vignette_bm25.pkl",
    #     split_name="test",
    #     default_source_name="vignette",
    # )
    # vignette_df_4 = builder.run()
    #
    # builder = OLDRadiologyVignetteMCQRetrieverBuilder(
    #     vignette_csv=os.path.join(base_path_chexpertData, "Vignette_5option.csv"),
    #     evidence_sentences_parquet=os.path.join(base_path_evidenceData, "evidence_sentences_trainval.parquet"),
    #     # evidence_sentences_parquet=os.path.join(base_path_evidenceData, "evidence_sentences_test.parquet"),
    #     out_dir=os.path.join(base_path_evidenceData, "vignette_option_retrieval_5"),
    #     seed=13,
    #     tokenizer_name=params["Preprocessing"]["tokenizer_name"],
    #     retrieval_top_k_full=30,
    #     max_evidence_sentences=2,
    #     bm25_cache_name="vignette_bm25.pkl",
    #     split_name="test",
    #     default_source_name="vignette",
    # )
    # vignette_df_5 = builder.run()
    #
    #
    # df = pd.concat([vignette_df_4, vignette_df_5], ignore_index=True)
    # os.makedirs(os.path.join(base_path_evidenceData, "vignette_option_retrieval"), exist_ok=True)
    # df.to_parquet(os.path.join(base_path_evidenceData, "vignette_option_retrieval/vignette_option_retrieval.parquet"), index=False)
    # df.to_csv(os.path.join(base_path_evidenceData, "vignette_option_retrieval/vignette_option_retrieval.csv"), index=False)








    builder = RadiologyVignetteMCQRetrieverBuilder(
        vignette_csv=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
        evidence_sentences_parquet=os.path.join(
            base_path_evidenceData,
            "evidence_sentences_trainval.parquet",
        ),
        # evidence_sentences_parquet=os.path.join(
        #     base_path_evidenceData,
        #     "evidence_sentences_test.parquet",
        # ),
        out_dir=os.path.join(base_path_evidenceData, "vignette_option_retrieval_4"),
        seed=13,
        tokenizer_name=params["Preprocessing"]["tokenizer_name"],
        retrieval_top_k_full=30,
        max_evidence_sentences=2,
        bm25_cache_name="vignette_bm25.pkl",
        split_name="test",
        default_source_name="vignette",
    )
    vignette_df_4 = builder.run()




