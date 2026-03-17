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





class MimicGroundingDatasetBuilder:
    """
    Steps 0-5 for a radiology grounding benchmark.

    Outputs:
      reports_master.parquet
      reports_sections.parquet
      reports_impression_labels.parquet
      qa_grounding_raw.parquet
      qa_grounding_main.parquet
      qa_grounding_easy_control.parquet
      qa_grounding_main_with_anchors.parquet
      benchmark_manifest.json
    """

    def __init__(self, master_csv: str, reports_root: str, out_dir: str, seed: int = 13):
        self.master_csv = Path(master_csv)
        self.reports_root = Path(reports_root)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)

        self.section_re = re.compile(r"^\s*(FINDINGS|IMPRESSION)\s*:\s*$", re.I)
        self.sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")
        self.word_re = re.compile(r"[a-z0-9]+")

        self.communication_re = re.compile(
            r"\b(discussed with|communicated to|called to|paged|notified|by phone|phone at|results were communicated)\b",
            re.I,
        )
        self.admin_inline_re = re.compile(
            r"\b(recommend short[- ]term follow[- ]up|these findings were discovered|covering md|immediately paged)\b",
            re.I,
        )

        self.negation_re = re.compile(
            r"\b(no|without|absent|negative for|free of|no evidence of|no focal|neither)\b", re.I
        )
        self.uncertainty_re = re.compile(
            r"\b(possible|possibly|may|might|likely|unlikely|cannot exclude|could represent|suggestive of|question of)\b",
            re.I,
        )
        self.resolution_re = re.compile(
            r"\b(resolved|improved|decreased|cleared|near complete resolution)\b", re.I
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

        # final concept set
        self.concepts = {
            "pleural effusion": re.compile(r"\bpleural effusions?\b|\beffusions?\b", re.I),
            "pneumothorax": re.compile(r"\bpneumothorax\b|\bpneumothoraces\b", re.I),
            "cardiomegaly": re.compile(
                r"\bcardiomegaly\b|"
                r"\benlarged cardiac silhouette\b|"
                r"\bcardiac silhouette is enlarged\b|"
                r"\bcardiac silhouette enlarged\b|"
                r"\benlarged heart\b|"
                r"\bheart is enlarged\b|"
                r"\bheart size is enlarged\b|"
                r"\bheart size enlarged\b|"
                r"\bheart is mildly enlarged\b|"
                r"\bmild cardiomegaly\b",
                re.I,
            ),
            "atelectasis": re.compile(
                r"\batelectasis\b|"
                r"\batelectatic\b|"
                r"\bsubsegmental atelectatic\b|"
                r"\bbibasilar atelectatic\b|"
                r"\bbasal atelectatic\b",
                re.I,
            ),
            "consolidation": re.compile(
                r"\bconsolidation\b|"
                r"\blobar consolidation\b|"
                r"\bfocal consolidation\b|"
                r"\bairspace consolidation\b",
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
                r"\bhilar vasculature\b",
                re.I,
            ),
            "interstitial edema": re.compile(
                r"\binterstitial edema\b|"
                r"\binterstitial pulmonary edema\b",
                re.I,
            ),
            "rib fracture": re.compile(
                r"\brib fractures?\b|"
                r"\bfractured ribs?\b",
                re.I,
            ),
            "fracture": re.compile(
                r"\bfracture\b|\bfractures\b",
                re.I,
            ),
        }

        # direct shortcut detection in FINDINGS
        self.shortcut_patterns = {
            "pleural effusion": re.compile(
                r"\b(no|without|trace|small|mild|moderate|large|tiny)?\s*pleural effusions?\b|"
                r"\bno effusion\b|\bno effusions\b|"
                r"\bthere is (a |an )?(trace |small |mild |moderate |large )?pleural effusion\b",
                re.I,
            ),
            "pneumothorax": re.compile(
                r"\b(no|without|small|tiny|trace)?\s*pneumothorax\b|"
                r"\bpneumothoraces\b|"
                r"\bthere is (a |an )?(small |tiny |trace )?pneumothorax\b",
                re.I,
            ),
            "cardiomegaly": self.concepts["cardiomegaly"],
            "atelectasis": self.concepts["atelectasis"],
            "consolidation": self.concepts["consolidation"],
            "pulmonary edema": re.compile(
                r"\bpulmonary edema\b|"
                r"\bedema\b",
                re.I,
            ),
            "vascular congestion": self.concepts["vascular congestion"],
            "interstitial edema": self.concepts["interstitial edema"],
            "rib fracture": self.concepts["rib fracture"],
            "fracture": re.compile(
                r"\bfracture\b|\bfractures\b",
                re.I,
            ),
        }

        self.anchor_stop = {
            "again", "noted", "seen", "demonstrated", "identified", "there", "here", "this", "that",
            "these", "those", "study", "exam", "report", "finding", "findings", "impression",
            "present", "absent", "uncertain", "possible", "likely", "question", "described",
            "showing", "shows", "show", "patient", "view", "views", "portable", "upright", "ap", "pa",
            "lateral", "frontal", "radiograph", "radiographs", "chest", "single", "obtained",
            "comparison", "history", "indication", "technique", "year", "old", "female", "male",
            "woman", "man", "normal", "limits", "within", "stable", "unchanged", "little", "change",
            "acute", "cardiopulmonary", "process", "abnormality",
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

    def _read_report(self, rel_path: str) -> str:
        p = self.reports_root / str(rel_path)
        if not p.exists():
            return ""
        return self._norm(p.read_text(errors="ignore"))

    def _clean_section_text(self, text: str) -> str:
        if not text:
            return ""
        lines = [x.strip() for x in text.split("\n") if x.strip()]
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
        if len(toks) < 5:
            return False
        clinical = {
            "lung", "lungs", "pleural", "effusion", "pneumothorax", "cardiac", "cardiomediastinal",
            "opacity", "opacities", "atelectasis", "atelectatic", "consolidation", "edema",
            "hilar", "mediastinal", "silhouette", "vascular", "retrocardiac", "basilar",
            "interstitial", "fracture", "fractures", "rib", "ribs", "osseous",
        }
        return len(toks & clinical) > 0

    def _split_patient_wise(self, df: pd.DataFrame) -> pd.DataFrame:
        subjects = sorted(df["subject_id"].dropna().astype(int).unique().tolist())
        rng = random.Random(self.seed)
        rng.shuffle(subjects)

        n = len(subjects)
        n_train = int(0.75 * n)
        n_val = int(0.10 * n)

        train_ids = set(subjects[:n_train])
        val_ids = set(subjects[n_train:n_train + n_val])

        def assign(subject_id: int) -> str:
            if subject_id in train_ids:
                return "train"
            if subject_id in val_ids:
                return "val"
            return "test"

        out = df.copy()
        out["split"] = out["subject_id"].astype(int).map(assign)
        return out

    def _choose_question(self, concept: str, state: str, qa_row_id: int) -> str:
        candidates = self.question_templates[concept][state]
        idx = (int(qa_row_id) + self.seed) % len(candidates)
        return candidates[idx]

    def _concept_in_text(self, text: str, concept: str) -> bool:
        return bool(self.concepts[concept].search(self._norm(text)))

    # ---------- Step 0 ----------

    def _step0_master(self) -> pd.DataFrame:
        df = pd.read_csv(self.master_csv)
        keep = [c for c in ["subject_id", "study_id", "report_rel_path"] if c in df.columns]
        df = df[keep].drop_duplicates(subset=["report_rel_path"]).reset_index(drop=True)
        df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
        df = df[~df["subject_id"].isna()].copy()
        df["subject_id"] = df["subject_id"].astype(int)
        df = self._split_patient_wise(df)
        df.to_parquet(self.out_dir / "reports_master.parquet", index=False)
        return df

    # ---------- Step 1 ----------

    def _parse_sections(self, raw_text: str) -> Tuple[str, str, str, str, str]:
        if not raw_text:
            return "", "", "unusable", "low", "empty_report"

        lines = [x.strip() for x in raw_text.split("\n") if x.strip()]
        sections: Dict[str, List[str]] = {}
        current = None

        for ln in lines:
            m = self.section_re.match(ln)
            if m:
                current = m.group(1).upper()
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(ln)

        findings = self._clean_section_text(" ".join(sections.get("FINDINGS", [])))
        impression = self._clean_section_text(" ".join(sections.get("IMPRESSION", [])))

        if findings and impression and self._clinical_text_ok(findings) and self._clinical_text_ok(impression):
            return findings, impression, "both_findings_and_impression", "high", "explicit_sections"
        if findings and self._clinical_text_ok(findings):
            return findings, "", "findings_only", "low", "missing_impression"
        if impression and self._clinical_text_ok(impression):
            return "", impression, "impression_only", "low", "missing_findings"
        return "", "", "unusable", "low", "no_reliable_sections"

    def _step1_sections(self, reports_master: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in reports_master.iterrows():
            raw = self._read_report(r["report_rel_path"])
            findings, impression, report_type, conf, notes = self._parse_sections(raw)
            rows.append({
                "subject_id": int(r["subject_id"]),
                "study_id": r.get("study_id", None),
                "report_rel_path": r["report_rel_path"],
                "split": r["split"],
                "raw_report_text": raw,
                "report_format_type": report_type,
                "parse_confidence": conf,
                "findings_text": findings,
                "impression_text": impression,
                "parse_notes": notes,
                "is_main_benchmark_eligible": report_type == "both_findings_and_impression" and conf == "high",
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
                    "study_id": r.get("study_id", None),
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
                "study_id": r.get("study_id", None),
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

            # prefer medium-length informative findings sentences
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
            "split_policy": "patient_wise_75_10_15",
            "main_benchmark_source": "both_findings_and_impression_high_confidence",
            "label_source": "impression_only",
            "future_verifier_input": "findings_only",
            "concepts": list(self.concepts.keys()),
            "notes": {
                "gold_state_column_added": True,
                "concept_presence_flags_added": True,
                "easy_control_rows_retained": True,
                "anchor_terms_interpreted_as_case_cues": True,
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
        (self.out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
class EvidenceSentence:
    sentence_id: int
    article_id: str
    article_title: str
    section_title: Optional[str]
    sentence_text: str
    sentence_token_count: int
    article_relpath: str
    sentence_hash: str

    is_chest_relevant: bool
    is_definition_like: bool
    is_metadata_like: bool
    is_fragment_like: bool

    matched_concepts_json: str
    concept_states_json: str
    num_matched_concepts: int

    evidence_split: str   # "trainval" or "test"


class RadiopaediaSentenceEvidenceBuilder:
    """
    Build sentence-level Radiopaedia evidence DB and create a held-out article split.

    Outputs:
      - evidence_sentences.parquet
      - evidence_sentences_trainval.parquet
      - evidence_sentences_test.parquet
      - evidence_articles_split.csv
      - manifest.json

    Split policy:
      - article-level split only
      - stratified approximately by concept coverage
      - trainval articles are used for verifier train/val supervision
      - test articles are reserved for oracle-test and retrieval-test evaluation
    """

    def __init__(
        self,
        in_dir: str,
        out_dir: str,
        snapshot_name: str = "RadiopaediaSentenceSnapshot_v6",
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        min_tokens: int = 7,
        max_tokens: int = 64,
        heldout_article_ratio: float = 0.20,
        seed: int = 13,
        write_csv: bool = True,
        write_parquet: bool = True,
        encoding: str = "utf-8",
        tqdm_desc: str = "Building Radiopaedia sentence evidence",
    ):
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_name = snapshot_name
        self.tokenizer_name = tokenizer_name
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.heldout_article_ratio = float(heldout_article_ratio)
        self.seed = int(seed)

        self.write_csv = bool(write_csv)
        self.write_parquet = bool(write_parquet)
        self.encoding = encoding
        self.tqdm_desc = tqdm_desc

        self.rng = random.Random(self.seed)

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        try:
            self._tokenizer.model_max_length = int(1e9)
        except Exception:
            pass

        self.sent_split_re = re.compile(r"(?<=[\.\?\!])\s+")
        self.word_re = re.compile(r"[A-Za-z0-9]+")
        self.ws_re = re.compile(r"[ \t]+")
        self.blankline_re = re.compile(r"\n{3,}")

        self.meta_line_re = re.compile(
            r"\b("
            r"reference article|summary|see also|role of imaging|history and etymology|"
            r"differential diagnosis|treatment and prognosis|epidemiology|etiology|aetiology|"
            r"pathology|classification|risk factors|peer review|last cme review|"
            r"more references needed|this is a basic article|for medical students|"
            r"micro - x ltd|shareholder|paid speaker|ongoing review"
            r")\b",
            re.I,
        )

        self.author_line_re = re.compile(
            r"\b("
            r"jeremy jones|craig hacking|henry knipe|liz silverstone|"
            r"yuranga weerakkody|daniel j bell|joachim feger|stefan tigges|"
            r"behrang amini|andrew murphy|magdi mahsoub"
            r")\b",
            re.I,
        )

        self.section_heading_re = re.compile(
            r"^\s*("
            r"summary|reference article|history and etymology|epidemiology|"
            r"pathology|treatment and prognosis|differential diagnosis|see also|"
            r"role of imaging|classification|risk factors|terminology|"
            r"imaging|ultrasound|ct|mri|plain radiograph|radiograph|"
            r"treatment|prognosis|history"
            r")\s*:?\s*$",
            re.I,
        )

        self.definition_re = re.compile(
            r"\b("
            r"refers to|is defined as|defined as|describes|is a type of|is a form of|"
            r"is characterized by|represents|consists of|is the presence of|"
            r"are collections of|collection of gas within"
            r")\b",
            re.I,
        )

        self.chest_relevant_re = re.compile(
            r"\b("
            r"chest|thoracic|lung|lungs|pleural|effusion|pneumothorax|"
            r"cardiac|cardiomegaly|heart|silhouette|vascular|edema|oedema|"
            r"interstitial|opacity|opacities|airspace|consolidation|"
            r"atelectasis|atelectatic|collapse|hilar|mediastinal|hemithorax|"
            r"costophrenic|rib|diaphragm|parenchymal"
            r")\b",
            re.I,
        )

        self.verb_like_re = re.compile(
            r"\b("
            r"is|are|was|were|be|been|being|"
            r"show|shows|shown|demonstrate|demonstrates|demonstrated|"
            r"appear|appears|appeared|"
            r"seen|visualized|visualised|noted|identified|suggest|suggests|"
            r"may|can|could|should|would|remains|becomes|become"
            r")\b",
            re.I,
        )

        self.alias_like_re = re.compile(r"\b(alias|also known as|aka)\b", re.I)

        self.negation_re = re.compile(
            r"\b("
            r"no|without|absent|negative for|free of|"
            r"no evidence of|no focal|neither|none"
            r")\b",
            re.I,
        )
        self.uncertainty_re = re.compile(
            r"\b("
            r"possible|possibly|may|might|likely|unlikely|"
            r"cannot exclude|could represent|suggestive of|question of|"
            r"suspicious for|concerning for|may represent|appears to"
            r")\b",
            re.I,
        )
        self.resolution_re = re.compile(
            r"\b("
            r"resolved|improved|decreased|cleared|near complete resolution"
            r")\b",
            re.I,
        )

        self.concepts: Dict[str, re.Pattern] = {
            "pleural effusion": re.compile(
                r"\b("
                r"pleural effusion|pleural effusions|effusion|effusions|"
                r"pleural fluid|pleural collection|"
                r"costophrenic blunting|blunting of (the )?(right |left )?costophrenic angle|"
                r"blunted costophrenic angle|blunted costophrenic angles"
                r")\b",
                re.I,
            ),
            "pneumothorax": re.compile(
                r"\b("
                r"pneumothorax|pneumothoraces|pleural air|collapsed lung"
                r")\b",
                re.I,
            ),
            "cardiomegaly": re.compile(
                r"\b("
                r"cardiomegaly|cardiac enlargement|cardiac silhouette enlargement|"
                r"enlarged cardiac silhouette|cardiac silhouette is enlarged|"
                r"cardiac silhouette enlarged|enlarged heart|heart is enlarged|"
                r"heart size is enlarged|heart size enlarged|mild cardiomegaly|"
                r"cardiomegalic"
                r")\b",
                re.I,
            ),
            "atelectasis": re.compile(
                r"\b("
                r"atelectasis|atelectatic|subsegmental atelectatic|"
                r"bibasilar atelectatic|basal atelectatic|"
                r"subsegmental collapse|lobar collapse|collapse"
                r")\b",
                re.I,
            ),
            "consolidation": re.compile(
                r"\b("
                r"consolidation|lobar consolidation|focal consolidation|"
                r"airspace consolidation|air space consolidation|"
                r"airspace opacity|airspace opacities|air space opacity|air space opacities|"
                r"alveolar opacity|alveolar opacities|parenchymal opacity|parenchymal opacities|"
                r"airspace disease"
                r")\b",
                re.I,
            ),
            "pulmonary edema": re.compile(
                r"\b("
                r"pulmonary edema|pulmonary oedema|edema|oedema|"
                r"alveolar edema|alveolar oedema"
                r")\b",
                re.I,
            ),
            "vascular congestion": re.compile(
                r"\b("
                r"vascular congestion|pulmonary vascular congestion|"
                r"central pulmonary vascular congestion|vascular indistinctness|"
                r"hilar vasculature|vascular prominence|prominent pulmonary vasculature|"
                r"cephalisation|cephalization"
                r")\b",
                re.I,
            ),
            "interstitial edema": re.compile(
                r"\b("
                r"interstitial edema|interstitial pulmonary edema|"
                r"interstitial oedema|interstitial pulmonary oedema|"
                r"septal edema|septal oedema|interlobular septal thickening"
                r")\b",
                re.I,
            ),
            "rib fracture": re.compile(
                r"\b("
                r"rib fracture|rib fractures|fractured rib|fractured ribs"
                r")\b",
                re.I,
            ),
            "fracture": re.compile(
                r"\b("
                r"fracture|fractures|fractured"
                r")\b",
                re.I,
            ),
        }

    def _list_txt_files(self) -> List[Path]:
        files = sorted(self.in_dir.rglob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found under: {self.in_dir}")
        return files

    def _read_text(self, p: Path) -> str:
        return p.read_text(encoding=self.encoding, errors="ignore")

    def _make_article_id(self, p: Path) -> str:
        return p.stem

    def _stable_hash(self, s: str) -> str:
        return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()

    def _first_nonempty_line(self, text: str) -> str:
        for ln in text.splitlines():
            s = ln.strip()
            if s:
                return s
        return ""

    def _token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not t:
            return ""
        lines = [self.ws_re.sub(" ", ln).strip() for ln in t.split("\n")]
        out = "\n".join(lines)
        out = self.blankline_re.sub("\n\n", out).strip()
        return out

    def _is_bad_line(self, line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if self.meta_line_re.search(s):
            return True
        if self.author_line_re.search(s):
            return True
        if self.section_heading_re.match(s):
            return True
        if len(self.word_re.findall(s)) <= 2 and s[-1:] not in ".!?":
            return True
        if s.count(":") >= 2:
            return True
        return False

    def _reconstruct_paragraphs(self, text: str) -> List[Tuple[Optional[str], str]]:
        lines = text.split("\n")
        paragraphs: List[Tuple[Optional[str], str]] = []

        current_section: Optional[str] = None
        current_lines: List[str] = []

        def flush():
            nonlocal current_lines
            if current_lines:
                para = " ".join(current_lines).strip()
                para = self.ws_re.sub(" ", para).strip()
                if para:
                    paragraphs.append((current_section, para))
            current_lines = []

        for raw_ln in lines:
            ln = raw_ln.strip()

            if not ln:
                flush()
                continue

            if self._is_bad_line(ln):
                flush()
                continue

            if self.section_heading_re.match(ln):
                flush()
                current_section = ln.rstrip(":").strip()
                continue

            n_words = len(self.word_re.findall(ln))
            if n_words <= 2 and ln[-1:] not in ".!?":
                flush()
                continue

            current_lines.append(ln)

        flush()
        return paragraphs

    def _title_case_fraction(self, sent: str) -> float:
        toks = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]*\b", sent)
        if not toks:
            return 0.0
        title_like = sum(1 for t in toks if t[:1].isupper() and not t.isupper())
        return float(title_like) / float(len(toks))

    def _looks_fragment_like(self, sent: str) -> bool:
        s = sent.strip()

        if self.alias_like_re.search(s):
            return True
        if ";" in s and s.count(";") >= 2:
            return True
        if ":" in s and s.count(":") >= 2:
            return True

        toks = self.word_re.findall(s)
        if len(toks) < self.min_tokens:
            return True
        if not self.verb_like_re.search(s):
            return True
        if self._title_case_fraction(s) > 0.45:
            return True

        if "." not in s and "?" not in s and "!" not in s:
            if len(toks) <= 14:
                return True

        if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b", s):
            if not self.verb_like_re.search(s):
                return True

        return False

    def _is_valid_sentence(self, sent: str, article_title: str) -> bool:
        s = sent.strip()
        if not s:
            return False

        toks = self.word_re.findall(s)
        if len(toks) < self.min_tokens:
            return False

        ntoks = self._token_count(s)
        if ntoks < self.min_tokens or ntoks > self.max_tokens:
            return False

        if len(set(t.lower() for t in toks)) < 5:
            return False
        if "\n" in s:
            return False
        if not s[0].isalnum():
            return False
        if self.meta_line_re.search(s):
            return False
        if self.author_line_re.search(s):
            return False

        title_norm = article_title.strip().lower()
        if title_norm and s.strip().lower() == title_norm:
            return False

        if self._looks_fragment_like(s):
            return False

        return True

    def _extract_sentences(
        self,
        paragraphs: List[Tuple[Optional[str], str]],
        article_title: str,
    ) -> List[Tuple[Optional[str], str]]:
        out: List[Tuple[Optional[str], str]] = []

        for section_title, para in paragraphs:
            sents = [x.strip() for x in self.sent_split_re.split(para) if x.strip()]
            for sent in sents:
                if self._is_valid_sentence(sent, article_title=article_title):
                    out.append((section_title, sent))

        return out

    def _sentence_flags(self, sent: str) -> Tuple[bool, bool, bool, bool]:
        s = sent.strip()
        is_definition_like = bool(self.definition_re.search(s))
        is_metadata_like = bool(self.meta_line_re.search(s) or self.author_line_re.search(s))
        is_chest_relevant = bool(self.chest_relevant_re.search(s))
        is_fragment_like = bool(self._looks_fragment_like(s))
        return is_chest_relevant, is_definition_like, is_metadata_like, is_fragment_like

    def _classify_state_for_span(self, text: str, start: int, end: int) -> str:
        lo = max(0, start - 100)
        hi = min(len(text), end + 100)
        window = text[lo:hi]

        has_neg = bool(self.negation_re.search(window))
        has_unc = bool(self.uncertainty_re.search(window))
        has_res = bool(self.resolution_re.search(window))

        if has_neg:
            return "absent"
        if has_unc:
            return "uncertain"
        if has_res:
            return "absent"
        return "present"

    def _match_concepts_in_sentence(self, sent: str) -> Dict[str, str]:
        out: Dict[str, str] = {}

        for concept, pat in self.concepts.items():
            m = pat.search(sent)
            if not m:
                continue
            state = self._classify_state_for_span(sent, m.start(), m.end())
            out[concept] = state

        return out

    def build_snapshot(self) -> pd.DataFrame:
        files = self._list_txt_files()

        rows: List[Dict[str, object]] = []
        sentence_id = 0

        pbar = tqdm(total=len(files), desc=self.tqdm_desc, unit="articles")
        for p in files:
            raw = self._read_text(p).strip()
            if not raw:
                pbar.update(1)
                continue

            title = self._first_nonempty_line(raw)
            body = self._normalize_text(raw)
            if not body:
                pbar.update(1)
                continue

            article_id = self._make_article_id(p)
            relpath = str(p.relative_to(self.in_dir))

            paragraphs = self._reconstruct_paragraphs(body)
            sentences = self._extract_sentences(paragraphs, article_title=title)

            for section_title, sent in sentences:
                concept_states = self._match_concepts_in_sentence(sent)
                matched_concepts = sorted(concept_states.keys())

                (
                    is_chest_relevant,
                    is_definition_like,
                    is_metadata_like,
                    is_fragment_like,
                ) = self._sentence_flags(sent)

                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "article_id": article_id,
                        "article_title": title,
                        "section_title": section_title,
                        "sentence_text": sent,
                        "sentence_token_count": int(self._token_count(sent)),
                        "article_relpath": relpath,
                        "sentence_hash": self._stable_hash(sent),
                        "is_chest_relevant": is_chest_relevant,
                        "is_definition_like": is_definition_like,
                        "is_metadata_like": is_metadata_like,
                        "is_fragment_like": is_fragment_like,
                        "matched_concepts_json": json.dumps(matched_concepts, ensure_ascii=False),
                        "concept_states_json": json.dumps(concept_states, ensure_ascii=False, sort_keys=True),
                        "num_matched_concepts": int(len(matched_concepts)),
                    }
                )
                sentence_id += 1

            pbar.update(1)
        pbar.close()

        return pd.DataFrame(rows)

    # ----------------------------
    # NEW: article-level split
    # ----------------------------

    def _article_concept_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        article_ids = sorted(df["article_id"].astype(str).unique().tolist())

        for aid in article_ids:
            sub = df[df["article_id"].astype(str) == aid]
            concepts_seen: Set[str] = set()

            for s in sub["matched_concepts_json"].tolist():
                try:
                    xs = json.loads(s)
                    if isinstance(xs, list):
                        concepts_seen.update(str(x) for x in xs)
                except Exception:
                    pass

            rows.append(
                {
                    "article_id": aid,
                    "article_title": sub["article_title"].iloc[0] if len(sub) else "",
                    "article_relpath": sub["article_relpath"].iloc[0] if len(sub) else "",
                    "num_sentences": int(len(sub)),
                    "num_chest_sentences": int((sub["is_chest_relevant"] == True).sum()),
                    "num_matched_sentences": int((sub["num_matched_concepts"] > 0).sum()),
                    "matched_concepts_json": json.dumps(sorted(concepts_seen), ensure_ascii=False),
                }
            )

        return pd.DataFrame(rows)

    def _make_article_split(self, df: pd.DataFrame) -> pd.DataFrame:
        art = self._article_concept_profile(df)

        # only stratify over articles that actually contain matched concepts
        informative = art[art["num_matched_sentences"] > 0].copy()
        noninformative = art[art["num_matched_sentences"] == 0].copy()

        informative["concept_key"] = informative["matched_concepts_json"].astype(str)

        test_article_ids: Set[str] = set()

        # stratify approximately by concept-profile bucket
        for key, g in informative.groupby("concept_key", sort=False):
            article_ids = g["article_id"].astype(str).tolist()
            self.rng.shuffle(article_ids)

            n_test = max(1, int(round(len(article_ids) * self.heldout_article_ratio))) if len(article_ids) >= 3 else 0
            test_article_ids.update(article_ids[:n_test])

        # fill remaining from noninformative articles
        all_article_ids = art["article_id"].astype(str).tolist()
        target_test_n = int(round(len(all_article_ids) * self.heldout_article_ratio))

        remaining_pool = [aid for aid in noninformative["article_id"].astype(str).tolist() if aid not in test_article_ids]
        self.rng.shuffle(remaining_pool)

        need = max(0, target_test_n - len(test_article_ids))
        test_article_ids.update(remaining_pool[:need])

        art["evidence_split"] = art["article_id"].astype(str).apply(
            lambda x: "test" if x in test_article_ids else "trainval"
        )
        return art

    def _attach_split_to_sentences(self, df: pd.DataFrame, article_split_df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(
            article_split_df[["article_id", "evidence_split"]],
            on="article_id",
            how="left",
        )
        out["evidence_split"] = out["evidence_split"].fillna("trainval")
        return out

    # ----------------------------
    # write
    # ----------------------------

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
        full_csv_path = self.out_dir / "evidence_sentences.csv"
        full_parquet_path = self.out_dir / "evidence_sentences.parquet"

        trainval_csv_path = self.out_dir / "evidence_sentences_trainval.csv"
        trainval_parquet_path = self.out_dir / "evidence_sentences_trainval.parquet"

        test_csv_path = self.out_dir / "evidence_sentences_test.csv"
        test_parquet_path = self.out_dir / "evidence_sentences_test.parquet"

        article_split_csv_path = self.out_dir / "evidence_articles_split.csv"

        article_split_df = self._make_article_split(df)
        df = self._attach_split_to_sentences(df, article_split_df)

        trainval_df = df[df["evidence_split"] == "trainval"].reset_index(drop=True)
        test_df = df[df["evidence_split"] == "test"].reset_index(drop=True)

        if self.write_csv:
            df.to_csv(full_csv_path, index=False)
            trainval_df.to_csv(trainval_csv_path, index=False)
            test_df.to_csv(test_csv_path, index=False)
            article_split_df.to_csv(article_split_csv_path, index=False)

        wrote_parquet = False
        if self.write_parquet:
            if self._parquet_available():
                df.to_parquet(full_parquet_path, index=False)
                trainval_df.to_parquet(trainval_parquet_path, index=False)
                test_df.to_parquet(test_parquet_path, index=False)
                wrote_parquet = True
            else:
                print("[WARN] Parquet requested but pyarrow/fastparquet not installed. Skipping parquet.")

        manifest = {
            "snapshot_name": self.snapshot_name,
            "input_dir": str(self.in_dir.resolve()),
            "tokenizer": self.tokenizer_name,
            "evidence_unit": "sentence",
            "split_policy": {
                "type": "article_level_stratified_holdout",
                "heldout_article_ratio": self.heldout_article_ratio,
                "seed": self.seed,
                "trainval_usage": "verifier_train_and_val_oracle_supervision",
                "test_usage": "oracle_test_and_retrieval_test_only",
            },
            "content_policy": {
                "content_rewritten": False,
                "whitespace_normalized_only": True,
                "metadata_removed": True,
                "sentence_tags_added": True,
                "strict_sentence_filtering": True,
                "concept_specific_state_annotation": True,
            },
            "concepts_supported": sorted(list(self.concepts.keys())),
            "counts": {
                "num_articles_total": int(article_split_df["article_id"].nunique()) if not article_split_df.empty else 0,
                "num_articles_trainval": int((article_split_df["evidence_split"] == "trainval").sum()) if not article_split_df.empty else 0,
                "num_articles_test": int((article_split_df["evidence_split"] == "test").sum()) if not article_split_df.empty else 0,
                "num_sentences_total": int(len(df)),
                "num_sentences_trainval": int(len(trainval_df)),
                "num_sentences_test": int(len(test_df)),
                "num_sentences_with_any_concept_match_total": int((df["num_matched_concepts"] > 0).sum()) if not df.empty else 0,
                "num_sentences_with_any_concept_match_trainval": int((trainval_df["num_matched_concepts"] > 0).sum()) if not trainval_df.empty else 0,
                "num_sentences_with_any_concept_match_test": int((test_df["num_matched_concepts"] > 0).sum()) if not test_df.empty else 0,
            },
            "files": {
                "full_csv": "evidence_sentences.csv" if self.write_csv else None,
                "full_parquet": "evidence_sentences.parquet" if wrote_parquet else None,
                "trainval_csv": "evidence_sentences_trainval.csv" if self.write_csv else None,
                "trainval_parquet": "evidence_sentences_trainval.parquet" if wrote_parquet else None,
                "test_csv": "evidence_sentences_test.csv" if self.write_csv else None,
                "test_parquet": "evidence_sentences_test.parquet" if wrote_parquet else None,
                "article_split_csv": "evidence_articles_split.csv" if self.write_csv else None,
            },
        }

        manifest_path = self.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {manifest_path}")
        if self.write_csv:
            print(f"Wrote: {full_csv_path}")
            print(f"Wrote: {trainval_csv_path}")
            print(f"Wrote: {test_csv_path}")
            print(f"Wrote: {article_split_csv_path}")
        if wrote_parquet:
            print(f"Wrote: {full_parquet_path}")
            print(f"Wrote: {trainval_parquet_path}")
            print(f"Wrote: {test_parquet_path}")

    def run(self) -> pd.DataFrame:
        df = self.build_snapshot()
        self.write_outputs(df)
        return df







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


class RadiologyVerifierBinaryDatasetBuilder:
    """
    Build binary verifier supervision:
        input:  case + evidence + claim
        target: supported / not_supported

    Row types:
        C = claim matches gold state + supporting evidence -> 1
        D = opposite claim + opposite-support evidence -> 0
        A = positive claim + non-support evidence -> 0
        B = negative claim + non-support evidence -> 0
    """

    def __init__(
        self,
        qa_parquet: str,
        evidence_sentences_parquet: str,
        out_dir: str,
        seed: int = 13,
        max_evidence_sentences: int = 2,
        max_rows_per_type_per_concept_split: Optional[int] = None,
    ):
        self.qa_parquet = Path(qa_parquet)
        self.evidence_sentences_parquet = Path(evidence_sentences_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.max_rows_per_type_per_concept_split = max_rows_per_type_per_concept_split

        self.out_parquet = self.out_dir / "verifier_binary_trainvaltest.parquet"
        self.manifest_out = self.out_dir / "manifest_verifier_binary.json"

        self.word_re = re.compile(r"[a-z0-9]+")

        # Keep only concepts that are realistically supportable from current Radiopaedia DB
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
    # basic helpers
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

    def _take_evidence(self, candidates: List[Dict], source_type: str) -> Optional[Tuple[List[int], List[str], List[str], str]]:
        """
        Returns:
            sentence_ids, sentence_texts, article_titles, evidence_source_type
        """
        if not candidates:
            return None

        chosen = self._sample_n(candidates, self.max_evidence_sentences)
        chosen = sorted(chosen, key=lambda x: int(x["sentence_id"]))

        sids = [int(x["sentence_id"]) for x in chosen]
        sents = [self._norm(x["sentence_text"]) for x in chosen]
        arts = [self._norm(x["article_title"]) for x in chosen]

        return sids, sents, arts, source_type

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

        # Step 5 revised code has gold_state. Fall back if needed.
        if "gold_state" not in qa.columns:
            qa["gold_state"] = qa["claim_text"].astype(str).str.extract(r"\s+is\s+(present|absent|uncertain)\s*$", expand=False)

        qa["qa_row_id"] = pd.to_numeric(qa["qa_row_id"], errors="coerce")
        qa["subject_id"] = pd.to_numeric(qa["subject_id"], errors="coerce")
        qa = qa[~qa["qa_row_id"].isna() & ~qa["subject_id"].isna()].copy()
        qa["qa_row_id"] = qa["qa_row_id"].astype(int)
        qa["subject_id"] = qa["subject_id"].astype(int)

        qa["concept"] = qa["concept"].astype(str).str.strip()
        qa["gold_state"] = qa["gold_state"].astype(str).str.strip()
        qa["case_context"] = qa["findings_text"].astype(str)

        qa = qa[qa["concept"].isin(self.core_concepts)].copy()
        qa = qa[qa["gold_state"].isin(["present", "absent"])].copy()

        qa = qa.reset_index(drop=True)
        return qa


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



    def _load_evidence(self) -> pd.DataFrame:
        ev = pd.read_parquet(self.evidence_sentences_parquet).copy()

        required = {
            "sentence_id",
            "sentence_text",
            "article_title",
            "is_chest_relevant",
            "is_definition_like",
            "is_metadata_like",
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

        # hard global cleanup
        ev = ev[ev["is_metadata_like"] == False].copy()
        ev = ev[ev["is_fragment_like"] == False].copy()
        ev = ev[ev["is_definition_like"] == False].copy()

        ev = ev.reset_index(drop=True)
        return ev

    # --------------------------------------------------
    # evidence pools
    # --------------------------------------------------

    def _build_evidence_pools(self, ev: pd.DataFrame) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Per concept create:
            support_present
            support_absent
            nonsupport_hard
            nonsupport_easy
        """
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
                        # uncertain mention -> non-support hard
                        nonsupport_hard.append(row)



                else:
                    # chest relevant but not this concept
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

    def _select_nonsupport_evidence(self, concept_pools: Dict[str, List[Dict]], hard_first: bool = True) -> Optional[Tuple[List[int], List[str], List[str], str]]:
        hard_pool = concept_pools["nonsupport_hard"]
        easy_pool = concept_pools["nonsupport_easy"]

        # mix hard and easy, prefer hard
        mixed: List[Dict] = []
        if hard_first:
            mixed.extend(self._sample_n(hard_pool, min(len(hard_pool), max(1, self.max_evidence_sentences))))
            if len(mixed) < self.max_evidence_sentences:
                need = self.max_evidence_sentences - len(mixed)
                mixed.extend(self._sample_n(easy_pool, min(len(easy_pool), need)))
        else:
            mixed.extend(self._sample_n(easy_pool, min(len(easy_pool), max(1, self.max_evidence_sentences))))
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

            # C: claim matches gold + evidence supports that claim -> 1
            c_pack = self._take_evidence(support_gold_pool, source_type=f"support_{gold_state}")
            if c_pack is not None:
                rows.append(self._make_one_row(
                    verifier_row_id=verifier_row_id,
                    qa_row=q,
                    claim_state=gold_state,
                    evidence_pack=c_pack,
                    row_type="C",
                    label_id=1,
                ))
                verifier_row_id += 1

            # D: claim opposite + evidence supports opposite claim -> 0
            d_pack = self._take_evidence(support_opp_pool, source_type=f"support_{opp_state}")
            if d_pack is not None:
                rows.append(self._make_one_row(
                    verifier_row_id=verifier_row_id,
                    qa_row=q,
                    claim_state=opp_state,
                    evidence_pack=d_pack,
                    row_type="D",
                    label_id=0,
                ))
                verifier_row_id += 1

            # A: positive claim + non-support evidence -> 0
            a_pack = self._select_nonsupport_evidence(concept_pools, hard_first=True)
            if a_pack is not None:
                rows.append(self._make_one_row(
                    verifier_row_id=verifier_row_id,
                    qa_row=q,
                    claim_state="present",
                    evidence_pack=a_pack,
                    row_type="A",
                    label_id=0,
                ))
                verifier_row_id += 1

            # B: negative claim + non-support evidence -> 0
            b_pack = self._select_nonsupport_evidence(concept_pools, hard_first=False)
            if b_pack is not None:
                rows.append(self._make_one_row(
                    verifier_row_id=verifier_row_id,
                    qa_row=q,
                    claim_state="absent",
                    evidence_pack=b_pack,
                    row_type="B",
                    label_id=0,
                ))
                verifier_row_id += 1

        out = pd.DataFrame(rows)

        if out.empty:
            raise ValueError("No verifier rows were created. Check evidence pools and concept coverage.")

        # --------------------------------------------------
        # optional balancing per split x concept x row_type
        # --------------------------------------------------
        if self.max_rows_per_type_per_concept_split is not None:
            kept_parts = []
            for (split, concept, row_type), g in out.groupby(["split", "concept", "row_type"], sort=False):
                if len(g) <= self.max_rows_per_type_per_concept_split:
                    kept_parts.append(g)
                else:
                    kept_parts.append(g.sample(self.max_rows_per_type_per_concept_split, random_state=self.seed))
            out = pd.concat(kept_parts, axis=0).reset_index(drop=True)
            out["verifier_row_id"] = range(len(out))

        # # subsampling for training
        # pos = out[out["label_id"] == 1]
        # neg = out[out["label_id"] == 0]
        # neg_sampled = neg.sample(len(pos), random_state=13)
        # out = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=13).reset_index(drop=True)


        out.to_parquet(self.out_parquet, index=False)

        manifest = {
            "task": "binary_verifier_support",
            "input_definition": "case_context + evidence + claim -> supported / not_supported",
            "labels": {"not_supported": 0, "supported": 1},
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

        return out








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


class RadiologyVerifierBinaryHeldoutRetrievedTestBuilder:
    """
    Build non-oracle held-out test data for the verifier.

    Design:
      - use only QA rows with split == "test"
      - retrieve evidence only from held-out Radiopaedia test articles
      - no oracle evidence selection
      - create 2 rows per QA:
            1) gold-state claim
            2) opposite-state claim
      - label = 1 only if:
            claim_state == gold_state
            AND at least one retrieved evidence sentence supports that claim
      - otherwise label = 0

    This is the realistic retrieval-based held-out test.
    """

    def __init__(
        self,
        qa_parquet: str,
        evidence_sentences_test_parquet: str,
        out_dir: str,
        seed: int = 13,
        retrieval_top_k_full: int = 8,
        max_evidence_sentences: int = 5,
        bm25_cache_name: str = "heldout_test_bm25.pkl",
    ):
        self.qa_parquet = Path(qa_parquet)
        self.evidence_sentences_test_parquet = Path(evidence_sentences_test_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        self.retrieval_top_k_full = int(retrieval_top_k_full)
        self.max_evidence_sentences = int(max_evidence_sentences)

        self.out_parquet = self.out_dir / "verifier_binary_heldout_retrieval_test.parquet"
        self.manifest_out = self.out_dir / "manifest_verifier_binary_heldout_retrieval_test.json"
        self.bm25_cache_path = self.out_dir / bm25_cache_name

        self.word_re = re.compile(r"[a-z0-9]+")
        self.negation_re = re.compile(
            r"\b(no|without|absent|negative for|free of|no evidence of|no focal|neither|none)\b",
            re.I,
        )
        self.uncertainty_re = re.compile(
            r"\b(possible|possibly|may|might|likely|unlikely|cannot exclude|could represent|"
            r"suggestive of|question of|suspicious for|concerning for|may represent|appears to)\b",
            re.I,
        )
        self.resolution_re = re.compile(
            r"\b(resolved|improved|decreased|cleared|near complete resolution)\b",
            re.I,
        )

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

    def _tokens(self, s: str) -> List[str]:
        return self.word_re.findall(self._norm(s).lower())

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

    def _build_query(self, claim_text: str, anchor_terms: List[str]) -> str:
        if anchor_terms:
            return f"{claim_text}\n\n" + " ".join(anchor_terms[:8])
        return claim_text

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
    # load
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
                r"\s+is\s+(present|absent|uncertain)\s*$", expand=False
            )

        qa["qa_row_id"] = pd.to_numeric(qa["qa_row_id"], errors="coerce")
        qa["subject_id"] = pd.to_numeric(qa["subject_id"], errors="coerce")
        qa = qa[~qa["qa_row_id"].isna() & ~qa["subject_id"].isna()].copy()
        qa["qa_row_id"] = qa["qa_row_id"].astype(int)
        qa["subject_id"] = qa["subject_id"].astype(int)

        qa["concept"] = qa["concept"].astype(str).str.strip()
        qa["gold_state"] = qa["gold_state"].astype(str).str.strip()
        qa["case_context"] = qa["findings_text"].astype(str)
        qa["anchor_terms_json"] = qa.get("anchor_terms_json", "[]")
        qa["anchor_terms_json"] = qa["anchor_terms_json"].fillna("[]").astype(str)

        qa = qa[qa["split"].astype(str) == "test"].copy()
        qa = qa[qa["concept"].isin(self.core_concepts)].copy()
        qa = qa[qa["gold_state"].isin(["present", "absent"])].copy()

        return qa.reset_index(drop=True)

    def _load_evidence(self) -> pd.DataFrame:
        ev = pd.read_parquet(self.evidence_sentences_test_parquet).copy()

        required = {
            "sentence_id",
            "sentence_text",
            "article_title",
            "matched_concepts_json",
            "concept_states_json",
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
        ev["sentence_text"] = ev["sentence_text"].astype(str)
        ev["article_title"] = ev["article_title"].astype(str)

        ev["matched_concepts"] = ev["matched_concepts_json"].apply(self._loads_json_list)
        ev["concept_states"] = ev["concept_states_json"].apply(self._loads_json_dict)

        ev = ev[ev["is_metadata_like"] == False].copy()
        ev = ev[ev["is_definition_like"] == False].copy()
        ev = ev[ev["is_fragment_like"] == False].copy()
        ev = ev[ev["is_chest_relevant"] == True].copy()

        return ev.reset_index(drop=True)

    # --------------------------------------------------
    # bm25
    # --------------------------------------------------

    def _build_bm25(self, ev: pd.DataFrame):
        if self.bm25_cache_path.exists():
            with open(self.bm25_cache_path, "rb") as f:
                return pickle.load(f)

        postings: Dict[str, List[Tuple[int, int]]] = {}
        dfreq: Dict[str, int] = {}
        sent_ids = ev["sentence_id"].tolist()
        doc_len = np.zeros(len(ev), dtype=np.int32)

        for i, text in enumerate(tqdm(ev["sentence_text"].tolist(), desc="Building heldout BM25", unit="sent")):
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
    # support check on retrieved evidence
    # --------------------------------------------------

    def _retrieved_sentence_supports_claim(
        self,
        sent_meta: Dict[int, Dict[str, object]],
        sentence_id: int,
        concept: str,
        claim_state: str,
    ) -> bool:
        meta = sent_meta.get(int(sentence_id))
        if meta is None:
            return False

        concept_states = meta["concept_states"]
        if concept not in concept_states:
            return False

        st = str(concept_states[concept]).strip()
        sent_text = self._norm(meta["sentence_text"])

        if st != claim_state:
            return False

        if concept not in sent_text.lower():
            return False

        if not self._is_imaging_like_support_sentence(sent_text):
            return False

        return True

    def _make_one_row(
        self,
        verifier_row_id: int,
        qa_row: pd.Series,
        claim_state: str,
        retrieved_ids: List[int],
        retrieved_sents: List[str],
        retrieved_titles: List[str],
        label_id: int,
    ) -> Dict:
        label = "supported" if int(label_id) == 1 else "not_supported"

        return asdict(
            VerifierBinaryRow(
                verifier_row_id=int(verifier_row_id),
                qa_row_id=int(qa_row["qa_row_id"]),
                split="test",
                subject_id=int(qa_row["subject_id"]),
                report_rel_path=str(qa_row["report_rel_path"]),
                concept=str(qa_row["concept"]),
                gold_state=str(qa_row["gold_state"]),
                claim_state=str(claim_state),
                claim_text=self._claim_text(str(qa_row["concept"]), str(claim_state)),
                case_context=self._norm(qa_row["case_context"]),
                evidence_sentence_ids_json=json.dumps([int(x) for x in retrieved_ids]),
                evidence_sentences_json=json.dumps([self._norm(x) for x in retrieved_sents], ensure_ascii=False),
                evidence_article_titles_json=json.dumps([self._norm(x) for x in retrieved_titles], ensure_ascii=False),
                row_type="retrieved",
                evidence_source_type=f"heldout_bm25_top{self.retrieval_top_k_full}",
                label=label,
                label_id=int(label_id),
            )
        )

    # --------------------------------------------------
    # main
    # --------------------------------------------------

    def run(self) -> pd.DataFrame:
        qa = self._load_qa()
        ev = self._load_evidence()
        bm25_state = self._build_bm25(ev)

        sent_meta = {
            int(r["sentence_id"]): {
                "sentence_text": str(r["sentence_text"]),
                "article_title": str(r["article_title"]),
                "concept_states": r["concept_states"],
            }
            for _, r in ev.iterrows()
        }

        rows: List[Dict] = []
        verifier_row_id = 0

        pbar = tqdm(total=len(qa), desc="Building heldout retrieval test", unit="qa")
        for _, q in qa.iterrows():
            concept = str(q["concept"])
            gold_state = str(q["gold_state"])
            opp_state = self._opposite_state(gold_state)
            if opp_state is None:
                pbar.update(1)
                continue

            anchor_terms = self._loads_json_list(q["anchor_terms_json"])
            claim_states = [gold_state, opp_state]

            for claim_state in claim_states:
                claim_text = self._claim_text(concept, claim_state)
                query = self._build_query(claim_text, anchor_terms)

                retrieved_ids_full = self._bm25_search(query, bm25_state, self.retrieval_top_k_full)
                retrieved_ids_full = [sid for sid in retrieved_ids_full if sid in sent_meta]

                # label uses full retrieval set
                any_support = any(
                    self._retrieved_sentence_supports_claim(
                        sent_meta=sent_meta,
                        sentence_id=sid,
                        concept=concept,
                        claim_state=claim_state,
                    )
                    for sid in retrieved_ids_full
                )

                # model input uses truncated visible evidence only
                retrieved_ids_shown = retrieved_ids_full[: self.max_evidence_sentences]
                retrieved_sents = [self._norm(sent_meta[sid]["sentence_text"]) for sid in retrieved_ids_shown]
                retrieved_titles = [self._norm(sent_meta[sid]["article_title"]) for sid in retrieved_ids_shown]

                # label 1 only if claim matches case gold AND retrieved evidence supports that claim
                label_id = int((claim_state == gold_state) and any_support)

                rows.append(
                    self._make_one_row(
                        verifier_row_id=verifier_row_id,
                        qa_row=q,
                        claim_state=claim_state,
                        retrieved_ids=retrieved_ids_shown,
                        retrieved_sents=retrieved_sents,
                        retrieved_titles=retrieved_titles,
                        label_id=label_id,
                    )
                )
                verifier_row_id += 1

            pbar.update(1)
        pbar.close()

        out = pd.DataFrame(rows)
        if out.empty:
            raise ValueError("No heldout retrieval test rows were created.")

        out.to_parquet(self.out_parquet, index=False)

        manifest = {
            "task": "binary_verifier_support",
            "evaluation_mode": "heldout_retrieval_test",
            "input_definition": "case_context + retrieved_heldout_evidence + claim -> supported / not_supported",
            "labels": {"not_supported": 0, "supported": 1},
            "retrieval": {
                "method": "bm25",
                "retrieval_top_k": self.retrieval_top_k_full,
                "max_evidence_sentences": self.max_evidence_sentences,
                "evidence_pool": str(self.evidence_sentences_test_parquet.resolve()),
            },
            "inputs": {
                "qa_parquet": str(self.qa_parquet.resolve()),
                "evidence_sentences_test_parquet": str(self.evidence_sentences_test_parquet.resolve()),
            },
            "outputs": {
                "heldout_retrieval_test_parquet": str(self.out_parquet.resolve()),
            },
            "params": {
                "seed": self.seed,
                "core_concepts": sorted(self.core_concepts),
            },
            "counts": {
                "num_rows": int(len(out)),
                "num_supported": int((out["label_id"] == 1).sum()),
                "num_not_supported": int((out["label_id"] == 0).sum()),
                "num_unique_qa_rows": int(out["qa_row_id"].nunique()),
            },
            "counts_by_concept": {k: int(v) for k, v in out["concept"].value_counts().to_dict().items()},
        }

        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.out_parquet}")
        print(f"Wrote: {self.manifest_out}")
        print("\nLabel counts:")
        print(out["label"].value_counts(dropna=False).to_string())

        return out






if __name__ == "__main__":

    with open(global_config_path, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    base_path_mimicData = params['Preprocessing']['base_path_mimicData']
    base_path_evidenceData = params['Preprocessing']['base_path_evidenceData']
    mimic_reports_base = params['Preprocessing']['mimic_reports_base']


    ''' steps 0 to 5 '''
    # builder = MimicGroundingDatasetBuilder(
    #     master_csv=os.path.join(base_path_mimicData, "mimic_master_list_forReports.csv"),
    #     reports_root=mimic_reports_base,
    #     out_dir=os.path.join(base_path_mimicData, "grounding_dataset"),
    #     seed=13,
    # )
    # builder.run_all()



    ''' RadiopaediaSnapshot '''
    # evidence_builder = RadiopaediaSentenceEvidenceBuilder(
    #     in_dir=os.path.join(base_path_evidenceData, "articles_clean_txt"),
    #     out_dir=base_path_evidenceData,
    #     snapshot_name=params["Preprocessing"]["snapshot_name"],
    #     tokenizer_name=params["Preprocessing"]["tokenizer_name"],
    #     min_tokens=7,
    #     max_tokens=50,
    #     heldout_article_ratio=0.20,
    #     seed=13,
    #     write_csv=True,
    #     write_parquet=True,
    #     encoding="utf-8",
    # )
    # evidence_df = evidence_builder.run()



    ''' step 6 '''
    # verifier_builder = RadiologyVerifierBinaryDatasetBuilder(
    #     qa_parquet=os.path.join(base_path_mimicData, "grounding_dataset", "qa_grounding_main_with_anchors.parquet"),
    #     evidence_sentences_parquet=os.path.join(base_path_evidenceData, "evidence_sentences_trainval.parquet"),
    #     out_dir=os.path.join(base_path_evidenceData, "verifier_binary_dataset"),
    #     seed=13,
    #     max_evidence_sentences=2,
    #     max_rows_per_type_per_concept_split=None,
    # )
    # verifier_df = verifier_builder.run()



    ''' mimic held-out test '''
    heldout_test_builder = RadiologyVerifierBinaryHeldoutRetrievedTestBuilder(
        qa_parquet=os.path.join(base_path_mimicData, "grounding_dataset", "qa_grounding_main_with_anchors.parquet"),
        evidence_sentences_test_parquet=os.path.join(base_path_evidenceData, "evidence_sentences_test.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "verifier_binary_dataset_heldout_test"),
        seed=13,
        retrieval_top_k_full=30,
        max_evidence_sentences=8,
    )
    heldout_test_df = heldout_test_builder.run()


