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


class RadiopaediaSentenceEvidenceBuilder:
    """
    Build a frozen sentence-level evidence database from Radiopaedia .txt files.
    """

    def __init__(
        self,
        in_dir: str,
        out_dir: str,
        snapshot_name: str = "RadiopaediaSentenceSnapshot_v3",
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        min_tokens: int = 7,
        max_tokens: int = 50,
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
        self.write_csv = bool(write_csv)
        self.write_parquet = bool(write_parquet)
        self.encoding = encoding
        self.tqdm_desc = tqdm_desc

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
            r"cardiac|cardiomegaly|heart|silhouette|vascular|edema|"
            r"interstitial|opacity|opacities|airspace|consolidation|"
            r"atelectasis|atelectatic|hilar|mediastinal|hemithorax|"
            r"costophrenic|rib|diaphragm"
            r")\b",
            re.I,
        )

        self.verb_like_re = re.compile(
            r"\b("
            r"is|are|was|were|be|been|being|"
            r"show|shows|shown|demonstrate|demonstrates|demonstrated|"
            r"appear|appears|appeared|"
            r"seen|visualized|noted|identified|suggest|suggests|"
            r"may|can|could|should|would|remains|becomes|become"
            r")\b",
            re.I,
        )

        self.alias_like_re = re.compile(r"\b(alias|also known as|aka)\b", re.I)

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

    def _sentence_flags(self, sent: str) -> Tuple[bool, bool, bool]:
        s = sent.strip()
        is_definition_like = bool(self.definition_re.search(s))
        is_metadata_like = bool(self.meta_line_re.search(s) or self.author_line_re.search(s))
        is_chest_relevant = bool(self.chest_relevant_re.search(s))
        return is_chest_relevant, is_definition_like, is_metadata_like

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

    def _extract_sentences(self, paragraphs: List[Tuple[Optional[str], str]], article_title: str) -> List[Tuple[Optional[str], str]]:
        out: List[Tuple[Optional[str], str]] = []

        for section_title, para in paragraphs:
            sents = [x.strip() for x in self.sent_split_re.split(para) if x.strip()]
            for sent in sents:
                if self._is_valid_sentence(sent, article_title=article_title):
                    out.append((section_title, sent))

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
                is_chest_relevant, is_definition_like, is_metadata_like = self._sentence_flags(sent)
                rows.append(
                    asdict(
                        EvidenceSentence(
                            sentence_id=sentence_id,
                            article_id=article_id,
                            article_title=title,
                            section_title=section_title,
                            sentence_text=sent,
                            sentence_token_count=int(self._token_count(sent)),
                            article_relpath=relpath,
                            sentence_hash=self._stable_hash(sent),
                            is_chest_relevant=is_chest_relevant,
                            is_definition_like=is_definition_like,
                            is_metadata_like=is_metadata_like,
                        )
                    )
                )
                sentence_id += 1

            pbar.update(1)
        pbar.close()

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
        csv_path = self.out_dir / "evidence_sentences.csv"
        parquet_path = self.out_dir / "evidence_sentences.parquet"

        if self.write_csv:
            df.to_csv(csv_path, index=False)

        wrote_parquet = False
        if self.write_parquet:
            if self._parquet_available():
                df.to_parquet(parquet_path, index=False)
                wrote_parquet = True
            else:
                print("[WARN] Parquet requested but pyarrow/fastparquet not installed. Skipping parquet.")

        manifest = {
            "snapshot_name": self.snapshot_name,
            "input_dir": str(self.in_dir.resolve()),
            "num_articles": int(df["article_id"].nunique()) if not df.empty else 0,
            "num_sentences": int(len(df)),
            "tokenizer": self.tokenizer_name,
            "evidence_unit": "sentence",
            "content_policy": {
                "content_rewritten": False,
                "whitespace_normalized_only": True,
                "metadata_removed": True,
                "sentence_tags_added": True,
                "strict_sentence_filtering": True,
            },
            "files": {
                "csv": "evidence_sentences.csv" if self.write_csv else None,
                "parquet": "evidence_sentences.parquet" if wrote_parquet else None,
            },
        }

        manifest_path = self.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {manifest_path}")
        if self.write_csv:
            print(f"Wrote: {csv_path}  (rows={len(df):,}, articles={manifest['num_articles']:,})")
        if wrote_parquet:
            print(f"Wrote: {parquet_path}  (rows={len(df):,}, articles={manifest['num_articles']:,})")

    def run(self) -> pd.DataFrame:
        df = self.build_snapshot()
        self.write_outputs(df)
        return df




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

            rows.append({
                "qa_row_id": int(i),
                "subject_id": int(r["subject_id"]),
                "study_id": r.get("study_id", None),
                "report_rel_path": r["report_rel_path"],
                "split": r["split"],
                "findings_text": r["findings_text"],
                "impression_text": r["impression_text"],
                "concept": concept,
                "question": self._choose_question(concept, state, i),
                "claim_text": f"{concept} is {state}",
                "label_state": state,
                "label_source_text": r["label_source_text"],
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

    def _best_anchor_sentence(self, findings_text: str, concept: str) -> str:
        sents = [x.strip() for x in self.sent_split_re.split(findings_text) if x.strip()]
        concept_pat = self.concepts[concept]

        best_sent = ""
        best_score = -10**9

        for s in sents:
            s_norm = s.lower()
            phrase_hits = sum(1 for p in self.anchor_phrases if p in s_norm)
            concept_penalty = 3 if concept_pat.search(s) else 0
            informative_len = min(len(self._tokens(s)), 30) * 0.01
            score = phrase_hits - concept_penalty + informative_len

            if score > best_score:
                best_score = score
                best_sent = s

        return best_sent

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
            anchor_sent = self._best_anchor_sentence(r["findings_text"], r["concept"])
            terms = self._extract_anchor_terms(anchor_sent, r["concept"])
            rows.append({
                "qa_row_id": int(r["qa_row_id"]),
                "anchor_sentence_text": anchor_sent,
                "anchor_terms_json": json.dumps(terms),
                "anchor_confidence": "medium" if len(terms) >= 2 else "low",
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






class RadiologyStep6SupervisionBuilder:
    """
    Step 6
    Build final supervision for retriever and verifier.
    """

    def __init__(
        self,
        qa_parquet: str,
        evidence_sentences_parquet: str,
        out_dir: str,
        model_name: str = "BAAI/bge-base-en-v1.5",
        seed: int = 13,
        device: str = "cuda",
    ):
        self.qa_parquet = Path(qa_parquet)
        self.evidence_sentences_parquet = Path(evidence_sentences_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.seed = int(seed)
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        self.max_query_len = 128
        self.max_doc_len = 128
        self.batch_size = 128

        self.dense_topk = 80
        self.lexical_topk = 80
        self.merge_topk = 48

        self.max_retriever_pos = 2
        self.max_verifier_evidence = 2

        self.min_support_score = 1.75
        self.min_contradict_score = 1.75
        self.min_insufficient_score = 0.45
        self.max_insufficient_score = 1.35
        self.min_insufficient_anchor_overlap = 0.15

        self.retriever_out = self.out_dir / "retriever_supervision.parquet"
        self.verifier_out = self.out_dir / "verifier_supervision.parquet"
        self.manifest_out = self.out_dir / "manifest_step6.json"

        self.embeds_path = self.out_dir / "evidence_sentence_embeds.npy"
        self.index_path = self.out_dir / "evidence_sentence.index"
        self.bm25_path = self.out_dir / "evidence_sentence_bm25.pkl"

        self.allowed_concepts = {
            "pleural effusion",
            "pneumothorax",
            "atelectasis",
            "consolidation",
            "pulmonary edema",
            "interstitial edema",
        }

        self.word_re = re.compile(r"[a-z0-9]+")
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

        self.population_mismatch_re = re.compile(
            r"\b(fetal|foetal|neonatal|newborn|infant|child|children|pediatric|paediatric|congenital)\b",
            re.I,
        )

        self.generic_style_re = re.compile(
            r"\b("
            r"may occur|can occur|often|usually|typically|classically|commonly|"
            r"is defined as|defined as|refers to|is characterized by|describes|represents|"
            r"summary|reference article|see also|history and etymology|epidemiology|"
            r"treatment and prognosis|differential diagnosis|classification|etiology|aetiology|"
            r"pathology|risk factors|for medical students|this is a basic article"
            r")\b",
            re.I,
        )

        self.non_findings_re = re.compile(
            r"\b("
            r"treatment|management|surgery|surgical|therapy|prognosis|mortality|"
            r"incidence|prevalence|cause|causes|caused by|risk factor|complication|complications"
            r")\b",
            re.I,
        )

        self.fragment_style_re = re.compile(r"^[A-Z][A-Za-z0-9\-\s/]{0,45}$")
        self.alias_like_re = re.compile(r"\b(alias|also known as|aka)\b", re.I)

        self.imaging_terms_re = re.compile(
            r"\b("
            r"radiograph|radiographs|x-ray|xray|ct|scan|imaging|film|films|"
            r"appearance|appears|seen|visualized|demonstrates|demonstrated|show|shows|"
            r"opacity|opacities|opacification|consolidation|effusion|pneumothorax|"
            r"atelectasis|edema|vascular|silhouette|cardiomediastinal|hilar|mediastinal|"
            r"pleural|airspace|lung|lungs|thorax|chest"
            r")\b",
            re.I,
        )

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
            "pulmonary edema": re.compile(r"\bpulmonary edema\b|\bedema\b", re.I),
            "vascular congestion": re.compile(
                r"\bvascular congestion\b|"
                r"\bpulmonary vascular congestion\b|"
                r"\bcentral pulmonary vascular congestion\b|"
                r"\bvascular indistinctness\b|"
                r"\bhilar vasculature\b",
                re.I,
            ),
            "interstitial edema": re.compile(r"\binterstitial edema\b|\binterstitial pulmonary edema\b", re.I),
        }

        self.concept_synonyms = {
            "pleural effusion": [
                r"\beffusion\b", r"\beffusions\b", r"\bpleural fluid\b", r"\bpleural collection\b"
            ],
            "pneumothorax": [
                r"\bpneumothorax\b", r"\bpneumothoraces\b", r"\bpleural air\b"
            ],
            "atelectasis": [
                r"\batelectasis\b", r"\batelectatic\b", r"\bcollapse\b", r"\bsubsegmental collapse\b"
            ],
            "consolidation": [
                r"\bconsolidation\b", r"\bairspace consolidation\b", r"\bairspace opacity\b", r"\bairspace opacities\b"
            ],
            "pulmonary edema": [
                r"\bpulmonary edema\b", r"\bedema\b", r"\boedema\b"
            ],
            "interstitial edema": [
                r"\binterstitial edema\b", r"\binterstitial pulmonary edema\b", r"\binterstitial oedema\b"
            ],
        }

        self.question_templates = [
            "Is there {concept}?",
            "Does the study show {concept}?",
            "Is {concept} described in the report?",
            "Are findings of {concept} present?",
        ]

        self.answer_templates = {
            "present": [
                "Yes.",
                "{Concept} is present.",
                "The study shows {concept}.",
            ],
            "absent": [
                "No.",
                "{Concept} is absent.",
                "The study does not show {concept}.",
            ],
            "uncertain": [
                "Uncertain.",
                "{Concept} is uncertain.",
                "The report does not clearly establish {concept}.",
            ],
        }

        self._set_seed()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _norm(self, s: Optional[str]) -> str:
        if s is None:
            return ""
        s = str(s).replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _tokens(self, s: str) -> List[str]:
        return self.word_re.findall(self._norm(s).lower())

    def _infer_claim(self, claim_text: str) -> Tuple[str, str]:
        claim_text = self._norm(claim_text)
        for suffix in [" is present", " is absent", " is uncertain"]:
            if claim_text.endswith(suffix):
                return claim_text[: -len(suffix)].strip(), suffix.replace(" is ", "")
        raise ValueError(f"Cannot parse claim_text: {claim_text}")

    def _claim_text(self, concept: str, state: str) -> str:
        return f"{concept} is {state}"

    def _question_text(self, concept: str, qa_row_id: int) -> str:
        return self.question_templates[qa_row_id % len(self.question_templates)].format(concept=concept)

    def _answer_text(self, concept: str, state: str, qa_row_id: int) -> str:
        tmpls = self.answer_templates[state]
        tmpl = tmpls[(qa_row_id + len(state)) % len(tmpls)]
        return tmpl.format(concept=concept, Concept=concept.capitalize())

    def _state_order(self, gold_state: str) -> List[Tuple[str, str]]:
        if gold_state == "present":
            return [("present", "supported"), ("absent", "contradicted"), ("uncertain", "insufficient")]

        if gold_state == "absent":
            return [("absent", "supported"), ("present", "contradicted"), ("uncertain", "insufficient")]

        if gold_state == "uncertain":
            return [("uncertain", "supported"), ("present", "insufficient")]

    def _anchor_overlap(self, sent: str, anchor_terms: List[str]) -> float:
        if not anchor_terms:
            return 0.0
        stoks = set(self._tokens(sent))
        hit = sum(1 for t in anchor_terms if t in stoks)
        return float(hit) / float(len(anchor_terms))

    def _build_query(self, claim_text: str, anchor_terms: List[str]) -> str:
        return f"{claim_text}\n\n" + " ".join(anchor_terms[:8]) if anchor_terms else claim_text

    def _concept_mentioned(self, sent: str, concept: str) -> bool:
        return bool(self.concepts[concept].search(sent))

    def _concept_or_synonym_mentioned(self, sent: str, concept: str) -> bool:
        if self._concept_mentioned(sent, concept):
            return True
        syns = self.concept_synonyms.get(concept, [])
        for pat in syns:
            if re.search(pat, sent, re.I):
                return True
        return False

    def _classify_sentence_state(self, sent: str, concept: str) -> str:
        if not self._concept_mentioned(sent, concept):
            return ""
        if self.negation_re.search(sent):
            return "absent"
        if self.uncertainty_re.search(sent):
            return "uncertain"
        if self.resolution_re.search(sent):
            return "absent"
        return "present"

    def _is_good_findings_sentence(self, sent: str, is_definition_like: bool) -> bool:
        s = self._norm(sent)
        toks = self._tokens(s)

        if len(toks) < 7 or len(toks) > 50:
            return False
        if len(set(toks)) < 5:
            return False
        if is_definition_like:
            return False
        if self.population_mismatch_re.search(s):
            return False
        if self.generic_style_re.search(s):
            return False
        if self.non_findings_re.search(s):
            return False
        if self.alias_like_re.search(s):
            return False
        if self.fragment_style_re.match(s) and "." not in s:
            return False
        if not self.imaging_terms_re.search(s):
            return False

        return True

    def _load_sentence_corpus(self) -> pd.DataFrame:
        sent_df = pd.read_parquet(self.evidence_sentences_parquet)

        required = {
            "sentence_id",
            "sentence_text",
            "article_id",
            "article_title",
            "is_chest_relevant",
            "is_definition_like",
            "is_metadata_like",
        }
        missing = required - set(sent_df.columns)
        if missing:
            raise ValueError(f"evidence_sentences_parquet missing columns: {missing}")

        sent_df = sent_df.copy()
        sent_df["sentence_id"] = pd.to_numeric(sent_df["sentence_id"], errors="coerce")
        sent_df = sent_df[~sent_df["sentence_id"].isna()].copy()
        sent_df["sentence_id"] = sent_df["sentence_id"].astype(np.int64)
        sent_df["sentence_text"] = sent_df["sentence_text"].astype(str)
        sent_df = sent_df[sent_df["is_metadata_like"] == False].reset_index(drop=True)
        return sent_df

    def _encode_texts(self, texts: List[str], max_len: int, batch_size: int = 128) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = [self._norm(x) for x in texts[i:i + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                vec = out.last_hidden_state[:, 0]
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            all_vecs.append(vec.cpu().numpy().astype(np.float32))

        return np.vstack(all_vecs) if all_vecs else np.zeros((0, 768), dtype=np.float32)

    def _build_sentence_index(self, sent_df: pd.DataFrame) -> faiss.Index:
        if self.embeds_path.exists() and self.index_path.exists():
            return faiss.read_index(str(self.index_path))

        texts = sent_df["sentence_text"].tolist()
        embs = self._encode_texts(texts, max_len=self.max_doc_len, batch_size=self.batch_size)
        np.save(self.embeds_path, embs)

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, str(self.index_path))
        return index

    def _build_bm25(self, sent_df: pd.DataFrame):
        if self.bm25_path.exists():
            with open(self.bm25_path, "rb") as f:
                return pickle.load(f)

        postings: Dict[str, List[Tuple[int, int]]] = {}
        dfreq: Dict[str, int] = {}
        sent_ids = sent_df["sentence_id"].tolist()
        doc_len = np.zeros(len(sent_df), dtype=np.int32)

        pbar = tqdm(total=len(sent_df), desc="Building BM25", unit="sentences")
        for i, text in enumerate(sent_df["sentence_text"].tolist()):
            toks = self._tokens(text)
            doc_len[i] = len(toks)

            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                postings.setdefault(t, []).append((i, c))
                dfreq[t] = dfreq.get(t, 0) + 1
            pbar.update(1)
        pbar.close()

        n = len(sent_df)
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
        with open(self.bm25_path, "wb") as f:
            pickle.dump(state, f)
        return state

    def _bm25_search(self, query: str, bm25_state, topk: int) -> Tuple[List[int], List[float]]:
        toks = self._tokens(query)
        if not toks:
            return [], []

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
            return [], []

        best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        ids = [int(sent_ids[i]) for i, _ in best]
        vals = [float(v) for _, v in best]
        return ids, vals

    def _zscore(self, xs: List[float]) -> List[float]:
        if not xs:
            return []
        arr = np.asarray(xs, dtype=np.float32)
        sd = float(arr.std())
        if sd < 1e-8:
            return [0.0] * len(xs)
        mu = float(arr.mean())
        return [float((x - mu) / sd) for x in arr.tolist()]

    def _merge_candidates(
        self,
        dense_ids: List[int],
        dense_scores: List[float],
        lex_ids: List[int],
        lex_scores: List[float],
    ) -> List[Tuple[int, float]]:
        all_ids = sorted(set(dense_ids) | set(lex_ids))
        dmap = {i: s for i, s in zip(dense_ids, dense_scores)}
        lmap = {i: s for i, s in zip(lex_ids, lex_scores)}

        d_raw = [dmap.get(i, 0.0) for i in all_ids]
        l_raw = [lmap.get(i, 0.0) for i in all_ids]

        dz = self._zscore(d_raw)
        lz = self._zscore(l_raw)

        rows = [(sid, 0.5 * ds + 0.5 * ls) for sid, ds, ls in zip(all_ids, dz, lz)]
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows[: self.merge_topk]

    def _select_support_rows(
        self,
        candidates: List[Tuple[int, float]],
        sent_meta: Dict[int, Dict[str, object]],
        concept: str,
        target_state: str,
        anchor_terms: List[str],
        top_k: int,
    ) -> List[Tuple[float, int, str, str]]:
        rows = []
        for sid, prior in candidates:
            meta = sent_meta[sid]
            sent = meta["sentence_text"]

            if not self._is_good_findings_sentence(sent, meta["is_definition_like"]):
                continue
            if not self._concept_mentioned(sent, concept):
                continue

            state = self._classify_sentence_state(sent, concept)
            if state != target_state:
                continue

            score = prior
            score += 2.0
            score += 1.5 * self._anchor_overlap(sent, anchor_terms)
            if meta["is_chest_relevant"]:
                score += 0.25

            rows.append((float(score), sid, meta["article_title"], sent))

        rows.sort(key=lambda x: x[0], reverse=True)
        rows = [x for x in rows if x[0] >= self.min_support_score]
        return rows[:top_k]

    def _select_insufficient_rows(
        self,
        candidates: List[Tuple[int, float]],
        sent_meta: Dict[int, Dict[str, object]],
        concept: str,
        anchor_terms: List[str],
        top_k: int,
    ) -> List[Tuple[float, int, str, str]]:
        rows = []
        for sid, prior in candidates:
            meta = sent_meta[sid]
            sent = meta["sentence_text"]

            if not self._is_good_findings_sentence(sent, meta["is_definition_like"]):
                continue
            if not bool(meta["is_chest_relevant"]):
                continue
            if self._concept_or_synonym_mentioned(sent, concept):
                continue

            ao = self._anchor_overlap(sent, anchor_terms)
            score = prior + 0.8 * ao + 0.20

            if ao < self.min_insufficient_anchor_overlap:
                continue
            if score < self.min_insufficient_score or score > self.max_insufficient_score:
                continue

            rows.append((float(score), sid, meta["article_title"], sent))

        rows.sort(key=lambda x: x[0], reverse=True)
        return rows[:top_k]

    def _dedup_rows(self, rows: List[Tuple[float, int, str, str]]) -> List[Tuple[float, int, str, str]]:
        out = []
        seen = set()
        for row in rows:
            key = (int(row[1]), self._norm(row[3]).lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def run(self):
        qa = pd.read_parquet(self.qa_parquet)
        # TODO; remove this
        # qa = qa.iloc[: int(len(qa) * 0.1)].reset_index(drop=True)

        qa["anchor_terms_json"] = qa["anchor_terms_json"].fillna("[]")
        qa = qa[qa["concept"].isin(self.allowed_concepts)].reset_index(drop=True)

        sent_df = self._load_sentence_corpus()
        sent_meta = {
            int(r["sentence_id"]): {
                "sentence_text": r["sentence_text"],
                "article_title": r["article_title"],
                "is_chest_relevant": bool(r["is_chest_relevant"]),
                "is_definition_like": bool(r["is_definition_like"]),
            }
            for _, r in sent_df.iterrows()
        }

        index = self._build_sentence_index(sent_df)
        bm25_state = self._build_bm25(sent_df)

        retr_rows = []
        ver_rows = []

        concept_label_counts = {}
        support_yield = {}
        dropped_no_clean_evidence = 0

        pbar = tqdm(total=len(qa), desc="Building step-6 supervision", unit="qa")
        for _, r in qa.iterrows():
            qa_row_id = int(r["qa_row_id"])
            split = r["split"]
            subject_id = int(r["subject_id"])
            report_rel_path = r["report_rel_path"]
            case_context = r.get("case_context", r.get("findings_text", ""))
            supporting_span = r.get("supporting_span", "")
            claim_text_gold = r["claim_text"]
            concept, gold_state = self._infer_claim(claim_text_gold)
            anchor_terms = json.loads(r["anchor_terms_json"])

            retr_query = self._build_query(claim_text_gold, anchor_terms)
            retr_emb = self._encode_texts([retr_query], max_len=self.max_query_len, batch_size=1)
            D, I = index.search(retr_emb, self.dense_topk)

            dense_ids, dense_scores, seen = [], [], set()
            for j, sc in zip(I[0].tolist(), D[0].tolist()):
                if j < 0:
                    continue
                sid = int(sent_df.iloc[j]["sentence_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                dense_ids.append(sid)
                dense_scores.append(float(sc))

            lex_ids, lex_scores = self._bm25_search(retr_query, bm25_state, self.lexical_topk)
            merged = self._merge_candidates(dense_ids, dense_scores, lex_ids, lex_scores)

            retr_pos = self._select_support_rows(
                candidates=merged,
                sent_meta=sent_meta,
                concept=concept,
                target_state=gold_state,
                anchor_terms=anchor_terms,
                top_k=self.max_retriever_pos,
            )
            retr_pos = self._dedup_rows(retr_pos)

            if retr_pos:
                support_yield[concept] = support_yield.get(concept, 0) + 1
                retr_rows.append({
                    "qa_row_id": qa_row_id,
                    "split": split,
                    "subject_id": subject_id,
                    "report_rel_path": report_rel_path,
                    "concept": concept,
                    "gold_state": gold_state,
                    "retrieval_query": retr_query,
                    "positive_sentence_ids_json": json.dumps([int(x[1]) for x in retr_pos]),
                    "positive_article_titles_json": json.dumps([x[2] for x in retr_pos]),
                    "positive_sentences_json": json.dumps([x[3] for x in retr_pos]),
                })

            for cand_state, label in self._state_order(gold_state):
                cand_claim = self._claim_text(concept, cand_state)
                cand_answer = self._answer_text(concept, cand_state, qa_row_id)
                cand_query = self._build_query(cand_claim, anchor_terms)

                cand_emb = self._encode_texts([cand_query], max_len=self.max_query_len, batch_size=1)
                Dv, Iv = index.search(cand_emb, self.dense_topk)

                dense_ids_v, dense_scores_v, seen_v = [], [], set()
                for j, sc in zip(Iv[0].tolist(), Dv[0].tolist()):
                    if j < 0:
                        continue
                    sid = int(sent_df.iloc[j]["sentence_id"])
                    if sid in seen_v:
                        continue
                    seen_v.add(sid)
                    dense_ids_v.append(sid)
                    dense_scores_v.append(float(sc))

                lex_ids_v, lex_scores_v = self._bm25_search(cand_query, bm25_state, self.lexical_topk)
                merged_v = self._merge_candidates(dense_ids_v, dense_scores_v, lex_ids_v, lex_scores_v)

                if label in {"supported", "contradicted"}:
                    evidence_rows = self._select_support_rows(
                        candidates=merged_v,
                        sent_meta=sent_meta,
                        concept=concept,
                        target_state=cand_state,
                        anchor_terms=anchor_terms,
                        top_k=self.max_verifier_evidence,
                    )
                else:
                    evidence_rows = self._select_insufficient_rows(
                        candidates=merged_v,
                        sent_meta=sent_meta,
                        concept=concept,
                        anchor_terms=anchor_terms,
                        top_k=self.max_verifier_evidence,
                    )

                evidence_rows = self._dedup_rows(evidence_rows)

                if not evidence_rows:
                    dropped_no_clean_evidence += 1
                    continue

                concept_label_counts[(concept, label)] = concept_label_counts.get((concept, label), 0) + 1

                ver_rows.append({
                    "verifier_row_id": len(ver_rows),
                    "qa_row_id": qa_row_id,
                    "split": split,
                    "subject_id": subject_id,
                    "report_rel_path": report_rel_path,
                    "concept": concept,
                    "gold_state": gold_state,
                    "candidate_state": cand_state,
                    "question": self._question_text(concept, qa_row_id),
                    "answer_candidate": cand_answer,
                    "claim_text": cand_claim,
                    "case_context": case_context,
                    "supporting_span": supporting_span,
                    "evidence_sentence_ids_json": json.dumps([int(x[1]) for x in evidence_rows]),
                    "evidence_article_titles_json": json.dumps([x[2] for x in evidence_rows]),
                    "evidence_sentences_json": json.dumps([x[3] for x in evidence_rows]),
                    "label": label,
                    "label_id": {"insufficient": 0, "contradicted": 1, "supported": 2}[label],
                })

            pbar.update(1)
        pbar.close()

        retr_df = pd.DataFrame(retr_rows)
        ver_df = pd.DataFrame(ver_rows)

        retr_df.to_parquet(self.retriever_out, index=False)
        ver_df.to_parquet(self.verifier_out, index=False)

        print("\n[Step 6 audit]")
        print(f"input qa rows: {len(qa):,}")
        print(f"retriever rows: {len(retr_df):,}")
        print(f"verifier rows: {len(ver_df):,}")
        print(f"dropped verifier rows for no clean evidence: {dropped_no_clean_evidence:,}")
        print("\nPer concept support yield:")
        for k in sorted(support_yield):
            print(f"  {k}: {support_yield[k]:,}")
        print("\nPer concept x label counts:")
        for k in sorted(concept_label_counts):
            print(f"  {k[0]} | {k[1]}: {concept_label_counts[k]:,}")

        manifest = {
            "step": "step6_report_grounded_verifier_v4",
            "qa_parquet": str(self.qa_parquet.resolve()),
            "evidence_sentences_parquet": str(self.evidence_sentences_parquet.resolve()),
            "retriever_supervision": str(self.retriever_out.resolve()),
            "verifier_supervision": str(self.verifier_out.resolve()),
            "model_name_for_weak_retrieval": self.model_name,
            "allowed_concepts": sorted(self.allowed_concepts),
            "split_policy": "inherited_from_steps_0_5_patient_wise_75_10_15",
            "verifier_task": "case_context + question + answer_candidate + retrieved_evidence -> 3-way label",
            "verifier_labels": {"insufficient": 0, "contradicted": 1, "supported": 2},
            "selection_thresholds": {
                "min_support_score": self.min_support_score,
                "min_contradict_score": self.min_contradict_score,
                "min_insufficient_score": self.min_insufficient_score,
                "max_insufficient_score": self.max_insufficient_score,
                "min_insufficient_anchor_overlap": self.min_insufficient_anchor_overlap,
            },
            "counts": {
                "num_qa_rows_input": int(len(qa)),
                "num_retriever_rows": int(len(retr_df)),
                "num_verifier_rows": int(len(ver_df)),
                "num_evidence_sentences": int(len(sent_df)),
                "dropped_verifier_rows_for_no_clean_evidence": int(dropped_no_clean_evidence),
            },
        }
        self.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote: {self.retriever_out}")
        print(f"Wrote: {self.verifier_out}")
        print(f"Wrote: {self.manifest_out}")

        return {
            "retriever_supervision": retr_df,
            "verifier_supervision": ver_df,
        }





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
    mimic_reports_base = params['Preprocessing']['mimic_reports_base']


    ''' steps 0 to 5 '''
    # builder = MimicGroundingDatasetBuilder(
    #     master_csv=os.path.join(base_path_mimicData, "mimic_master_list_forReports.csv"),
    #     reports_root=mimic_reports_base,
    #     out_dir=os.path.join(base_path_mimicData, "grounding_dataset"),
    #     seed=13,
    # )
    # builder.run_all()



    ''' step 6a: RadiopaediaSnapshot_v1 (chunking only, no embeddings) '''
    # evidence_builder = RadiopaediaSentenceEvidenceBuilder(
    #     in_dir=os.path.join(base_path_evidenceData, "articles_clean_txt"),
    #     out_dir=base_path_evidenceData,
    #     snapshot_name=params['Preprocessing']['snapshot_name'],
    #     tokenizer_name=params["Preprocessing"]["tokenizer_name"],
    #     min_tokens=7,
    #     max_tokens=50,
    #     write_csv=True,
    #     write_parquet=True,
    #     encoding="utf-8",
    # )
    # evidence_df = evidence_builder.run()


    ''' step 6 '''
    builder = RadiologyStep6SupervisionBuilder(
        qa_parquet=os.path.join(base_path_mimicData, "grounding_dataset/qa_grounding_main_with_anchors.parquet"),
        evidence_sentences_parquet=os.path.join(base_path_evidenceData, "evidence_sentences.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "step6_final_supervision"),
        model_name=params["Preprocessing"]["tokenizer_name"],
        seed=int(params["Network"]["seed"]),
        device="cuda")
    outputs = builder.run()
    retriever_df = outputs["retriever_supervision"]
    verifier_df = outputs["verifier_supervision"]




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

