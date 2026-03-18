"""
Created on Feb 23, 2026.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import json
import os.path
import pdb
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import math
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional
import random
import re

from config.serde import read_config






@dataclass(frozen=True)
class RetrieverBatch:
    queries: List[str]
    pos_docs: List[str]
    neg_docs: List[List[str]]
    qa_row_ids: Optional[List[int]]


class RadiopaediaSentenceStore:
    """
    Sentence-level evidence store with lightweight BM25 support.
    """

    def __init__(
        self,
        evidence_sentences_parquet: str,
        sentence_id_col: str = "sentence_id",
        sentence_text_col: str = "sentence_text",
        chest_col: str = "is_chest_relevant",
        definition_col: str = "is_definition_like",
        metadata_col: str = "is_metadata_like",
        article_title_col: str = "article_title",
    ):
        if not os.path.exists(evidence_sentences_parquet):
            raise FileNotFoundError(f"Evidence sentences parquet not found: {evidence_sentences_parquet}")

        cols = [sentence_id_col, sentence_text_col, chest_col, definition_col, metadata_col, article_title_col]
        df = pd.read_parquet(evidence_sentences_parquet, columns=cols)

        df[sentence_id_col] = pd.to_numeric(df[sentence_id_col], errors="coerce")
        df = df.dropna(subset=[sentence_id_col]).copy()
        df[sentence_id_col] = df[sentence_id_col].astype(int)
        df[sentence_text_col] = df[sentence_text_col].astype(str)

        for c in [chest_col, definition_col, metadata_col]:
            if c in df.columns:
                df[c] = df[c].fillna(False).astype(bool)
            else:
                df[c] = False

        if article_title_col not in df.columns:
            df[article_title_col] = ""

        self.sentence_id_col = sentence_id_col
        self.sentence_text_col = sentence_text_col
        self.chest_col = chest_col
        self.definition_col = definition_col
        self.metadata_col = metadata_col
        self.article_title_col = article_title_col

        self.df = df.reset_index(drop=True)

        self.id2text: Dict[int, str] = dict(zip(df[sentence_id_col].tolist(), df[sentence_text_col].tolist()))
        self.id2meta: Dict[int, Dict[str, object]] = {
            int(r[sentence_id_col]): {
                "sentence_text": str(r[sentence_text_col]),
                "is_chest_relevant": bool(r[chest_col]),
                "is_definition_like": bool(r[definition_col]),
                "is_metadata_like": bool(r[metadata_col]),
                "article_title": str(r[article_title_col]),
            }
            for _, r in df.iterrows()
        }
        self.all_sentence_ids: List[int] = list(self.id2text.keys())

        self.word_re = re.compile(r"[a-z0-9]+")
        self._build_bm25()

    def _tokens(self, s: str) -> List[str]:
        return self.word_re.findall(str(s).lower())

    def _build_bm25(self):
        postings: Dict[str, List[Tuple[int, int]]] = {}
        dfreq: Dict[str, int] = {}
        doc_len = np.zeros(len(self.df), dtype=np.int32)
        sent_ids = self.df[self.sentence_id_col].tolist()

        for i, text in enumerate(self.df[self.sentence_text_col].tolist()):
            toks = self._tokens(text)
            doc_len[i] = len(toks)

            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            for t, c in tf.items():
                postings.setdefault(t, []).append((i, c))
                dfreq[t] = dfreq.get(t, 0) + 1

        n = len(self.df)
        avgdl = float(doc_len.mean()) if n > 0 else 0.0
        idf = {t: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for t, df in dfreq.items()}

        self._bm25_state = {
            "sent_ids": sent_ids,
            "postings": postings,
            "idf": idf,
            "doc_len": doc_len,
            "avgdl": avgdl,
            "k1": 1.2,
            "b": 0.75,
        }

    def bm25_search(self, query: str, topk: int) -> List[int]:
        toks = self._tokens(query)
        if not toks:
            return []

        sent_ids = self._bm25_state["sent_ids"]
        postings = self._bm25_state["postings"]
        idf = self._bm25_state["idf"]
        doc_len = self._bm25_state["doc_len"]
        avgdl = self._bm25_state["avgdl"]
        k1 = self._bm25_state["k1"]
        b = self._bm25_state["b"]

        scores = {}
        for t in toks:
            plist = postings.get(t, [])
            idft = idf.get(t, 0.0)
            for idx, tf in plist:
                dl = float(doc_len[idx])
                denom = tf + k1 * (1.0 - b + b * dl / (avgdl + 1e-12))
                s = idft * (tf * (k1 + 1.0)) / (denom + 1e-12)
                scores[idx] = scores.get(idx, 0.0) + float(s)

        best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [int(sent_ids[i]) for i, _ in best]

    def get(self, sentence_id: int) -> str:
        return self.id2text.get(int(sentence_id), "")

    def get_meta(self, sentence_id: int) -> Dict[str, object]:
        return self.id2meta.get(int(sentence_id), {})


class RetrieverSupervisionDataset(Dataset):
    """
    Reads Step 6 retriever supervision and returns one positive + hard negatives.
    Config-driven version.
    """

    def __init__(
        self,
        cfg_path: str,
        sentence_store: RadiopaediaSentenceStore,
        split_name: str,
        supervision_relpath: Optional[str] = None,
        split_col: str = "split",
        qa_row_id_col: str = "qa_row_id",
        query_col: str = "retrieval_query",
        concept_col: str = "concept",
        gold_state_col: str = "gold_state",
        pos_ids_col: str = "positive_sentence_ids_json",
        pos_texts_col: str = "positive_sentences_json",
        bm25_pool_size: Optional[int] = None,
        num_hard_negatives: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.base_path = self.params["Preprocessing"]["base_path_evidenceData"]

        self.sentence_store = sentence_store
        self.split_name = str(split_name)

        self.split_col = str(split_col)
        self.qa_row_id_col = str(qa_row_id_col)
        self.query_col = str(query_col)
        self.concept_col = str(concept_col)
        self.gold_state_col = str(gold_state_col)
        self.pos_ids_col = str(pos_ids_col)
        self.pos_texts_col = str(pos_texts_col)

        if supervision_relpath is None:
            supervision_relpath = self.params["retriever"].get(
                "supervision_relpath",
                "step6_final_supervision/retriever_supervision.parquet",
            )
        self.supervision_relpath = str(supervision_relpath)

        if bm25_pool_size is None:
            bm25_pool_size = int(self.params["retriever"]["bm25_pool_size"])
        if num_hard_negatives is None:
            num_hard_negatives = int(self.params["retriever"]["num_hard_negatives"])
        if seed is None:
            seed = int(self.params["Network"]["seed"])

        self.bm25_pool_size = int(bm25_pool_size)
        self.num_hard_negatives = int(num_hard_negatives)
        self.base_seed = int(seed)

        retriever_supervision_path = os.path.join(self.base_path, self.supervision_relpath)
        if not os.path.exists(retriever_supervision_path):
            raise FileNotFoundError(f"Retriever supervision parquet not found: {retriever_supervision_path}")

        df = pd.read_parquet(retriever_supervision_path)

        required = [self.split_col, self.qa_row_id_col, self.query_col, self.pos_ids_col]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column in retriever supervision: {c}")

        df[self.split_col] = df[self.split_col].astype(str)
        df = df[df[self.split_col] == self.split_name].reset_index(drop=True)
        df[self.query_col] = df[self.query_col].astype(str)

        self.has_pos_texts = self.pos_texts_col in df.columns
        self.has_concept = self.concept_col in df.columns
        self.has_gold_state = self.gold_state_col in df.columns

        self.df = df

        self.negation_re = re.compile(r"\b(no|without|absent|negative for|free of|no evidence of|no focal|neither)\b", re.I)
        self.uncertainty_re = re.compile(r"\b(possible|possibly|may|might|likely|unlikely|cannot exclude|could represent|suggestive of|question of)\b", re.I)
        self.resolution_re = re.compile(r"\b(resolved|improved|decreased|cleared|near complete resolution)\b", re.I)

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
                r"\bmild cardiomegaly\b", re.I
            ),
            "atelectasis": re.compile(
                r"\batelectasis\b|"
                r"\batelectatic\b|"
                r"\bsubsegmental atelectatic\b|"
                r"\bbibasilar atelectatic\b|"
                r"\bbasal atelectatic\b", re.I
            ),
            "consolidation": re.compile(
                r"\bconsolidation\b|"
                r"\blobar consolidation\b|"
                r"\bfocal consolidation\b|"
                r"\bairspace consolidation\b", re.I
            ),
            "pulmonary edema": re.compile(r"\bpulmonary edema\b|\bedema\b", re.I),
            "vascular congestion": re.compile(
                r"\bvascular congestion\b|"
                r"\bpulmonary vascular congestion\b|"
                r"\bcentral pulmonary vascular congestion\b|"
                r"\bvascular indistinctness\b|"
                r"\bhilar vasculature\b", re.I
            ),
            "interstitial edema": re.compile(r"\binterstitial edema\b|\binterstitial pulmonary edema\b", re.I),
        }

    def __len__(self) -> int:
        return len(self.df)

    def _loads(self, s: str) -> List:
        try:
            out = json.loads(str(s))
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _rng_for_row(self, qa_row_id: int) -> random.Random:
        return random.Random(self.base_seed + int(qa_row_id))

    def _concept_mentioned(self, sent: str, concept: str) -> bool:
        pat = self.concepts.get(concept)
        if pat is None:
            return False
        return bool(pat.search(sent))

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

    def _is_sentence_usable_negative(self, sid: int) -> bool:
        meta = self.sentence_store.get_meta(sid)
        if not meta:
            return False
        if bool(meta.get("is_metadata_like", False)):
            return False
        if bool(meta.get("is_definition_like", False)):
            return False

        txt = str(meta.get("sentence_text", "")).strip()
        if not txt:
            return False

        toks = re.findall(r"[a-z0-9]+", txt.lower())
        return len(toks) >= 6

    def _mine_hard_negative_ids(
        self,
        query: str,
        concept: str,
        gold_state: str,
        positive_ids: List[int],
        qa_row_id: int,
    ) -> List[int]:
        rng = self._rng_for_row(qa_row_id)
        pos_set = set(int(x) for x in positive_ids)

        bm25_ids = self.sentence_store.bm25_search(query, topk=self.bm25_pool_size)

        wrong_state = []
        near_topic_no_concept = []
        fallback = []

        for sid in bm25_ids:
            sid = int(sid)
            if sid in pos_set:
                continue
            if not self._is_sentence_usable_negative(sid):
                continue

            txt = self.sentence_store.get(sid)
            if not txt.strip():
                continue

            has_concept = self._concept_mentioned(txt, concept)
            if has_concept:
                st = self._classify_sentence_state(txt, concept)
                if st and st != gold_state:
                    wrong_state.append(sid)
            else:
                near_topic_no_concept.append(sid)

        if len(wrong_state) + len(near_topic_no_concept) < self.num_hard_negatives:
            all_ids = self.sentence_store.all_sentence_ids[:]
            rng.shuffle(all_ids)
            for sid in all_ids:
                sid = int(sid)
                if sid in pos_set:
                    continue
                if sid in wrong_state or sid in near_topic_no_concept or sid in fallback:
                    continue
                if not self._is_sentence_usable_negative(sid):
                    continue

                meta = self.sentence_store.get_meta(sid)
                if not bool(meta.get("is_chest_relevant", False)):
                    continue

                fallback.append(sid)
                if len(wrong_state) + len(near_topic_no_concept) + len(fallback) >= self.num_hard_negatives:
                    break

        ordered = wrong_state + near_topic_no_concept + fallback

        uniq = []
        seen = set()
        for sid in ordered:
            if sid not in seen:
                seen.add(sid)
                uniq.append(sid)

        return uniq[: self.num_hard_negatives]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        qa_row_id = int(row[self.qa_row_id_col])
        query = str(row[self.query_col]).strip()

        pos_ids = [int(x) for x in self._loads(row[self.pos_ids_col])]
        pos_docs = []

        for sid in pos_ids:
            txt = self.sentence_store.get(sid)
            if txt and txt.strip():
                pos_docs.append((sid, txt.strip()))

        if len(pos_docs) == 0:
            return {
                "qa_row_id": qa_row_id,
                "query": query,
                "pos_doc": "",
                "neg_docs": [],
            }

        rng = self._rng_for_row(qa_row_id)
        _, pos_doc = rng.choice(pos_docs)

        concept = str(row[self.concept_col]).strip() if self.has_concept else ""
        gold_state = str(row[self.gold_state_col]).strip() if self.has_gold_state else ""

        neg_ids = self._mine_hard_negative_ids(
            query=query,
            concept=concept,
            gold_state=gold_state,
            positive_ids=[sid for sid, _ in pos_docs],
            qa_row_id=qa_row_id,
        )

        neg_docs = []
        for sid in neg_ids:
            txt = self.sentence_store.get(sid)
            if txt and txt.strip():
                neg_docs.append(txt.strip())

        return {
            "qa_row_id": qa_row_id,
            "query": query,
            "pos_doc": pos_doc,
            "neg_docs": neg_docs,
        }


def retriever_collate_fn(batch: List[Dict]) -> RetrieverBatch:
    queries = []
    pos_docs = []
    neg_docs = []
    qa_row_ids = []

    for item in batch:
        queries.append(str(item.get("query", "")).strip())
        pos_docs.append(str(item.get("pos_doc", "")).strip())
        neg_docs.append([str(x).strip() for x in item.get("neg_docs", []) if str(x).strip()])
        qa_row_ids.append(int(item.get("qa_row_id", -1)))

    return RetrieverBatch(
        queries=queries,
        pos_docs=pos_docs,
        neg_docs=neg_docs,
        qa_row_ids=qa_row_ids,
    )





@dataclass(frozen=True)
class VerifierBatch:
    inputs: List[str]
    targets: torch.Tensor
    qa_row_ids: List[int]
    verifier_row_ids: List[int]


class VerifierBinaryDataset(Dataset):
    """
    Reads binary verifier supervision parquet.

    Input format:
        CASE + EVIDENCE + CLAIM

    Expected parquet columns:
        verifier_row_id
        qa_row_id
        split
        case_context
        claim_text
        evidence_sentences_json
        label_id
    """

    def __init__(
        self,
        cfg_path: str,
        split_name: str,
        supervision_relpath: str = "verifier_binary_dataset/verifier_binary_trainvaltest.parquet",
        max_evidence_sentences: int = 2,
        max_evidence_chars: int = 12000,
    ) -> None:
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.base_path = self.params["Preprocessing"]["base_path_evidenceData"]

        self.split_name = str(split_name)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.max_evidence_chars = int(max_evidence_chars)

        self.supervision_path = os.path.join(self.base_path, supervision_relpath)
        if not os.path.exists(self.supervision_path):
            raise FileNotFoundError(f"Verifier supervision parquet not found: {self.supervision_path}")

        self.pf = pq.ParquetFile(self.supervision_path)

        required = [
            "verifier_row_id",
            "qa_row_id",
            "split",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        schema_names = set(self.pf.schema.names)
        for c in required:
            if c not in schema_names:
                raise ValueError(f"Missing required column in verifier parquet: {c}")

        self._rg_row_index: List[tuple[int, int]] = []
        self._build_split_index()

        self._rg_cache: Dict[int, Dict[str, List]] = {}
        self._label_counts_cache: Optional[Dict[int, int]] = None

    def _build_split_index(self) -> None:
        for rg in range(self.pf.num_row_groups):
            t = self.pf.read_row_group(rg, columns=["split"])
            split_list = t.column(0).to_pylist()
            for i, s in enumerate(split_list):
                if str(s) == self.split_name:
                    self._rg_row_index.append((rg, i))

        if len(self._rg_row_index) == 0:
            print(f"[WARN] No rows found for split='{self.split_name}' in {self.supervision_path}")

    def __len__(self) -> int:
        return len(self._rg_row_index)

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(self, s: str) -> List:
        try:
            out = json.loads(s)
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _truncate_text(self, s: str) -> str:
        s = self._safe_str(s)
        if self.max_evidence_chars > 0 and len(s) > self.max_evidence_chars:
            return s[: self.max_evidence_chars]
        return s

    def _materialize_evidence(self, evidence_sentences_json: str) -> str:
        sents = self._loads_json_list(self._safe_str(evidence_sentences_json))
        sents = [self._safe_str(x).strip() for x in sents if self._safe_str(x).strip()]
        sents = sents[: self.max_evidence_sentences]
        return self._truncate_text("\n\n".join(sents))

    def _build_input_text(
            self,
            case_context: str,
            evidence_text: str,
            claim_text: str,
    ) -> str:
        case_text = self._safe_str(case_context).strip()
        evidence_text = self._safe_str(evidence_text).strip()
        claim_text = self._safe_str(claim_text).strip()

        return (
            "[TASK] support_verification\n\n"
            "[CASE_BEGIN]\n"
            f"{case_text}\n"
            "[CASE_END]\n\n"
            "[EVIDENCE_BEGIN]\n"
            f"{evidence_text}\n"
            "[EVIDENCE_END]\n\n"
            "[CLAIM_BEGIN]\n"
            f"{claim_text}\n"
            "[CLAIM_END]"
        )


    def _get_row_group_columns(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached

        cols = [
            "verifier_row_id",
            "qa_row_id",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        t = self.pf.read_row_group(rg, columns=cols)
        payload = {c: t.column(c).to_pylist() for c in cols}
        self._rg_cache[rg] = payload
        return payload

    def get_label_counts(self) -> Dict[int, int]:
        if self._label_counts_cache is not None:
            return dict(self._label_counts_cache)

        counts = {0: 0, 1: 0}
        for rg, row_in_group in self._rg_row_index:
            payload = self._get_row_group_columns(rg)
            try:
                y = int(payload["label_id"][row_in_group])
            except Exception:
                y = 0
            counts[y] = counts.get(y, 0) + 1

        self._label_counts_cache = counts
        return dict(counts)

    def __getitem__(self, idx: int) -> Dict:
        rg, row_in_group = self._rg_row_index[idx]
        payload = self._get_row_group_columns(rg)

        verifier_row_id = int(payload["verifier_row_id"][row_in_group])
        qa_row_id = int(payload["qa_row_id"][row_in_group])

        case_context = payload["case_context"][row_in_group]
        claim_text = payload["claim_text"][row_in_group]
        evidence_sentences_json = payload["evidence_sentences_json"][row_in_group]
        label_id = int(payload["label_id"][row_in_group])

        evidence_text = self._materialize_evidence(evidence_sentences_json)
        input_text = self._build_input_text(
            case_context=case_context,
            evidence_text=evidence_text,
            claim_text=claim_text,
        )

        return {
            "verifier_row_id": verifier_row_id,
            "qa_row_id": qa_row_id,
            "input_text": input_text,
            "target": label_id,
        }


def verifier_binary_collate_fn(batch: List[Dict]) -> VerifierBatch:
    inputs = [str(x["input_text"]).strip() for x in batch]
    targets = torch.tensor([int(x["target"]) for x in batch], dtype=torch.long)
    qa_row_ids = [int(x["qa_row_id"]) for x in batch]
    verifier_row_ids = [int(x["verifier_row_id"]) for x in batch]

    return VerifierBatch(
        inputs=inputs,
        targets=targets,
        qa_row_ids=qa_row_ids,
        verifier_row_ids=verifier_row_ids,
    )






class VerifierBinaryTestDataset(Dataset):
    """
    Test dataset for verifier evaluation on a specific scenario parquet.
    """

    def __init__(
            self,
            cfg_path: str,
            test_parquet_path: str,
            split_name: str = "test",
            max_evidence_sentences: int = 2,
            max_evidence_chars: int = 12000,
    ) -> None:
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)

        self.test_parquet_path = str(test_parquet_path)
        self.split_name = str(split_name)
        self.max_evidence_sentences = int(max_evidence_sentences)
        self.max_evidence_chars = int(max_evidence_chars)

        if not os.path.exists(self.test_parquet_path):
            raise FileNotFoundError(f"Test parquet not found: {self.test_parquet_path}")

        self.pf = pq.ParquetFile(self.test_parquet_path)

        required = [
            "verifier_row_id",
            "qa_row_id",
            "split",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        schema_names = set(self.pf.schema.names)
        for c in required:
            if c not in schema_names:
                raise ValueError(f"Missing required column in verifier parquet: {c}")

        self._rg_row_index = []
        self._build_index()
        self._rg_cache: Dict[int, Dict[str, List]] = {}


    def _build_index(self) -> None:
        for rg in range(self.pf.num_row_groups):
            t = self.pf.read_row_group(rg, columns=["split"])
            split_list = t.column(0).to_pylist()

            for i, s in enumerate(split_list):
                if str(s) == self.split_name:
                    self._rg_row_index.append((rg, i))

        if len(self._rg_row_index) == 0:
            print(f"[WARN] No rows found for split='{self.split_name}' in {self.test_parquet_path}")


    def __len__(self) -> int:
        return len(self._rg_row_index)

    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        s = str(x)
        return "" if s.lower() == "nan" else s

    def _loads_json_list(self, s: str) -> List:
        try:
            out = json.loads(s)
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _truncate_text(self, s: str) -> str:
        s = self._safe_str(s)
        if self.max_evidence_chars > 0 and len(s) > self.max_evidence_chars:
            return s[: self.max_evidence_chars]
        return s

    def _materialize_evidence(self, evidence_sentences_json: str) -> str:
        sents = self._loads_json_list(self._safe_str(evidence_sentences_json))
        sents = [self._safe_str(x).strip() for x in sents if self._safe_str(x).strip()]
        sents = sents[: self.max_evidence_sentences]
        return self._truncate_text("\n\n".join(sents))

    def _build_input_text(
        self,
        case_context: str,
        evidence_text: str,
        claim_text: str,
    ) -> str:
        case_text = self._safe_str(case_context).strip()
        evidence_text = self._safe_str(evidence_text).strip()
        claim_text = self._safe_str(claim_text).strip()

        return (
            "[TASK] support_verification\n\n"
            "[CASE_BEGIN]\n"
            f"{case_text}\n"
            "[CASE_END]\n\n"
            "[EVIDENCE_BEGIN]\n"
            f"{evidence_text}\n"
            "[EVIDENCE_END]\n\n"
            "[CLAIM_BEGIN]\n"
            f"{claim_text}\n"
            "[CLAIM_END]"
        )

    def _get_row_group_columns(self, rg: int) -> Dict[str, List]:
        cached = self._rg_cache.get(rg)
        if cached is not None:
            return cached

        cols = [
            "verifier_row_id",
            "qa_row_id",
            "case_context",
            "claim_text",
            "evidence_sentences_json",
            "label_id",
        ]
        t = self.pf.read_row_group(rg, columns=cols)
        payload = {c: t.column(c).to_pylist() for c in cols}
        self._rg_cache[rg] = payload
        return payload

    def __getitem__(self, idx: int) -> Dict:
        rg, row_in_group = self._rg_row_index[idx]
        payload = self._get_row_group_columns(rg)

        verifier_row_id = int(payload["verifier_row_id"][row_in_group])
        qa_row_id = int(payload["qa_row_id"][row_in_group])

        case_context = payload["case_context"][row_in_group]
        claim_text = payload["claim_text"][row_in_group]
        evidence_sentences_json = payload["evidence_sentences_json"][row_in_group]
        label_id = int(payload["label_id"][row_in_group])

        evidence_text = self._materialize_evidence(evidence_sentences_json)
        input_text = self._build_input_text(
            case_context=case_context,
            evidence_text=evidence_text,
            claim_text=claim_text,
        )

        return {
            "verifier_row_id": verifier_row_id,
            "qa_row_id": qa_row_id,
            "input_text": input_text,
            "target": label_id,
        }

