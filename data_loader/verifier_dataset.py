"""
data_loader/verifier_dataset.py
Created on July 24, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import re
from typing import Dict, List, Optional

import pandas as pd


MASK_LEXICON: Dict[str, re.Pattern] = {
    "pleural effusion": re.compile(
        r"\b(pleural effusion|pleural effusions|effusion|effusions|pleural fluid|"
        r"pleural collection|costophrenic blunting|blunted costophrenic angle|"
        r"blunted costophrenic angles)\b", re.I),
    "pneumothorax": re.compile(
        r"\b(pneumothorax|pneumothoraces|pleural air|collapsed lung)\b", re.I),
    "cardiomegaly": re.compile(
        r"\b(cardiomegaly|cardiac enlargement|cardiac silhouette enlargement|"
        r"enlarged cardiac silhouette|cardiac silhouette is enlarged|"
        r"cardiac silhouette enlarged|enlarged heart|heart is enlarged|"
        r"heart size is enlarged|heart size enlarged|mild cardiomegaly|cardiomegalic)\b", re.I),
    "atelectasis": re.compile(
        r"\b(atelectasis|atelectatic|subsegmental atelectatic|bibasilar atelectatic|"
        r"basal atelectatic|subsegmental collapse|lobar collapse|collapse)\b", re.I),
    "consolidation": re.compile(
        r"\b(consolidation|lobar consolidation|focal consolidation|airspace consolidation|"
        r"air space consolidation|airspace opacity|airspace opacities|air space opacity|"
        r"air space opacities|alveolar opacity|alveolar opacities|parenchymal opacity|"
        r"parenchymal opacities|airspace disease)\b", re.I),
    "pulmonary edema": re.compile(
        r"\b(pulmonary edema|pulmonary oedema|edema|oedema|alveolar edema|alveolar oedema)\b", re.I),
    "vascular congestion": re.compile(
        r"\b(vascular congestion|pulmonary vascular congestion|central pulmonary vascular congestion|"
        r"vascular indistinctness|hilar vasculature|vascular prominence|prominent pulmonary vasculature|"
        r"cephalisation|cephalization)\b", re.I),
    "interstitial edema": re.compile(
        r"\b(interstitial edema|interstitial pulmonary edema|interstitial oedema|"
        r"interstitial pulmonary oedema|septal edema|septal oedema|interlobular septal thickening)\b", re.I),
    "rib fracture": re.compile(
        r"\b(rib fracture|rib fractures|fractured rib|fractured ribs)\b", re.I),
    "fracture": re.compile(r"\b(fracture|fractures|fractured)\b", re.I),
}

VERIFIER_COLUMNS = [
    "verifier_row_id", "qa_row_id", "split", "case_context", "claim_text",
    "evidence_sentences_json", "label_id", "concept", "gold_state",
    "claim_state", "row_type",
]


def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s


def loads_json_list(s: str) -> List:
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def mask_concepts(text: str, mask_token: str) -> str:
    out = safe_str(text)
    for pat in MASK_LEXICON.values():
        out = pat.sub(mask_token, out)
    return out


def materialize_evidence(
    evidence_sentences_json: str,
    max_evidence_sentences: int,
    max_evidence_chars: int = 12000,
    mask_token: Optional[str] = None,
) -> str:
    sents = loads_json_list(safe_str(evidence_sentences_json))
    sents = [safe_str(x).strip() for x in sents if safe_str(x).strip()]
    sents = sents[: int(max_evidence_sentences)]
    if mask_token is not None:
        sents = [mask_concepts(s, mask_token) for s in sents]
    text = "\n\n".join(sents)
    if max_evidence_chars > 0 and len(text) > max_evidence_chars:
        text = text[:max_evidence_chars]
    return text


def build_input_text(case_context: str, evidence_text: str, claim_text: str) -> str:
    case_text = safe_str(case_context).strip()
    evidence_text = safe_str(evidence_text).strip()
    claim_text = safe_str(claim_text).strip()
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


def fit_inputs(tokenizer, cases: List[str], evidence: List[str], claims: List[str], max_tokens: int):
    texts = [build_input_text(c, e, s) for c, e, s in zip(cases, evidence, claims)]
    lengths = [len(x) for x in tokenizer(texts, truncation=False)["input_ids"]]
    shortened = 0
    for i, n in enumerate(lengths):
        if n <= max_tokens:
            continue
        words = safe_str(cases[i]).split()
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            t = build_input_text(" ".join(words[:mid]), evidence[i], claims[i])
            if len(tokenizer(t, truncation=False)["input_ids"]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        t = build_input_text(" ".join(words[:lo]), evidence[i], claims[i])
        if len(tokenizer(t, truncation=False)["input_ids"]) > max_tokens:
            raise ValueError(f"the evidence and statement alone are longer than {max_tokens} tokens")
        texts[i] = t
        shortened += 1
    return texts, shortened


def load_verifier_records(
    parquet_path: str,
    split: Optional[str] = "test",
    max_evidence_sentences: int = 2,
    max_evidence_chars: int = 12000,
    mask_variant: Optional[str] = None,
    mask_token: str = "[FINDING]",
    fit_tokenizer=None,
    max_tokens: Optional[int] = None,
) -> List[Dict]:
    df = pd.read_parquet(parquet_path)
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == str(split)].copy()
    df = df.reset_index(drop=True)

    ev_mask = mask_token if mask_variant in ("mask_evidence", "mask_evidence_and_claim") else None
    records: List[Dict] = []
    for _, r in df.iterrows():
        claim_text = safe_str(r.get("claim_text"))
        if mask_variant == "mask_evidence_and_claim":
            claim_text = mask_concepts(claim_text, mask_token)
        evidence_text = materialize_evidence(
            r.get("evidence_sentences_json"),
            max_evidence_sentences=max_evidence_sentences,
            max_evidence_chars=max_evidence_chars,
            mask_token=ev_mask,
        )
        input_text = build_input_text(
            case_context=r.get("case_context"),
            evidence_text=evidence_text,
            claim_text=claim_text,
        )
        records.append({
            "verifier_row_id": int(r["verifier_row_id"]) if "verifier_row_id" in r and pd.notna(r["verifier_row_id"]) else -1,
            "qa_row_id": int(r["qa_row_id"]) if "qa_row_id" in r and pd.notna(r["qa_row_id"]) else -1,
            "concept": safe_str(r.get("concept")),
            "row_type": safe_str(r.get("row_type")),
            "gold_state": safe_str(r.get("gold_state")),
            "claim_state": safe_str(r.get("claim_state")),
            "label_id": int(r["label_id"]) if pd.notna(r.get("label_id")) else 0,
            "input_text": input_text,
            "_parts": (safe_str(r.get("case_context")), evidence_text, claim_text),
        })
    if fit_tokenizer is not None:
        texts, n = fit_inputs(fit_tokenizer, [x["_parts"][0] for x in records], [x["_parts"][1] for x in records],
                              [x["_parts"][2] for x in records], int(max_tokens))
        for rec, t in zip(records, texts):
            rec["input_text"] = t
    for rec in records:
        del rec["_parts"]
    return records
