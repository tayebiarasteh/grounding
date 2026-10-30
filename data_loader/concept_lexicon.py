"""
data_loader/concept_lexicon.py
Created on July 24, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import re
from typing import Dict, List

REPORT_PATTERNS: Dict[str, re.Pattern] = {
    "pleural effusion": re.compile(r"\bpleural effusions?\b|\beffusions?\b", re.I),
    "pneumothorax": re.compile(r"\bpneumothorax\b|\bpneumothoraces\b", re.I),
    "cardiomegaly": re.compile(
        r"\bcardiomegaly\b|\benlarged cardiac silhouette\b|\bcardiac silhouette is enlarged\b|"
        r"\bcardiac silhouette enlarged\b|\benlarged heart\b|\bheart is enlarged\b|\bheart size is enlarged\b|"
        r"\bheart size enlarged\b|\bheart is mildly enlarged\b|\bmild cardiomegaly\b", re.I),
    "atelectasis": re.compile(
        r"\batelectasis\b|\batelectatic\b|\bsubsegmental atelectatic\b|\bbibasilar atelectatic\b|"
        r"\bbasal atelectatic\b", re.I),
    "consolidation": re.compile(
        r"\bconsolidation\b|\blobar consolidation\b|\bfocal consolidation\b|\bairspace consolidation\b", re.I),
    "pulmonary edema": re.compile(r"\bpulmonary edema\b|\bedema\b", re.I),
    "vascular congestion": re.compile(
        r"\bvascular congestion\b|\bpulmonary vascular congestion\b|\bcentral pulmonary vascular congestion\b|"
        r"\bvascular indistinctness\b|\bhilar vasculature\b", re.I),
    "interstitial edema": re.compile(r"\binterstitial edema\b|\binterstitial pulmonary edema\b", re.I),
    "rib fracture": re.compile(r"\brib fractures?\b|\bfractured ribs?\b", re.I),
    "fracture": re.compile(r"\bfracture\b|\bfractures\b", re.I),
}
REPORT_CORE = {"vascular congestion": "pulmonary edema", "interstitial edema": "pulmonary edema",
               "rib fracture": "fracture"}


def report_findings(sentence: str) -> List[str]:
    s = "" if sentence is None else str(sentence)
    return sorted({REPORT_CORE.get(c, c) for c, pat in REPORT_PATTERNS.items() if pat.search(s)})
