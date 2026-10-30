"""
experiments/chexbert.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List

import numpy as np

_LOADED: Dict = {}


def codes(cfg: Dict, texts: List[str]) -> np.ndarray:
    from transformers import AutoTokenizer
    from experiments._common import cuda_device
    from Inference.chexbert_labeler import fetch_checkpoint, label_texts, load_labeler
    ch = cfg["chexbert"]
    if not _LOADED:
        _LOADED["device"] = cuda_device("chexbert")
        _LOADED["model"] = load_labeler(fetch_checkpoint(ch, cfg["hf_cache_dir"]), ch["tokenizer"],
                                        _LOADED["device"])
        _LOADED["tok"] = AutoTokenizer.from_pretrained(ch["tokenizer"])
    return label_texts(_LOADED["model"], _LOADED["tok"], list(texts), _LOADED["device"], batch_size=int(ch["batch_size"]))
