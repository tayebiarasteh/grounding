"""
Inference/llm_client.py
Created on July 24, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
import re
import threading
import time
from typing import Dict, Optional, Tuple

_NOTICE_LOCK = threading.Lock()
_NOTICES: Dict[Tuple[str, str], list] = {}


def notice(kind: str, key: str, message: str, every_s: float = 60.0) -> None:
    now = time.time()
    with _NOTICE_LOCK:
        last, held = _NOTICES.get((kind, key), [0.0, 0])
        if now - last < every_s:
            _NOTICES[(kind, key)] = [last, held + 1]
            return
        _NOTICES[(kind, key)] = [now, 0]


def resolve_secret(value: str, env_name: Optional[str]) -> str:
    placeholder = (value is None) or (isinstance(value, str) and set(value.strip()) <= {"x"}) \
        or (isinstance(value, str) and value.startswith("REPLACE_"))
    if placeholder and env_name and os.environ.get(env_name):
        return os.environ[env_name]
    return value


def _is_timeout(e: Exception) -> bool:
    return "timeout" in type(e).__name__.lower()


def _is_rate_limit(e: Exception) -> bool:
    return getattr(e, "status_code", None) == 429 or type(e).__name__ == "RateLimitError"


def call_with_retry(fn, max_retries: int, connect_retry_max_s: float, description: str, timeout_s: float):
    slept, attempt, long_fails, throttled = 0.0, 0, 0, 0
    while True:
        t0 = time.time()
        try:
            return fn()
        except Exception as e:
            spent = time.time() - t0
            attempt += 1
            if _is_rate_limit(e) and throttled < 8:
                throttled += 1
                pause = min(2.0 ** throttled, 30.0)
                notice("429", description, f"[llm] {description}: throttled (429); waiting up to {pause:.0f} s; lower "
                                           f"the concurrency if it repeats")
                time.sleep(pause)
                continue
            if _is_timeout(e) and spent >= 0.9 * timeout_s:
                notice("timeout", description, f"[llm] {description}: a call used its whole {timeout_s:.0f} s timeout; "
                                               f"not retrying it")
                raise
            long_fails += int(spent >= 0.5 * timeout_s)
            remaining = connect_retry_max_s - slept
            if long_fails > max_retries or remaining <= 0:
                raise
            pause = min(2.0 ** min(attempt, 5), 60.0, remaining)
            notice("retry", description, f"[llm] {description}: retry {attempt} after {type(e).__name__}; sleeping "
                                         f"{pause:.0f} s")
            time.sleep(pause)
            slept += pause


def _split_harmony(raw: str) -> str:
    if raw is None:
        return ""
    text = raw
    end_markers = ["<|return|>", "<|end|>", "<|endoftext|>"]
    if "final<|message|>" in text:
        after = text.split("final<|message|>", 1)[1]
        for m in end_markers:
            after = after.split(m, 1)[0]
        return after.strip()
    if "assistantfinal" in text:
        after = text.split("assistantfinal", 1)[1]
        for m in end_markers:
            after = after.split(m, 1)[0]
        return after.strip()
    return text.strip()


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def served_models(serving_cfg: dict, timeout_s: float = 30.0) -> list:
    import httpx
    from openai import OpenAI
    client = OpenAI(base_url=resolve_secret(serving_cfg["api_base_url"], serving_cfg.get("api_base_url_env")),
                    api_key=resolve_secret(serving_cfg["api_key"], serving_cfg.get("api_key_env")), max_retries=0,
                    http_client=httpx.Client(verify=serving_cfg.get("verify_ssl", False), timeout=timeout_s))
    return [m.id for m in client.models.list().data]
