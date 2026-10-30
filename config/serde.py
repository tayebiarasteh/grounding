"""
config/serde.py
Created on July 24, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import re

import yaml


def _resolve_once(s: str, flat_vars: dict) -> str:
    return re.sub(
        r'\$\{([^}]+)\}',
        lambda m: flat_vars.get(m.group(1).strip(), m.group(0)),
        s,
    )


def _resolve_all(obj, flat_vars: dict):
    if isinstance(obj, str):
        prev = None
        while prev != obj:
            prev = obj
            obj = _resolve_once(obj, flat_vars)
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_all(v, flat_vars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_all(item, flat_vars) for item in obj]
    return obj


def read_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        cfg = yaml.safe_load(f)

    flat_vars: dict = {k: v for k, v in cfg.items() if isinstance(v, str)}

    for _ in range(10):
        resolved = {k: _resolve_once(v, flat_vars) for k, v in flat_vars.items()}
        if resolved == flat_vars:
            break
        flat_vars = resolved

    return _resolve_all(cfg, flat_vars)
