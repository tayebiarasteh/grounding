"""
experiments/overlay.py
Created on October 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from typing import Tuple

from experiments._common import MissingInput, atomic_write_json


def package_overlay(path: str, pins: Tuple[str, ...], tag: str) -> str:
    record = os.path.join(path, "worker_pins.json")

    def current() -> bool:
        if not os.path.exists(record):
            return False
        with open(record) as f:
            return json.load(f).get("pins") == list(pins)
    if current():
        return path
    if os.path.exists(path):
        aside = f"{path}.refused_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            os.replace(path, aside)
        except FileNotFoundError:
            pass
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-deps",
                         "--target", tmp, *pins]).returncode
    if rc != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise MissingInput(f"pip could not build the worker's package overlay (exit code {rc}); run it again "
                           f"on a node that reaches PyPI")
    atomic_write_json({"pins": list(pins)}, os.path.join(tmp, "worker_pins.json"))
    try:
        os.rename(tmp, path)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not current():
            raise
    return path
