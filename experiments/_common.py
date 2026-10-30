"""
experiments/_common.py
Created on July 24, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config


def get_cfg(cfg_path: str) -> dict:
    return read_config(cfg_path)["Grounding"]


def stats_params(cfg: dict):
    s = cfg["stats"]
    return int(s["n_boot"]), int(s["boot_seed"])


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o)


def atomic_write(path: str, write: Callable[[str], None], suffix: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{suffix}"
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _fsync(path: str) -> None:
    with open(path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    def _write(tmp):
        df.to_csv(tmp, index=False)
        _fsync(tmp)
    atomic_write(path, _write)


def atomic_write_json(obj, path: str) -> None:
    def _write(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=True)
            f.flush()
            os.fsync(f.fileno())
    atomic_write(path, _write)


def read_results_csv(path: str, **kw) -> pd.DataFrame:
    kw.setdefault("low_memory", False)
    kw.setdefault("float_precision", "round_trip")
    return pd.read_csv(path, **kw)


def read_source_csv(path: str, **kw) -> pd.DataFrame:
    kw.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, encoding="utf-8", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kw)


def flush_status(path: str, message: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(message.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())


class MissingInput(Exception):
    pass


class GateFailure(Exception):
    pass


def cuda_device(stage: str):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"[{stage}] no usable GPU: {node_and_gpu_line()}")
    return torch.device("cuda")


def oom_batches(order, batch_size: int, run: Callable[[np.ndarray], None], tag: str) -> None:
    import gc
    import torch
    order = np.asarray(order)
    size, i = max(int(batch_size), 1), 0
    bar = tqdm(total=len(order), desc=tag, unit="item")
    while i < len(order):
        idx = order[i:i + size]
        failed = False
        try:
            run(idx)
        except torch.OutOfMemoryError:
            failed = True
        if failed:
            gc.collect()
            torch.cuda.empty_cache()
            if size == 1:
                free, total = torch.cuda.mem_get_info()
                raise RuntimeError(f"[{tag}] one item does not fit on the card ({free / 1024 ** 3:.1f} of "
                                   f"{total / 1024 ** 3:.1f} GiB free); no batch size can fix this")
            size = max(1, size // 2)
            continue
        i += len(idx)
        bar.update(len(idx))
    bar.close()


def fingerprint(path: str) -> Dict:
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def digest_file(path: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def require(path: str, producer: str) -> str:
    if not os.path.exists(path):
        raise MissingInput(f"{path} is missing; run {producer} first")
    return path


def node_and_gpu_line() -> str:
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME") or socket.gethostname()
    driver = "no NVIDIA driver on this node"
    try:
        with open("/proc/driver/nvidia/version") as f:
            head = f.readline().strip()
        m = re.search(r"\b\d+\.\d+(?:\.\d+)?\b", head)
        driver = f"NVIDIA kernel module {m.group(0)}" if m else head
    except OSError:
        pass
    return (f"node {node}, SLURM job {os.environ.get('SLURM_JOB_ID', 'none')}, {driver}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")


_CLAIM_STALE_AFTER_S = 1800.0
_TAKEOVER_LOCK_S = 600.0
_HEARTBEAT_EVERY_S = 30.0
_LIVE_CLAIMS: set = set()
_CLAIM_TOKENS: Dict[str, str] = {}
_LAST_HEARTBEAT: Dict[str, float] = {}


def _safe(s) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _claim_path(claims_dir: str, tag: str) -> str:
    return os.path.join(claims_dir, f"claim__{_safe(tag)}.json")


def _slurm_job_state(job_id: str) -> str:
    if not job_id:
        return "unknown"
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id)], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "gone" if "invalid job id" in (out.stderr or "").lower() else "unknown"
    return "gone" if out.stdout.strip() == "" else "alive"


def _owner_state(claim: Dict) -> str:
    state = _slurm_job_state(str(claim.get("slurm_job_id", "")))
    if state != "unknown" or claim.get("host") != socket.gethostname():
        return state
    try:
        pid = int(claim.get("pid", 0))
    except (TypeError, ValueError):
        return "unknown"
    if pid <= 0:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "gone"
    except OSError:
        return "unknown"
    return "alive"


def _read_claim(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _remove_own_claim(path: str) -> None:
    token = _CLAIM_TOKENS.pop(path, None)
    _LIVE_CLAIMS.discard(path)
    _LAST_HEARTBEAT.pop(path, None)
    if not os.path.exists(path):
        return
    if token is not None and _read_claim(path).get("token") not in (None, token):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _create_claim(path: str, token: str) -> bool:
    tmp = f"{path}.{token}.tmp"
    with open(tmp, "w") as f:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                   "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "token": token}, f, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f, open(tmp) as src:
            f.write(src.read())
        return True
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _take_over(path: str, expired: Dict, token: str) -> bool:
    lock = f"{path}.takeover"
    if not _create_claim(lock, token):
        try:
            if time.time() - os.path.getmtime(lock) > _TAKEOVER_LOCK_S:
                os.remove(lock)
        except OSError:
            pass
        return False
    try:
        if _read_claim(path) != expired:
            return False
        try:
            os.remove(path)
        except OSError:
            return False
        return _create_claim(path, token)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def claim_unit(claims_dir: str, tag: str, stale_after_s: float = _CLAIM_STALE_AFTER_S) -> bool:
    os.makedirs(claims_dir, exist_ok=True)
    path = _claim_path(claims_dir, tag)
    token = uuid.uuid4().hex
    if not _create_claim(path, token):
        claim = _read_claim(path)
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return False
        owner = _owner_state(claim)
        if owner == "alive" or (owner == "unknown" and age < stale_after_s):
            return False
        if not claim or not _take_over(path, claim, token):
            return False
    _LIVE_CLAIMS.add(path)
    _CLAIM_TOKENS[path] = token
    return True


def heartbeat_claim(claims_dir: str, tag: str) -> None:
    path = _claim_path(claims_dir, tag)
    now = time.time()
    if now - _LAST_HEARTBEAT.get(path, 0.0) < _HEARTBEAT_EVERY_S:
        return
    _LAST_HEARTBEAT[path] = now
    if _CLAIM_TOKENS.get(path) is not None and os.path.exists(path):
        os.utime(path, None)


@contextmanager
def claim_beat(claims_dir: str, tag: str, every_s: float = 60.0):
    path = _claim_path(claims_dir, tag)
    stop = threading.Event()

    def beat():
        while not stop.wait(every_s):
            token = _CLAIM_TOKENS.get(path)
            if token is not None and _read_claim(path).get("token") == token:
                try:
                    os.utime(path, None)
                except OSError:
                    pass
    thread = threading.Thread(target=beat, name=f"claim beat {tag}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def release_claim(claims_dir: str, tag: str) -> None:
    _remove_own_claim(_claim_path(claims_dir, tag))


def as_stored(obj):
    return json.loads(json.dumps(obj, default=_json_default, ensure_ascii=True))


def _params_differ(stored: Dict, current: Dict, legacy: Optional[Dict] = None, prefix: str = "") -> List[str]:
    legacy = legacy or {}
    if not prefix:
        current = as_stored(current)
    moved = []
    for k, v in current.items():
        if k not in stored:
            if k in legacy and legacy[k] != v:
                moved.append(f"{prefix}{k}: {legacy[k]!r} (before the key existed) -> {v!r}")
            continue
        if isinstance(v, dict) and isinstance(stored[k], dict):
            moved += _params_differ(stored[k], v, None, f"{prefix}{k}.")
        elif stored[k] != v:
            moved.append(f"{prefix}{k}: {stored[k]!r} -> {v!r}")
    return moved


def run_units(stage: str, units: List[Dict], compute_fn: Callable[[Dict], List[Dict]],
              out_dir: str, status_path: str, validate: Optional[Callable[[Dict, pd.DataFrame], bool]] = None,
              stale_s: float = _CLAIM_STALE_AFTER_S) -> Tuple[List[Dict], int]:
    part_dir = os.path.join(out_dir, "partial")
    claims_dir = os.path.join(part_dir, "claims")
    os.makedirs(part_dir, exist_ok=True)
    rows_by_key: Dict[str, List[Dict]] = {}

    def cached(unit, say: bool) -> Optional[List[Dict]]:
        key = unit["key"]
        csv_path, rec_path = os.path.join(part_dir, f"{key}.csv"), os.path.join(part_dir, f"{key}.params.json")
        if not (os.path.exists(rec_path) and os.path.exists(csv_path)):
            return None
        with open(rec_path) as f:
            moved = _params_differ(json.load(f), unit["params"], unit.get("legacy"))
        if moved:
            return None
        done = read_results_csv(csv_path)
        if not len(done) or (validate is not None and not validate(unit, done)):
            return None
        return done.to_dict("records")

    def attempt(unit) -> str:
        key = unit["key"]
        rows = cached(unit, say=False)
        if rows is not None:
            rows_by_key[key] = rows
            return "cached"
        if not claim_unit(claims_dir, key, stale_s):
            return "held"
        try:
            rows = cached(unit, say=True)
            if rows is not None:
                rows_by_key[key] = rows
                return "cached"
            try:
                with claim_beat(claims_dir, key):
                    out = compute_fn(unit)
            except MissingInput:
                return "refused"
            if not out:
                raise RuntimeError(f"[{stage}] {key} produced no rows")
            csv_path = os.path.join(part_dir, f"{key}.csv")
            atomic_write_csv(pd.DataFrame(out), csv_path)
            done = read_results_csv(csv_path)
            if validate is not None and not validate(unit, done):
                raise RuntimeError(f"[{stage}] {key}: the rows just written do not validate")
            atomic_write_json(unit["params"], os.path.join(part_dir, f"{key}.params.json"))
            rows_by_key[key] = done.to_dict("records")
            flush_status(status_path, f"[{stage}] {key}: {len(out)} rows")
            return "computed"
        finally:
            release_claim(claims_dir, key)

    counts, held = {"cached": 0, "computed": 0, "refused": 0}, []
    for unit in tqdm(units, desc=stage, unit="unit"):
        outcome = attempt(unit)
        if outcome == "held":
            held.append(unit)
        else:
            counts[outcome] += 1
    pause, last_note = 1.0, 0.0
    while held:
        still = []
        for unit in held:
            outcome = attempt(unit)
            if outcome == "held":
                still.append(unit)
            else:
                counts[outcome] += 1
        held = still
        if held:
            if time.time() - last_note > 60.0:
                last_note = time.time()
            time.sleep(pause)
            pause = min(2.0 * pause, 10.0)
    return [r for u in units for r in rows_by_key.get(u["key"], [])], counts["refused"]


def write_table(df: pd.DataFrame, path: str) -> bool:
    new = df.to_csv(index=False)
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == new:
                return False

    def _write(tmp):
        with open(tmp, "w") as f:
            f.write(new)
            f.flush()
            os.fsync(f.fileno())
    atomic_write(path, _write)
    return True
