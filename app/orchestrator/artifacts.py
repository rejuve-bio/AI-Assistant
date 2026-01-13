from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import shutil


@dataclass
class Artifact:
    id: str
    type: str  # figure | table | log | manifest
    name: str
    format: str  # png | svg | pdf | html | json | csv | txt
    path_or_url: str
    bytes_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


def ensure_run_dirs(root: str, run_id: str) -> Dict[str, str]:
    base = os.path.join(root, run_id)
    figures = os.path.join(base, "figures")
    tables = os.path.join(base, "tables")
    logs = os.path.join(base, "logs")
    for p in (base, figures, tables, logs):
        os.makedirs(p, exist_ok=True)
    return {"base": base, "figures": figures, "tables": tables, "logs": logs}


def build_manifest(run_id: str, inputs: Dict[str, Any], params: Dict[str, Any], artifacts: List[Artifact], environment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "inputs": inputs,
        "params": params,
        "environment": environment,
        "artifacts": [a.__dict__ for a in artifacts],
    }


def cleanup_expired(root: str, ttl_minutes: int) -> Dict[str, Any]:
    now = time.time()
    ttl = ttl_minutes * 60
    removed: List[str] = []
    if not os.path.isdir(root):
        return {"removed": removed}
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        try:
            if os.path.isdir(path):
                mtime = os.path.getmtime(path)
                if now - mtime > ttl:
                    shutil.rmtree(path, ignore_errors=True)
                    removed.append(path)
        except Exception:
            continue
    return {"removed": removed}


