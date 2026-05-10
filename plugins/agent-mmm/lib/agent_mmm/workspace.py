"""Workspace path helpers and run-ID generation for ./mmm-workspace/."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path


WORKSPACE_DIR = "mmm-workspace"


def workspace_root(base: str | Path = ".") -> Path:
    return Path(base) / WORKSPACE_DIR


def ensure_workspace(base: str | Path = ".") -> Path:
    root = workspace_root(base)
    for sub in ["audit", "controls", "priors", "runs", "reports", "data"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def new_run_id() -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
    return f"{ts}_v01"


def run_dir(run_id: str, base: str | Path = ".") -> Path:
    return workspace_root(base) / "runs" / run_id


def ensure_run_dir(run_id: str, base: str | Path = ".") -> Path:
    d = run_dir(run_id, base)
    d.mkdir(parents=True, exist_ok=True)
    return d
