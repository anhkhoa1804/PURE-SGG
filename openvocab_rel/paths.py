"""Portable project paths with backward-compatible repository defaults.

Core historical evaluators intentionally retain their recorded CLI defaults.
New tooling can use these helpers, while shell launchers may map the same
variables to their existing ``DATA_ROOT``/``OUT_DIR``/``SAVE_PATH`` flags.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _root(env_name: str, legacy_env: str | None, default: str) -> Path:
    value = os.environ.get(env_name)
    if not value and legacy_env:
        value = os.environ.get(legacy_env)
    path = Path(value).expanduser() if value else REPO_ROOT / default
    return path.resolve(strict=False)


def data_root() -> Path:
    """VG150/data root; ``DATA_ROOT`` remains a supported legacy fallback."""

    return _root("RESEARCH_NO1_DATA_ROOT", "DATA_ROOT", "datasets_vg150_clean")


def checkpoint_root() -> Path:
    return _root("RESEARCH_NO1_CHECKPOINT_ROOT", None, "checkpoints")


def run_root() -> Path:
    return _root("RESEARCH_NO1_RUN_ROOT", None, "runs")


def cache_root() -> Path:
    return _root("RESEARCH_NO1_CACHE_ROOT", "HF_HOME", "caches")
