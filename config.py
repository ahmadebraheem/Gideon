"""Shared configuration, paths, and IO helpers for Gideon.

This module deliberately holds **only** constants and pure helper functions
(paths, logging, atomic IO). It carries no runtime state, which preserves the
project's core rule: agents communicate *exclusively* through files in the
``artifacts/`` folder, never through shared memory.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
INBOX_DIR = BASE_DIR / "inbox"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# --------------------------------------------------------------------------- #
# Artifact files — the inter-agent contract.
# Each agent reads the upstream artifact(s) and writes exactly one primary
# artifact (plus an optional sidecar metadata file).
# --------------------------------------------------------------------------- #
RAW_PARQUET = ARTIFACTS_DIR / "raw.parquet"            # ingestor  -> output
INGEST_META = ARTIFACTS_DIR / "ingest_meta.json"

CLEANED_PARQUET = ARTIFACTS_DIR / "cleaned.parquet"    # cleaner   -> output
CLEAN_REPORT = ARTIFACTS_DIR / "clean_report.json"

FEATURES_PARQUET = ARTIFACTS_DIR / "features.parquet"  # feature_eng -> output
FEATURE_META = ARTIFACTS_DIR / "feature_meta.json"

MODEL_FILE = ARTIFACTS_DIR / "model.joblib"            # trainer   -> output
TRAIN_META = ARTIFACTS_DIR / "train_meta.json"

METRICS_FILE = ARTIFACTS_DIR / "metrics.json"          # evaluator -> output
PREDICTIONS_PARQUET = ARTIFACTS_DIR / "predictions.parquet"

DEPLOYMENT_FILE = ARTIFACTS_DIR / "deployment.json"    # deployer  -> output

DASHBOARD_DATA = ARTIFACTS_DIR / "dashboard_data.json"  # dashboard_gen -> output

MANIFEST_FILE = ARTIFACTS_DIR / "manifest.json"        # boss run manifest

# --------------------------------------------------------------------------- #
# Behaviour constants
# --------------------------------------------------------------------------- #
# Input dataset formats accepted in the inbox.
SUPPORTED_EXTENSIONS = (".csv", ".json", ".parquet")

# Column names (case-insensitive) that strongly indicate the prediction target.
TARGET_NAME_CANDIDATES = ("target", "label", "y", "class", "outcome", "result")

# A numeric column is treated as a *classification* target when it has at most
# this many distinct integer-like values.
MAX_CLASSIFICATION_CLASSES = 20

# Categorical columns with at most this many distinct values are one-hot
# encoded; anything above is frequency-encoded.
MAX_ONEHOT_CARDINALITY = 20

# Drop columns whose fraction of missing values exceeds this threshold.
MAX_MISSING_FRACTION = 0.6

TEST_SIZE = 0.2
RANDOM_STATE = 42

PIPELINE_STAGES = (
    "ingestor",
    "cleaner",
    "feature_eng",
    "trainer",
    "evaluator",
    "deployer",
    "dashboard_gen",
)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Return a configured, non-duplicating logger."""
    logger = logging.getLogger(f"gideon.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
                "%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    """Create the runtime directories if they do not yet exist."""
    for directory in (INBOX_DIR, ARTIFACTS_DIR, DASHBOARD_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def inbox_datasets() -> list[Path]:
    """Dataset files currently present in the inbox (top level only)."""
    if not INBOX_DIR.exists():
        return []
    return sorted(
        p for p in INBOX_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith(".")
    )


def clear_artifacts() -> None:
    """Remove all generated artifacts so the dashboard resets to a clean state.
    Keeps the directory itself and its git placeholders."""
    if not ARTIFACTS_DIR.exists():
        return
    for path in ARTIFACTS_DIR.iterdir():
        if path.name in (".gitkeep", ".gitignore"):
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass


def clear_inbox() -> int:
    """Delete the dataset files in the inbox. Returns how many were removed."""
    removed = 0
    for path in inbox_datasets():
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


# --------------------------------------------------------------------------- #
# Atomic IO — write to a temp file then os.replace so readers (the dashboard)
# never observe a half-written artifact.
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def write_json(path: os.PathLike | str, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    os.replace(tmp, path)
    return path


def read_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_parquet(df, path: os.PathLike | str) -> Path:
    """Write a DataFrame to Parquet atomically (index is dropped on purpose:
    row position is the stable key shared between agents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path
