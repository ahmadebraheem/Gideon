"""Monitor — observability agent for Gideon.

Runs after every pipeline execution (called by the boss). Logs everything to
``logs/`` and writes ``logs/monitor.json`` as a structured summary the dashboard
can optionally read.

Tracks:
  - Per-stage timing, status, and errors from the run manifest
  - Artifact health — presence, size, and age of every expected artifact
  - Pipeline run history — last 50 runs with success/fail/duration
  - Data drift — column means/stds vs previous run (warns on large shifts)
  - System snapshot — CPU and memory at the moment monitor runs
  - Model metrics — RMSE/R2 or accuracy/F1 trend across runs

Logs written:
  logs/gideon.log         — rotating, human-readable, every event
  logs/runs.jsonl         — one JSON line per pipeline run (append-only history)
  logs/monitor.json       — latest structured summary (atomic write)

Integration with boss.py:
  Call ``monitor.observe(manifest)`` at the end of ``run_pipeline()``, after
  ``_write_manifest(manifest)``. That's the only change needed in boss.py.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import config

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
LOGS_DIR = config.BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "gideon.log"
RUNS_JSONL = LOGS_DIR / "runs.jsonl"
MONITOR_JSON = LOGS_DIR / "monitor.json"

MAX_HISTORY = 50  # runs kept in runs.jsonl summary

# Artifact files the monitor checks for health
EXPECTED_ARTIFACTS: dict[str, Path] = {
    "raw.parquet":          config.RAW_PARQUET,
    "ingest_meta.json":     config.INGEST_META,
    "cleaned.parquet":      config.CLEANED_PARQUET,
    "clean_report.json":    config.CLEAN_REPORT,
    "features.parquet":     config.FEATURES_PARQUET,
    "feature_meta.json":    config.FEATURE_META,
    "model.joblib":         config.MODEL_FILE,
    "train_meta.json":      config.TRAIN_META,
    "metrics.json":         config.METRICS_FILE,
    "predictions.parquet":  config.PREDICTIONS_PARQUET,
    "deployment.json":      config.DEPLOYMENT_FILE,
    "dashboard_data.json":  config.DASHBOARD_DATA,
    "manifest.json":        config.MANIFEST_FILE,
}

# Data drift: warn if a column mean shifts by more than this fraction of its std
DRIFT_WARN_THRESHOLD = 2.0


# --------------------------------------------------------------------------- #
# Logging setup — rotating file + console
# --------------------------------------------------------------------------- #
def _get_file_logger() -> logging.Logger:
    """Return a logger that writes to logs/gideon.log (rotating, 5 MB × 3)."""
    logger = logging.getLogger("gideon.monitor.file")
    if logger.handlers:
        return logger

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-26s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


log = config.get_logger("monitor")       # console logger (existing pattern)
flog = _get_file_logger()                # file logger (everything)


def _log(level: str, msg: str, *args) -> None:
    """Write to both console logger and file logger."""
    getattr(log, level)(msg, *args)
    getattr(flog, level)(msg, *args)


# --------------------------------------------------------------------------- #
# Section 1 — Pipeline run summary
# --------------------------------------------------------------------------- #
def _log_run_summary(manifest: dict) -> dict:
    run_id   = manifest.get("run_id", "unknown")
    status   = manifest.get("status", "unknown")
    source   = manifest.get("source_name", "unknown")
    duration = manifest.get("duration_seconds", 0)
    stages   = manifest.get("stages", [])

    _log("info", "=" * 60)
    _log("info", "RUN SUMMARY  run_id=%s", run_id)
    _log("info", "  source   : %s", source)
    _log("info", "  status   : %s", status.upper())
    _log("info", "  duration : %.3fs", duration)
    _log("info", "  stages   : %d", len(stages))

    stage_summaries = []
    for s in stages:
        name     = s.get("name", "?")
        sstatus  = s.get("status", "?")
        sdur     = s.get("duration_seconds", 0)
        err      = s.get("error")
        icon     = "OK" if sstatus == "success" else "FAIL"
        _log("info", "    [%s] %-16s  %.3fs", icon, name, sdur)
        if err:
            _log("error", "         error: %s", err)
            tb = s.get("traceback")
            if tb:
                for line in tb.splitlines():
                    flog.debug("         %s", line)
        stage_summaries.append({
            "name": name,
            "status": sstatus,
            "duration_seconds": sdur,
            "error": err,
        })

    _log("info", "=" * 60)

    return {
        "run_id": run_id,
        "source": source,
        "status": status,
        "duration_seconds": duration,
        "stages": stage_summaries,
    }


# --------------------------------------------------------------------------- #
# Section 2 — Artifact health
# --------------------------------------------------------------------------- #
def _log_artifact_health() -> list[dict]:
    _log("info", "ARTIFACT HEALTH")
    now = time.time()
    results = []

    for name, path in EXPECTED_ARTIFACTS.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            age_s   = now - path.stat().st_mtime
            _log("info", "  OK   %-28s  %.1f KB  age %.0fs", name, size_kb, age_s)
            results.append({"artifact": name, "present": True,
                             "size_kb": round(size_kb, 1), "age_seconds": round(age_s)})
        else:
            _log("warning", "  MISS %-28s  (not found)", name)
            results.append({"artifact": name, "present": False,
                             "size_kb": None, "age_seconds": None})

    return results


# --------------------------------------------------------------------------- #
# Section 3 — Model metrics
# --------------------------------------------------------------------------- #
def _log_model_metrics() -> dict:
    _log("info", "MODEL METRICS")
    if not config.METRICS_FILE.exists():
        _log("warning", "  metrics.json not found — skipping")
        return {}

    try:
        metrics = config.read_json(config.METRICS_FILE)
    except Exception as exc:
        _log("error", "  Failed to read metrics.json: %s", exc)
        return {}

    for k, v in metrics.items():
        if isinstance(v, float):
            _log("info", "  %-20s : %.6f", k, v)
        else:
            _log("info", "  %-20s : %s", k, v)

    return metrics


# --------------------------------------------------------------------------- #
# Section 4 — Data drift
# --------------------------------------------------------------------------- #
def _load_column_stats(parquet_path: Path) -> dict[str, dict]:
    """Return {column: {mean, std}} for numeric columns."""
    if not parquet_path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        stats = {}
        for col in df.select_dtypes(include="number").columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std":  float(df[col].std()),
                "min":  float(df[col].min()),
                "max":  float(df[col].max()),
                "nulls": int(df[col].isna().sum()),
                "rows":  len(df),
            }
        return stats
    except Exception as exc:
        _log("warning", "  Could not compute column stats from %s: %s", parquet_path.name, exc)
        return {}


def _log_data_drift(current_stats: dict, previous_stats: dict) -> list[dict]:
    _log("info", "DATA DRIFT CHECK")
    if not previous_stats:
        _log("info", "  No previous run stats — drift check skipped (first run)")
        return []

    drift_events = []
    for col, cur in current_stats.items():
        if col not in previous_stats:
            _log("info", "  NEW  column: %s", col)
            continue
        prev = previous_stats[col]
        prev_std = prev.get("std") or 0
        cur_mean  = cur.get("mean", 0)
        prev_mean = prev.get("mean", 0)

        if prev_std > 0:
            shift = abs(cur_mean - prev_mean) / prev_std
            if shift >= DRIFT_WARN_THRESHOLD:
                _log("warning",
                     "  DRIFT %-20s  mean %.3f -> %.3f  (%.1f stds)",
                     col, prev_mean, cur_mean, shift)
                drift_events.append({
                    "column": col,
                    "prev_mean": prev_mean,
                    "curr_mean": cur_mean,
                    "shift_stds": round(shift, 2),
                })
            else:
                _log("debug",
                     "  OK    %-20s  mean %.3f -> %.3f  (%.1f stds)",
                     col, prev_mean, cur_mean, shift)
        else:
            _log("debug", "  SKIP  %-20s  prev std=0", col)

    for col in previous_stats:
        if col not in current_stats:
            _log("warning", "  DROP column no longer present: %s", col)
            drift_events.append({"column": col, "dropped": True})

    if not drift_events:
        _log("info", "  No significant drift detected")

    return drift_events


# --------------------------------------------------------------------------- #
# Section 5 — System snapshot
# --------------------------------------------------------------------------- #
def _log_system_snapshot() -> dict:
    _log("info", "SYSTEM SNAPSHOT")
    snapshot: dict[str, Any] = {}

    try:
        import psutil
        cpu  = psutil.cpu_percent(interval=0.5)
        mem  = psutil.virtual_memory()
        disk = psutil.disk_usage(str(config.BASE_DIR))
        snapshot = {
            "cpu_percent":        cpu,
            "memory_used_gb":     round(mem.used / 1e9, 2),
            "memory_total_gb":    round(mem.total / 1e9, 2),
            "memory_percent":     mem.percent,
            "disk_used_gb":       round(disk.used / 1e9, 2),
            "disk_free_gb":       round(disk.free / 1e9, 2),
            "disk_percent":       disk.percent,
        }
        _log("info", "  CPU         : %.1f%%", cpu)
        _log("info", "  Memory      : %.2f / %.2f GB (%.1f%%)",
             snapshot["memory_used_gb"], snapshot["memory_total_gb"], mem.percent)
        _log("info", "  Disk free   : %.2f GB (%.1f%% used)",
             snapshot["disk_free_gb"], disk.percent)
    except ImportError:
        _log("info", "  psutil not installed — install it for system metrics (uv add psutil)")
        snapshot["note"] = "psutil not available"
    except Exception as exc:
        _log("warning", "  System snapshot failed: %s", exc)
        snapshot["error"] = str(exc)

    return snapshot


# --------------------------------------------------------------------------- #
# Section 6 — Run history (append to runs.jsonl)
# --------------------------------------------------------------------------- #
def _append_run_history(run_record: dict) -> list[dict]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing history
    history: list[dict] = []
    if RUNS_JSONL.exists():
        try:
            with open(RUNS_JSONL, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        except Exception as exc:
            flog.warning("Could not read runs.jsonl: %s", exc)

    # Append new record
    history.append(run_record)

    # Keep only last MAX_HISTORY
    history = history[-MAX_HISTORY:]

    # Rewrite file
    tmp = RUNS_JSONL.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in history:
            fh.write(json.dumps(record, default=str) + "\n")
    os.replace(tmp, RUNS_JSONL)

    # Log history summary
    total    = len(history)
    success  = sum(1 for r in history if r.get("status") == "success")
    failed   = total - success
    avg_dur  = sum(r.get("duration_seconds", 0) for r in history) / max(total, 1)

    _log("info", "RUN HISTORY (last %d runs)", total)
    _log("info", "  success : %d", success)
    _log("info", "  failed  : %d", failed)
    _log("info", "  avg dur : %.2fs", avg_dur)

    return history


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def observe(manifest: dict) -> None:
    """Called by boss.py after every pipeline run. Logs everything, writes
    logs/monitor.json and appends to logs/runs.jsonl."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    _log("info", "MONITOR START  %s", started_at)

    # 1. Run summary
    run_summary = _log_run_summary(manifest)

    # 2. Artifact health
    artifact_health = _log_artifact_health()

    # 3. Model metrics
    model_metrics = _log_model_metrics()

    # 4. Data drift — compare current features.parquet to previous run stats
    prev_stats_path = LOGS_DIR / "prev_column_stats.json"
    current_stats   = _load_column_stats(config.FEATURES_PARQUET)

    previous_stats: dict = {}
    if prev_stats_path.exists():
        try:
            previous_stats = config.read_json(prev_stats_path)
        except Exception:
            pass

    drift_events = _log_data_drift(current_stats, previous_stats)

    # Save current stats as previous for next run
    if current_stats:
        config.write_json(prev_stats_path, current_stats)

    # 5. System snapshot
    system_snapshot = _log_system_snapshot()

    # 6. Run history
    run_record = {
        "run_id":           run_summary["run_id"],
        "timestamp":        started_at,
        "source":           run_summary["source"],
        "status":           run_summary["status"],
        "duration_seconds": run_summary["duration_seconds"],
        "metrics":          model_metrics,
        "drift_events":     len(drift_events),
        "artifacts_ok":     sum(1 for a in artifact_health if a["present"]),
        "artifacts_miss":   sum(1 for a in artifact_health if not a["present"]),
    }
    history = _append_run_history(run_record)

    # 7. Write monitor.json summary
    summary = {
        "generated_at":   started_at,
        "run":            run_summary,
        "artifact_health": artifact_health,
        "model_metrics":  model_metrics,
        "drift_events":   drift_events,
        "system":         system_snapshot,
        "history": {
            "total_runs":    len(history),
            "success_count": sum(1 for r in history if r.get("status") == "success"),
            "failed_count":  sum(1 for r in history if r.get("status") == "failed"),
            "avg_duration":  round(
                sum(r.get("duration_seconds", 0) for r in history) / max(len(history), 1), 3
            ),
            "recent":        history[-10:],
        },
    }
    config.write_json(MONITOR_JSON, summary)
    _log("info", "Monitor summary written to %s", MONITOR_JSON)
    _log("info", "Full log at %s", LOG_FILE)
    _log("info", "MONITOR DONE")
