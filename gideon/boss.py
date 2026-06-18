"""Boss — the orchestrator.

The boss runs the seven specialist agents in a fixed sequence for a single CSV.
It owns the *control flow* only; all data flows through the ``artifacts/``
folder. Every run produces ``artifacts/manifest.json`` describing per-stage
status, timing, and any error, so the dashboard can show pipeline health.

Run a single file manually::

    python -m gideon.boss inbox/my_data.csv
"""
from __future__ import annotations

import shutil
import sys
import time
import traceback
from pathlib import Path

from gideon import config
from gideon.agents import (
    cleaner,
    dashboard_gen,
    deployer,
    evaluator,
    feature_eng,
    ingestor,
    trainer,
)

log = config.get_logger("boss")


def _build_stages(csv_path: Path):
    """Return the ordered (name, thunk) pipeline. Each thunk takes no args and
    relies on the standard artifact paths — one input, one output per agent."""
    return [
        ("ingestor", lambda: ingestor.ingest(csv_path)),
        ("cleaner", cleaner.clean),
        ("feature_eng", feature_eng.engineer),
        ("trainer", trainer.train),
        ("evaluator", evaluator.evaluate),
        ("deployer", deployer.deploy),
        ("dashboard_gen", dashboard_gen.generate),
    ]


def _write_manifest(manifest: dict) -> None:
    config.write_json(config.MANIFEST_FILE, manifest)


def run_pipeline(csv_path: str | Path) -> dict:
    """Run the full pipeline for one CSV. Returns the run manifest."""
    config.ensure_dirs()
    csv_path = Path(csv_path)

    manifest = {
        "run_id": config.now_iso(),
        "source_csv": str(csv_path),
        "source_name": csv_path.name,
        "status": "running",
        "started_at": config.now_iso(),
        "finished_at": None,
        "duration_seconds": None,
        "stages": [],
    }
    _write_manifest(manifest)
    log.info("=== Pipeline start: %s ===", csv_path.name)

    run_start = time.perf_counter()
    failed = False

    for name, func in _build_stages(csv_path):
        stage = {"name": name, "status": "running", "duration_seconds": None, "error": None}
        manifest["stages"].append(stage)
        _write_manifest(manifest)

        stage_start = time.perf_counter()
        try:
            output = func()
            stage["status"] = "success"
            stage["output"] = str(output) if output is not None else None
            log.info("[%s] OK", name)
        except Exception as exc:  # noqa: BLE001 — record and stop the pipeline
            stage["status"] = "failed"
            stage["error"] = f"{type(exc).__name__}: {exc}"
            stage["traceback"] = traceback.format_exc()
            log.error("[%s] FAILED: %s", name, exc)
            failed = True
        finally:
            stage["duration_seconds"] = round(time.perf_counter() - stage_start, 3)
            _write_manifest(manifest)

        if failed:
            break

    manifest["status"] = "failed" if failed else "success"
    manifest["finished_at"] = config.now_iso()
    manifest["duration_seconds"] = round(time.perf_counter() - run_start, 3)
    _write_manifest(manifest)

    if not failed:
        _archive_csv(csv_path)
        log.info("=== Pipeline success: %s (%.2fs) ===", csv_path.name, manifest["duration_seconds"])
    else:
        log.error("=== Pipeline failed: %s ===", csv_path.name)

    return manifest


def _archive_csv(csv_path: Path) -> None:
    """Move a successfully processed CSV into ``inbox/_processed/`` so it is not
    re-triggered, keeping a timestamped copy if a same-named file exists."""
    try:
        if not csv_path.exists():
            return
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = config.PROCESSED_DIR / csv_path.name
        if dest.exists():
            stem, suffix = csv_path.stem, csv_path.suffix
            dest = config.PROCESSED_DIR / f"{stem}_{int(time.time())}{suffix}"
        shutil.move(str(csv_path), str(dest))
    except OSError as exc:
        log.warning("Could not archive %s: %s", csv_path.name, exc)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m gideon.boss <path/to.csv>", file=sys.stderr)
        return 2
    manifest = run_pipeline(argv[0])
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
