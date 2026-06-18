"""Watcher — triggers the boss when a dataset (CSV / JSON / Parquet) lands in
``inbox/`` and resets the dashboard when the inbox is emptied.

Design notes:
  * A single background worker thread consumes a queue, so pipeline runs are
    *serialised* — dropping a newer CSV mid-run simply queues another run that
    starts once the current one finishes (and the dashboard then updates).
  * Files are only processed once their size has stopped changing (so a CSV that
    is still being copied in is not read half-written).
  * **Content-hash de-duplication**: re-saving the same file (or duplicate
    filesystem events) does not re-run the pipeline; only new content does.
  * Processed CSVs **stay in the inbox**. When the last CSV is removed, the
    artifacts are cleared so the dashboard returns to a clean "waiting" state.

Run::

    python -m watcher
"""
from __future__ import annotations

import hashlib
import queue
import sys
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import config
import boss

log = config.get_logger("watcher")

_SETTLE_POLL_SECONDS = 0.5
_SETTLE_STABLE_CHECKS = 3  # consecutive equal-size checks => file finished copying

# path -> last processed content hash (in-memory; rebuilt on restart)
_processed_hashes: dict[str, str] = {}
_RESET = "__reset__"  # sentinel queued when the inbox becomes empty


def _is_supported(path: Path) -> bool:
    return (
        path.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and path.parent == config.INBOX_DIR
        and not path.name.startswith(".")
    )


def _file_hash(path: Path) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _wait_until_stable(path: Path) -> bool:
    """Return True once the file size is stable; False if it disappears."""
    last_size = -1
    stable = 0
    for _ in range(120):  # ~60s ceiling
        if not path.exists():
            return False
        size = path.stat().st_size
        if size == last_size:
            stable += 1
            if stable >= _SETTLE_STABLE_CHECKS:
                return True
        else:
            stable = 0
            last_size = size
        time.sleep(_SETTLE_POLL_SECONDS)
    return True


class _CsvHandler(FileSystemEventHandler):
    def __init__(self, work_queue: "queue.Queue") -> None:
        self._queue = work_queue

    def _maybe_enqueue(self, raw_path: str) -> None:
        path = Path(raw_path)
        if _is_supported(path) and path.exists():
            self._queue.put(path)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.dest_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            _processed_hashes.pop(str(path), None)
            # If no datasets remain, queue a reset so the dashboard goes clean.
            if not config.inbox_datasets():
                self._queue.put(_RESET)


def _worker(work_queue: "queue.Queue") -> None:
    while True:
        item = work_queue.get()
        try:
            if item is _RESET:
                if not config.inbox_datasets():
                    config.clear_artifacts()
                    log.info("Inbox empty — cleared artifacts; dashboard reset.")
                continue

            path = item
            if not _wait_until_stable(path):
                log.info("Skipped (vanished before stable): %s", path.name)
                continue

            digest = _file_hash(path)
            if digest is None:
                continue
            if _processed_hashes.get(str(path)) == digest:
                continue  # unchanged content — skip duplicate/no-op events
            _processed_hashes[str(path)] = digest

            log.info("Processing: %s", path.name)
            boss.run_pipeline(path)
        except Exception as exc:  # noqa: BLE001 — keep the watcher alive
            log.error("Run errored: %s", exc)
        finally:
            work_queue.task_done()


def _enqueue_existing(work_queue: "queue.Queue") -> None:
    existing = sorted(config.inbox_datasets(), key=lambda p: p.stat().st_mtime)
    for path in existing:
        log.info("Queuing existing file: %s", path.name)
        work_queue.put(path)


def main() -> int:
    config.ensure_dirs()
    work_queue: "queue.Queue" = queue.Queue()

    threading.Thread(target=_worker, args=(work_queue,), daemon=True).start()
    _enqueue_existing(work_queue)

    handler = _CsvHandler(work_queue)
    observer = Observer()
    observer.schedule(handler, str(config.INBOX_DIR), recursive=False)
    observer.start()
    log.info("Watching %s for datasets %s (Ctrl+C to stop)…",
             config.INBOX_DIR, config.SUPPORTED_EXTENSIONS)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping watcher…")
    finally:
        observer.stop()
        observer.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
