"""Watcher — triggers the boss when a new CSV lands in ``inbox/``.

Design notes:
  * A single background worker thread consumes a queue, so pipeline runs are
    *serialised* — dropping a newer CSV mid-run simply queues another run that
    starts once the current one finishes (and the dashboard then updates).
  * New files are only enqueued once their size has stopped changing, to avoid
    reading a CSV that is still being copied in.

Run::

    python -m watcher
"""
from __future__ import annotations

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


def _is_target_csv(path: Path) -> bool:
    return (
        path.suffix.lower() == ".csv"
        and path.is_file()
        and config.PROCESSED_DIR not in path.parents
        and not path.name.startswith(".")
    )


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
    def __init__(self, work_queue: "queue.Queue[Path]") -> None:
        self._queue = work_queue

    def _maybe_enqueue(self, raw_path: str) -> None:
        path = Path(raw_path)
        if _is_target_csv(path):
            log.info("Detected: %s", path.name)
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


def _worker(work_queue: "queue.Queue[Path]") -> None:
    seen_recent: dict[str, float] = {}
    while True:
        path = work_queue.get()
        try:
            # Debounce duplicate events for the same file within a short window.
            now = time.time()
            last = seen_recent.get(str(path), 0)
            if now - last < 1.0 and not path.exists():
                continue
            seen_recent[str(path)] = now

            if not _wait_until_stable(path):
                log.info("Skipped (vanished before stable): %s", path.name)
                continue
            boss.run_pipeline(path)
        except Exception as exc:  # noqa: BLE001 — keep the watcher alive
            log.error("Run errored for %s: %s", path.name, exc)
        finally:
            work_queue.task_done()


def _enqueue_existing(work_queue: "queue.Queue[Path]") -> None:
    existing = sorted(
        (p for p in config.INBOX_DIR.glob("*.csv") if _is_target_csv(p)),
        key=lambda p: p.stat().st_mtime,
    )
    for path in existing:
        log.info("Queuing existing file: %s", path.name)
        work_queue.put(path)


def main() -> int:
    config.ensure_dirs()
    work_queue: "queue.Queue[Path]" = queue.Queue()

    threading.Thread(target=_worker, args=(work_queue,), daemon=True).start()
    _enqueue_existing(work_queue)

    handler = _CsvHandler(work_queue)
    observer = Observer()
    observer.schedule(handler, str(config.INBOX_DIR), recursive=False)
    observer.start()
    log.info("Watching %s for new CSVs (Ctrl+C to stop)…", config.INBOX_DIR)

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
