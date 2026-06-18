"""Agent 1 — Ingestor.

Input  : a raw CSV file (from ``inbox/``).
Output : ``artifacts/raw.parquet`` (+ ``artifacts/ingest_meta.json``).

Responsibility: load the CSV robustly, normalise column names, run light
sanity checks, and persist a typed copy other agents can rely on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from gideon import config

log = config.get_logger("ingestor")


def _dedupe_columns(columns: list[str]) -> list[str]:
    """Make column names unique (``a, a`` -> ``a, a.1``)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            out.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            out.append(col)
    return out


def _read_csv(csv_path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(csv_path, encoding=encoding, skipinitialspace=True)
        except UnicodeDecodeError:
            continue
    # Last resort: let pandas pick and replace undecodable bytes.
    return pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")


def ingest(
    csv_path: str | Path,
    out_parquet: Path = config.RAW_PARQUET,
    out_meta: Path = config.INGEST_META,
) -> Path:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = _read_csv(csv_path)

    # Normalise column names.
    df.columns = _dedupe_columns([str(c).strip() for c in df.columns])

    # Drop unnamed index columns that pandas/Excel often leave behind.
    junk = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if junk:
        df = df.drop(columns=junk)

    if df.shape[1] < 2:
        raise ValueError(
            "CSV must contain at least 2 columns (features + a target column)."
        )
    if len(df) < 5:
        log.warning("Very small dataset (%d rows); model quality may be poor.", len(df))

    config.write_parquet(df, out_parquet)

    meta = {
        "source_file": str(csv_path.resolve()),
        "source_name": csv_path.name,
        "ingested_at": config.now_iso(),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "dropped_unnamed_columns": junk,
    }
    config.write_json(out_meta, meta)

    log.info(
        "Ingested '%s' -> %s (%d rows x %d cols)",
        csv_path.name, out_parquet.name, len(df), df.shape[1],
    )
    return out_parquet


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit("usage: python -m gideon.agents.ingestor <path/to.csv>")
    config.ensure_dirs()
    ingest(argv[0])


if __name__ == "__main__":
    main()
