"""Agent 1 — Ingestor.

Input  : a raw dataset file from ``inbox/`` — CSV, JSON, or Parquet.
Output : ``artifacts/raw.parquet`` (+ ``artifacts/ingest_meta.json``).

Responsibility: load the dataset robustly (whatever the supported format),
normalise column names, run light sanity checks, and persist a typed copy other
agents can rely on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import config

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


def _read_json(json_path: Path) -> pd.DataFrame:
    # Try a regular JSON array/object first, then newline-delimited JSON (JSONL).
    try:
        df = pd.read_json(json_path)
    except ValueError:
        df = pd.read_json(json_path, lines=True)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def _read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}."
    )


def ingest(
    source_path: str | Path,
    out_parquet: Path = config.RAW_PARQUET,
    out_meta: Path = config.INGEST_META,
) -> Path:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    df = _read_dataset(source_path)

    # Normalise column names.
    df.columns = _dedupe_columns([str(c).strip() for c in df.columns])

    # Drop unnamed index columns that pandas/Excel often leave behind.
    junk = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if junk:
        df = df.drop(columns=junk)

    # JSON inputs often carry nested or mixed-type columns that Parquet/Arrow
    # cannot represent; coerce those to text so the rest of the pipeline is safe.
    df = config.make_parquet_safe(df)

    if df.shape[1] < 2:
        raise ValueError(
            "Dataset must contain at least 2 columns (features + a target column)."
        )
    if len(df) < 5:
        log.warning("Very small dataset (%d rows); model quality may be poor.", len(df))

    config.write_parquet(df, out_parquet)

    meta = {
        "source_file": str(source_path.resolve()),
        "source_name": source_path.name,
        "source_format": source_path.suffix.lower().lstrip("."),
        "ingested_at": config.now_iso(),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "dropped_unnamed_columns": junk,
    }
    config.write_json(out_meta, meta)

    log.info(
        "Ingested '%s' (%s) -> %s (%d rows x %d cols)",
        source_path.name, meta["source_format"], out_parquet.name, len(df), df.shape[1],
    )
    return out_parquet


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit("usage: python -m agents.ingestor <path/to/dataset.{csv,json,parquet}>")
    config.ensure_dirs()
    ingest(argv[0])


if __name__ == "__main__":
    main()
