"""Agent 2 — Cleaner.

Input  : ``artifacts/raw.parquet``.
Output : ``artifacts/cleaned.parquet`` (+ ``artifacts/clean_report.json``).

Responsibility: produce a tidy, analysis-ready table — drop empty/duplicate
rows, coerce obvious numeric/datetime strings, drop mostly-empty columns, and
impute the remaining missing values deterministically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gideon import config

log = config.get_logger("cleaner")

_NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "?", "-"}
_DATE_HINTS = ("-", "/", ":")


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object"):
        stripped = df[col].astype(str).str.strip()
        stripped = stripped.mask(stripped.str.lower().isin(_NULL_TOKENS), np.nan)
        df[col] = stripped
    return df


def _coerce_numeric(df: pd.DataFrame) -> list[str]:
    coerced: list[str] = []
    for col in list(df.select_dtypes(include="object")):
        as_num = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna()
        if non_null.any() and as_num[non_null].notna().mean() >= 0.95:
            df[col] = as_num
            coerced.append(col)
    return coerced


def _coerce_datetime(df: pd.DataFrame) -> list[str]:
    coerced: list[str] = []
    for col in list(df.select_dtypes(include="object")):
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty or not sample.str.contains("|".join(map(_re_escape, _DATE_HINTS))).any():
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed", dayfirst=False)
        non_null = df[col].notna()
        if non_null.any() and parsed[non_null].notna().mean() >= 0.95:
            df[col] = parsed
            coerced.append(col)
    return coerced


def _re_escape(ch: str) -> str:
    import re
    return re.escape(ch)


def clean(
    in_parquet: Path = config.RAW_PARQUET,
    out_parquet: Path = config.CLEANED_PARQUET,
    out_report: Path = config.CLEAN_REPORT,
) -> Path:
    df = pd.read_parquet(in_parquet)
    rows_before, cols_before = df.shape

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = _strip_strings(df)

    n_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    coerced_numeric = _coerce_numeric(df)
    coerced_datetime = _coerce_datetime(df)

    # Drop columns that are mostly missing.
    high_missing = [c for c in df.columns if df[c].isna().mean() > config.MAX_MISSING_FRACTION]
    df = df.drop(columns=high_missing)

    # Impute the remainder.
    imputed: dict[str, str] = {}
    for col in df.columns:
        if not df[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            imputed[col] = "median"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            imputed[col] = "median(datetime)"
        else:
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "missing"
            df[col] = df[col].fillna(fill).astype(str)
            imputed[col] = "mode"

    # Guard: a dataset with no rows or a single column left is unusable.
    if df.empty or df.shape[1] < 2:
        raise ValueError("Cleaning removed too much data; nothing usable remains.")

    config.write_parquet(df, out_parquet)

    report = {
        "cleaned_at": config.now_iso(),
        "rows_before": int(rows_before),
        "rows_after": int(len(df)),
        "cols_before": int(cols_before),
        "cols_after": int(df.shape[1]),
        "duplicate_rows_removed": n_duplicates,
        "high_missing_columns_dropped": high_missing,
        "coerced_to_numeric": coerced_numeric,
        "coerced_to_datetime": coerced_datetime,
        "imputed_columns": imputed,
    }
    config.write_json(out_report, report)

    log.info(
        "Cleaned: %d->%d rows, %d->%d cols (%d dup removed, %d cols dropped)",
        rows_before, len(df), cols_before, df.shape[1], n_duplicates, len(high_missing),
    )
    return out_parquet


def main() -> None:
    config.ensure_dirs()
    clean()


if __name__ == "__main__":
    sys.exit(main())
