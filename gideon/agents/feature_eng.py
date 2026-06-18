"""Agent 3 — Feature Engineer.

Input  : ``artifacts/cleaned.parquet``.
Output : ``artifacts/features.parquet`` (model-ready X + target column)
         (+ ``artifacts/feature_meta.json``).

Responsibility (all automatic — "nobody touches code"):
  * detect the target column,
  * decide the task type (classification vs regression),
  * expand datetime columns, encode categoricals, and label-encode a
    string/categorical classification target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gideon import config

log = config.get_logger("feature_eng")


# --------------------------------------------------------------------------- #
# Target & task-type detection
# --------------------------------------------------------------------------- #
def detect_target(df: pd.DataFrame) -> str:
    """Pick the target column: a well-known name if present, else the last
    column (the most common convention for tabular CSVs)."""
    lower = {c.lower(): c for c in df.columns}
    for candidate in config.TARGET_NAME_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    return df.columns[-1]


def detect_task_type(target: pd.Series) -> str:
    """Return ``"classification"`` or ``"regression"``."""
    if target.dtype == bool or isinstance(target.dtype, pd.CategoricalDtype):
        return "classification"
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    # Numeric target: few distinct, integer-like values => classification.
    n_unique = target.nunique(dropna=True)
    is_intlike = bool(np.all(np.equal(np.mod(target.dropna(), 1), 0)))
    if n_unique <= config.MAX_CLASSIFICATION_CLASSES and is_intlike:
        return "classification"
    return "regression"


# --------------------------------------------------------------------------- #
# Feature transforms
# --------------------------------------------------------------------------- #
def _expand_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for col in list(df.select_dtypes(include=["datetime", "datetimetz"])):
        s = df[col]
        df[f"{col}__year"] = s.dt.year
        df[f"{col}__month"] = s.dt.month
        df[f"{col}__day"] = s.dt.day
        df[f"{col}__dayofweek"] = s.dt.dayofweek
        df = df.drop(columns=[col])
    return df


def _encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {"one_hot": [], "frequency": []}
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]))
    for col in cat_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique <= config.MAX_ONEHOT_CARDINALITY:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, dummy_na=False)
            df = pd.concat([df.drop(columns=[col]), dummies.astype(np.int8)], axis=1)
            info["one_hot"].append(col)
        else:
            freq = df[col].astype(str).map(df[col].astype(str).value_counts(normalize=True))
            df[col] = freq.fillna(0.0)
            info["frequency"].append(col)
    return df, info


def engineer(
    in_parquet: Path = config.CLEANED_PARQUET,
    out_parquet: Path = config.FEATURES_PARQUET,
    out_meta: Path = config.FEATURE_META,
) -> Path:
    df = pd.read_parquet(in_parquet)

    target_name = detect_target(df)
    task_type = detect_task_type(df[target_name])

    target = df[target_name]
    features = df.drop(columns=[target_name])

    # --- target encoding ---
    label_mapping: dict | None = None
    class_names: list | None = None
    if task_type == "classification":
        codes, uniques = pd.factorize(target.astype(str), sort=True)
        if (codes < 0).any():  # NaN slipped through -> own class
            codes = np.where(codes < 0, len(uniques), codes)
            uniques = list(uniques) + ["missing"]
        target = pd.Series(codes, index=target.index, name=target_name).astype(np.int64)
        class_names = [str(u) for u in uniques]
        label_mapping = {str(u): int(i) for i, u in enumerate(uniques)}
    else:
        target = pd.to_numeric(target, errors="coerce").astype(float)

    # --- feature transforms ---
    features = _expand_datetimes(features)
    features, encoding_info = _encode_categoricals(features)

    # Keep only numeric features, fill any residual NaNs.
    features = features.select_dtypes(include=[np.number]).copy()
    features = features.fillna(0.0)

    if features.shape[1] == 0:
        raise ValueError("No usable feature columns after engineering.")

    out = features.copy()
    out[target_name] = target.values
    # Drop rows where a regression target could not be parsed.
    if task_type == "regression":
        out = out[np.isfinite(out[target_name])]
    out = out.reset_index(drop=True)

    config.write_parquet(out, out_parquet)

    meta = {
        "engineered_at": config.now_iso(),
        "target_name": target_name,
        "task_type": task_type,
        "feature_columns": list(features.columns),
        "n_features": int(features.shape[1]),
        "n_rows": int(len(out)),
        "encoding": encoding_info,
        "label_mapping": label_mapping,
        "class_names": class_names,
        "n_classes": (len(class_names) if class_names else None),
    }
    config.write_json(out_meta, meta)

    log.info(
        "Engineered %d features for %s task (target='%s'%s)",
        features.shape[1], task_type, target_name,
        f", {len(class_names)} classes" if class_names else "",
    )
    return out_parquet


def main() -> None:
    config.ensure_dirs()
    engineer()


if __name__ == "__main__":
    sys.exit(main())
