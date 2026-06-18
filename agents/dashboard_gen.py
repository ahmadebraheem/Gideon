"""Agent 7 — Dashboard Generator.

Input  : ``artifacts/deployment.json`` plus the upstream artifacts
         (cleaned data, metadata, metrics, predictions).
Output : ``artifacts/dashboard_data.json`` — the single bundle the Streamlit
         app renders.

Responsibility: turn the raw artifacts into everything the dashboard needs —
dataset overview, per-column EDA summaries, target-inclusive correlations, model
metrics, feature importances, per-feature input ranges (for the what-if
simulator), and pointers to the predictions table and the live model. The
Streamlit app mostly renders this bundle; the only computation it does at
runtime is the interactive what-if `model.predict()` and forecast projection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = config.get_logger("dashboard_gen")

_MAX_TOP_FEATURES = 25
_MAX_CATEGORY_LEVELS = 12
_MAX_CORR_COLS = 15


def _column_summaries(df: pd.DataFrame) -> list[dict]:
    summaries = []
    for col in df.columns:
        s = df[col]
        info = {
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            info["kind"] = "numeric"
            desc = s.describe()
            info.update({
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
                "min": float(desc.get("min", np.nan)),
                "q25": float(desc.get("25%", np.nan)),
                "median": float(desc.get("50%", np.nan)),
                "q75": float(desc.get("75%", np.nan)),
                "max": float(desc.get("max", np.nan)),
            })
            hist_counts, hist_edges = np.histogram(s.dropna(), bins=min(20, max(5, info["unique"])))
            info["histogram"] = {"counts": hist_counts.tolist(), "edges": hist_edges.tolist()}
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["kind"] = "datetime"
            info["min"] = str(s.min())
            info["max"] = str(s.max())
        else:
            info["kind"] = "categorical"
            top = s.astype(str).value_counts().head(_MAX_CATEGORY_LEVELS)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        summaries.append(info)
    return summaries


def _correlations_with_target(df: pd.DataFrame, feature_meta: dict) -> dict | None:
    """Correlation matrix over numeric columns, always including the target
    (label-encoding it first when it is categorical)."""
    df = df.copy()
    target = feature_meta.get("target_name")
    if target in df.columns and not pd.api.types.is_numeric_dtype(df[target]):
        mapping = feature_meta.get("label_mapping")
        if mapping:
            df[target] = df[target].astype(str).map(mapping)

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None

    if target in num.columns:
        others = [c for c in num.columns if c != target][: _MAX_CORR_COLS - 1]
        cols = others + [target]
    else:
        cols = list(num.columns)[:_MAX_CORR_COLS]

    corr = num[cols].corr().fillna(0.0)
    return {
        "columns": list(corr.columns),
        "matrix": corr.round(4).values.tolist(),
        "target": target if target in cols else None,
    }


def _feature_inputs(features_df: pd.DataFrame, target: str, importances: dict) -> dict:
    """Per-feature min/max/mean for the what-if sliders, ordered by importance
    (most influential first)."""
    feat_cols = [c for c in features_df.columns if c != target]
    ordered = [c for c in importances if c in feat_cols]
    ordered += [c for c in feat_cols if c not in importances]
    out: dict[str, dict] = {}
    for col in ordered:
        s = pd.to_numeric(features_df[col], errors="coerce")
        out[col] = {
            "min": float(np.nanmin(s.values)),
            "max": float(np.nanmax(s.values)),
            "mean": float(np.nanmean(s.values)),
        }
    return out


def generate(
    in_cleaned: Path = config.CLEANED_PARQUET,
    in_deployment: Path = config.DEPLOYMENT_FILE,
    out_data: Path = config.DASHBOARD_DATA,
) -> Path:
    cleaned = pd.read_parquet(in_cleaned)

    ingest_meta = config.read_json(config.INGEST_META)
    clean_report = config.read_json(config.CLEAN_REPORT)
    feature_meta = config.read_json(config.FEATURE_META)
    train_meta = config.read_json(config.TRAIN_META)
    metrics = config.read_json(config.METRICS_FILE)
    deployment = config.read_json(in_deployment)

    importances = train_meta.get("feature_importances", {})
    top_importances = dict(list(importances.items())[:_MAX_TOP_FEATURES])

    features_df = pd.read_parquet(config.FEATURES_PARQUET)
    target_name = feature_meta.get("target_name")
    feature_inputs = _feature_inputs(features_df, target_name, importances)

    bundle = {
        "generated_at": config.now_iso(),
        "dataset": {
            "name": ingest_meta.get("source_name"),
            "rows": int(len(cleaned)),
            "columns": int(cleaned.shape[1]),
            "ingested_at": ingest_meta.get("ingested_at"),
            "raw_rows": ingest_meta.get("n_rows"),
            "raw_columns": ingest_meta.get("n_cols"),
        },
        "cleaning": clean_report,
        "task": {
            "type": feature_meta.get("task_type"),
            "target": feature_meta.get("target_name"),
            "n_features": feature_meta.get("n_features"),
            "n_classes": feature_meta.get("n_classes"),
            "class_names": feature_meta.get("class_names"),
        },
        "metrics": metrics,
        "deployment": deployment,
        "feature_importances": top_importances,
        "model": {
            "path": str(config.MODEL_FILE.resolve()),
            "type": train_meta.get("model_type"),
            "features": train_meta.get("feature_columns"),
        },
        "feature_inputs": feature_inputs,
        "column_summaries": _column_summaries(cleaned),
        "correlations": _correlations_with_target(cleaned, feature_meta),
        "artifacts": {
            "cleaned_parquet": str(config.CLEANED_PARQUET.resolve()),
            "features_parquet": str(config.FEATURES_PARQUET.resolve()),
            "predictions_parquet": str(config.PREDICTIONS_PARQUET.resolve()),
            "model_file": str(config.MODEL_FILE.resolve()),
        },
        "preview_rows": cleaned.head(50).astype(object).where(
            pd.notna(cleaned.head(50)), None
        ).to_dict(orient="records"),
    }
    config.write_json(out_data, bundle)

    log.info("Dashboard data generated (%d columns summarised)", cleaned.shape[1])
    return out_data


def main() -> None:
    config.ensure_dirs()
    generate()


if __name__ == "__main__":
    sys.exit(main())
