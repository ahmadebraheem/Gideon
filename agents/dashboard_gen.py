"""Agent 7 — Dashboard Generator.

Input  : ``artifacts/deployment.json`` plus the upstream artifacts
         (cleaned data, metadata, metrics, predictions).
Output : ``artifacts/dashboard_data.json`` — the single bundle the Streamlit
         app renders.

Responsibility: turn the raw artifacts into everything the dashboard needs —
dataset overview, per-column EDA summaries, numeric correlations, model
metrics, feature importances, and a pointer to the predictions table. The
Streamlit app does no analysis of its own; it just renders this bundle.
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


def _numeric_correlations(df: pd.DataFrame) -> dict | None:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    num = num.iloc[:, :_MAX_CORR_COLS]
    corr = num.corr(numeric_only=True).fillna(0.0)
    return {"columns": list(corr.columns), "matrix": corr.values.round(4).tolist()}


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
        "column_summaries": _column_summaries(cleaned),
        "correlations": _numeric_correlations(cleaned),
        "artifacts": {
            "cleaned_parquet": str(config.CLEANED_PARQUET.resolve()),
            "predictions_parquet": str(config.PREDICTIONS_PARQUET.resolve()),
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
