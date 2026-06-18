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


_METRIC_HINTS = (
    "sales", "revenue", "amount", "price", "total", "profit", "cost", "gmv",
    "spend", "qty", "quantity", "count", "volume", "value", "income", "units",
)
_ID_HINTS = ("id", "index", "code", "zip", "postal", "phone", "year", "month", "day")


def _detect_date_column(df: pd.DataFrame) -> str | None:
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not date_cols:
        return None
    # Prefer the column spanning the widest time range.
    return max(date_cols, key=lambda c: (df[c].max() - df[c].min()))


def _detect_metric_column(df: pd.DataFrame, date_col: str | None, target: str | None) -> str | None:
    numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != date_col and not any(h in c.lower() for h in _ID_HINTS)
    ]
    if not numeric:
        return None
    # 1) name hints (sales/revenue/...).
    for col in numeric:
        if any(h in col.lower() for h in _METRIC_HINTS):
            return col
    # 2) the target, if it is numeric.
    if target in numeric:
        return target
    # 3) highest coefficient-of-variation numeric column.
    def cov(col: str) -> float:
        s = df[col]
        m = s.mean()
        return float(s.std() / abs(m)) if m else 0.0

    return max(numeric, key=cov)


def _pct(curr: float, prev: float) -> float | None:
    if prev is None or prev == 0 or pd.isna(prev) or pd.isna(curr):
        return None
    return float((curr - prev) / abs(prev) * 100)


def _time_kpis(df: pd.DataFrame, date_col: str | None, metric_col: str | None) -> dict:
    if not date_col or not metric_col:
        return {"available": False,
                "reason": "No datetime column detected — time-based KPIs need a date field."}

    s = df[[date_col, metric_col]].dropna().copy()
    s = s.set_index(date_col).sort_index()
    if s.empty:
        return {"available": False, "reason": "No usable date/metric rows."}

    monthly_sum = s[metric_col].resample("MS").sum()
    monthly_mean = s[metric_col].resample("MS").mean()
    monthly_count = s[metric_col].resample("MS").count()
    if len(monthly_sum) < 2:
        return {"available": False,
                "reason": "Data spans less than two months — not enough for trends."}

    primary = monthly_sum  # totals per month
    mom = primary.pct_change() * 100
    yoy = primary.pct_change(12) * 100 if len(primary) > 12 else pd.Series(index=primary.index, dtype=float)

    periods = [d.strftime("%Y-%m") for d in primary.index]
    values = [float(v) for v in primary.values]

    # Trend via linear slope over the monthly totals.
    x = np.arange(len(values))
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) > 1 else 0.0
    mean_val = float(np.mean(values)) or 1.0
    rel = slope / abs(mean_val)
    trend = "up" if rel > 0.01 else "down" if rel < -0.01 else "flat"

    best_i = int(np.argmax(values))
    worst_i = int(np.argmin(values))
    mom_clean = mom.dropna()
    rise = {"period": mom_clean.idxmax().strftime("%Y-%m"), "pct": float(mom_clean.max())} if not mom_clean.empty else None
    drop = {"period": mom_clean.idxmin().strftime("%Y-%m"), "pct": float(mom_clean.min())} if not mom_clean.empty else None

    def _last(series: pd.Series):
        v = series.dropna()
        return float(v.iloc[-1]) if not v.empty else None

    return {
        "available": True,
        "date_column": date_col,
        "metric_column": metric_col,
        "agg": "monthly total",
        "periods": periods,
        "values_sum": values,
        "values_mean": [float(v) for v in monthly_mean.values],
        "values_count": [int(v) for v in monthly_count.values],
        "mom_pct": [None if pd.isna(v) else float(v) for v in mom.values],
        "yoy_pct": [None if pd.isna(v) else float(v) for v in yoy.values],
        "summary": {
            "latest_period": periods[-1],
            "latest_value": values[-1],
            "mom_pct": _last(mom),
            "yoy_pct": _last(yoy),
            "best_period": periods[best_i], "best_value": values[best_i],
            "worst_period": periods[worst_i], "worst_value": values[worst_i],
            "total": float(np.sum(values)),
            "trend": trend, "trend_slope": slope,
            "biggest_rise": rise, "biggest_drop": drop,
            "n_months": len(values),
        },
    }


def _strength(r: float) -> str:
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    return "weak"


def _insights(df: pd.DataFrame, feature_meta: dict) -> dict:
    df = df.copy()
    target = feature_meta.get("target_name")
    if target in df.columns and not pd.api.types.is_numeric_dtype(df[target]):
        mapping = feature_meta.get("label_mapping")
        if mapping:
            df[target] = df[target].astype(str).map(mapping)

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return {"top_correlations": [], "target_correlations": []}

    corr = num.corr().fillna(0.0)
    cols = list(corr.columns)

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if abs(r) >= 0.3:  # only "interesting" relationships
                direction = "positively" if r > 0 else "negatively"
                pairs.append({
                    "a": cols[i], "b": cols[j], "r": round(r, 3),
                    "strength": _strength(r), "direction": direction,
                    "text": f"`{cols[i]}` and `{cols[j]}` are {_strength(r)}ly {direction} "
                            f"correlated (r = {r:+.2f}).",
                })
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)

    target_corr = []
    if target in corr.columns:
        tser = corr[target].drop(labels=[target], errors="ignore")
        for feat, r in tser.reindex(tser.abs().sort_values(ascending=False).index).items():
            r = float(r)
            if abs(r) < 0.1:
                continue
            direction = "increase" if r > 0 else "decrease"
            target_corr.append({
                "feature": feat, "r": round(r, 3),
                "text": f"As `{feat}` rises, `{target}` tends to {direction} (r = {r:+.2f}).",
            })

    return {"top_correlations": pairs[:6], "target_correlations": target_corr[:5]}


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

    date_col = _detect_date_column(cleaned)
    metric_col = _detect_metric_column(cleaned, date_col, target_name)
    kpis = _time_kpis(cleaned, date_col, metric_col)
    insights = _insights(cleaned, feature_meta)

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
        "kpis": kpis,
        "insights": insights,
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

    log.info(
        "Dashboard data generated (%d columns summarised; KPIs=%s, metric=%s)",
        cleaned.shape[1], kpis.get("available"), kpis.get("metric_column"),
    )
    return out_data


def main() -> None:
    config.ensure_dirs()
    generate()


if __name__ == "__main__":
    sys.exit(main())
