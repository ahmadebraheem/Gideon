"""Agent 5 — Evaluator.

Input  : ``artifacts/model.joblib`` + ``artifacts/features.parquet``
         + ``artifacts/train_meta.json`` (for the holdout indices).
Output : ``artifacts/metrics.json`` (+ ``artifacts/predictions.parquet``).

Responsibility: score the model on the untouched holdout set and persist both
the headline metrics and the row-level predictions (actual vs predicted) used
by the dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from gideon import config

log = config.get_logger("evaluator")


def _classification_metrics(y_true, y_pred, y_proba, class_names) -> dict:
    out = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision_weighted": float(skm.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(skm.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred).tolist(),
        "labels": list(range(len(class_names))) if class_names else sorted(np.unique(y_true).tolist()),
    }
    # ROC-AUC where it is well defined.
    try:
        if y_proba is not None and len(np.unique(y_true)) == 2:
            out["roc_auc"] = float(skm.roc_auc_score(y_true, y_proba[:, 1]))
        elif y_proba is not None and len(np.unique(y_true)) > 2:
            out["roc_auc_ovr_weighted"] = float(
                skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
    except (ValueError, IndexError):
        pass
    out["headline_metric"] = "accuracy"
    out["headline_value"] = out["accuracy"]
    return out


def _regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    out = {
        "r2": float(skm.r2_score(y_true, y_pred)),
        "mae": float(skm.mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    if np.isfinite(mape):
        out["mape"] = float(mape)
    out["headline_metric"] = "r2"
    out["headline_value"] = out["r2"]
    return out


def evaluate(
    in_model: Path = config.MODEL_FILE,
    in_parquet: Path = config.FEATURES_PARQUET,
    in_meta: Path = config.TRAIN_META,
    out_metrics: Path = config.METRICS_FILE,
    out_predictions: Path = config.PREDICTIONS_PARQUET,
) -> Path:
    train_meta = config.read_json(in_meta)
    task_type = train_meta["task_type"]
    target_name = train_meta["target_name"]
    feature_cols = train_meta["feature_columns"]
    class_names = train_meta.get("class_names")
    test_idx = np.array(train_meta["test_indices"], dtype=int)

    model = joblib.load(in_model)
    df = pd.read_parquet(in_parquet)
    X_test = df[feature_cols].iloc[test_idx]
    y_test = df[target_name].iloc[test_idx]

    y_pred = model.predict(X_test)

    preds = pd.DataFrame({"row": test_idx})
    if task_type == "classification":
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        result = _classification_metrics(y_test.to_numpy(), y_pred, y_proba, class_names)
        if class_names:
            preds["actual"] = [class_names[int(i)] for i in y_test.to_numpy()]
            preds["predicted"] = [class_names[int(i)] for i in y_pred]
        else:
            preds["actual"] = y_test.to_numpy()
            preds["predicted"] = y_pred
        if y_proba is not None:
            preds["confidence"] = y_proba.max(axis=1)
        preds["correct"] = preds["actual"] == preds["predicted"]
    else:
        result = _regression_metrics(y_test.to_numpy(), np.asarray(y_pred, dtype=float))
        preds["actual"] = y_test.to_numpy()
        preds["predicted"] = np.asarray(y_pred, dtype=float)
        preds["abs_error"] = (preds["actual"] - preds["predicted"]).abs()

    result["task_type"] = task_type
    result["target_name"] = target_name
    result["n_test"] = int(len(test_idx))
    result["evaluated_at"] = config.now_iso()

    config.write_json(out_metrics, result)
    config.write_parquet(preds, out_predictions)

    log.info(
        "Evaluated %s: %s=%.4f on %d holdout rows",
        task_type, result["headline_metric"], result["headline_value"], len(test_idx),
    )
    return out_metrics


def main() -> None:
    config.ensure_dirs()
    evaluate()


if __name__ == "__main__":
    sys.exit(main())
