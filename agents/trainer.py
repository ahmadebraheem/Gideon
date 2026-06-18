"""Agent 4 — Trainer.

Input  : ``artifacts/features.parquet`` (+ ``artifacts/feature_meta.json``).
Output : ``artifacts/model.joblib`` (+ ``artifacts/train_meta.json``).

Responsibility: split the data deterministically, train an XGBoost model that
matches the detected task type, and persist the fitted model together with the
exact holdout indices so the evaluator scores on unseen rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import config

log = config.get_logger("trainer")


def _build_model(task_type: str, n_classes: int | None):
    common = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    if task_type == "classification":
        objective = "binary:logistic" if (n_classes or 2) <= 2 else "multi:softprob"
        return XGBClassifier(objective=objective, eval_metric="logloss", **common)
    return XGBRegressor(objective="reg:squarederror", **common)


def train(
    in_parquet: Path = config.FEATURES_PARQUET,
    in_meta: Path = config.FEATURE_META,
    out_model: Path = config.MODEL_FILE,
    out_meta: Path = config.TRAIN_META,
) -> Path:
    meta = config.read_json(in_meta)
    target_name = meta["target_name"]
    task_type = meta["task_type"]
    n_classes = meta.get("n_classes")

    df = pd.read_parquet(in_parquet)
    feature_cols = [c for c in df.columns if c != target_name]
    X = df[feature_cols]
    y = df[target_name]

    # Stratify classification splits when every class has at least 2 members.
    stratify = None
    if task_type == "classification":
        counts = y.value_counts()
        if (counts >= 2).all() and len(counts) > 1:
            stratify = y

    if len(df) >= 10:
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            idx, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=stratify,
        )
    else:  # too small to hold out — train on everything, evaluate on everything
        train_idx = test_idx = np.arange(len(df))
        log.warning("Dataset too small for a holdout split; evaluating on training data.")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

    model = _build_model(task_type, n_classes)
    model.fit(X_train, y_train)

    config.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)

    importances = dict(
        sorted(
            zip(feature_cols, (float(v) for v in model.feature_importances_)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    train_meta = {
        "trained_at": config.now_iso(),
        "task_type": task_type,
        "target_name": target_name,
        "feature_columns": feature_cols,
        "model_type": type(model).__name__,
        "model_params": {k: v for k, v in model.get_params().items() if not callable(v)},
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_indices": [int(i) for i in test_idx],
        "feature_importances": importances,
        "n_classes": n_classes,
        "class_names": meta.get("class_names"),
    }
    config.write_json(out_meta, train_meta)

    log.info(
        "Trained %s on %d rows (%d feats); holdout=%d",
        type(model).__name__, len(train_idx), len(feature_cols), len(test_idx),
    )
    return out_model


def main() -> None:
    config.ensure_dirs()
    train()


if __name__ == "__main__":
    sys.exit(main())
