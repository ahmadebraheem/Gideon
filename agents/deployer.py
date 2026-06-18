"""Agent 6 — Deployer.

Input  : ``artifacts/model.joblib`` + ``artifacts/metrics.json``
         + ``artifacts/train_meta.json``.
Output : ``artifacts/deployment.json``.

Responsibility: promote the freshly trained model to "live". It versions the
model into ``artifacts/deployments/<version>/``, records a quality verdict, and
writes the deployment record the dashboard treats as the source of truth for
"what is currently serving".
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import config

log = config.get_logger("deployer")

DEPLOYMENTS_DIR = config.ARTIFACTS_DIR / "deployments"

# Below these headline scores we still deploy, but flag the model as weak.
_MIN_ACCURACY = 0.5
_MIN_R2 = 0.0


def _quality_verdict(task_type: str, headline_value: float) -> tuple[str, str]:
    if task_type == "classification":
        if headline_value >= _MIN_ACCURACY:
            return "healthy", f"accuracy {headline_value:.3f} >= {_MIN_ACCURACY}"
        return "weak", f"accuracy {headline_value:.3f} < {_MIN_ACCURACY}"
    if headline_value >= _MIN_R2:
        return "healthy", f"R2 {headline_value:.3f} >= {_MIN_R2}"
    return "weak", f"R2 {headline_value:.3f} < {_MIN_R2}"


def deploy(
    in_model: Path = config.MODEL_FILE,
    in_metrics: Path = config.METRICS_FILE,
    in_train_meta: Path = config.TRAIN_META,
    out_deployment: Path = config.DEPLOYMENT_FILE,
) -> Path:
    metrics = config.read_json(in_metrics)
    train_meta = config.read_json(in_train_meta)
    ingest_meta = config.read_json(config.INGEST_META) if config.INGEST_META.exists() else {}

    version = config.now_iso().replace(":", "").replace("-", "").replace(".", "")[:15]
    version_dir = DEPLOYMENTS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(in_model, version_dir / "model.joblib")
    config.write_json(version_dir / "metrics.json", metrics)

    status, reason = _quality_verdict(metrics["task_type"], metrics["headline_value"])

    deployment = {
        "deployed_at": config.now_iso(),
        "version": version,
        "status": status,
        "status_reason": reason,
        "task_type": metrics["task_type"],
        "target_name": metrics["target_name"],
        "headline_metric": metrics["headline_metric"],
        "headline_value": metrics["headline_value"],
        "model_type": train_meta.get("model_type"),
        "model_path": str((version_dir / "model.joblib").resolve()),
        "live_model_path": str(config.MODEL_FILE.resolve()),
        "source_dataset": ingest_meta.get("source_name"),
        "n_train": train_meta.get("n_train"),
        "n_test": train_meta.get("n_test"),
    }
    config.write_json(out_deployment, deployment)

    log.info("Deployed version %s (status=%s, %s=%.4f)",
             version, status, metrics["headline_metric"], metrics["headline_value"])
    return out_deployment


def main() -> None:
    config.ensure_dirs()
    deploy()


if __name__ == "__main__":
    sys.exit(main())
