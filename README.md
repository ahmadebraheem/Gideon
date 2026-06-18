# Gideon

A **local, zero-cloud, zero-API** automated ML pipeline.

Drop a CSV into `inbox/` and Gideon automatically cleans the data, engineers
features, trains a model, evaluates it, deploys it, and refreshes a live
Streamlit dashboard. Drop a newer CSV and the whole pipeline reruns and the
dashboard updates. **Nobody touches any code.**

Everything runs on your machine. No network calls, no external services.

---

## Architecture

```
                 inbox/*.csv
                     │
            ┌────────▼────────┐
            │     watcher     │  watchdog detects new CSVs, serialises runs
            └────────┬────────┘
                     │ triggers
            ┌────────▼────────┐
            │      boss       │  orchestrates the 7 agents in sequence
            └────────┬────────┘
                     │
   ingestor → cleaner → feature_eng → trainer → evaluator → deployer → dashboard_gen
        │         │          │           │          │          │            │
        └─────────┴──────────┴─── artifacts/ ───────┴──────────┴────────────┘
                     │
            ┌────────▼────────┐
            │  dashboard/app  │  Streamlit renders artifacts/dashboard_data.json
            └─────────────────┘
```

**Design rules**

- **7 specialist agents**, each owning exactly one task.
- **One agent, one input, one output — no shared in-memory state.** Agents
  communicate *only* by reading/writing files in `artifacts/`.
- A **boss** orchestrates them sequentially and writes a run manifest.
- A **watcher** triggers the boss when a new CSV lands.

### The seven agents

| # | Agent           | Input                       | Output                                |
|---|-----------------|-----------------------------|---------------------------------------|
| 1 | `ingestor`      | `inbox/*.csv`               | `raw.parquet` + `ingest_meta.json`    |
| 2 | `cleaner`       | `raw.parquet`               | `cleaned.parquet` + `clean_report.json` |
| 3 | `feature_eng`   | `cleaned.parquet`           | `features.parquet` + `feature_meta.json` |
| 4 | `trainer`       | `features.parquet`          | `model.joblib` + `train_meta.json`    |
| 5 | `evaluator`     | `model.joblib` + features   | `metrics.json` + `predictions.parquet` |
| 6 | `deployer`      | model + metrics             | `deployment.json` (+ versioned model) |
| 7 | `dashboard_gen` | all upstream artifacts      | `dashboard_data.json`                 |

The pipeline is fully automatic:

- **Target detection** — uses a well-known column name (`target`, `label`, `y`,
  `class`, `outcome`, `result`) if present, otherwise the last column.
- **Task detection** — non-numeric or few integer-like values ⇒ classification,
  otherwise regression.
- **Feature engineering** — datetime expansion, one-hot for low-cardinality
  categoricals, frequency encoding for high-cardinality ones.
- **Modeling** — XGBoost classifier/regressor with sensible defaults.

---

## Stack

`pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `joblib`, `watchdog`,
`streamlit`, `plotly`. Python 3.10+. Dependencies are managed with `uv`.

---

## Quick start

Dependencies are managed with [uv](https://docs.astral.sh/uv/). All commands are
run from the repository root.

```bash
# 1. Install dependencies into a managed virtual environment
uv sync

# 2. Start the watcher
uv run python -m watcher

# 3. In another terminal, start the dashboard
uv run streamlit run dashboard/app.py

# 4. Drop a CSV into inbox/ and watch it flow through.
```

### Run a single CSV without the watcher

```bash
uv run python -m boss inbox/my_data.csv
```

### Run one agent standalone (handy for debugging)

```bash
uv run python -m agents.cleaner
```

> Prefer plain pip? A `requirements.txt` is also provided:
> `pip install -r requirements.txt`, then run the same commands without the
> `uv run` prefix.

---

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) renders the bundle from the
`dashboard_gen` agent and the live model. Tabs:

- **Overview** — dataset, target, deployment status, graded headline metric.
- **KPIs & Insights** — auto-detects a date column and a metric column (e.g. sales/
  revenue) and shows **MoM** and **YoY** trends, monthly **ups & downs** (green/red),
  best/worst months, overall direction, plus **auto-discovered interesting
  correlations** described in plain language. Falls back gracefully (insights only)
  when the dataset has no date column.
- **Data & EDA** — cleaned preview, cleaning report, per-column summaries and distributions.
- **Model health** — RMSE/MAE/R² (or accuracy/F1) cards, R²/accuracy colour-coded
  green/amber/red, and a rolling RMSE (regression) / rolling accuracy (classification) chart.
- **Predictions** — actual vs predicted chart, anomalies flagged where deviation
  exceeds 2σ (in red), and an anomaly-count summary.
- **Forecast** *(regression)* — configurable 1–12 step horizon, dashed projection
  from the last actual, ±1.5×RMSE confidence band, up/down/stable direction
  indicator, and a spike-alert banner when the change exceeds a threshold.
- **Feature importance** — horizontal bars sorted descending, top feature
  highlighted, with an auto-generated one-line summary.
- **What-if** — one slider per influential feature (auto-built from each feature's
  min/max/mean), live `model.predict()` on change, and a prediction card showing
  the delta from the baseline (mean) prediction.
- **Correlations** — annotated Plotly heatmap over numeric columns, target included.

## Layout

```
.
├── boss.py              # orchestrator
├── watcher.py           # inbox file watcher
├── config.py            # paths, constants, atomic IO helpers (no runtime state)
├── agents/              # the 7 specialist agents
├── artifacts/           # all inter-agent files live here (git-ignored)
├── inbox/               # drop CSVs here (processed files move to _processed/)
├── dashboard/app.py     # Streamlit renderer
├── pyproject.toml       # uv-managed dependencies
└── uv.lock              # pinned dependency lockfile
```

## Notes

- Artifacts are written **atomically** (temp file + rename) so the dashboard
  never reads a half-written file.
- Pipeline runs are **serialised**: dropping a newer CSV mid-run queues another
  run that starts when the current one finishes.
- Processed CSVs are moved to `inbox/_processed/` so they are not re-triggered.
