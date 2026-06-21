# Gideon

A **local, zero-cloud, zero-API** automated ML pipeline.

Drop a dataset (**CSV, JSON, or Parquet**) into `inbox/` and Gideon automatically
cleans the data, engineers features, trains a model, evaluates it, deploys it, and
refreshes a live Streamlit dashboard. Drop a newer file and the whole pipeline
reruns and the dashboard updates. **Nobody touches any code.**

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
            │      boss       │  orchestrates the 8 agents in sequence
            └────────┬────────┘
                     │
   ingestor → cleaner → feature_eng → trainer → evaluator → deployer → dashboard_gen → monitor
        │         │          │           │          │          │            │             │
        └─────────┴──────────┴─── artifacts/ ───────┴──────────┴────────────┘           logs/
                     │
            ┌────────▼────────┐
            │  dashboard/app  │  Streamlit renders artifacts/dashboard_data.json
            └─────────────────┘
```

**Design rules**

- *8 specialist agents**, each owning exactly one task.
- **One agent, one input, one output — no shared in-memory state.** Agents
  communicate *only* by reading/writing files in `artifacts/`.
- A **boss** orchestrates them sequentially and writes a run manifest.
- A **watcher** triggers the boss when a new CSV lands.

### The eight agents

| # | Agent           | Input                       | Output                                |
|---|-----------------|-----------------------------|---------------------------------------|
| 1 | `ingestor`      | `inbox/*.{csv,json,parquet}` | `raw.parquet` + `ingest_meta.json`   |
| 2 | `cleaner`       | `raw.parquet`               | `cleaned.parquet` + `clean_report.json` |
| 3 | `feature_eng`   | `cleaned.parquet`           | `features.parquet` + `feature_meta.json` |
| 4 | `trainer`       | `features.parquet`          | `model.joblib` + `train_meta.json`    |
| 5 | `evaluator`     | `model.joblib` + features   | `metrics.json` + `predictions.parquet` |
| 6 | `deployer`      | model + metrics             | `deployment.json` (+ versioned model) |
| 7 | `dashboard_gen` | all upstream artifacts      | `dashboard_data.json`                 |
| 8 | `monitor` | `manifest.json` +  all artifacts     | `logs/monitor.json`                 |

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

`pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `joblib`, `watchdog`, `psutil`,
`streamlit`, `plotly`. Python 3.10+. Dependencies are managed with `uv`.

---

## Prerequisites

- **Python 3.10+** and [uv](https://docs.astral.sh/uv/).
- **macOS only — install the OpenMP runtime** that XGBoost depends on:

  ```bash
  brew install libomp
  ```

  Without it, importing XGBoost fails with
  `XGBoostError: Library not loaded: @rpath/libomp.dylib`.
  See [Troubleshooting](#troubleshooting) below.

  (Linux wheels bundle their own OpenMP, so no extra step is needed there.)

---

## Quick start

Dependencies are managed with [uv](https://docs.astral.sh/uv/). All commands are
run from the repository root.

```bash
# 0. macOS only: install the OpenMP runtime XGBoost needs
brew install libomp

# 1. Install dependencies into a managed virtual environment
uv sync

# 2. Start the watcher
uv run python -m watcher

# 3. In another terminal, start the dashboard
uv run streamlit run dashboard/app.py

# 4. Drop a CSV/JSON/Parquet file into inbox/ and watch it flow through.
```

### Run a single dataset without the watcher

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
├── agents/              # the 8 specialist agents
├── artifacts/           # all inter-agent files live here (git-ignored)
├── inbox/               # drop CSV/JSON/Parquet here (they stay; empty it to reset)
├── dashboard/app.py     # Streamlit renderer
├── logs/                # monitor logs — gideon.log, runs.jsonl, monitor.json
├── pyproject.toml       # uv-managed dependencies
└── uv.lock              # pinned dependency lockfile
```

## Notes

- Artifacts are written **atomically** (temp file + rename) so the dashboard
  never reads a half-written file.
- Pipeline runs are **serialised**: dropping a newer CSV mid-run queues another
  run that starts when the current one finishes.
- Dropped CSVs **stay in `inbox/`**. The watcher de-duplicates by content hash,
  so re-saving the same file does not re-run the pipeline; only new content does.
- **Emptying the inbox resets the dashboard**: when the last CSV is removed, the
  artifacts are cleared and the dashboard returns to its clean "waiting" state
  (the app also hides stale results whenever the inbox is empty).
- **Reset button**: the dashboard sidebar has a **⚠️ Reset** control (behind a
  confirm checkbox) that wipes all artifacts and, optionally, the inbox CSVs —
  returning everything to a clean slate without touching the terminal.

---

## Troubleshooting

### `XGBoostError: Library not loaded: @rpath/libomp.dylib` (macOS)

XGBoost's native library needs the **OpenMP runtime**, which macOS does not ship
by default. Install it with Homebrew:

```bash
brew install libomp
```

Then re-run your command. If you use MacPorts instead of Homebrew, install
`libomp` via `sudo port install libomp`. On Apple Silicon, make sure you are
using a matching (arm64) Python/Homebrew so the architectures line up.

Full error for reference:

```
xgboost.core.XGBoostError:
XGBoost Library (libxgboost.dylib) could not be loaded.
  ...
  Reason: tried: '/usr/local/opt/libomp/lib/libomp.dylib' (no such file) ...
```

### `ModuleNotFoundError: No module named 'config'` / `agents`

Run all commands **from the repository root** (the folder containing `boss.py`),
e.g. `uv run python -m boss inbox/my_data.csv`. The flat module layout resolves
`config` and `agents` relative to that directory.
