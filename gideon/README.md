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
`streamlit`, `plotly`. Python 3.10+.

---

## Quick start

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r gideon/requirements.txt

# 2. Start the watcher (run from the repository root)
python -m gideon.watcher

# 3. In another terminal, start the dashboard
streamlit run gideon/dashboard/app.py

# 4. Drop a CSV into gideon/inbox/ and watch it flow through.
```

### Run a single CSV without the watcher

```bash
python -m gideon.boss gideon/inbox/my_data.csv
```

### Run one agent standalone (handy for debugging)

```bash
python -m gideon.agents.cleaner
```

---

## Layout

```
gideon/
├── boss.py              # orchestrator
├── watcher.py           # inbox file watcher
├── config.py            # paths, constants, atomic IO helpers (no runtime state)
├── agents/              # the 7 specialist agents
├── artifacts/           # all inter-agent files live here (git-ignored)
├── inbox/               # drop CSVs here (processed files move to _processed/)
└── dashboard/app.py     # Streamlit renderer
```

## Notes

- Artifacts are written **atomically** (temp file + rename) so the dashboard
  never reads a half-written file.
- Pipeline runs are **serialised**: dropping a newer CSV mid-run queues another
  run that starts when the current one finishes.
- Processed CSVs are moved to `inbox/_processed/` so they are not re-triggered.
