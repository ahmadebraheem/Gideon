"""Gideon live dashboard (Streamlit).

Renders the analytics bundle produced by the dashboard_gen agent
(``artifacts/dashboard_data.json``) plus the Parquet artifacts and the live
model it points to. Beyond rendering, the only runtime computation is the
interactive what-if ``model.predict()`` and the forecast projection.

Launch::

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root modules (config.py) importable when Streamlit runs
# this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_autorefresh import st_autorefresh  # noqa: E402

import config  # noqa: E402

st.set_page_config(page_title="Gideon", page_icon="🤖", layout="wide")

# Colour scale for grading scores.
_GREEN, _AMBER, _RED = "#16a34a", "#d97706", "#dc2626"


# --------------------------------------------------------------------------- #
# Loading helpers (cached, keyed by file mtime so they refresh on new runs)
# --------------------------------------------------------------------------- #
def _mtime(path) -> float:
    path = Path(path)
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_bundle(_mtime: float) -> dict | None:
    if not config.DASHBOARD_DATA.exists():
        return None
    return config.read_json(config.DASHBOARD_DATA)


@st.cache_data(show_spinner=False)
def _load_parquet(path: str, _mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def _load_model(path: str, _mtime: float):
    return joblib.load(path)


def _manifest() -> dict | None:
    if config.MANIFEST_FILE.exists():
        return config.read_json(config.MANIFEST_FILE)
    return None


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def _grade(value: float) -> tuple[str, str]:
    """Map a 0–1 score (R²/accuracy/F1) to (colour, label)."""
    if value >= 0.8:
        return _GREEN, "Good"
    if value >= 0.5:
        return _AMBER, "Fair"
    return _RED, "Poor"


def _graded_metric(container, title: str, value: float, fmt: str = "{:.3f}") -> None:
    colour, label = _grade(value)
    container.metric(title, fmt.format(value))
    container.markdown(
        f"<span style='color:{colour};font-weight:600'>● {label}</span>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _sidebar() -> None:
    st.sidebar.title("🤖 Gideon")
    st.sidebar.caption("Local automated ML pipeline")

    auto = st.sidebar.checkbox("Auto-refresh", value=True, key="auto_refresh")
    interval = st.sidebar.slider(
        "Refresh interval (s)", 2, 30, 5, disabled=not auto, key="refresh_interval"
    )
    # st_autorefresh schedules a *websocket* rerun (not a full page reload), so
    # widget/session state is preserved and it stops cleanly when unchecked.
    if auto:
        st_autorefresh(interval=interval * 1000, key="auto_refresh_tick")
    if st.sidebar.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

    manifest = _manifest()
    if manifest:
        status = manifest.get("status", "unknown")
        emoji = {"running": "⏳", "success": "✅", "failed": "❌"}.get(status, "❔")
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Pipeline:** {emoji} `{status}`")
        st.sidebar.caption(f"Source: {manifest.get('source_name', '—')}")
        if manifest.get("duration_seconds") is not None:
            st.sidebar.caption(f"Last run: {manifest['duration_seconds']}s")
        with st.sidebar.expander("Stage timeline"):
            for stage in manifest.get("stages", []):
                mark = {"success": "✅", "failed": "❌", "running": "⏳"}.get(stage["status"], "•")
                dur = stage.get("duration_seconds")
                st.write(f"{mark} {stage['name']} " + (f"({dur}s)" if dur else ""))
                if stage.get("error"):
                    st.error(stage["error"])

    st.sidebar.markdown("---")
    with st.sidebar.expander("⚠️ Reset"):
        st.caption("Wipe all analysis and return the dashboard to a clean slate.")
        also_inbox = st.checkbox("Also remove CSVs from the inbox", value=True,
                                 key="reset_inbox")
        confirm = st.checkbox("I understand this clears the current results",
                              key="reset_confirm")
        if st.button("Reset now", type="primary", disabled=not confirm):
            config.clear_artifacts()
            removed = config.clear_inbox() if also_inbox else 0
            st.cache_data.clear()
            st.cache_resource.clear()
            st.toast(f"Reset complete — removed {removed} inbox file(s).")
            st.rerun()


# --------------------------------------------------------------------------- #
# Data access for predictions
# --------------------------------------------------------------------------- #
def _predictions(bundle: dict) -> pd.DataFrame | None:
    path = bundle.get("artifacts", {}).get("predictions_parquet")
    if not path or not Path(path).exists():
        return None
    df = _load_parquet(path, _mtime(path))
    if "row" in df.columns:
        df = df.sort_values("row").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Tab: Overview
# --------------------------------------------------------------------------- #
def _tab_overview(bundle: dict) -> None:
    ds, task, deploy = bundle["dataset"], bundle["task"], bundle["deployment"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ds['rows']:,}")
    c2.metric("Columns", ds["columns"])
    c3.metric("Task", str(task["type"]).title())
    headline = bundle["metrics"]
    _graded_metric(c4, headline["headline_metric"].upper(), headline["headline_value"])

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("Target")
        st.write(f"**Column:** `{task['target']}`")
        st.write(f"**Type:** {task['type']}")
        if task.get("n_classes"):
            st.write(f"**Classes:** {task['n_classes']} — {', '.join(map(str, task['class_names']))}")
        st.write(f"**Features used:** {task['n_features']}")
    with right:
        st.subheader("Deployment")
        status = deploy.get("status", "unknown")
        badge = {"healthy": "🟢", "weak": "🟡"}.get(status, "⚪")
        st.write(f"**Status:** {badge} {status} — {deploy.get('status_reason', '')}")
        st.write(f"**Model:** `{deploy.get('model_type')}`")
        st.write(f"**Version:** `{deploy.get('version')}`")
        st.write(f"**Deployed at:** {deploy.get('deployed_at')}")


# --------------------------------------------------------------------------- #
# Tab: Data & EDA
# --------------------------------------------------------------------------- #
def _tab_data(bundle: dict) -> None:
    st.subheader("Cleaned data preview")
    preview = pd.DataFrame(bundle.get("preview_rows", []))
    if not preview.empty:
        st.dataframe(preview, use_container_width=True, height=320)

    clean = bundle.get("cleaning", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows kept", f"{clean.get('rows_after', 0):,}",
              f"{clean.get('rows_after', 0) - clean.get('rows_before', 0)}")
    c2.metric("Duplicate rows removed", clean.get("duplicate_rows_removed", 0))
    c3.metric("Columns dropped", len(clean.get("high_missing_columns_dropped", [])))

    st.markdown("---")
    summaries = bundle.get("column_summaries", [])
    st.subheader("Column summaries")
    sdf = pd.DataFrame([{k: v for k, v in s.items() if k not in ("histogram", "top_values")}
                       for s in summaries])
    # Some stat columns (min/max) mix floats and datetime-strings across rows;
    # stringify object columns so Arrow can serialise the table.
    for col in sdf.columns:
        if sdf[col].dtype == object:
            sdf[col] = sdf[col].astype(str)
    st.dataframe(sdf, use_container_width=True)

    st.subheader("Distribution")
    names = [s["name"] for s in summaries]
    chosen = st.selectbox("Column", names, key="eda_col") if names else None
    if chosen:
        info = next(s for s in summaries if s["name"] == chosen)
        if info["kind"] == "numeric" and info.get("histogram"):
            edges = info["histogram"]["edges"]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
            st.plotly_chart(
                px.bar(x=centers, y=info["histogram"]["counts"], labels={"x": chosen, "y": "count"}),
                use_container_width=True,
            )
        elif info["kind"] == "categorical" and info.get("top_values"):
            tv = info["top_values"]
            st.plotly_chart(
                px.bar(x=list(tv.keys()), y=list(tv.values()), labels={"x": chosen, "y": "count"}),
                use_container_width=True,
            )
        else:
            st.info("No chart available for this column type.")

    with st.expander("Cleaning report (details)"):
        st.json(clean)


# --------------------------------------------------------------------------- #
# Tab: Model health
# --------------------------------------------------------------------------- #
def _tab_model_health(bundle: dict) -> None:
    metrics = bundle["metrics"]
    task = metrics["task_type"]
    st.subheader("Performance (holdout set)")

    if task == "regression":
        c1, c2, c3, c4 = st.columns(4)
        _graded_metric(c1, "R²", metrics.get("r2", 0.0))
        c2.metric("RMSE", f"{metrics.get('rmse', 0):,.3f}")
        c3.metric("MAE", f"{metrics.get('mae', 0):,.3f}")
        if "mape" in metrics:
            c4.metric("MAPE %", f"{metrics['mape']:.2f}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        _graded_metric(c1, "Accuracy", metrics.get("accuracy", 0.0))
        _graded_metric(c2, "F1", metrics.get("f1_weighted", 0.0))
        c3.metric("Precision", f"{metrics.get('precision_weighted', 0):.3f}")
        c4.metric("Recall", f"{metrics.get('recall_weighted', 0):.3f}")
        for key in ("roc_auc", "roc_auc_ovr_weighted"):
            if key in metrics:
                st.caption(f"{key}: {metrics[key]:.3f}")

    st.markdown("---")
    preds = _predictions(bundle)
    if preds is None or preds.empty:
        st.info("No predictions available yet.")
        return

    if task == "regression" and {"actual", "predicted"} <= set(preds.columns):
        st.subheader("Rolling RMSE over holdout sequence")
        window = max(5, len(preds) // 10)
        sq_err = (preds["actual"] - preds["predicted"]) ** 2
        rolling_rmse = np.sqrt(sq_err.rolling(window, min_periods=1).mean())
        fig = px.line(x=preds.index, y=rolling_rmse,
                      labels={"x": "holdout row order", "y": f"rolling RMSE (window={window})"})
        fig.add_hline(y=metrics.get("rmse", 0), line_dash="dot", line_color=_AMBER,
                      annotation_text="overall RMSE")
        st.plotly_chart(fig, use_container_width=True)
    elif task == "classification" and "correct" in preds.columns:
        st.subheader("Rolling accuracy over holdout sequence")
        window = max(5, len(preds) // 10)
        rolling_acc = preds["correct"].astype(float).rolling(window, min_periods=1).mean()
        fig = px.line(x=preds.index, y=rolling_acc,
                      labels={"x": "holdout row order", "y": f"rolling accuracy (window={window})"})
        fig.add_hline(y=metrics.get("accuracy", 0), line_dash="dot", line_color=_AMBER,
                      annotation_text="overall accuracy")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    if task == "classification":
        cm = metrics.get("confusion_matrix")
        if cm:
            labels = bundle["task"].get("class_names") or metrics.get("labels")
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                colorscale="Blues", text=cm, texttemplate="%{text}"))
            fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab: Predictions (actual vs predicted + anomalies)
# --------------------------------------------------------------------------- #
def _tab_predictions(bundle: dict) -> None:
    preds = _predictions(bundle)
    if preds is None or preds.empty:
        st.info("No predictions available yet.")
        return
    task = bundle["task"]["type"]

    if task == "regression" and {"actual", "predicted"} <= set(preds.columns):
        residual = preds["actual"] - preds["predicted"]
        mu, sigma = residual.mean(), residual.std(ddof=0)
        threshold = 2 * sigma
        anomaly = (residual - mu).abs() > threshold
        n_anom = int(anomaly.sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Holdout points", len(preds))
        c2.metric("Anomalies (>2σ)", n_anom)
        c3.metric("Anomaly rate", f"{(n_anom / len(preds) * 100):.1f}%")

        st.subheader("Actual vs predicted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=preds.index, y=preds["actual"], mode="lines", name="actual"))
        fig.add_trace(go.Scatter(x=preds.index, y=preds["predicted"], mode="lines",
                                 name="predicted", line=dict(dash="dot")))
        if n_anom:
            fig.add_trace(go.Scatter(
                x=preds.index[anomaly], y=preds["actual"][anomaly], mode="markers",
                name="anomaly (>2σ)", marker=dict(color=_RED, size=9, symbol="x")))
        fig.update_layout(xaxis_title="holdout row order", yaxis_title=bundle["task"]["target"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predictions table")
        show = preds.copy()
        show["residual"] = residual
        show["anomaly"] = anomaly
        st.dataframe(show, use_container_width=True, height=320)

    elif task == "classification" and {"actual", "predicted"} <= set(preds.columns):
        mism = preds["actual"].astype(str) != preds["predicted"].astype(str)
        n_anom = int(mism.sum())
        low_conf = None
        if "confidence" in preds.columns:
            low_conf = int((preds["confidence"] < 0.5).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Holdout points", len(preds))
        c2.metric("Misclassifications", n_anom)
        if low_conf is not None:
            c3.metric("Low-confidence (<50%)", low_conf)

        st.subheader("Actual vs predicted (misclassifications in red)")
        class_names = bundle["task"].get("class_names")
        order = {c: i for i, c in enumerate(class_names)} if class_names else None
        if order:
            act_code = preds["actual"].astype(str).map(order)
            pred_code = preds["predicted"].astype(str).map(order)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=preds.index, y=act_code, mode="lines+markers",
                                     name="actual", opacity=0.6))
            fig.add_trace(go.Scatter(x=preds.index, y=pred_code, mode="markers",
                                     name="predicted", marker=dict(size=6)))
            if n_anom:
                fig.add_trace(go.Scatter(x=preds.index[mism], y=act_code[mism], mode="markers",
                                         name="misclassified", marker=dict(color=_RED, size=10, symbol="x")))
            fig.update_layout(
                yaxis=dict(tickmode="array", tickvals=list(order.values()),
                           ticktext=list(order.keys())),
                xaxis_title="holdout row order")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predictions table")
        st.dataframe(preds, use_container_width=True, height=320)


# --------------------------------------------------------------------------- #
# Tab: Forecast (regression only)
# --------------------------------------------------------------------------- #
def _tab_forecast(bundle: dict) -> None:
    task = bundle["task"]["type"]
    target = bundle["task"]["target"]
    if task != "regression":
        st.info("Forecasting applies to **regression** targets. This dataset is a "
                f"`{task}` task, so the Forecast view is not available.")
        return

    cleaned_path = bundle.get("artifacts", {}).get("cleaned_parquet")
    if not cleaned_path or not Path(cleaned_path).exists():
        st.info("Cleaned data not available.")
        return
    cleaned = _load_parquet(cleaned_path, _mtime(cleaned_path))
    if target not in cleaned.columns:
        st.info("Target column not found in cleaned data.")
        return

    y = pd.to_numeric(cleaned[target], errors="coerce").dropna().reset_index(drop=True)
    if len(y) < 5:
        st.info("Not enough data to forecast.")
        return

    c1, c2 = st.columns(2)
    horizon = c1.slider("Forecast horizon (steps)", 1, 12, 6, key="fc_horizon")
    spike_pct = c2.slider("Spike alert threshold (%)", 1, 50, 10, key="fc_spike")

    n = len(y)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y.values, 1)
    future_x = np.arange(n, n + horizon)
    forecast = intercept + slope * future_x

    rmse = float(bundle["metrics"].get("rmse", 0.0))
    band = 1.5 * rmse

    last_actual = float(y.iloc[-1])
    change = float(forecast[-1]) - last_actual
    pct_change = (change / abs(last_actual) * 100) if last_actual != 0 else 0.0

    if abs(change) <= max(band, 1e-9):
        direction, arrow, colour = "Stable", "→", _AMBER
    elif change > 0:
        direction, arrow, colour = "Up", "↑", _GREEN
    else:
        direction, arrow, colour = "Down", "↓", _RED

    if abs(pct_change) > spike_pct:
        st.warning(f"⚠️ Spike alert: forecast changes by {pct_change:+.1f}% over "
                   f"{horizon} steps (threshold ±{spike_pct}%).")

    m1, m2, m3 = st.columns(3)
    m1.metric("Last actual", f"{last_actual:,.2f}")
    m2.metric(f"Forecast (+{horizon})", f"{forecast[-1]:,.2f}", f"{change:,.2f}")
    m3.markdown(f"**Direction**<br><span style='color:{colour};font-size:1.6rem;"
                f"font-weight:700'>{arrow} {direction}</span>", unsafe_allow_html=True)

    tail = min(n, 60)
    hist_x = list(range(n - tail, n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=y.values[-tail:], mode="lines", name="actual"))
    fc_x = [n - 1] + list(future_x)
    fc_y = [last_actual] + list(forecast)
    fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode="lines+markers", name="forecast",
                             line=dict(dash="dash", color=colour)))
    upper = [last_actual] + list(forecast + band)
    lower = [last_actual] + list(forecast - band)
    fig.add_trace(go.Scatter(x=fc_x + fc_x[::-1], y=upper + lower[::-1], fill="toself",
                             fillcolor="rgba(99,102,241,0.15)", line=dict(width=0),
                             name="±1.5×RMSE", hoverinfo="skip"))
    fig.update_layout(xaxis_title="row order", yaxis_title=target)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Forecast is a linear trend projection of the target's recent trajectory; "
               "the band is ±1.5×RMSE from the holdout evaluation.")


# --------------------------------------------------------------------------- #
# Tab: Feature importance
# --------------------------------------------------------------------------- #
def _tab_importance(bundle: dict) -> None:
    imp = bundle.get("feature_importances", {})
    st.subheader("Feature importance")
    if not imp:
        st.info("No feature importances available.")
        return

    items = list(imp.items())  # already sorted descending
    top_name, top_val = items[0]
    total = sum(v for _, v in items) or 1.0
    runners = ", ".join(f"`{k}`" for k, _ in items[1:3])
    st.markdown(
        f"**Summary:** the most influential feature is **`{top_name}`** "
        f"({top_val / total * 100:.1f}% of the shown importance)"
        + (f", followed by {runners}." if runners else ".")
    )

    rev = items[::-1]
    names = [k for k, _ in rev]
    vals = [v for _, v in rev]
    colours = [_GREEN if k == top_name else "#6366f1" for k in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colours))
    fig.update_layout(height=max(300, 22 * len(names)), xaxis_title="importance", yaxis_title="feature")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab: What-if simulator
# --------------------------------------------------------------------------- #
_MAX_SLIDERS = 12


def _tab_whatif(bundle: dict) -> None:
    st.subheader("What-if simulator")
    model_info = bundle.get("model", {})
    finputs = bundle.get("feature_inputs", {})
    feats_order = model_info.get("features")
    model_path = model_info.get("path")

    if not feats_order or not finputs or not model_path or not Path(model_path).exists():
        st.info("Model or feature metadata not available.")
        return

    model = _load_model(model_path, _mtime(model_path))
    baseline = {f: finputs.get(f, {}).get("mean", 0.0) for f in feats_order}

    st.caption("Adjust the most influential features; the rest are held at their mean. "
               "The prediction updates live.")
    values = dict(baseline)
    slider_feats = [f for f in finputs.keys() if finputs[f]["max"] > finputs[f]["min"]][:_MAX_SLIDERS]
    cols = st.columns(2)
    for i, f in enumerate(slider_feats):
        info = finputs[f]
        lo, hi, mean = float(info["min"]), float(info["max"]), float(info["mean"])
        step = (hi - lo) / 100 or None
        with cols[i % 2]:
            values[f] = st.slider(f, lo, hi, mean, step=step, key=f"wi_{f}")

    row = pd.DataFrame([[values.get(f, 0.0) for f in feats_order]], columns=feats_order)
    base_row = pd.DataFrame([[baseline.get(f, 0.0) for f in feats_order]], columns=feats_order)

    st.markdown("---")
    task = bundle["task"]["type"]
    if task == "classification":
        class_names = bundle["task"].get("class_names")
        pred = int(model.predict(row)[0])
        base_pred = int(model.predict(base_row)[0])
        pname = class_names[pred] if class_names else str(pred)
        bname = class_names[base_pred] if class_names else str(base_pred)
        c1, c2 = st.columns(2)
        c1.metric("Predicted class", pname, delta=None if pname == bname else f"baseline: {bname}")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            c2.metric("Confidence", f"{float(np.max(proba)):.1%}")
            st.bar_chart(pd.DataFrame({"probability": proba},
                                      index=[str(c) for c in (class_names or range(len(proba)))]))
    else:
        pred = float(model.predict(row)[0])
        base = float(model.predict(base_row)[0])
        st.metric(f"Predicted {bundle['task']['target']}", f"{pred:,.3f}",
                  delta=f"{pred - base:,.3f} vs baseline")


# --------------------------------------------------------------------------- #
# Tab: KPIs & Insights
# --------------------------------------------------------------------------- #
def _fmt(v: float) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1_000_000:
        return f"{v/1_000_000:,.2f}M"
    if a >= 1_000:
        return f"{v/1_000:,.1f}K"
    return f"{v:,.2f}"


def _delta(pct: float | None) -> str | None:
    return None if pct is None else f"{pct:+.1f}%"


def _tab_kpis(bundle: dict) -> None:
    kpis = bundle.get("kpis", {})
    insights = bundle.get("insights", {})

    st.subheader("Business KPIs")
    if kpis.get("available"):
        s = kpis["summary"]
        metric, date_col = kpis["metric_column"], kpis["date_column"]
        st.caption(f"Tracking **{metric}** (monthly total) over **{date_col}** "
                   f"— {s['n_months']} months.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Latest ({s['latest_period']})", _fmt(s["latest_value"]),
                  _delta(s.get("mom_pct")), help="Δ vs previous month (MoM)")
        c2.metric("YoY", _delta(s.get("yoy_pct")) or "—",
                  help="vs same month last year")
        arrow = {"up": "↑", "down": "↓", "flat": "→"}[s["trend"]]
        c3.metric("Overall trend", f"{arrow} {s['trend'].title()}")
        c4.metric("Total", _fmt(s["total"]))

        b1, b2, b3 = st.columns(3)
        b1.metric(f"Best ({s['best_period']})", _fmt(s["best_value"]))
        b2.metric(f"Worst ({s['worst_period']})", _fmt(s["worst_value"]))
        if s.get("biggest_rise") and s.get("biggest_drop"):
            b3.metric("Biggest MoM swing",
                      f"{s['biggest_rise']['pct']:+.0f}% / {s['biggest_drop']['pct']:+.0f}%",
                      help=f"rise in {s['biggest_rise']['period']}, "
                           f"drop in {s['biggest_drop']['period']}")

        agg = st.radio("Monthly aggregation", ["total", "average", "count"],
                       horizontal=True, key="kpi_agg")
        series = {"total": kpis["values_sum"], "average": kpis["values_mean"],
                  "count": kpis["values_count"]}[agg]
        periods = kpis["periods"]

        st.subheader(f"{metric} — monthly {agg} (ups & downs)")
        df = pd.DataFrame({"period": periods, "value": series})
        df["change"] = df["value"].diff()
        colours = ["#94a3b8"] + [(_GREEN if c >= 0 else _RED) for c in df["change"][1:]]
        fig = go.Figure(go.Bar(x=df["period"], y=df["value"], marker_color=colours))
        fig.add_trace(go.Scatter(x=df["period"], y=df["value"], mode="lines",
                                 line=dict(color="#6366f1", width=2), name="trend"))
        fig.update_layout(showlegend=False, xaxis_title="month", yaxis_title=f"{agg} {metric}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Month-over-month change (%)")
        mom = kpis["mom_pct"]
        mom_colours = [(_GREEN if (m or 0) >= 0 else _RED) for m in mom]
        fig2 = go.Figure(go.Bar(x=periods, y=[m if m is not None else 0 for m in mom],
                                marker_color=mom_colours))
        fig2.add_hline(y=0, line_color="#475569")
        fig2.update_layout(xaxis_title="month", yaxis_title="MoM %")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"⏳ Time-based KPIs unavailable: {kpis.get('reason', 'no date column')}.")

    st.markdown("---")
    st.subheader("Interesting correlations")
    top = insights.get("top_correlations", [])
    tgt = insights.get("target_correlations", [])
    if not top and not tgt:
        st.info("No strong correlations found in this dataset.")
        return

    if tgt:
        st.markdown(f"**What moves the target (`{bundle['task']['target']}`):**")
        for item in tgt:
            st.markdown(f"- {item['text']}")
    if top:
        st.markdown("**Strongest relationships between columns:**")
        for item in top:
            st.markdown(f"- {item['text']}")
        names = [f"{p['a']} ↔ {p['b']}" for p in top][::-1]
        vals = [abs(p["r"]) for p in top][::-1]
        signs = [_GREEN if p["r"] > 0 else _RED for p in top][::-1]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=signs))
        fig.update_layout(height=max(250, 40 * len(names)),
                          xaxis_title="|correlation|", xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab: Correlations
# --------------------------------------------------------------------------- #
def _tab_correlations(bundle: dict) -> None:
    corr = bundle.get("correlations")
    st.subheader("Correlation matrix (numeric columns + target)")
    if not corr:
        st.info("Not enough numeric columns to compute correlations.")
        return
    z = np.array(corr["matrix"])
    fig = go.Figure(data=go.Heatmap(
        z=z, x=corr["columns"], y=corr["columns"], zmin=-1, zmax=1, colorscale="RdBu",
        text=np.round(z, 2), texttemplate="%{text}", colorbar=dict(title="r")))
    fig.update_layout(height=max(450, 32 * len(corr["columns"])))
    st.plotly_chart(fig, use_container_width=True)
    if corr.get("target"):
        st.caption(f"Target column `{corr['target']}` is included in the matrix.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _sidebar()
    st.title("Gideon")

    # When the inbox has no datasets, show a clean waiting state instead of
    # lingering on the previous analysis — even if stale artifacts remain on disk.
    has_input = bool(config.inbox_csvs())
    bundle = _load_bundle(_mtime(config.DASHBOARD_DATA))
    if not has_input or bundle is None:
        st.info("⏳ Waiting for a dataset. Drop a CSV into the `inbox/` folder to begin.")
        if not has_input and bundle is not None:
            st.caption("The inbox is empty, so previous results are hidden. "
                       "Add a CSV to run the pipeline again.")
        manifest = _manifest()
        if has_input and manifest and manifest.get("status") == "running":
            st.warning("A pipeline run is in progress…")
        return

    st.caption(f"Dataset: **{bundle['dataset']['name']}**  •  generated {bundle.get('generated_at', '')}")

    tabs = st.tabs([
        "Overview", "KPIs & Insights", "Data & EDA", "Model health", "Predictions",
        "Forecast", "Feature importance", "What-if", "Correlations",
    ])
    with tabs[0]:
        _tab_overview(bundle)
    with tabs[1]:
        _tab_kpis(bundle)
    with tabs[2]:
        _tab_data(bundle)
    with tabs[3]:
        _tab_model_health(bundle)
    with tabs[4]:
        _tab_predictions(bundle)
    with tabs[5]:
        _tab_forecast(bundle)
    with tabs[6]:
        _tab_importance(bundle)
    with tabs[7]:
        _tab_whatif(bundle)
    with tabs[8]:
        _tab_correlations(bundle)


main()
