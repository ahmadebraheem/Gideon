"""Gideon live dashboard (Streamlit).

This app is a pure *renderer*: every number and chart comes from
``artifacts/dashboard_data.json`` (produced by the dashboard_gen agent) and the
two Parquet artifacts it points to. It performs no modelling itself.

Launch::

    streamlit run gideon/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the `gideon` package importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from gideon import config  # noqa: E402

st.set_page_config(page_title="Gideon — Live ML Dashboard", page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (cached, keyed by file mtime so it refreshes on new runs)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_bundle(_mtime: float) -> dict | None:
    if not config.DASHBOARD_DATA.exists():
        return None
    return config.read_json(config.DASHBOARD_DATA)


@st.cache_data(show_spinner=False)
def _load_parquet(path: str, _mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _manifest() -> dict | None:
    if config.MANIFEST_FILE.exists():
        return config.read_json(config.MANIFEST_FILE)
    return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _sidebar() -> None:
    st.sidebar.title("🤖 Gideon")
    st.sidebar.caption("Local automated ML pipeline")

    auto = st.sidebar.checkbox("Auto-refresh", value=True)
    interval = st.sidebar.slider("Refresh interval (s)", 2, 30, 5, disabled=not auto)
    if auto:
        st.markdown(
            f'<meta http-equiv="refresh" content="{interval}">',
            unsafe_allow_html=True,
        )
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


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def _tab_overview(bundle: dict) -> None:
    ds, task, deploy = bundle["dataset"], bundle["task"], bundle["deployment"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ds['rows']:,}")
    c2.metric("Columns", ds["columns"])
    c3.metric("Task", str(task["type"]).title())
    headline = bundle["metrics"]
    c4.metric(headline["headline_metric"].upper(), f"{headline['headline_value']:.3f}")

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


def _tab_data(bundle: dict) -> None:
    st.subheader("Cleaned data preview")
    preview = pd.DataFrame(bundle.get("preview_rows", []))
    if not preview.empty:
        st.dataframe(preview, use_container_width=True, height=380)
    clean = bundle.get("cleaning", {})
    st.subheader("Cleaning report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows kept", f"{clean.get('rows_after', 0):,}", f"{clean.get('rows_after',0)-clean.get('rows_before',0)}")
    c2.metric("Duplicate rows removed", clean.get("duplicate_rows_removed", 0))
    c3.metric("Columns dropped", len(clean.get("high_missing_columns_dropped", [])))
    with st.expander("Details"):
        st.json(clean)


def _tab_eda(bundle: dict) -> None:
    summaries = bundle.get("column_summaries", [])
    st.subheader("Column summaries")
    st.dataframe(
        pd.DataFrame([{k: v for k, v in s.items() if k not in ("histogram", "top_values")}
                      for s in summaries]),
        use_container_width=True,
    )

    st.subheader("Distributions")
    names = [s["name"] for s in summaries]
    chosen = st.selectbox("Column", names) if names else None
    if chosen:
        info = next(s for s in summaries if s["name"] == chosen)
        if info["kind"] == "numeric" and info.get("histogram"):
            edges = info["histogram"]["edges"]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
            fig = px.bar(x=centers, y=info["histogram"]["counts"],
                         labels={"x": chosen, "y": "count"})
            st.plotly_chart(fig, use_container_width=True)
        elif info["kind"] == "categorical" and info.get("top_values"):
            tv = info["top_values"]
            fig = px.bar(x=list(tv.keys()), y=list(tv.values()),
                         labels={"x": chosen, "y": "count"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chart available for this column type.")

    corr = bundle.get("correlations")
    if corr:
        st.subheader("Numeric correlations")
        fig = go.Figure(data=go.Heatmap(
            z=corr["matrix"], x=corr["columns"], y=corr["columns"],
            zmin=-1, zmax=1, colorscale="RdBu",
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def _tab_model(bundle: dict) -> None:
    metrics = bundle["metrics"]
    task = metrics["task_type"]
    st.subheader("Performance (holdout set)")

    if task == "classification":
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        cols[1].metric("Precision", f"{metrics.get('precision_weighted', 0):.3f}")
        cols[2].metric("Recall", f"{metrics.get('recall_weighted', 0):.3f}")
        cols[3].metric("F1", f"{metrics.get('f1_weighted', 0):.3f}")
        for key in ("roc_auc", "roc_auc_ovr_weighted"):
            if key in metrics:
                st.caption(f"{key}: {metrics[key]:.3f}")
        cm = metrics.get("confusion_matrix")
        if cm:
            labels = bundle["task"].get("class_names") or metrics.get("labels")
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                colorscale="Blues", text=cm, texttemplate="%{text}",
            ))
            fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
    else:
        cols = st.columns(4)
        cols[0].metric("R²", f"{metrics.get('r2', 0):.3f}")
        cols[1].metric("MAE", f"{metrics.get('mae', 0):.3f}")
        cols[2].metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
        if "mape" in metrics:
            cols[3].metric("MAPE %", f"{metrics['mape']:.2f}")


def _tab_importance(bundle: dict) -> None:
    imp = bundle.get("feature_importances", {})
    st.subheader("Feature importance")
    if not imp:
        st.info("No feature importances available.")
        return
    items = list(imp.items())[::-1]
    fig = px.bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                 labels={"x": "importance", "y": "feature"})
    fig.update_layout(height=max(300, 22 * len(items)))
    st.plotly_chart(fig, use_container_width=True)


def _tab_predictions(bundle: dict) -> None:
    st.subheader("Predictions (holdout)")
    path = bundle.get("artifacts", {}).get("predictions_parquet")
    if not path or not Path(path).exists():
        st.info("No predictions available yet.")
        return
    preds = _load_parquet(path, _mtime(Path(path)))
    st.dataframe(preds, use_container_width=True, height=360)

    if bundle["task"]["type"] == "regression" and {"actual", "predicted"} <= set(preds.columns):
        fig = px.scatter(preds, x="actual", y="predicted", title="Actual vs Predicted")
        lo, hi = float(preds[["actual", "predicted"]].min().min()), float(preds[["actual", "predicted"]].max().max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(dash="dash"), name="ideal"))
        st.plotly_chart(fig, use_container_width=True)
    elif bundle["task"]["type"] == "classification" and "correct" in preds.columns:
        rate = preds["correct"].mean()
        st.metric("Holdout accuracy", f"{rate:.3f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _sidebar()
    st.title("Gideon — Live ML Dashboard")

    bundle = _load_bundle(_mtime(config.DASHBOARD_DATA))
    if bundle is None:
        st.info("⏳ Waiting for the first dataset. Drop a CSV into the `inbox/` folder to begin.")
        manifest = _manifest()
        if manifest and manifest.get("status") == "running":
            st.warning("A pipeline run is in progress…")
        return

    st.caption(
        f"Dataset: **{bundle['dataset']['name']}**  •  "
        f"generated {bundle.get('generated_at', '')}"
    )

    tabs = st.tabs(
        ["Overview", "Data", "EDA", "Model", "Feature importance", "Predictions"]
    )
    with tabs[0]:
        _tab_overview(bundle)
    with tabs[1]:
        _tab_data(bundle)
    with tabs[2]:
        _tab_eda(bundle)
    with tabs[3]:
        _tab_model(bundle)
    with tabs[4]:
        _tab_importance(bundle)
    with tabs[5]:
        _tab_predictions(bundle)


main()
