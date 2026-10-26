"""Remote MLflow dashboard for experiments tracked in DagsHub."""
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# Importing app_utils loads the project's root .env before MLflow is initialized.
from app_utils import get_mlflow_tracking_uri, load_env_file

load_env_file()
st.set_page_config(page_title="MLflow | EMIPredict AI", layout="wide")
st.title("MLflow experiment dashboard")
st.caption("Live view of the experiments and model runs logged to DagsHub, not the local mlflow.db.")

default_uri = get_mlflow_tracking_uri()
with st.sidebar:
    st.subheader("DagsHub tracking connection")
    tracking_uri = st.text_input(
        "MLflow tracking URI",
        value=default_uri,
        help="Loaded from .env as MLFLOW_TRACKING_URI. Credentials stay in environment variables and are never displayed.",
    ).strip()
    st.caption("Credentials are read from MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD in the project .env.")
    max_runs = st.select_slider(
        "Maximum runs to load",
        options=[10, 25, 50, 100, 250, 500],
        value=100,
    )
    refresh = st.button("Refresh remote data", use_container_width=True)
    st.markdown("[Open DagsHub MLflow](https://dagshub.com/shahjahanIqbal/emipredict-ai.mlflow)")

try:
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient
except ImportError:
    st.warning(
        "MLflow is not installed in this environment. Install it with "
        "`python -m pip install mlflow`."
    )
    st.stop()

if refresh:
    st.cache_data.clear()


@st.cache_data(ttl=30, show_spinner="Connecting to DagsHub and loading experiments...")
def fetch_experiments(uri: str):
    client = MlflowClient(tracking_uri=uri)
    experiments = client.search_experiments(view_type=ViewType.ALL)
    return [
        {
            "experiment_id": str(exp.experiment_id),
            "name": exp.name,
            "lifecycle_stage": getattr(exp, "lifecycle_stage", "active"),
            "artifact_location": getattr(exp, "artifact_location", ""),
        }
        for exp in experiments
    ]


@st.cache_data(ttl=30, show_spinner="Loading tracked runs...")
def fetch_runs(uri: str, experiment_id: str, limit: int):
    client = MlflowClient(tracking_uri=uri)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ALL,
        max_results=limit,
        order_by=["attributes.start_time DESC"],
    )
    rows = []
    for run in runs:
        info, data = run.info, run.data
        tags = data.tags or {}
        start_time = getattr(info, "start_time", None)
        end_time = getattr(info, "end_time", None)
        rows.append({
            "run_id": info.run_id,
            "run_name": tags.get("mlflow.runName") or getattr(info, "run_name", None) or info.run_id[:8],
            "status": info.status,
            "start_time": (
                datetime.fromtimestamp(start_time / 1000).astimezone().isoformat(timespec="seconds")
                if start_time else None
            ),
            "duration_seconds": (
                round((end_time - start_time) / 1000, 2)
                if start_time and end_time else None
            ),
            "artifact_uri": getattr(info, "artifact_uri", ""),
            **{f"metric.{key}": value for key, value in (data.metrics or {}).items()},
            **{f"param.{key}": value for key, value in (data.params or {}).items()},
            **{f"tag.{key}": value for key, value in tags.items() if key != "mlflow.runName"},
        })
    return rows


@st.cache_data(ttl=30, show_spinner="Loading registered models...")
def fetch_registered_models(uri: str):
    client = MlflowClient(tracking_uri=uri)
    try:
        models = client.search_registered_models()
    except AttributeError:
        # Compatibility with older MLflow client versions.
        models = client.list_registered_models()
    records = []
    for model in models:
        versions = getattr(model, "latest_versions", []) or []
        records.append({
            "model_name": model.name,
            "latest_versions": ", ".join(
                f"v{v.version} ({getattr(v, 'current_stage', 'None')})" for v in versions
            ) or "No versions returned",
            "description": getattr(model, "description", "") or "",
        })
    return records


if not tracking_uri:
    st.error("No tracking URI configured. Set MLFLOW_TRACKING_URI in the root .env file.")
    st.stop()

try:
    experiments = fetch_experiments(tracking_uri)
except Exception as exc:
    st.error("Could not connect to the DagsHub MLflow tracking server.")
    with st.expander("Connection details"):
        st.code(str(exc))
    st.markdown(
        "Check that the root `.env` contains `MLFLOW_TRACKING_URI`, "
        "`MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD`. "
        "Confirm the token is valid and has access to the DagsHub repository."
    )
    st.stop()

st.caption(f"Tracking URI: `{tracking_uri}`")
if not experiments:
    st.info("No experiments were returned by DagsHub for this account.")
    st.stop()

exp_df = pd.DataFrame(experiments)
st.subheader("Remote experiments")
st.dataframe(exp_df, use_container_width=True, hide_index=True)

labels = {f"{row['name']} (ID: {row['experiment_id']})": row["experiment_id"] for row in experiments}
selected_label = st.selectbox("Choose an experiment", list(labels))
experiment_id = labels[selected_label]

try:
    run_rows = fetch_runs(tracking_uri, experiment_id, int(max_runs))
except Exception as exc:
    st.error("The experiment was found, but its runs could not be loaded.")
    with st.expander("Run-query details"):
        st.code(str(exc))
    st.stop()

if run_rows:
    runs_df = pd.DataFrame(run_rows)
    metric_cols = [col for col in runs_df.columns if col.startswith("metric.")]
    param_cols = [col for col in runs_df.columns if col.startswith("param.")]
    tag_cols = [col for col in runs_df.columns if col.startswith("tag.")]

    st.divider()
    st.subheader("Run summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Runs loaded", len(runs_df))
    m2.metric("Finished", int((runs_df["status"] == "FINISHED").sum()))
    m3.metric("Failed", int((runs_df["status"] == "FAILED").sum()))

    display_cols = [
        col for col in ["run_name", "status", "start_time", "duration_seconds", "run_id"]
        if col in runs_df.columns
    ] + metric_cols
    st.dataframe(runs_df[display_cols], use_container_width=True, hide_index=True)

    if metric_cols:
        st.subheader("Compare runs by metric")
        metric_labels = {col.removeprefix("metric."): col for col in metric_cols}
        metric_name = st.selectbox("Metric to compare", list(metric_labels))
        metric_col = metric_labels[metric_name]
        chart_df = runs_df[["run_name", "status", metric_col]].copy()
        chart_df[metric_col] = pd.to_numeric(chart_df[metric_col], errors="coerce")
        chart_df = chart_df.dropna(subset=[metric_col])
        if not chart_df.empty:
            # Metric direction differs: lower is better for loss/error metrics.
            lower_is_better = any(token in metric_name.lower() for token in ("loss", "rmse", "mae", "mape", "error"))
            chart_df = chart_df.sort_values(metric_col, ascending=lower_is_better)
            fig = px.bar(
                chart_df, x="run_name", y=metric_col, color="status",
                hover_data=["status"], title=f"Logged {metric_name} by run",
            )
            fig.update_layout(xaxis_title="Run", yaxis_title=metric_name)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No numeric values are available for this metric.")
    else:
        st.info("No metrics were logged for this experiment.")

    with st.expander("Run parameters and tags"):
        if param_cols:
            st.dataframe(runs_df[["run_name", "run_id"] + param_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No parameters were logged for these runs.")
        if tag_cols:
            st.dataframe(runs_df[["run_name", "run_id"] + tag_cols], use_container_width=True, hide_index=True)

    with st.expander("Artifact locations"):
        st.dataframe(
            runs_df[[col for col in ["run_name", "run_id", "artifact_uri"] if col in runs_df.columns]],
            use_container_width=True, hide_index=True,
        )
        st.caption("Artifact URIs are shown for reference. This page does not download or modify model artifacts.")
else:
    st.info("This experiment has no runs to display.")

st.divider()
st.subheader("DagsHub model registry")
try:
    registered = fetch_registered_models(tracking_uri)
    if registered:
        st.dataframe(pd.DataFrame(registered), use_container_width=True, hide_index=True)
    else:
        st.info("No registered models were returned. Run registration may be disabled for this tracking server.")
except Exception as exc:
    st.warning("Registered models could not be queried. Experiment tracking can still work.")
    with st.expander("Registry details"):
        st.code(str(exc))
