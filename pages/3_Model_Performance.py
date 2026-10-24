"""Regression model metrics and candidate comparison."""
import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import get_artifact_dir, read_json

st.set_page_config(page_title="Model Performance - EMIPredict AI", layout="wide")
st.title("Model Performance")


artifact_dir = get_artifact_dir()
metrics = read_json(artifact_dir / "test_metrics.json")
info = read_json(artifact_dir / "model_info.json")
comparison_path = artifact_dir / "regression_comparison.csv"

if not metrics:
    st.error(f"Test metrics not found at `{artifact_dir / 'test_metrics.json'}`.")
    st.stop()

best_name = info.get("model_name", "Selected regression model") if info else "Selected regression model"
st.subheader(f"Selected model: {best_name}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Test RMSE", f"₹{metrics.get('rmse', float('nan')):,.2f}", help="Lower is better.")
m2.metric("Test MAE", f"₹{metrics.get('mae', float('nan')):,.2f}", help="Lower is better.")
m3.metric("Test R²", f"{metrics.get('r2', float('nan')):.4f}", help="Closer to 1 is generally better.")
m4.metric("Test MAPE", f"{metrics.get('mape_pct', float('nan')):.2f}%")

st.caption(
    "These metrics are from the held-out test split. MAPE can be unstable when actual target "
    "values are small, so interpret it alongside RMSE and MAE."
)

if comparison_path.is_file():
    comparison = pd.read_csv(comparison_path)
    if "model" not in comparison.columns and len(comparison.columns):
        comparison = comparison.rename(columns={comparison.columns[0]: "model"})
    st.divider()
    st.subheader("Candidate model comparison (validation split)")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if {"model", "rmse"}.issubset(comparison.columns):
        fig = px.bar(comparison.sort_values("rmse"), x="model", y="rmse", title="Validation RMSE (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
    if {"model", "r2"}.issubset(comparison.columns):
        fig2 = px.bar(comparison.sort_values("r2", ascending=False), x="model", y="r2", title="Validation R² (higher is better)")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info(f"Candidate comparison file not found: `{comparison_path}`.")

with st.expander("Model details"):
    st.write(f"**Target:** `{info.get('target', 'max_monthly_emi') if info else 'max_monthly_emi'}`")
    if info:
        st.write(f"**Validation RMSE:** {info.get('validation_rmse', float('nan')):,.2f}")
        st.write(f"**Input features:** {len(info.get('input_features', []))}")
        st.code("\n".join(info.get("input_features", [])) or "Feature list not available")
    st.info("Eligibility classification is intentionally marked unavailable until a classifier is trained.")
