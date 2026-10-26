"""EMIPredict AI home page. Classification is not yet available."""
from pathlib import Path

import streamlit as st

from app_utils import APP_ROOT, get_artifact_dir, read_json

st.set_page_config(page_title="EMIPredict AI", page_icon="📊", layout="wide")
st.title("EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown(
    """
This application currently provides **maximum affordable monthly EMI regression**
using the team's trained Random Forest pipeline. The eligibility classifier has
not been implemented yet, so this version does not claim to predict eligibility.

Use the sidebar to:
- **Predict**: estimate maximum affordable monthly EMI for an applicant.
- **Data Explorer**: inspect a sample of the cleaned EMI dataset.
- **Model Performance**: review validation comparisons and held-out test metrics.
- **Data Management**: manage a session-local sample and export changes.
"""
)

artifact_dir = get_artifact_dir()
info = read_json(artifact_dir / "model_info.json")
metrics = read_json(artifact_dir / "test_metrics.json")
quality = read_json(APP_ROOT / "reports" / "data_quality_report.json")

st.divider()
c1, c2, c3, c4 = st.columns(4)

model_name = info.get("model_name", "Regression model") if info else "Regression model"
c1.metric("Active task", "EMI regression")
c2.metric("Selected model", model_name)

if metrics:
    c3.metric("Test R²", f"{metrics.get('r2', float('nan')):.4f}")
    c4.metric("Test MAE", f"₹{metrics.get('mae', float('nan')):,.0f}")
else:
    c3.metric("Test R²", "Not available")
    c4.metric("Test MAE", "Not available")

if info:
    st.caption(
        f"Target: `{info.get('target', 'max_monthly_emi')}` · "
        f"Expected raw input features: {len(info.get('input_features', [])) or 25}"
    )
else:
    st.warning(
        f"Model metadata was not found in `{artifact_dir}`. The app can still load "
        "if the model path is configured, but the model artifacts should be checked."
    )

if quality:
    after = quality.get("after_cleaning", [])
    clean_log = quality.get("cleaning_log", {})
    if isinstance(after, list):
        rows_after = sum(int(part.get("rows", 0)) for part in after if isinstance(part, dict))
    elif isinstance(after, dict):
        rows_after = int(after.get("rows", 0))
    else:
        rows_after = 0
    rows_dropped = int(clean_log.get("rows_dropped_total", 0) or 0)
    q1, q2 = st.columns(2)
    q1.metric("Rows after cleaning", f"{rows_after:,}" if rows_after else "Not available")
    q2.metric("Rows removed during cleaning", f"{rows_dropped:,}" if rows_dropped else "0")
else:
    st.info(
        "Data-quality report not found at `reports/data_quality_report.json`. "
        "This does not prevent regression predictions."
    )

st.divider()
st.subheader("Current model evaluation")
if metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"₹{metrics.get('rmse', float('nan')):,.2f}")
    m2.metric("MAE", f"₹{metrics.get('mae', float('nan')):,.2f}")
    m3.metric("R²", f"{metrics.get('r2', float('nan')):.4f}")
    m4.metric("MAPE", f"{metrics.get('mape_pct', float('nan')):.2f}%")
    st.caption(
        "Metrics are from the held-out test split. They describe aggregate performance, "
        "not a guarantee that any individual prediction will be accurate."
    )
else:
    st.info(f"No `test_metrics.json` found in `{artifact_dir}` yet.")
