"""Exploratory charts for the generated EMI sample."""
import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import load_sample_data, get_sample_path

st.set_page_config(page_title="Data Explorer - EMIPredict AI", layout="wide")
st.title("Data Explorer")
st.caption("Charts use a sample of the cleaned dataset for responsiveness.")

try:
    df = load_sample_data()
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.code(
        "python scripts/build_frontend_sample.py "
        "--input /path/to/backend/data/processed/train.parquet "
        "--output data/EMI_sample.csv",
        language="bash",
    )
    st.stop()

required = {
    "emi_scenario", "emi_eligibility", "max_monthly_emi", "monthly_salary",
    "credit_score", "education", "years_of_employment", "bank_balance",
    "emergency_fund", "current_emi_amount",
}
missing = sorted(required - set(df.columns))
if missing:
    st.error(f"Sample dataset is missing required columns: {', '.join(missing)}")
    st.stop()

st.caption(f"Loaded {len(df):,} rows from `{get_sample_path()}`.")
scenarios = sorted(df["emi_scenario"].dropna().unique().tolist())
scenario_filter = st.multiselect("Filter by EMI scenario", scenarios, default=scenarios)
df_view = df[df["emi_scenario"].isin(scenario_filter)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records in view", f"{len(df_view):,}")
c2.metric("Average monthly salary", f"₹{df_view['monthly_salary'].mean():,.0f}" if len(df_view) else "N/A")
c3.metric("Average credit score", f"{df_view['credit_score'].mean():.0f}" if len(df_view) else "N/A")
eligible_rate = (df_view["emi_eligibility"].astype(str) == "Eligible").mean() * 100 if len(df_view) else float("nan")
c4.metric("Eligible rate in sample", f"{eligible_rate:.1f}%" if len(df_view) else "N/A")

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader("Eligibility label distribution by scenario")
    ct = pd.crosstab(df_view["emi_scenario"], df_view["emi_eligibility"], normalize="index") * 100
    if not ct.empty:
        ct = ct.reset_index().melt(id_vars="emi_scenario", var_name="eligibility", value_name="percent")
        fig = px.bar(ct, x="emi_scenario", y="percent", color="eligibility", barmode="stack")
        fig.update_layout(xaxis_title="", yaxis_title="Percent of records")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rows match the selected scenarios.")

with col2:
    st.subheader("Maximum monthly EMI distribution")
    fig2 = px.histogram(df_view, x="max_monthly_emi", nbins=50, color="emi_eligibility")
    fig2.update_layout(xaxis_title="Maximum monthly EMI (INR)")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Salary vs maximum monthly EMI")
    if len(df_view):
        plot_df = df_view.sample(min(3_000, len(df_view)), random_state=1)
        fig3 = px.scatter(plot_df, x="monthly_salary", y="max_monthly_emi", color="emi_eligibility", opacity=0.5)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No rows to plot.")

with col4:
    st.subheader("Credit score by education")
    fig4 = px.box(df_view, x="education", y="credit_score", color="education")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Correlation with maximum monthly EMI")
numeric_cols = [
    c for c in [
        "monthly_salary", "years_of_employment", "credit_score", "bank_balance",
        "emergency_fund", "current_emi_amount", "max_monthly_emi",
    ] if c in df_view.columns
]
if len(df_view) and len(numeric_cols) >= 2:
    corr = df_view[numeric_cols].corr(numeric_only=True)
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.subheader("Sample records")
st.dataframe(df_view.head(200), use_container_width=True)
