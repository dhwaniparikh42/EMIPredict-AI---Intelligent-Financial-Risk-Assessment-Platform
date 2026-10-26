"""Applicant-facing maximum affordable EMI prediction."""
import math

import pandas as pd
import streamlit as st

from app_utils import (
    RAW_COLS, SCENARIO_RULES, VALID_CATEGORIES, load_regression_model,
)

st.set_page_config(page_title="Predict - EMIPredict AI", layout="wide")
st.title("Maximum Monthly EMI Prediction")


try:
    model = load_regression_model()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Could not load the regression model. {exc}")
    st.stop()

with st.form("applicant_form"):
    st.subheader("Applicant details")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 25, 60, 35)
        gender = st.selectbox("Gender", VALID_CATEGORIES["gender"])
        marital_status = st.selectbox("Marital status", VALID_CATEGORIES["marital_status"])
        education = st.selectbox("Education", VALID_CATEGORIES["education"])
        employment_type = st.selectbox("Employment type", VALID_CATEGORIES["employment_type"])
        company_type = st.selectbox("Company type", VALID_CATEGORIES["company_type"])
        years_of_employment = st.number_input("Years of employment", 0.0, 40.0, 5.0, step=0.5)

    with c2:
        monthly_salary = st.number_input("Monthly salary (INR)", 15_000, 200_000, 50_000, step=1_000)
        house_type = st.selectbox("House type", VALID_CATEGORIES["house_type"])
        monthly_rent = st.number_input("Monthly rent (INR)", 0, 100_000, 0, step=500)
        family_size = st.slider("Family size", 1, 10, 3)
        dependents = st.slider("Dependents", 0, 9, 1)
        school_fees = st.number_input("School fees (INR/month)", 0, 100_000, 0, step=100)
        college_fees = st.number_input("College fees (INR/month)", 0, 100_000, 0, step=100)

    with c3:
        travel_expenses = st.number_input("Travel expenses (INR/month)", 0, 100_000, 3_000, step=100)
        groceries_utilities = st.number_input("Groceries & utilities (INR/month)", 0, 100_000, 8_000, step=100)
        other_monthly_expenses = st.number_input("Other monthly expenses (INR)", 0, 100_000, 2_000, step=100)
        existing_loans = st.selectbox("Existing loans", VALID_CATEGORIES["existing_loans"])
        current_emi_amount = st.number_input("Current EMI amount (INR)", 0, 200_000, 0, step=500)
        credit_score = st.slider("Credit score", 300, 850, 700)
        bank_balance = st.number_input("Bank balance (INR)", 0, 10_000_000, 100_000, step=1_000)

    st.subheader("Loan request")
    c4, c5, c6 = st.columns(3)
    with c4:
        emergency_fund = st.number_input("Emergency fund (INR)", 0, 10_000_000, 20_000, step=1_000)
    with c5:
        emi_scenario = st.selectbox("EMI scenario", VALID_CATEGORIES["emi_scenario"])
        min_amount, max_amount = SCENARIO_RULES[emi_scenario]["amount"]
        min_tenure, max_tenure = SCENARIO_RULES[emi_scenario]["tenure"]
        st.caption(f"Scenario request range: ₹{min_amount:,}–₹{max_amount:,}; {min_tenure}–{max_tenure} months.")
    with c6:
        requested_amount = st.number_input(
            "Requested loan amount (INR)", 10_000, 1_500_000,
            min(max(200_000, min_amount), max_amount), step=5_000,
        )
        requested_tenure = st.slider(
            "Requested tenure (months)", 3, 84,
            min(max(24, min_tenure), max_tenure),
        )
        annual_interest_rate = st.number_input(
            "Assumed annual interest rate (%)", min_value=0.1, max_value=50.0,
            value=12.0, step=0.5,
        )

    submitted = st.form_submit_button("Predict maximum monthly EMI", type="primary")

if submitted:
    applicant = pd.DataFrame([{
        "age": age,
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "company_type": company_type,
        "house_type": house_type,
        "existing_loans": existing_loans,
        "emi_scenario": emi_scenario,
    }], columns=RAW_COLS)

    if dependents >= family_size:
        st.warning("The dataset's business rules generally expect dependents to be fewer than family size.")
    if house_type != "Rented" and monthly_rent > 0:
        st.warning("The dataset's business rules expect monthly rent to be zero for non-rented housing.")
    if not (min_amount <= requested_amount <= max_amount):
        st.warning("The requested amount is outside the usual range for the selected EMI scenario.")
    if not (min_tenure <= requested_tenure <= max_tenure):
        st.warning("The requested tenure is outside the usual range for the selected EMI scenario.")

    try:
        prediction = float(model.predict(applicant)[0])
        if not math.isfinite(prediction):
            raise ValueError("The model returned a non-finite prediction.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    monthly_rate = annual_interest_rate / 1200.0
    if monthly_rate == 0:
        implied_emi = requested_amount / requested_tenure
    else:
        factor = (1 + monthly_rate) ** requested_tenure
        implied_emi = requested_amount * monthly_rate * factor / (factor - 1)

    st.divider()
    st.subheader("Prediction result")
    r1, r2 = st.columns(2)
    r1.metric("Estimated maximum affordable monthly EMI", f"₹{prediction:,.0f}")
    r2.metric("Illustrative EMI for requested loan", f"₹{implied_emi:,.0f}")

    if implied_emi > prediction:
        st.warning("The illustrative EMI is above the model's estimated affordable monthly EMI.")
    else:
        st.success("The illustrative EMI is within the model's estimated affordable monthly EMI.")

    st.caption(
        "This is a model estimate, not a lending decision or financial advice. "
        "The model was trained to predict `max_monthly_emi`; it does not classify eligibility."
    )
