"""Smoke test for the DEPLOYABLE models committed in models/. Skipped if absent."""
import joblib
import pandas as pd
import pytest

from src import config as C

ROW = {"age": 35, "gender": "Male", "marital_status": "Married", "education": "Graduate",
       "monthly_salary": 60000.0, "employment_type": "Private", "years_of_employment": 8.0,
       "company_type": "MNC", "house_type": "Rented", "monthly_rent": 12000.0, "family_size": 4,
       "dependents": 2, "school_fees": 5000.0, "college_fees": 0.0, "travel_expenses": 3000.0,
       "groceries_utilities": 9000.0, "other_monthly_expenses": 4000.0, "existing_loans": "Yes",
       "current_emi_amount": 6000.0, "credit_score": 720.0, "bank_balance": 150000.0,
       "emergency_fund": 80000.0, "emi_scenario": "Personal Loan EMI",
       "requested_amount": 300000.0, "requested_tenure": 36}


def _load(name):
    p = C.ROOT / "models" / name
    if not p.exists():
        pytest.skip(f"{name} not committed yet")
    return joblib.load(p)


def test_classifier_predicts_valid_class_from_raw_row():
    b = _load("best_classifier.joblib")
    pred = b["pipeline"].predict(pd.DataFrame([ROW]))[0]
    assert b["classes"][pred] in C.VALID_CATEGORIES[C.TARGET_CLF]


def test_regressor_predicts_sane_emi_from_raw_row():
    b = _load("best_regressor.joblib")
    emi = float(b["pipeline"].predict(pd.DataFrame([ROW]))[0])
    assert -1000 < emi < 60_000
