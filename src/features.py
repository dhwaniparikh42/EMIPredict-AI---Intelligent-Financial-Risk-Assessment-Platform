"""Feature engineering + preprocessing.

Everything is wrapped in ONE sklearn Pipeline that accepts the RAW columns, so
the exact same object can be used for training here and for real-time
prediction in the Streamlit app later (no separate "prepare input" code).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

EXPENSE_COLS = ["monthly_rent", "school_fees", "college_fees", "travel_expenses",
                "groceries_utilities", "other_monthly_expenses"]

BASE_NUMERIC = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size",
    "dependents", "school_fees", "college_fees", "travel_expenses",
    "groceries_utilities", "other_monthly_expenses", "current_emi_amount",
    "credit_score", "bank_balance", "emergency_fund", "requested_amount",
    "requested_tenure",
]
ENGINEERED = [
    "total_expenses", "requested_emi_est", "expense_to_income", "debt_to_income",
    "new_debt_to_income", "disposable_income", "affordability_ratio",
    "emi_to_disposable", "loan_to_annual_income", "savings_to_income",
    "emergency_fund_months", "dependents_ratio", "income_per_member",
    "employment_stability", "credit_band", "salary_x_credit",
]
NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED
CATEGORICAL_FEATURES = ["gender", "marital_status", "education", "employment_type",
                        "company_type", "house_type", "existing_loans", "emi_scenario"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived financial ratios, risk scores and interactions (no target used)."""
    d = df.copy()
    sal = d["monthly_salary"].clip(lower=1)

    d["total_expenses"] = d[EXPENSE_COLS].sum(axis=1)
    d["requested_emi_est"] = d["requested_amount"] / d["requested_tenure"].clip(lower=1)

    # --- ratios
    d["expense_to_income"] = d["total_expenses"] / sal
    d["debt_to_income"] = d["current_emi_amount"] / sal
    d["new_debt_to_income"] = (d["current_emi_amount"] + d["requested_emi_est"]) / sal
    d["disposable_income"] = d["monthly_salary"] - d["total_expenses"] - d["current_emi_amount"]
    d["affordability_ratio"] = d["disposable_income"] / sal
    d["emi_to_disposable"] = d["requested_emi_est"] / d["disposable_income"].clip(lower=1)
    d["loan_to_annual_income"] = d["requested_amount"] / (sal * 12)
    d["savings_to_income"] = (d["bank_balance"] + d["emergency_fund"]) / sal
    d["emergency_fund_months"] = d["emergency_fund"] / (d["total_expenses"] + d["current_emi_amount"]).clip(lower=1)
    d["dependents_ratio"] = d["dependents"] / d["family_size"].clip(lower=1)
    d["income_per_member"] = d["monthly_salary"] / d["family_size"].clip(lower=1)

    # --- risk scores
    d["employment_stability"] = d["years_of_employment"] / (d["age"] - 18).clip(lower=1)
    d["credit_band"] = pd.cut(d["credit_score"], [-np.inf, 579, 669, 739, 799, np.inf],
                              labels=[0, 1, 2, 3, 4]).astype(float)  # poor .. exceptional

    # --- interaction
    d["salary_x_credit"] = d["monthly_salary"] * d["credit_score"] / 1e6
    return d


def build_pipeline(model) -> Pipeline:
    """raw columns -> engineered features -> scale/one-hot -> model"""
    prep = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("fe", FunctionTransformer(add_features, validate=False)),
        ("prep", prep),
        ("model", model),
    ])
