"""
Feature engineering utility (app copy).

Provides add_derived_features() which computes derived financial ratios
from raw imputed columns.  Used by the Streamlit app for inference.

The training pipeline in preprocessing.py contains its own copy of this
logic inside DerivedFeaturesTransformer so that feature computation is
identical between training and serving.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

ANNUAL_INTEREST_RATE = 12.0
EDUCATION_ORDER = {"High School": 0, "Graduate": 1, "Post Graduate": 2, "Professional": 3}

RAW_NUMERIC_COLS = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
]
RAW_CATEGORICAL_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]


def add_derived_features(df):
    """Compute derived financial features from raw imputed columns.

    Expects the input to already have NaN values filled (no missing data).
    Returns a copy with new columns added.
    """
    df = df.copy()
    salary = df.monthly_salary.replace(0, np.nan)

    total_fixed_expenses = (
        df.monthly_rent + df.school_fees + df.college_fees + df.travel_expenses
        + df.groceries_utilities + df.other_monthly_expenses
    )

    df["debt_to_income"] = (df.current_emi_amount / salary).fillna(0)
    df["expense_to_income"] = (total_fixed_expenses / salary).fillna(0)
    df["disposable_income"] = df.monthly_salary - total_fixed_expenses - df.current_emi_amount
    df["affordability_ratio"] = (df.disposable_income / salary).fillna(0)
    df["savings_cushion"] = (df.emergency_fund / salary.replace(0, np.nan) / 3).fillna(0)

    r = ANNUAL_INTEREST_RATE / (12 * 100)
    df["implied_new_emi"] = (
        df.requested_amount * r * (1 + r) ** df.requested_tenure
        / ((1 + r) ** df.requested_tenure - 1)
    )
    df["requested_to_income"] = (df.implied_new_emi / salary).fillna(0)
    df["total_emi_to_income"] = ((df.current_emi_amount + df.implied_new_emi) / salary).fillna(0)

    df["employment_stability"] = np.clip(df.years_of_employment / 15, 0, 1)
    df["credit_band"] = pd.cut(
        df.credit_score,
        bins=[0, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    ).astype(str)

    df["income_per_dependent"] = df.monthly_salary / (df.dependents + 1)
    df["balance_to_expense"] = df.bank_balance / total_fixed_expenses.replace(0, np.nan)
    df["balance_to_expense"] = df.balance_to_expense.fillna(df.bank_balance)
    df["education_level_num"] = df.education.map(EDUCATION_ORDER)

    return df


class DerivedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Compute derived financial features, then one-hot encode categoricals.

    Input:  numpy array  [RAW_NUMERIC_COLS ..., RAW_CATEGORICAL_COLS ...]
    Output: numeric numpy array with derived + encoded features.

    This class must be importable at deserialization time because the fitted
    pipeline saved by preprocessing.py contains it.
    """

    def fit(self, X, y=None):
        df = self._to_dataframe(X)
        df = self._add_derived(df)

        cat_cols = RAW_CATEGORICAL_COLS + ["credit_band"]
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(df[cat_cols])

        self.all_cat_cols_ = cat_cols
        self.all_numeric_cols_ = [c for c in df.columns if c not in cat_cols]
        self.feature_names_ = (
            self.all_numeric_cols_
            + list(self.encoder_.get_feature_names_out(cat_cols))
        )
        return self

    def transform(self, X):
        df = self._to_dataframe(X)
        df = self._add_derived(df)

        numeric = df[self.all_numeric_cols_].to_numpy(dtype=float)
        encoded = self.encoder_.transform(df[self.all_cat_cols_])
        return np.hstack([numeric, encoded])

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS)
        for c in RAW_NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _add_derived(self, df):
        df = df.copy()
        salary = df.monthly_salary.replace(0, np.nan)

        total_fixed_expenses = (
            df.monthly_rent + df.school_fees + df.college_fees
            + df.travel_expenses + df.groceries_utilities
            + df.other_monthly_expenses
        )

        df["debt_to_income"] = (df.current_emi_amount / salary).fillna(0)
        df["expense_to_income"] = (total_fixed_expenses / salary).fillna(0)
        df["disposable_income"] = (
            df.monthly_salary - total_fixed_expenses - df.current_emi_amount
        )
        df["affordability_ratio"] = (df.disposable_income / salary).fillna(0)
        df["savings_cushion"] = (
            df.emergency_fund / salary.replace(0, np.nan) / 3
        ).fillna(0)

        r = ANNUAL_INTEREST_RATE / (12 * 100)
        df["implied_new_emi"] = (
            df.requested_amount * r * (1 + r) ** df.requested_tenure
            / ((1 + r) ** df.requested_tenure - 1)
        )
        df["requested_to_income"] = (df.implied_new_emi / salary).fillna(0)
        df["total_emi_to_income"] = (
            (df.current_emi_amount + df.implied_new_emi) / salary
        ).fillna(0)

        df["employment_stability"] = np.clip(df.years_of_employment / 15, 0, 1)
        df["credit_band"] = pd.cut(
            df.credit_score,
            bins=[0, 580, 670, 740, 800, 850],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        ).astype(str)
        df["income_per_dependent"] = df.monthly_salary / (df.dependents + 1)
        df["balance_to_expense"] = df.bank_balance / total_fixed_expenses.replace(0, np.nan)
        df["balance_to_expense"] = df.balance_to_expense.fillna(df.bank_balance)
        df["education_level_num"] = df.education.map(EDUCATION_ORDER)

        return df
