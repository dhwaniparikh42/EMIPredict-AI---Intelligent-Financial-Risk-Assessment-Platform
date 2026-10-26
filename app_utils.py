"""Shared paths and artifact/data loading for the EMIPredict Streamlit app."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent


def get_artifact_dir() -> Path:
    """Directory containing model metadata and metrics; configurable for local integration."""
    return Path(os.environ.get("EMIPREDICT_ARTIFACT_DIR", APP_ROOT / "artifacts")).expanduser()


def get_model_path() -> Path:
    """Allow a separate model path so a large artifact need not be copied into the frontend repo."""
    explicit = os.environ.get("EMIPREDICT_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return get_artifact_dir() / "best_regression_model.joblib"


@st.cache_resource(show_spinner="Loading EMI regression model...")
def load_regression_model():
    path = get_model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Regression model not found at {path}. Set EMIPREDICT_MODEL_PATH to the "
            "team's best_regression_model.joblib, or place it in the configured artifacts directory."
        )
    return joblib.load(path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def get_sample_path() -> Path:
    explicit = os.environ.get("EMIPREDICT_SAMPLE_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return APP_ROOT / "data" / "EMI_sample.csv"


@st.cache_data(show_spinner="Loading sample dataset...")
def load_sample_data() -> pd.DataFrame:
    path = get_sample_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Sample dataset not found at {path}. Create data/EMI_sample.csv with "
            "scripts/build_frontend_sample.py, or set EMIPREDICT_SAMPLE_PATH."
        )
    return pd.read_csv(path)


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
RAW_COLS = RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS

SCENARIO_RULES = {
    "E-commerce Shopping EMI": {"amount": (10_000, 200_000), "tenure": (3, 24)},
    "Home Appliances EMI": {"amount": (20_000, 300_000), "tenure": (6, 36)},
    "Vehicle EMI": {"amount": (80_000, 1_500_000), "tenure": (12, 84)},
    "Personal Loan EMI": {"amount": (50_000, 1_000_000), "tenure": (12, 60)},
    "Education EMI": {"amount": (50_000, 500_000), "tenure": (6, 48)},
}

VALID_CATEGORIES = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married"],
    "education": ["High School", "Graduate", "Post Graduate", "Professional"],
    "employment_type": ["Private", "Government", "Self-employed"],
    "company_type": ["Large Indian", "MNC", "Mid-size", "Startup", "Small"],
    "house_type": ["Rented", "Own", "Family"],
    "existing_loans": ["Yes", "No"],
    "emi_scenario": list(SCENARIO_RULES),
}
