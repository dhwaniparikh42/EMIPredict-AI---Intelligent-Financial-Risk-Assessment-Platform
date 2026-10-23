"""Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import preprocessing as P
from src.quality import assert_clean, quality_report


def make_raw(n=3000, seed=0) -> pd.DataFrame:
    """Synthetic *valid* raw-style frame (all strings, like load_raw returns)."""
    rng = np.random.default_rng(seed)
    scen = rng.choice(list(C.SCENARIO_RULES), n)
    amt = [rng.integers(*C.SCENARIO_RULES[s]["amount"]) for s in scen]
    ten = [rng.integers(*C.SCENARIO_RULES[s]["tenure"]) for s in scen]
    loans = rng.choice(["Yes", "No"], n)
    house = rng.choice(["Rented", "Own", "Family"], n)
    df = pd.DataFrame({
        "age": rng.integers(25, 60, n), "gender": rng.choice(["Male", "Female"], n),
        "marital_status": rng.choice(["Single", "Married"], n),
        "education": rng.choice(C.VALID_CATEGORIES["education"], n),
        "monthly_salary": rng.integers(20_000, 150_000, n).astype(float),
        "employment_type": rng.choice(C.VALID_CATEGORIES["employment_type"], n),
        "years_of_employment": rng.uniform(0.5, 6, n).round(1),
        "company_type": rng.choice(C.VALID_CATEGORIES["company_type"], n),
        "house_type": house, "monthly_rent": np.where(house == "Rented", 8000.0, 0.0),
        "family_size": 4, "dependents": rng.integers(0, 3, n),
        "school_fees": 0.0, "college_fees": 0.0, "travel_expenses": 2000.0,
        "groceries_utilities": 5000.0, "other_monthly_expenses": 2000.0,
        "existing_loans": loans, "current_emi_amount": np.where(loans == "Yes", 3000.0, 0.0),
        "credit_score": rng.integers(300, 850, n).astype(float),
        "bank_balance": 100_000.0, "emergency_fund": 50_000.0,
        "emi_scenario": scen, "requested_amount": np.array(amt, dtype=float),
        "requested_tenure": ten,
        "emi_eligibility": rng.choice(C.VALID_CATEGORIES[C.TARGET_CLF], n, p=[.2, .05, .75]),
        "max_monthly_emi": rng.integers(500, 15_000, n).astype(float),
    })
    return df[C.ALL_COLS].astype(str)


def test_standardize_repairs_malformed_numbers_and_gender():
    raw = make_raw(10)
    raw.loc[0, "monthly_salary"] = "18000.0.0"
    raw.loc[1, "bank_balance"] = "nan.0"
    raw.loc[2, "gender"] = "MALE"
    raw.loc[3, "gender"] = "f"
    std, _ = P.standardize(raw)
    assert std.loc[0, "monthly_salary"] == 18000.0
    assert np.isnan(std.loc[1, "bank_balance"])
    assert std.loc[2, "gender"] == "Male" and std.loc[3, "gender"] == "Female"


def test_clean_fixes_and_drops():
    raw = make_raw(200)
    raw.loc[0, "credit_score"] = "999"        # invalid -> null (row kept)
    raw.loc[1, "monthly_salary"] = "500000"   # out of range -> dropped
    raw.loc[2, ["house_type", "monthly_rent"]] = ["Own", "9000.0"]  # rent -> 0
    std, _ = P.standardize(raw)
    clean, log = P.clean_data(std)
    assert len(clean) == len(std) - log["rows_dropped_total"] <= 199
    assert (clean["monthly_salary"] <= 200_000).all()
    assert clean["credit_score"].isna().sum() >= 1
    assert (clean.loc[clean["house_type"] != "Rented", "monthly_rent"] == 0).all()


def test_split_is_disjoint_stratified_and_imputer_uses_train_only():
    std, _ = P.standardize(make_raw(3000))
    std.loc[::50, "credit_score"] = np.nan
    clean, _ = P.clean_data(std)
    splits = P.split_data(clean)
    n = sum(len(v) for v in splits.values())
    assert n == len(clean)
    assert abs(len(splits["train"]) / n - 0.70) < 0.01
    # disjoint (every row is unique in the synthetic data via a row id)
    ref = splits["train"][C.TARGET_CLF].value_counts(normalize=True)
    for v in splits.values():
        got = v[C.TARGET_CLF].value_counts(normalize=True)
        assert (got - ref).abs().max() < 0.01   # class mix preserved
    vals = P.fit_imputer(splits["train"])
    assert vals["credit_score"] == pytest.approx(splits["train"]["credit_score"].median())
    final = {k: P.apply_imputer(v, vals) for k, v in splits.items()}
    assert_clean([quality_report(v, k) for k, v in final.items()])
