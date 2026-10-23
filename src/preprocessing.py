"""Step 1 of the project: load, standardise, clean, split and impute.

Order matters (to avoid data leakage):
    load_raw -> standardize -> clean_data -> split_data -> fit_imputer(train) -> apply_imputer(all)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .quality import rule_masks

# Matches broken numbers such as "18000.0.0" or "38.0.0" -> "18000.0"
_MALFORMED_NUM = re.compile(r"^(-?\d+\.\d+)(?:\.\d+)+$")


# --------------------------------------------------------------- loading
def load_raw(path: Path = C.RAW_DATA_PATH) -> pd.DataFrame:
    """Read everything as text so no column is silently mis-typed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Put emi_prediction_dataset.csv in data/raw/."
        )
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    missing = set(C.ALL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df[C.ALL_COLS].copy()


# --------------------------------------------------------- standardising
def standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Fix types and spelling only. No rows are dropped here."""
    df = df.copy()
    log: Dict[str, int] = {}

    # --- numeric columns: repair malformed strings, then coerce
    n_malformed = n_unparseable = 0
    for col in C.NUMERIC_COLS:
        s = df[col].astype("string").str.strip()
        n_malformed += int(s.str.match(_MALFORMED_NUM).fillna(False).sum())
        s = s.str.replace(_MALFORMED_NUM, r"\1", regex=True)
        num = pd.to_numeric(s, errors="coerce").astype("float64")
        n_unparseable += int((num.isna() & s.notna()).sum())
        df[col] = num
    log["malformed_numbers_repaired (e.g. '18000.0.0')"] = n_malformed
    log["unparseable_numbers_set_to_null (e.g. 'nan.0')"] = n_unparseable

    # --- categorical columns: trim, fix gender spelling, null-out unknown labels
    for col in C.CATEGORICAL_COLS:
        df[col] = df[col].astype("string").str.strip().astype(object)
    before = df["gender"].copy()
    key = df["gender"].str.lower()
    df["gender"] = key.map(C.GENDER_MAP).where(key.isin(C.GENDER_MAP), df["gender"])
    log["gender_labels_normalised (MALE/M/male/F/... -> Male/Female)"] = int(
        (before.notna() & (before != df["gender"])).sum()
    )

    n_invalid = 0
    for col, allowed in C.VALID_CATEGORIES.items():
        bad = df[col].notna() & ~df[col].isin(allowed)
        n_invalid += int(bad.sum())
        df.loc[bad, col] = np.nan
    log["unknown_category_labels_set_to_null"] = n_invalid
    return df, log



# -------------------------------------------------------------- cleaning
def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Row-level, deterministic cleaning (safe to do before splitting).

    1. drop exact duplicates
    2. credit_score outside 300-850 (0 / 999 / 1200 placeholders) -> null (imputed later)
    3. rent on Own/Family houses -> 0 ; missing rent on Own/Family -> 0
    4. drop rows with a null in any column that is not imputable
    5. drop rows violating business rules in config.DROP_RULES
    """
    df = df.copy()
    actions: Dict[str, int] = {}
    n_start = len(df)

    # 1. duplicates
    dup = df.duplicated()
    actions["duplicate_rows_dropped"] = int(dup.sum())
    df = df[~dup]

    # 2. impossible credit scores
    lo, hi = C.VALID_RANGES["credit_score"]
    bad_cs = (df["credit_score"] < lo) | (df["credit_score"] > hi)
    actions["invalid_credit_scores_set_to_null (0/999/1200)"] = int(bad_cs.sum())
    df.loc[bad_cs, "credit_score"] = np.nan

    # 3. rent consistency
    not_rented = df["house_type"] != "Rented"
    actions["rent_set_to_0_for_Own/Family_houses"] = int((not_rented & (df["monthly_rent"] != 0)).sum())
    df.loc[not_rented, "monthly_rent"] = 0.0

    # 4. rows with nulls in non-imputable columns
    critical = [c for c in C.ALL_COLS if c not in C.IMPUTE_NUMERIC + C.IMPUTE_CATEGORICAL]
    crit_null = df[critical].isna().any(axis=1)
    actions["rows_dropped_null_in_critical_column"] = int(crit_null.sum())
    df = df[~crit_null]

    # 5. business-rule violations (counted independently, dropped as a union)
    masks = rule_masks(df)
    drop = pd.Series(False, index=df.index)
    for rule in C.DROP_RULES:
        actions[f"rows_flagged: {rule}"] = int(masks[rule].sum())
        drop |= masks[rule]
    actions["rows_dropped_business_rules (unique rows)"] = int(drop.sum())
    df = df[~drop]

    for col in C.INTEGER_COLS:
        df[col] = df[col].astype("int64")

    df = df.reset_index(drop=True)
    return df, {"actions": actions, "rows_in": n_start, "rows_out": len(df),
                "rows_dropped_total": n_start - len(df)}


# ------------------------------------------------------------- splitting
def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """70/15/15 split stratified on eligibility class x EMI scenario."""
    assert abs(C.TRAIN_SIZE + C.VAL_SIZE + C.TEST_SIZE - 1.0) < 1e-9
    strata = df[C.TARGET_CLF] + "|" + df["emi_scenario"]
    train, temp = train_test_split(
        df, test_size=C.VAL_SIZE + C.TEST_SIZE, stratify=strata, random_state=C.RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=C.TEST_SIZE / (C.VAL_SIZE + C.TEST_SIZE),
        stratify=strata.loc[temp.index], random_state=C.RANDOM_STATE,
    )
    return {k: v.reset_index(drop=True) for k, v in
            {"train": train, "val": val, "test": test}.items()}


# ------------------------------------------------------------ imputation
def fit_imputer(train: pd.DataFrame) -> dict:
    """Learn fill values from TRAIN only."""
    values: Dict[str, object] = {}
    for col in C.IMPUTE_NUMERIC:
        if col == "monthly_rent":  # rent only applies to renters
            values[col] = float(train.loc[train["house_type"] == "Rented", col].median())
        else:
            values[col] = float(train[col].median())
    for col in C.IMPUTE_CATEGORICAL:
        values[col] = str(train[col].mode().iloc[0])
    return values


def apply_imputer(df: pd.DataFrame, values: dict) -> pd.DataFrame:
    df = df.copy()
    for col, v in values.items():
        df[col] = df[col].fillna(v)
    return df


# ---------------------------------------------------------------- saving
def save_splits(splits: Dict[str, pd.DataFrame], out_dir: Path = C.PROCESSED_DIR,
                fmt: str = "parquet") -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            print("      [warn] pyarrow not installed -> saving CSV instead (pip install pyarrow for parquet)")
            fmt = "csv"
    paths = []
    for name, part in splits.items():
        p = out_dir / f"{name}.{fmt}"
        if fmt == "parquet":
            part.to_parquet(p, index=False)
        else:
            part.to_csv(p, index=False)
        paths.append(p)
    return paths


def save_imputer(values: dict, path: Path = C.ARTIFACTS_DIR / "imputation_values.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2))
