"""Helper for teammates: load the processed splits in one line.

    from src.data_loader import load_split, get_xy
    train = load_split("train")
    X_train, y_clf, y_reg = get_xy(train)
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_split(name: str) -> pd.DataFrame:
    """name in {'train', 'val', 'test'}."""
    for ext in ("parquet", "csv"):
        p = C.PROCESSED_DIR / f"{name}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"{name} split not found in {C.PROCESSED_DIR}. Run `python run_pipeline.py` first.")


def get_xy(df: pd.DataFrame):
    """Return features and both targets. Both targets are removed from X
    (max_monthly_emi must NOT be a feature of the classifier, and vice versa)."""
    X = df.drop(columns=[C.TARGET_CLF, C.TARGET_REG])
    return X, df[C.TARGET_CLF], df[C.TARGET_REG]
