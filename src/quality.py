"""Data-quality rules + report generation.

`rule_masks` is the single source of truth for what counts as a violation. It
is used (a) to *report* violations before/after cleaning and (b) by the
cleaning step to decide which rows to drop, so the two can never disagree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from . import config as C


def rule_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Boolean mask (True = violation) for every data-quality rule.
    NaN never counts as a violation here; missingness is reported separately."""
    m: Dict[str, pd.Series] = {}

    for col, (lo, hi) in C.VALID_RANGES.items():
        m[f"{col}_out_of_range"] = (df[col] < lo) | (df[col] > hi)

    amt_bad = pd.Series(False, index=df.index)
    ten_bad = pd.Series(False, index=df.index)
    for scenario, r in C.SCENARIO_RULES.items():
        in_sc = df["emi_scenario"] == scenario
        a_lo, a_hi = r["amount"]
        t_lo, t_hi = r["tenure"]
        amt_bad |= in_sc & ((df["requested_amount"] < a_lo) | (df["requested_amount"] > a_hi))
        ten_bad |= in_sc & ((df["requested_tenure"] < t_lo) | (df["requested_tenure"] > t_hi))
    m["requested_amount_out_of_range_for_scenario"] = amt_bad
    m["requested_tenure_out_of_range_for_scenario"] = ten_bad

    m["negative_values"] = (df[C.NUMERIC_COLS] < 0).any(axis=1)

    for col, allowed in C.VALID_CATEGORIES.items():
        m[f"{col}_invalid_category"] = df[col].notna() & ~df[col].isin(allowed)

    m["rent_on_non_rented_house"] = (df["house_type"] != "Rented") & (df["monthly_rent"] > 0)
    m["existing_loans_flag_mismatch"] = (
        ((df["existing_loans"] == "No") & (df["current_emi_amount"] > 0))
        | ((df["existing_loans"] == "Yes") & (df["current_emi_amount"] == 0))
    )
    m["dependents_ge_family_size"] = df["dependents"] >= df["family_size"]
    m["employment_exceeds_working_age"] = df["years_of_employment"] > (df["age"] - 18)
    m["max_emi_exceeds_salary"] = df[C.TARGET_REG] > df["monthly_salary"]
    m["duplicate_rows"] = df.duplicated()
    return m


def quality_report(df: pd.DataFrame, name: str) -> dict:
    """Compact, JSON-serialisable quality summary of one dataframe."""
    nulls = df.isna().sum()
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "rows_with_any_null": int(df.isna().any(axis=1).sum()),
        "null_counts": {k: int(v) for k, v in nulls[nulls > 0].items()},
        "rule_violations": {k: int(v.sum()) for k, v in rule_masks(df).items()},
        "class_distribution": df[C.TARGET_CLF].value_counts(normalize=True).round(4).to_dict(),
        "target_reg_summary": df[C.TARGET_REG].describe().round(2).to_dict(),
    }


def assert_clean(reports: list[dict]) -> None:
    """Hard gate: fail loudly if any final split still has nulls/violations."""
    problems = []
    for r in reports:
        if r["null_counts"]:
            problems.append(f"{r['name']}: nulls {r['null_counts']}")
        enforced = set(C.DROP_RULES) | set(C.ALWAYS_ENFORCED)
        bad = {k: v for k, v in r["rule_violations"].items() if v and k in enforced}
        if bad:
            problems.append(f"{r['name']}: violations {bad}")
    if problems:
        raise AssertionError("Final data failed validation:\n" + "\n".join(problems))


# ------------------------------------------------------------ reporting
def _table(rows: list, header: tuple) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_reports(before: dict, final: list, log: dict, splits_info: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"before_cleaning": before, "after_cleaning": final,
               "cleaning_log": log, "splits": splits_info}
    (out_dir / "data_quality_report.json").write_text(json.dumps(payload, indent=2, default=str))

    rules = sorted(before["rule_violations"])
    after = {k: sum(r["rule_violations"].get(k, 0) for r in final) for k in rules}
    rule_rows = [(k, f"{before['rule_violations'][k]:,}", f"{after[k]:,}")
                 for k in rules if before["rule_violations"][k] or after[k]]
    classes = C.VALID_CATEGORIES[C.TARGET_CLF]

    md = [
        "# Data Quality Report", "",
        f"- Rows entering cleaning (after parsing): **{before['rows']:,}**",
        f"- Rows after cleaning: **{sum(r['rows'] for r in final):,}** "
        f"(dropped {log['rows_dropped_total']:,})", "",
        "## 1. Missing values (before cleaning)", "",
        _table(sorted(before["null_counts"].items(), key=lambda x: -x[1]), ("column", "nulls")), "",
        "## 2. Rule violations: before vs after", "",
        _table(rule_rows, ("rule", "before", "after (train+val+test)")), "",
        "## 3. Cleaning actions", "",
        _table([(k, f"{v:,}") for k, v in log["actions"].items()], ("action", "rows / values affected")), "",
        "## 4. Train / validation / test split", "",
        _table([(n, f"{i['rows']:,}", *[f"{i['class_distribution'].get(c, 0):.2%}" for c in classes])
                for n, i in splits_info.items()], ("split", "rows", *classes)), "",
        "Stratified on `emi_eligibility` x `emi_scenario`, so the class mix is preserved in every split.", "",
        "## 5. Imputation", "",
        "Fitted on the **train split only** and applied to val/test (no leakage). "
        "Values are stored in `artifacts/imputation_values.json`.", "",
    ]
    (out_dir / "data_quality_report.md").write_text("\n".join(md))
