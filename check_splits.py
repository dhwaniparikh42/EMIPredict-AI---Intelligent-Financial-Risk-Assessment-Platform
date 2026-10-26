"""Verify the train / val / test split.   Usage:  python check_splits.py

Run it after `python run_pipeline.py`. Prints a PASS/FAIL checklist and exits
with code 1 if anything is wrong.
"""
import sys

import pandas as pd

from src import config as C
from src.quality import rule_masks

OK, BAD = "[PASS]", "[FAIL]"
failures = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"{OK if cond else BAD} {label}" + (f"  ->  {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def load(name: str) -> pd.DataFrame:
    for ext in ("parquet", "csv"):
        p = C.PROCESSED_DIR / f"{name}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    sys.exit(f"{BAD} {name} split not found in {C.PROCESSED_DIR}. Run:  python run_pipeline.py")


splits = {n: load(n) for n in ("train", "val", "test")}
total = sum(len(d) for d in splits.values())

# 1 ------------------------------------------------------------ sizes
print("\n=== 1. SIZES ===")
want = {"train": C.TRAIN_SIZE, "val": C.VAL_SIZE, "test": C.TEST_SIZE}
for n, d in splits.items():
    share = len(d) / total
    print(f"   {n:<5} {len(d):>9,} rows  {share:6.2%}  x {d.shape[1]} columns")
    check(abs(share - want[n]) < 0.005, f"{n} is ~{want[n]:.0%} of the data")
check(len({tuple(d.columns) for d in splits.values()}) == 1, "all splits have identical columns")

# 2 ------------------------------------------------- no leakage between splits
print("\n=== 2. NO ROW APPEARS IN TWO SPLITS (leakage) ===")
hashes = {n: set(pd.util.hash_pandas_object(d, index=False)) for n, d in splits.items()}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = len(hashes[a] & hashes[b])
    check(overlap == 0, f"{a} / {b} overlap", f"{overlap} shared rows")
for n, d in splits.items():
    check(d.duplicated().sum() == 0, f"no duplicate rows inside {n}")

# 3 ------------------------------------------------------- data quality
print("\n=== 3. DATA QUALITY ===")
for n, d in splits.items():
    nulls = int(d.isna().sum().sum())
    enforced = set(C.DROP_RULES) | set(C.ALWAYS_ENFORCED)
    bad = {k: int(v.sum()) for k, v in rule_masks(d).items() if k in enforced and v.sum()}
    check(nulls == 0, f"{n}: no missing values", f"{nulls} nulls")
    check(not bad, f"{n}: no business-rule violations", str(bad) if bad else "")

# 4 ---------------------------------------------------- stratification
print("\n=== 4. CLASS BALANCE (should look the same in every split) ===")
cls = pd.DataFrame({n: d[C.TARGET_CLF].value_counts(normalize=True) for n, d in splits.items()})
print((cls * 100).round(2).add_suffix(" %").to_string())
check((cls.max(axis=1) - cls.min(axis=1)).max() < 0.005, "eligibility class mix matches across splits (<0.5 pt)")

print("\n   EMI scenario mix:")
sc = pd.DataFrame({n: d["emi_scenario"].value_counts(normalize=True) for n, d in splits.items()})
print((sc * 100).round(2).add_suffix(" %").to_string())
check((sc.max(axis=1) - sc.min(axis=1)).max() < 0.005, "scenario mix matches across splits (<0.5 pt)")

# 5 ----------------------------------------------------- regression target
print("\n=== 5. REGRESSION TARGET (max_monthly_emi) ===")
reg = pd.DataFrame({n: d[C.TARGET_REG].describe() for n, d in splits.items()}).round(1)
print(reg.to_string())
means = reg.loc["mean"]
check((means.max() - means.min()) / means.mean() < 0.02, "target mean within 2% across splits")

# summary ---------------------------------------------------------------
print("\n" + "=" * 60)
if failures:
    print(f"{BAD} {len(failures)} check(s) failed:")
    for f in failures:
        print("   -", f)
    sys.exit(1)
print(f"{OK} All checks passed. The split is safe to use for modelling.")
