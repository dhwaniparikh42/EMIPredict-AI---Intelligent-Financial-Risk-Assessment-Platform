"""Run the full data pipeline:  python run_pipeline.py

Outputs
    data/processed/{train,val,test}.parquet   cleaned + imputed splits
    artifacts/imputation_values.json          fill values learned on train
    reports/data_quality_report.{md,json}     before/after validation report
"""
import argparse
import sys
import time

from src import config as C
from src import preprocessing as P
from src.quality import assert_clean, quality_report, write_reports


def main() -> int:
    ap = argparse.ArgumentParser(description="EMIPredict AI - data cleaning, validation and splitting")
    ap.add_argument("--raw", default=str(C.RAW_DATA_PATH), help="path to raw CSV")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="output format")
    args = ap.parse_args()
    t0 = time.time()

    print("[1/6] Loading raw data ...")
    raw = P.load_raw(args.raw)
    print(f"      {len(raw):,} rows x {raw.shape[1]} columns")

    print("[2/6] Standardising types & labels ...")
    std, std_log = P.standardize(raw)
    before = quality_report(std, "before_cleaning")

    print("[3/6] Cleaning (duplicates, invalid values, business rules) ...")
    clean, clean_log = P.clean_data(std)
    clean_log["actions"] = {**std_log, **clean_log["actions"]}
    print(f"      kept {clean_log['rows_out']:,} rows, dropped {clean_log['rows_dropped_total']:,}")

    print("[4/6] Splitting 70/15/15 (stratified) ...")
    splits = P.split_data(clean)

    print("[5/6] Imputing (fit on train only) ...")
    values = P.fit_imputer(splits["train"])
    splits = {k: P.apply_imputer(v, values) for k, v in splits.items()}
    P.save_imputer(values)

    print("[6/6] Validating final data & saving ...")
    reports = [quality_report(v, k) for k, v in splits.items()]
    assert_clean(reports)  # raises if any null / rule violation remains
    paths = P.save_splits(splits, fmt=args.format)
    write_reports(before, reports, clean_log, {r["name"]: r for r in reports}, C.REPORTS_DIR)

    print("\nDone in %.1fs. Saved:" % (time.time() - t0))
    for p in paths:
        print("  ", p.relative_to(C.ROOT))
    print("   reports/data_quality_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
