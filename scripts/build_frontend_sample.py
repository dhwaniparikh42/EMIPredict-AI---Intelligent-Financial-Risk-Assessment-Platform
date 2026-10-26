"""Create the small CSV sample used by the frontend from backend processed data."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Backend train.parquet or cleaned CSV path")
    parser.add_argument("--output", default="data/EMI_sample.csv", help="Output CSV path")
    parser.add_argument("--rows", type=int, default=15000, help="Maximum sample rows (default: 15000)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if args.rows < 1:
        raise ValueError("--rows must be >= 1")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input must be a .parquet or .csv file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = df.sample(n=min(args.rows, len(df)), random_state=42) if len(df) else df
    sample.to_csv(output_path, index=False)
    print(f"Saved {len(sample):,} rows x {sample.shape[1]} columns to {output_path}")


if __name__ == "__main__":
    main()
