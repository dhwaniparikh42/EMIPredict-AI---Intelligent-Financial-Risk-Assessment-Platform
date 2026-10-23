# EMIPredict AI: Intelligent Financial Risk Assessment Platform

**Project Name:** EMIPredict AI  

**Authors:** Shahjahan Iqbal, Purva Maurya , Prakhar Singh, Dhwani Parikh, Harsh Bhalodiya
**Last Updated:** [23 Oct 2026]

---

## Table of Contents

- [EMIPredict AI: Data Cleaning, Validation & Splits](#emipredict-ai-data-cleaning-validation--splits)
- [Project Layout](#project-layout)
- [Run It Locally](#run-it-locally)
- [What the Pipeline Does](#what-the-pipeline-does)
- [Using the Data in Your Own Code](#using-the-data-in-your-own-code)
- [EMIPredict Frontend Integration](#emipredict-frontend-integration)
  - [What Changed](#what-changed)
- [Rules for Teammates](#rules-for-teammates)
- [Pushing to the Shared GitHub Repo](#pushing-to-the-shared-github-repo)

---


Step 1 of the project: turns the raw 404,800-row EMI dataset into clean,
validated, leakage-free **train / val / test** files. 

## Project layout

```
emipredict-ai/
├── run_pipeline.py          # one command to run everything
├── requirements.txt
├── src/
│   ├── config.py            # paths, columns, business rules, split ratios, seed
│   ├── preprocessing.py     # load -> standardise -> clean -> split -> impute -> save
│   ├── quality.py           # validation rules + before/after report
│   └── data_loader.py       # load_split("train") helper for teammates
├── tests/test_pipeline.py
├── data/raw/                # put emi_prediction_dataset.csv here (git-ignored)
├── data/processed/          # train/val/test written here (git-ignored)
├── artifacts/               # imputation_values.json
└── reports/                 # data_quality_report.md / .json
```

## Run it locally

Requires Python 3.9+.

```bash
# 1. get the code
git clone <your-repo-url>
cd emipredict-ai

# 2. virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
                                   # Windows cmd:        .venv\Scripts\activate.bat

# 3. dependencies
pip install -r requirements.txt

# 4. put the raw file in place
#    data/raw/emi_prediction_dataset.csv

# 5. run the pipeline (~15 seconds)
python run_pipeline.py

# 6. (optional) run the tests
pytest -q
```

Use `python run_pipeline.py --format csv` if you prefer CSV over parquet.

## What the pipeline does

| Stage | What happens |
|---|---|
| Load | Reads every column as text so nothing is silently mis-typed |
| Standardise | Repairs malformed numbers (`18000.0.0` -> `18000.0`, `nan.0` -> null), normalises gender (`MALE`, `M`, `male`, `F`... -> `Male`/`Female`) |
| Clean | Drops duplicates; credit scores outside 300-850 (`0`, `999`, `1200`) -> null; rent forced to 0 for Own/Family houses; drops rows that break business rules (see `config.DROP_RULES`) |
| Split | 70 / 15 / 15, stratified on `emi_eligibility` x `emi_scenario`, `random_state=42` |
| Impute | Median/mode fill for `monthly_rent`, `credit_score`, `bank_balance`, `emergency_fund`, `education`, **learned on train only**, applied to val/test |
| Validate | Re-checks every rule on the final splits; the script **fails** if any null or violation remains |

Every number (rows dropped, values repaired, class balance per split) is written
to `reports/data_quality_report.md`.

## Using the data in your own code

```python
from src.data_loader import load_split, get_xy

train, val, test = load_split("train"), load_split("val"), load_split("test")
X_train, y_clf_train, y_reg_train = get_xy(train)
```

`get_xy` removes **both** targets from `X`, so the classifier never sees
`max_monthly_emi` and the regressor never sees `emi_eligibility`.

# EMIPredict frontend integration

This frontend is adapted to the team's **regression-only** training output.

## What changed

- `pages/1_Predict.py` loads `best_regression_model.joblib` and passes the raw 25-feature DataFrame directly to the fitted scikit-learn pipeline. It does not apply a second preprocessing pipeline.
- The classifier and label encoder dependencies have been removed from the prediction flow. Eligibility is explicitly shown as unavailable until a classifier exists.
- `pages/3_Model_Performance.py` reads `test_metrics.json`, `model_info.json`, and `regression_comparison.csv` from the backend artifact directory.
- `Home.py` reads the backend artifact metadata and `reports/data_quality_report.json`.
- The Explorer and Data Management pages gracefully explain when `data/EMI_sample.csv` is missing.
- `.gitignore` excludes the large model file while leaving JSON/CSV metric artifacts trackable.

The backend artifact directory should contain:
- `best_regression_model.joblib`
- `model_info.json`
- `test_metrics.json`
- `regression_comparison.csv`

## Rules for teammates

* **Do not commit data.** `data/raw` and `data/processed` are git-ignored (the raw CSV is ~75 MB, close to GitHub's 100 MB hard limit). Share the raw file via Google Drive / OneDrive, or use Git LFS / DVC.
* **Fit anything learned (scalers, encoders, imputers, target stats) on `train` only**, then apply to val/test.
* **Use `val` for tuning, touch `test` once** for the final report.
* To change a business rule or the split, edit `src/config.py`, re-run `python run_pipeline.py`, and commit the updated `reports/`.
* The classes are imbalanced (~77% Not_Eligible, ~18% Eligible, ~4% High_Risk). Use stratified CV, `class_weight`, or SMOTE (on train only) and report macro-F1 as well as accuracy.

## Pushing to the shared GitHub repo

```bash
git init
git add .
git commit -m "Add data cleaning, validation and splitting pipeline"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

For teammates: `git pull`, create `data/raw/`, drop in the CSV, run `python run_pipeline.py`.
