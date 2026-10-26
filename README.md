# EMIPredict AI

Financial risk assessment: EMI eligibility (classification) + maximum safe EMI (regression),
tracked with MLflow. This repo covers **data cleaning -> features -> training -> model registry**.

See **CONTRIBUTING.md** for the team workflow and Git rules, and **docs/DAGSHUB_SETUP.md** for the shared MLflow setup.

## Layout
```
run_pipeline.py        # step 1: clean + validate + split data
start_mlflow_server.py # Member 1: start the shared MLflow server
check_mlflow.py        # check you can reach the MLflow server
check_splits.py        # step 1b: verify the split (sizes, leakage, balance)
train.py               # step 2: train 4 classifiers + 4 regressors, log to MLflow, pick best
push_to_github.py      # step 3: safe one-command push
src/  config.py preprocessing.py quality.py data_loader.py features.py
tests/                 # pytest
models/                # best_classifier.joblib, best_regressor.joblib (small, committed)
reports/               # data quality + model comparison reports (committed)
.vscode/tasks.json     # one-click tasks (Terminal > Run Task)
```

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
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

`train.py` options: `--task classification|regression|both`, `--models xgboost,random_forest`,
`--sample N`, `--no-balance` (turn off class weighting).

## What training does
* One sklearn `Pipeline` per model: raw columns -> feature engineering (16 ratios/risk scores,
  e.g. debt-to-income, affordability, EMI-to-disposable-income) -> scale + one-hot -> model.
  The same object predicts from raw user input, so Streamlit needs no extra preprocessing.
* Models: Logistic/Linear Regression, Decision Tree, Random Forest, XGBoost.
* Best model chosen on **validation** (classification: macro-F1, regression: RMSE);
  **test** metrics are computed once, only for the winner.
* MLflow logs params, metrics, plots and the model of every run; every model is a version in the
  registry (`EMI_Eligibility_Classifier`, `EMI_Max_EMI_Regressor`) and the winner gets alias `champion`.

## Teamwork rules
* Never commit data or `mlflow.db` (git-ignored). Share the CSV through Drive.
* Work on your own branch: `python push_to_github.py -m "msg" --branch feature/<name>`, then open a Pull Request.
* Fit anything learned on `train` only. Use `val` to tune; `test` only for the final report.
