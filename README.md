# EMIPredict AI

Financial risk assessment: EMI eligibility (classification) + maximum safe EMI (regression),
tracked with MLflow. This repo covers **data cleaning -> features -> training -> model registry**.

See **CONTRIBUTING.md** for the team workflow, shared MLflow setup and Git rules.

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
# put emi_prediction_dataset.csv in data/raw/
python run_pipeline.py           # ~15 s
python check_splits.py           # PASS/FAIL checklist for the split
python train.py --sample 20000   # smoke test, ~1-2 min
python train.py                  # full training
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
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
