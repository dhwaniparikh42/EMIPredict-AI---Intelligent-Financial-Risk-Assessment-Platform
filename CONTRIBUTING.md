# Team guide (remote team)

## Who owns what
| Member | Role | Owns |
|---|---|---|
| 1 | Data & MLOps lead | data pipeline, splits, MLflow + registry, repo, CI/CD, final docs |
| 2 | EDA & feature engineer | EDA report, `src/features.py` (final feature pipeline) |
| 3 | Classification engineer | classifiers, comparison report, registers best classifier |
| 4 | Regression engineer | regressors, comparison report, registers best regressor |
| 5 | App & deployment lead | Streamlit app (`app/`), Streamlit Cloud deployment |

`train.py` is a **reference baseline** (4 models per task + MLflow logging). Members 3 and 4 can extend it
or write their own scripts, as long as they use the shared splits, log to the shared MLflow
and keep the registry names `EMI_Eligibility_Classifier` / `EMI_Max_EMI_Regressor`.

---
## A. First-time setup (every member, once)
```powershell
git clone https://github.com/<owner>/<repo>.git
cd <repo>
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
1. Download `emi_prediction_dataset.csv` from the team Drive into `data/raw/`.
2. `python run_pipeline.py` then `python check_splits.py` -> must say **All checks passed**.
   (Seed is fixed, so everyone gets identical train/val/test files. Data is never committed.)
3. Quick test: `python train.py --sample 20000`

## B. Daily Git workflow
`main` is protected, nobody pushes to it directly.
```powershell
git pull                                          # on main, get the latest
python push_to_github.py -m "what I did" --branch feature/<name>-<topic>
```
Then open the **Pull Request** link it prints. Another member reviews, CI must be green, then merge.
Streamlit Cloud redeploys from `main` automatically.

---
## C. Shared MLflow for a team on different networks (DagsHub, free)
One hosted server for everyone, so all runs and the model registry are in ONE dashboard.
**Full step-by-step guide: [docs/DAGSHUB_SETUP.md](docs/DAGSHUB_SETUP.md).**

In short:
1. Member 1 creates a public repo on dagshub.com (blank, no GitHub connection needed) and adds every
   teammate as a collaborator. Only collaborators can log runs or open the dashboard.
2. Everyone creates their own DagsHub token (Settings -> Tokens).
3. Everyone copies `.env.example` to `.env` and fills in the shared URI
   (`https://dagshub.com/<user>/<repo>.mlflow`) + their own username + token. `.env` is git-ignored: **never commit it**.
4. `python check_mlflow.py` must print `[ok] Connected.`, then `python train.py ...` logs to the shared dashboard.

If the hosted server rejects the default model logging, use `python train.py --classic-logging`.
**Plan B (no hosted server):** results are reproducible (fixed seed), so Member 1 can run `python train.py`
once locally for the dashboard + registry, and teammates share results via Pull Requests
(`reports/model_comparison_*.csv` and `models/best_*.joblib`).

---
## D. Rules
* Never commit data, tokens, `.env`, `mlflow.db`, `mlartifacts/`. The CI blocks data/MLflow files.
* Fit anything learned (scaler / encoder / imputer) on `train` only; tune on `val`; use `test` once.
* Select classifiers on macro-F1 (imbalanced classes), regressors on RMSE.
* Commit only `models/best_classifier.joblib` and `models/best_regressor.joblib` (small).
