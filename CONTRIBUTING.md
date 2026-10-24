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

**Member 1 (once):**
1. Sign up at https://dagshub.com and **connect the GitHub repo** (New Repository -> Connect a repo ->
   GitHub). Use a **public** repo: DagsHub's free plan limits *private* repos to very few collaborators.
2. The repo's MLflow server is at `https://dagshub.com/<dagshub-user>/<repo>.mlflow`
   (also shown under the repo's "Experiments" tab -> "Remote").
3. Invite teammates as collaborators on the DagsHub repo (Settings -> Collaborators).

**Every member (once):** DagsHub -> your avatar -> Settings -> **Tokens** -> create a token. Never commit it.

**Every member (each new terminal):**
```powershell
$env:MLFLOW_TRACKING_URI      = "https://dagshub.com/<dagshub-user>/<repo>.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "<your DagsHub username>"
$env:MLFLOW_TRACKING_PASSWORD = "<your DagsHub token>"
python check_mlflow.py                       # must print: [ok] Connected.
python train.py --task classification        # runs now appear in the shared dashboard
```
macOS/Linux: use `export NAME="value"` instead of `$env:NAME = "value"`.

Notes
* Open the dashboard by visiting the `.mlflow` URL in a browser. Models tab = registry.
* If the server has no model aliases, `train.py` tags the best version `champion=true` instead.
* **Plan B (no hosted server):** everything is reproducible (fixed seed), so Member 1 can run
  `python train.py` once locally to produce the MLflow dashboard + registry, and teammates share results
  through Pull Requests (`reports/model_comparison_*.csv` and `models/best_*.joblib`).

---
## D. Rules
* Never commit data, tokens, `mlflow.db`, `mlartifacts/`. The CI blocks data/MLflow files.
* Fit anything learned (scaler / encoder / imputer) on `train` only; tune on `val`; use `test` once.
* Select classifiers on macro-F1 (imbalanced classes), regressors on RMSE.
* Commit only `models/best_classifier.joblib` and `models/best_regressor.joblib` (small).
