# Shared MLflow on DagsHub: setup guide

Our code lives on GitHub (`dhwaniparikh42/EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform`).
The shared MLflow dashboard + model registry live on **DagsHub**, in a separate repo that Member 1 creates.
It does **not** need to be connected to the GitHub repo and needs no admin rights on it.

> DagsHub's screens change now and then, so button names may differ slightly. The facts that matter:
> every DagsHub repo gets an MLflow server at `https://dagshub.com/<user>/<repo>.mlflow`, you log in with your
> DagsHub **username + access token**, and **only contributors (collaborators) of that DagsHub repo can log runs
> or open the dashboard**.

---
## Part 1 - Member 1 (once)

1. Sign up at https://dagshub.com (you can use "Sign in with GitHub").
2. **New Repository -> Create a blank repository.** Name: `emipredict-ai`. Visibility: **Public**
   (the free plan allows only very few collaborators on *private* repos; our data is not in the repo, so public is fine.
   Check dagshub.com/pricing if unsure).
3. Your MLflow address is  
   `https://dagshub.com/<your-dagshub-username>/emipredict-ai.mlflow`  
   (also reachable from the repo page via the **Remote** button -> MLflow).
4. Create a token: avatar (top right) -> **Settings -> Tokens** -> new token. Copy it. Treat it like a password.
5. In the project folder: copy `.env.example` to `.env` and fill in the three lines
   (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`).
   `.env` is git-ignored. **Never commit it or paste the token in chat.**
6. Test the connection:
   ```powershell
   python check_mlflow.py          # must print: [ok] Connected.
   ```
7. Log a first run:
   ```powershell
   python train.py --sample 20000
   ```
   Open the `.mlflow` address in a browser (log in with username + token if asked). You should see two
   experiments (`EMI_Eligibility_Classification`, `EMI_Max_EMI_Regression`) and, under **Models**,
   `EMI_Eligibility_Classifier` and `EMI_Max_EMI_Regressor`.
8. **Invite teammates:** DagsHub repo -> **Settings -> Collaborators** -> add each member's DagsHub username
   (they must sign up first). Without this they get 401/403 errors.
9. Send the team the message at the bottom of this file.

## Part 2 - every other member (once)

1. Sign up at DagsHub and send Member 1 your **DagsHub username**. Accept the collaborator invite.
2. Create your own token (Settings -> Tokens).
3. Copy `.env.example` to `.env`, fill in the URI Member 1 gave you + your username + your token.
4. `python check_mlflow.py` -> `[ok] Connected.`
5. Train as usual (`python train.py --task classification` etc.). Your runs appear in the shared dashboard.

---
## Troubleshooting
| Symptom | Fix |
|---|---|
| `401` / `403` / "permission" | You are not a collaborator yet, or wrong username/token. Re-check `.env` (no quotes needed, no spaces). |
| `Could not connect` | URL must end with `.mlflow` and match the repo name exactly. Check internet / VPN. |
| `404`, "endpoint not found", or model logging errors | The hosted server is older than your MLflow. `train.py` already falls back automatically; you can force it with `python train.py --classic-logging`. |
| Registry shows no alias `champion` | Older server without aliases: the best version is tagged `champion=true` instead. |
| Runs go to your local `mlflow.db` | `.env` is missing or in the wrong folder (must be next to `train.py`). `train.py` prints `MLflow tracking: ...` at the start. |
| Terminal variables vs `.env` | A variable set in the terminal (`$env:MLFLOW_TRACKING_URI=...`) overrides `.env`. |

## How the `champion` model is protected (shared registry)
Everyone logs to the same registry, so `train.py` only moves the alias `champion` when the new best model
beats the current champion on the **validation** set. Also:
* `--sample` runs (quick tests) never become champion and never overwrite `models/best_*.joblib` or the reports.
* A run that does not win is saved as `models/_candidate_*.joblib` (git-ignored).
* Only a real winner writes `models/best_*.joblib` and `reports/model_comparison_*.csv`: commit those in your PR.

## Final registry check (Member 1, Day 3)
In the DagsHub MLflow UI -> **Models**: each registered model should have the team's best version with alias
`champion` (or tag `champion=true`). Take screenshots of the experiment comparison and the registry for the report.

---
## Message to send the team
```
MLflow is ready. Dashboard: https://dagshub.com/<username>/emipredict-ai.mlflow
1. Make a DagsHub account and send me your DagsHub username (I'll add you as collaborator).
2. DagsHub -> Settings -> Tokens: create your own token.
3. Copy .env.example to .env and fill it in (URI above + your username + token). Never commit or share it.
4. python check_mlflow.py   -> should print "[ok] Connected."
5. python train.py --sample 20000 to test; your runs will show up in the shared dashboard.
Full guide: docs/DAGSHUB_SETUP.md
```
