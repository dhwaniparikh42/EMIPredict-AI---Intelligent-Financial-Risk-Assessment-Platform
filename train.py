"""Train, compare and register EMI models with MLflow.

    python train.py                        # both tasks, full data
    python train.py --task classification
    python train.py --task regression
    python train.py --sample 30000         # quick smoke test (~1 min)
    python train.py --models xgboost,decision_tree   # only some models

Then open the dashboard:   mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from src import config as C
from src.data_loader import get_xy, load_split
from src.env import load_env
from src.features import build_pipeline

load_env()  # reads MLFLOW_* settings from a local .env file (if present)

warnings.filterwarnings("ignore")
SEED = C.RANDOM_STATE
MODELS_DIR = C.ROOT / "models"
# Shared team server: set the MLFLOW_TRACKING_URI environment variable (see CONTRIBUTING.md).
# If it is not set, everything is stored locally in mlflow.db + mlartifacts/.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(C.ROOT / 'mlflow.db').as_posix()}"
IS_LOCAL_STORE = TRACKING_URI.startswith("sqlite")
ARTIFACT_ROOT = (C.ROOT / "mlartifacts").as_uri()

CLF_REGISTRY = "EMI_Eligibility_Classifier"
REG_REGISTRY = "EMI_Max_EMI_Regressor"


# ------------------------------------------------------------------ model zoo
def classifiers(n_jobs: int, balance: bool) -> dict:
    cw = "balanced" if balance else None
    return {
        "logistic_regression": LogisticRegression(max_iter=300, class_weight=cw, n_jobs=n_jobs),
        "decision_tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20,
                                                class_weight=cw, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=5,
                                                class_weight=cw, n_jobs=n_jobs, random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs,
                                 eval_metric="mlogloss", random_state=SEED),
    }


def regressors(n_jobs: int) -> dict:
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=SEED),
        "random_forest": RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=5,
                                               n_jobs=n_jobs, random_state=SEED),
        "xgboost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs,
                                random_state=SEED),
    }


# -------------------------------------------------------------------- metrics
def clf_metrics(pipe, X, y):
    pred, proba = pipe.predict(X), pipe.predict_proba(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_weighted": f1_score(y, pred, average="weighted"),
        "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
    }, pred


def reg_metrics(pipe, X, y):
    pred = pipe.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred),
        "mape": mean_absolute_percentage_error(y, pred) * 100,
    }, pred


# ---------------------------------------------------------------------- plots
def plot_importance(pipe, name):
    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(np.atleast_2d(model.coef_)).mean(axis=0)
    else:
        return None
    top = pd.Series(imp, index=names).nlargest(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax)
    ax.set_title(f"Top 15 features - {name}")
    fig.tight_layout()
    return fig


def plot_confusion(y, pred, labels, name):
    cm = confusion_matrix(y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"Confusion (val) - {name}")
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(y, pred, name):
    idx = np.random.default_rng(SEED).choice(len(y), min(3000, len(y)), replace=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(np.asarray(y)[idx], np.asarray(pred)[idx], s=4, alpha=0.4)
    lim = [0, max(np.max(y), np.max(pred))]
    ax.plot(lim, lim, "r--"); ax.set_xlabel("actual EMI"); ax.set_ylabel("predicted EMI")
    ax.set_title(f"Predicted vs actual (val) - {name}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- mlflow bits
def get_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    # on a remote server the server decides where artifacts live
    return (mlflow.create_experiment(name, artifact_location=ARTIFACT_ROOT) if IS_LOCAL_STORE
            else mlflow.create_experiment(name))


def log_sklearn_model(pipe, signature, registered_name, run_id, classic=False) -> int:
    """Log the model to the current run and register it. Returns the registry version.

    Tries MLflow's modern API first. Hosted servers (e.g. DagsHub) can be older and reject it,
    so on any failure it falls back to the classic route: upload files + register_model.
    """
    client = MlflowClient()
    kw = dict(signature=signature, serialization_format="cloudpickle")
    if not classic:
        try:
            try:  # MLflow >= 3
                info = mlflow.sklearn.log_model(pipe, name="model",
                                                registered_model_name=registered_name, **kw)
            except TypeError:  # MLflow 2.x
                info = mlflow.sklearn.log_model(pipe, artifact_path="model",
                                                registered_model_name=registered_name, **kw)
            version = getattr(info, "registered_model_version", None)
            if version is None:
                version = max(int(v.version) for v in client.search_model_versions(f"name='{registered_name}'")
                              if v.run_id == run_id)
            return int(version)
        except Exception as e:  # noqa: BLE001
            print(f"      [warn] modern model logging failed ({type(e).__name__}); using classic route")
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        mlflow.sklearn.save_model(pipe, str(model_dir), **kw)
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
    return int(mlflow.register_model(f"runs:/{run_id}/model", registered_name).version)


def simple_params(model) -> dict:
    return {k: v for k, v in model.get_params().items()
            if isinstance(v, (int, float, str, bool, type(None)))}


# ------------------------------------------------------------------ main loop
def run_task(task: str, train, val, test, args) -> pd.DataFrame:
    is_clf = task == "classification"
    X_tr, yc_tr, yr_tr = get_xy(train)
    X_va, yc_va, yr_va = get_xy(val)
    X_te, yc_te, yr_te = get_xy(test)

    if is_clf:
        le = LabelEncoder().fit(yc_tr)
        y_tr, y_va, y_te = le.transform(yc_tr), le.transform(yc_va), le.transform(yc_te)
        zoo, metric_fn = classifiers(args.n_jobs, not args.no_balance), clf_metrics
        select, higher = "f1_macro", True
        registry, exp_name = CLF_REGISTRY, "EMI_Eligibility_Classification"
    else:
        y_tr, y_va, y_te = yr_tr.values, yr_va.values, yr_te.values
        zoo, metric_fn = regressors(args.n_jobs), reg_metrics
        select, higher = "rmse", False
        registry, exp_name = REG_REGISTRY, "EMI_Max_EMI_Regression"

    if args.models:
        wanted = [m.strip() for m in args.models.split(",")]
        zoo = {k: v for k, v in zoo.items() if k in wanted}
        if not zoo:
            raise SystemExit(f"--models matched nothing. Choose from: {', '.join(wanted)}")

    exp_id = get_experiment(exp_name)
    client = MlflowClient()
    results, pipes = [], {}
    print(f"\n{'=' * 70}\n{task.upper()}  | train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}"
          f" | selecting on val {select}\n{'=' * 70}")

    for name, model in zoo.items():
        pipe = build_pipeline(model)
        fit_kw = {}
        if is_clf and name == "xgboost" and not args.no_balance:
            fit_kw["model__sample_weight"] = compute_sample_weight("balanced", y_tr)

        with mlflow.start_run(run_name=name, experiment_id=exp_id) as run:
            t0 = time.time()
            pipe.fit(X_tr, y_tr, **fit_kw)
            secs = time.time() - t0

            m_val, pred_val = metric_fn(pipe, X_va, y_va)
            mlflow.set_tags({"task": task, "model_type": name, "stage": "experiment"})
            mlflow.log_params(simple_params(model))
            mlflow.log_params({"n_train": len(X_tr), "n_val": len(X_va),
                               "n_features_in": X_tr.shape[1], "balanced": not args.no_balance})
            mlflow.log_metrics({f"val_{k}": v for k, v in m_val.items()})
            mlflow.log_metric("train_seconds", secs)

            fig = plot_importance(pipe, name)
            if fig is not None:
                mlflow.log_figure(fig, "feature_importance.png"); plt.close(fig)
            fig = (plot_confusion(y_va, pred_val, list(le.classes_), name) if is_clf
                   else plot_pred_vs_actual(y_va, pred_val, name))
            mlflow.log_figure(fig, "confusion_matrix.png" if is_clf else "pred_vs_actual.png")
            plt.close(fig)

            sig = infer_signature(X_va.head(50), pipe.predict(X_va.head(50)))
            version = log_sklearn_model(pipe, sig, registry, run.info.run_id, args.classic_logging)
            client.set_model_version_tag(registry, str(version), "model_type", name)

        results.append({"model": name, "run_id": run.info.run_id, "version": int(version),
                        "train_seconds": round(secs, 1), **m_val})
        pipes[name] = pipe
        print("  " + f"{name:<20}" + "  ".join(f"{k}={v:.4f}" for k, v in m_val.items())
              + f"  ({secs:.0f}s)")

    # ----- pick best on VALIDATION, then report TEST once
    table = pd.DataFrame(results).sort_values(select, ascending=not higher).reset_index(drop=True)
    best = table.iloc[0]
    best_pipe = pipes[best["model"]]
    m_test, _ = metric_fn(best_pipe, X_te, y_te)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})
        mlflow.set_tag("selected_as_best", "true")
    # ----- the registry alias 'champion' only moves if this run beats the current champion
    version = int(best["version"])
    if args.sample:
        promoted = False
        print("  [info] --sample run (smoke test): registry alias 'champion' NOT changed")
    else:
        promoted = maybe_set_champion(client, registry, version, select, higher, float(best[select]))

    status = "alias 'champion'" if promoted else "not promoted"
    print(f"\n  BEST {task}: {best['model']} (registry '{registry}' v{version}, {status})")
    print("  TEST  " + "  ".join(f"{k}={v:.4f}" for k, v in m_test.items()))

    # ----- save deployable bundle (+ reports) ONLY for a real champion.
    # Smoke tests and non-winning runs go to git-ignored files so they can never be committed by accident.
    MODELS_DIR.mkdir(exist_ok=True)
    kind = "classifier" if is_clf else "regressor"
    prefix = "best_" if promoted else ("_sample_" if args.sample else "_candidate_")
    bundle = {"task": task, "model_name": best["model"], "pipeline": best_pipe,
              "classes": list(le.classes_) if is_clf else None,
              "feature_columns": list(X_tr.columns), "val_metrics": m_val_of(best),
              "test_metrics": m_test, "registry": registry, "registry_version": version}
    out = MODELS_DIR / f"{prefix}{kind}.joblib"
    joblib.dump(bundle, out, compress=3)
    print(f"  saved {out.relative_to(C.ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    if promoted:
        C.REPORTS_DIR.mkdir(exist_ok=True)
        table.drop(columns=["run_id"]).to_csv(C.REPORTS_DIR / f"model_comparison_{task}.csv", index=False)
        (C.REPORTS_DIR / f"best_{task}_test_metrics.json").write_text(
            json.dumps({"model": best["model"], **m_test}, indent=2))
    return table


def maybe_set_champion(client, registry, version, metric, higher_is_better, new_value) -> bool:
    """Move alias 'champion' to `version` unless the current champion is at least as good
    (compared on the same VALIDATION split). Returns True if promoted."""
    try:
        cur = client.get_model_version_by_alias(registry, "champion")
    except Exception:  # no champion yet, or a server without aliases
        cur = None
    if cur is not None and str(cur.version) != str(version):
        old = client.get_run(cur.run_id).data.metrics.get(f"val_{metric}")
        if old is not None and ((new_value <= old) if higher_is_better else (new_value >= old)):
            print(f"  [info] current champion v{cur.version} (val {metric}={old:.4f}) is at least as good "
                  f"as this run ({new_value:.4f}); alias NOT moved")
            return False
    try:
        client.set_registered_model_alias(registry, "champion", str(version))
    except Exception as e:  # some hosted MLflow servers are older and have no aliases
        client.set_model_version_tag(registry, str(version), "champion", "true")
        print(f"  [warn] server has no model aliases ({type(e).__name__}); tagged champion=true instead")
    return True


def m_val_of(row) -> dict:
    return {k: float(v) for k, v in row.items() if k not in ("model", "run_id", "version", "train_seconds")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    ap.add_argument("--sample", type=int, default=0, help="use only N train rows (quick test)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--models", default="", help="comma list, e.g. xgboost,random_forest "
                    "(names: logistic_regression|linear_regression, decision_tree, random_forest, xgboost)")
    ap.add_argument("--classic-logging", action="store_true",
                    help="use the classic MLflow model upload (try this if a hosted server rejects the default)")
    ap.add_argument("--no-balance", action="store_true", help="disable class weighting")
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow tracking: {TRACKING_URI if not IS_LOCAL_STORE else 'local (mlflow.db)'}")
    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    if args.sample:
        train = train.sample(min(args.sample, len(train)), random_state=SEED)
        val = val.sample(min(args.sample // 4, len(val)), random_state=SEED)
        test = test.sample(min(args.sample // 4, len(test)), random_state=SEED)

    tasks = ["classification", "regression"] if args.task == "both" else [args.task]
    for t in tasks:
        tbl = run_task(t, train, val, test, args)
        print("\n" + tbl.drop(columns=["run_id", "version"]).round(4).to_string(index=False))
    print("\nOpen MLflow dashboard:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
