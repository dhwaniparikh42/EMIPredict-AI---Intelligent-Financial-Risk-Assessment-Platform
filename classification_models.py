"""
EMI Eligibility - Classification Models
Target: emi_eligibility (Not_Eligible / Eligible / High_Risk)

Models: Logistic Regression, Decision Tree, Random Forest,
        Gradient Boosting, XGBoost, Support Vector Classifier
Metrics: accuracy, precision, recall, F1 (macro), ROC-AUC (OvR macro)
The best model (by validation macro-F1) is evaluated once on the test set
and saved for deployment.

"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# settings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"

TARGET = "emi_eligibility"
DROP_COLS = ["max_monthly_emi"]  # regression target -> would leak the answer
RANDOM_STATE = 42
SVC_MAX_ROWS = 10000  # SVC is too slow on 276k rows, so it trains on a subset

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=None,
                    help="use only N training rows (fast test run)")
args = parser.parse_args()
os.makedirs(OUT_DIR, exist_ok=True)

#load data
train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
val = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
if args.sample:
    train = train.sample(args.sample, random_state=RANDOM_STATE)
    val = val.sample(min(args.sample, len(val)), random_state=RANDOM_STATE)
    test = test.sample(min(args.sample, len(test)), random_state=RANDOM_STATE)
print("Shapes  train/val/test:", train.shape, val.shape, test.shape)

le = LabelEncoder().fit(train[TARGET])
classes = list(le.classes_)
print("Classes:", classes)
print(train[TARGET].value_counts(normalize=True).round(3).to_string(), "\n")


def split_xy(df):
    X = df.drop(columns=[TARGET] + DROP_COLS)
    return X, le.transform(df[TARGET])


X_train, y_train = split_xy(train)
X_val, y_val = split_xy(val)
X_test, y_test = split_xy(test)

cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]
print("Numeric:", len(num_cols), "| Categorical:", cat_cols, "\n")


def make_preprocessor(scale):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


# models
# scale=True for models that need scaled inputs (LogReg, SVC)
models = {
    "Logistic Regression": (True, LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    "Decision Tree": (False, DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=20, class_weight="balanced",
        random_state=RANDOM_STATE)),
    "Random Forest": (False, RandomForestClassifier(
        n_estimators=100, max_depth=18, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
    "Gradient Boosting": (False, HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, random_state=RANDOM_STATE)),
    "XGBoost": (False, XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, objective="multi:softprob",
        eval_metric="mlogloss", tree_method="hist", n_jobs=-1,
        random_state=RANDOM_STATE)),
    "SVC": (True, SVC(
        kernel="rbf", C=1.0, probability=True, class_weight="balanced",
        random_state=RANDOM_STATE)),
}
# models without a class_weight option get balanced sample weights instead
needs_sample_weight = {"Gradient Boosting", "XGBoost"}


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
    }, pred


# train + validate all
results, fitted = [], {}
for name, (scale, clf) in models.items():
    Xtr, ytr = X_train, y_train
    if name == "SVC" and len(Xtr) > SVC_MAX_ROWS:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(Xtr), SVC_MAX_ROWS, replace=False)
        Xtr, ytr = X_train.iloc[idx], y_train[idx]

    pipe = Pipeline([("prep", make_preprocessor(scale)), ("model", clf)])
    fit_params = {}
    if name in needs_sample_weight:
        fit_params["model__sample_weight"] = compute_sample_weight("balanced", ytr)

    t0 = time.time()
    pipe.fit(Xtr, ytr, **fit_params)
    secs = time.time() - t0

    metrics, _ = evaluate(pipe, X_val, y_val)
    metrics.update({"model": name, "train_seconds": round(secs, 1)})
    results.append(metrics)
    fitted[name] = pipe
    print(f"{name:20s} val acc={metrics['accuracy']:.4f}  "
          f"F1={metrics['f1_macro']:.4f}  AUC={metrics['roc_auc_ovr']:.4f}  ({secs:.0f}s)")

res = (pd.DataFrame(results).set_index("model")
       .sort_values("f1_macro", ascending=False).round(4))
res.to_csv(os.path.join(OUT_DIR, "validation_results.csv"))
print("\nValidation comparison:\n", res.to_string())

#pick best, test it once
best_name = res.index[0]
best = fitted[best_name]
print(f"\nBest model (validation macro-F1): {best_name}")

test_metrics, test_pred = evaluate(best, X_test, y_test)
print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})
print("\nClassification report (test):\n",
      classification_report(y_test, test_pred, target_names=classes, digits=3))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred, display_labels=classes, cmap="Blues", ax=ax)
ax.set_title(f"{best_name} - Test confusion matrix")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

#feature importance (Random Forest)
rf = fitted["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = (pd.Series(rf.named_steps["model"].feature_importances_, index=names)
       .sort_values(ascending=False))
imp.to_csv(os.path.join(OUT_DIR, "rf_feature_importance.csv"), header=["importance"])
fig, ax = plt.subplots(figsize=(8, 6))
imp.head(15)[::-1].plot.barh(ax=ax, color="#2a6f97")
ax.set_title("Random Forest - Top 15 features")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rf_feature_importance.png"), dpi=150)
plt.close(fig)

#save
joblib.dump({"pipeline": best, "label_encoder": le, "model_name": best_name,
             "feature_columns": X_train.columns.tolist()},
            os.path.join(OUT_DIR, "best_classifier.joblib"))
with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
    json.dump({"model": best_name, **test_metrics}, f, indent=2)
print(f"\nSaved results, plots and best model to ./{OUT_DIR}/")
