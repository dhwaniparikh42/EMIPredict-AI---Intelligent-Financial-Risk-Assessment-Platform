import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)

PROJECT_DIR = r"C:\Users\dhwan\Desktop\EMI Project"
DATA_DIR = os.path.join(PROJECT_DIR, "data_processed")
# All classification models, results and charts live together in classification/
MODEL_DIR = os.path.join(PROJECT_DIR, "classification")
PLOT_DIR = MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_TARGET = "emi_eligibility"
REG_TARGET = "max_monthly_emi"
CLASS_LABELS = joblib.load(os.path.join(DATA_DIR, "class_labels.joblib"))  # name -> number
CLASS_NAMES = [name for name, _ in sorted(CLASS_LABELS.items(), key=lambda kv: kv[1])]


def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}_processed.csv"))
    y = df.pop(CLASS_TARGET)
    df = df.drop(columns=[REG_TARGET])  # the regression target must not be used as an input
    return df, y


# 1. Load the processed data from Feature Engineering.py
X_train, y_train = load("train")
X_val, y_val = load("val")
X_test, y_test = load("test")
print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

# 2. Define the models
models = {
    # Baseline, interpretable: coefficients show how each feature moves the odds
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    # Feature importance analysis
    "Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42),
    # High-performance gradient boosting
    "XGBoost": XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=8, subsample=0.9,
        colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
        early_stopping_rounds=30, tree_method="hist", n_jobs=-1, random_state=42),
}


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="macro"),
        "recall": recall_score(y, pred, average="macro"),
        "f1": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
    }


# 3. Train each model and score it on the validation set
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()
    if name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start

    scores = evaluate(model, X_val, y_val)
    scores["train_time_s"] = train_time
    results[name] = scores
    print("  " + " | ".join(f"{k} {v:.4f}" for k, v in scores.items()))

# 4. Compare models on validation (macro F1 treats all three classes equally)
comparison = pd.DataFrame(results).T.sort_values("f1", ascending=False)
print("\n=== Validation comparison (macro averages) ===")
print(comparison.round(4))
comparison.round(4).to_csv(os.path.join(MODEL_DIR, "classification_comparison.csv"))

best_name = comparison.index[0]
best_model = models[best_name]
print(f"\nBest model on validation: {best_name}")

# 5. Final check of every model on the unseen test set
print("\n=== Test set results ===")
test_results = pd.DataFrame({name: evaluate(m, X_test, y_test) for name, m in models.items()}).T
print(test_results.round(4))
test_results.round(4).to_csv(os.path.join(MODEL_DIR, "classification_test_results.csv"))

test_pred = best_model.predict(X_test)
print(f"\nClassification report for {best_name} (test set):")
print(classification_report(y_test, test_pred, target_names=CLASS_NAMES, digits=3))

target_acc = test_results.loc[best_name, "accuracy"]
print(f"Accuracy target > 90%: {'MET' if target_acc > 0.90 else 'NOT MET'} ({target_acc:.2%})")

# 6. Charts
# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test), normalize="true") * 100
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{name}\n(% of each true class)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "classification_confusion_matrices.png"), dpi=120)
plt.close(fig)

# Model comparison
fig, ax = plt.subplots(figsize=(10, 5))
test_results[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=ax)
ax.set_ylim(0.5, 1.0)
ax.set_title("Classification models: test set scores (macro)")
ax.tick_params(axis="x", rotation=0)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "classification_model_comparison.png"), dpi=120)
plt.close(fig)

# Random Forest feature importance
rf_importance = pd.Series(models["Random Forest"].feature_importances_, index=X_train.columns)
rf_importance = rf_importance.sort_values(ascending=False)
print("\nTop 15 features (Random Forest importance):")
print(rf_importance.head(15).round(4).to_string())
rf_importance.to_csv(os.path.join(MODEL_DIR, "rf_feature_importance.csv"), header=["importance"])

fig, ax = plt.subplots(figsize=(9, 7))
rf_importance.head(20).sort_values().plot(kind="barh", color="#2a9d8f", ax=ax)
ax.set_title("Random Forest: top 20 feature importances")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "rf_feature_importance.png"), dpi=120)
plt.close(fig)

# Logistic Regression coefficients (interpretability)
coefs = pd.DataFrame(models["Logistic Regression"].coef_.T, index=X_train.columns, columns=CLASS_NAMES)
print("\nLogistic Regression: features that most raise the odds of 'Eligible':")
print(coefs["Eligible"].sort_values(ascending=False).head(8).round(3).to_string())
print("\n...and most lower them:")
print(coefs["Eligible"].sort_values().head(8).round(3).to_string())
coefs.to_csv(os.path.join(MODEL_DIR, "logreg_coefficients.csv"))

# 7. Save all models; the best one is saved separately for the app
for name, model in models.items():
    joblib.dump(model, os.path.join(MODEL_DIR, f"clf_{name.lower().replace(' ', '_')}.joblib"))
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_classifier.joblib"))

# Record which model was selected for deployment and why
best_info = {
    "model": best_name,
    "selected_by": "highest macro F1 on validation set",
    "validation": comparison.loc[best_name].drop("train_time_s").round(4).to_dict(),
    "test": test_results.loc[best_name].round(4).to_dict(),
    "class_labels": CLASS_LABELS,
}
with open(os.path.join(MODEL_DIR, "best_classifier_info.json"), "w") as f:
    json.dump(best_info, f, indent=2)

print("\n=== Selected for deployment ===")
print(f"Model: {best_name} (highest macro F1 on validation)")
for metric, value in best_info["test"].items():
    print(f"  test {metric:9s}: {value:.4f}")

print(f"\nModels saved to: {MODEL_DIR}")
print(f"Charts saved to: {PLOT_DIR}")
