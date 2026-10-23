from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.data_loader import load_split, get_xy


# Output files
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET_NAME = "max_monthly_emi"


def load_regression_data(split_name):
    """Load features and the regression target from a shared split."""
    df = load_split(split_name)
    X, _, y_reg = get_xy(df)

    if y_reg.name != TARGET_NAME:
        raise ValueError(
            f"Expected target {TARGET_NAME}, got {y_reg.name}"
        )

    return X.copy(), y_reg.astype(float).copy()


def make_metrics(y_true, predictions):
    """Calculate regression evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if not np.isfinite(y_true).all():
        raise ValueError("Target contains non-finite values.")

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    nonzero = np.abs(y_true) > 1e-8

    mape = (
        float(
            np.mean(
                np.abs(
                    (y_true[nonzero] - predictions[nonzero])
                    / y_true[nonzero]
                )
            ) * 100
        )
        if nonzero.any()
        else None
    )

    return {
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
        "mape_pct": mape,
    }


def main():
    print("Loading existing train, validation and test splits...")

    X_train, y_train = load_regression_data("train")
    X_val, y_val = load_regression_data("val")
    X_test, y_test = load_regression_data("test")

    # Ensure all splits use the same features in the same order.
    if not X_train.columns.equals(X_val.columns):
        raise ValueError("Train and validation columns do not match.")

    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Train and test columns do not match.")

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)
    print("Target:", TARGET_NAME)

    # Detect numeric and categorical input features.
    numeric_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))

    # Fit imputers, scaling and encoding only on training data.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    # Start with three models.
    models = {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    mlflow.set_experiment("EMIPredict_Regression")

    results = []
    best_rmse = float("inf")
    best_name = None

    print("\nTraining and comparing models...")

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessing", clone(preprocessor)),
            ("model", clone(estimator)),
        ])

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)

            val_predictions = pipeline.predict(X_val)
            metrics = make_metrics(y_val, val_predictions)

            mlflow.log_param("model_name", name)
            mlflow.log_param("target", TARGET_NAME)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_metrics({
                key: value
                for key, value in metrics.items()
                if value is not None and np.isfinite(value)
            })

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append({
                "model": name,
                **metrics,
            })

            print(f"\n{name}")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

            # Select using validation RMSE only.
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_name = name

    # Save model comparison results.
    comparison = pd.DataFrame(results).sort_values("rmse")

    comparison.to_csv(
        ARTIFACT_DIR / "regression_comparison.csv",
        index=False,
    )

    print("\nValidation model comparison:")
    print(comparison.to_string(index=False))
    print("\nSelected model:", best_name)

    # Refit the winning algorithm using train + validation data.
    # The test split remains untouched until final evaluation.
    X_train_val = pd.concat(
        [X_train, X_val], ignore_index=True
    )
    y_train_val = pd.concat(
        [y_train, y_val], ignore_index=True
    )

    final_model = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", clone(models[best_name])),
    ])

    final_model.fit(X_train_val, y_train_val)

    # Final test evaluation.
    test_predictions = final_model.predict(X_test)
    test_metrics = make_metrics(y_test, test_predictions)

    print("\nFinal held-out test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value}")

    # Save the complete pipeline, including preprocessing.
    joblib.dump(
        final_model,
        ARTIFACT_DIR / "best_regression_model.joblib",
    )

    with open(
        ARTIFACT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(test_metrics, file, indent=2, allow_nan=False)

    with open(
        ARTIFACT_DIR / "model_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump({
            "model_name": best_name,
            "target": TARGET_NAME,
            "input_features": X_train.columns.tolist(),
            "validation_rmse": best_rmse,
        }, file, indent=2)

    # Log the final model and its test metrics in a separate MLflow run.
    with mlflow.start_run(run_name=f"{best_name}_final"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("target", TARGET_NAME)
        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in test_metrics.items()
            if value is not None and np.isfinite(value)
        })
        mlflow.sklearn.log_model(
            final_model,
            name="final_model",
            serialization_format="cloudpickle",
        )

    print("\nCompleted successfully.")
    print("Saved artifacts to:", ARTIFACT_DIR.resolve())


if __name__ == "__main__":
    main()
