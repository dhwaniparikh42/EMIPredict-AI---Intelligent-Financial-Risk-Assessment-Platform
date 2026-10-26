import joblib
import pandas as pd

path = "models/best_regressor.joblib"
loaded = joblib.load(path)

# The training workflow saves a bundle.
model = loaded["pipeline"] if isinstance(loaded, dict) else loaded

print("Model loaded:", type(model).__name__)
print("Has predict():", callable(getattr(model, "predict", None)))

# Inspect expected feature names.
feature_names = getattr(model, "feature_names_in_", None)

if feature_names is None and isinstance(loaded, dict):
    feature_names = loaded.get("feature_columns")

if feature_names is None:
    print("Feature names not found. Inspect the saved bundle.")
else:
    print("Expected input columns:", list(feature_names))
