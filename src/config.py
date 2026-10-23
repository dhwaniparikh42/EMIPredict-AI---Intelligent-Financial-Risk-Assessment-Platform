"""Central configuration for the EMIPredict AI data pipeline.

Every business rule / path / seed lives here so teammates can change it in
one place without touching pipeline code.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "emi_prediction_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

# ----------------------------------------------------------------- splits
RANDOM_STATE = 42
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15

# --------------------------------------------------------------- columns
TARGET_CLF = "emi_eligibility"      # classification target (3 classes)
TARGET_REG = "max_monthly_emi"      # regression target (INR)

NUMERIC_COLS = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure", TARGET_REG,
]
INTEGER_COLS = ["age", "family_size", "dependents", "requested_tenure"]

CATEGORICAL_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
    TARGET_CLF,
]
ALL_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Columns allowed to have gaps -> imputed (using TRAIN statistics only).
# Any other column with a gap makes the row unusable and it is dropped.
IMPUTE_NUMERIC = ["monthly_rent", "credit_score", "bank_balance", "emergency_fund"]
IMPUTE_CATEGORICAL = ["education"]

# --------------------------------------------------------- business rules
VALID_RANGES = {                    # inclusive, from the project brief
    "age": (25, 60),
    "monthly_salary": (15_000, 200_000),
    "credit_score": (300, 850),
    TARGET_REG: (500, 50_000),
}

SCENARIO_RULES = {                  # amount in INR, tenure in months
    "E-commerce Shopping EMI": {"amount": (10_000, 200_000),   "tenure": (3, 24)},
    "Home Appliances EMI":     {"amount": (20_000, 300_000),   "tenure": (6, 36)},
    "Vehicle EMI":             {"amount": (80_000, 1_500_000), "tenure": (12, 84)},
    "Personal Loan EMI":       {"amount": (50_000, 1_000_000), "tenure": (12, 60)},
    "Education EMI":           {"amount": (50_000, 500_000),   "tenure": (6, 48)},
}

VALID_CATEGORIES = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married"],
    "education": ["High School", "Graduate", "Post Graduate", "Professional"],
    "employment_type": ["Private", "Government", "Self-employed"],
    "company_type": ["Large Indian", "MNC", "Mid-size", "Startup", "Small"],
    "house_type": ["Rented", "Own", "Family"],
    "existing_loans": ["Yes", "No"],
    "emi_scenario": list(SCENARIO_RULES),
    TARGET_CLF: ["Eligible", "High_Risk", "Not_Eligible"],
}

GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}

# Rules whose violations cause the ROW TO BE DROPPED (see quality.rule_masks).
DROP_RULES = [
    "age_out_of_range",
    "monthly_salary_out_of_range",
    "max_monthly_emi_out_of_range",
    "requested_amount_out_of_range_for_scenario",
    "requested_tenure_out_of_range_for_scenario",
    "negative_values",
    "existing_loans_flag_mismatch",
    "dependents_ge_family_size",
    "employment_exceeds_working_age",
    "max_emi_exceeds_salary",
]

# Rules that must be ZERO in the final splits. DROP_RULES are always enforced;
# if you remove a rule from DROP_RULES (e.g. to keep high salaries) the final
# check simply stops enforcing it. These are always enforced:
ALWAYS_ENFORCED = [
    "credit_score_out_of_range", "rent_on_non_rented_house", "duplicate_rows",
    *[f"{c}_invalid_category" for c in VALID_CATEGORIES],
]
