"""Feature engineering and preprocessing pipeline for ChurnGuard.

Two layers:
1. engineer_features(): STATELESS, row-by-row. Learns nothing from data,
   so it is safe to apply to train, test, or a single API request.
2. build_preprocessor(): STATEFUL (imputer, scaler, encoders). Must be
   fit on training data only, then applied to anything else.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

TARGET = "churned"
ID_COL = "customer_id"

# Dropped: no predictive signal in EDA + sensitive attribute
DROP_COLS = ["gender"]

ADDON_COLS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]

# Feature groups AFTER engineer_features() has run
NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_addon_services", "charge_diff",
]
BINARY_FEATURES = ["senior_citizen", "is_auto_pay"]  # already 0/1
ORDINAL_FEATURES = ["contract"]
NOMINAL_FEATURES = [
    "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", *ADDON_COLS, "paperless_billing", "payment_method",
]


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from the target (y); drop the ID column."""
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET]
    return X, y


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning and feature creation (see docs/eda_findings.md)."""
    df = df.copy()  # never modify the caller's data

    # Decision 1: structural missingness → total ≈ tenure × monthly (0 for new customers)
    df["total_charges"] = df["total_charges"].fillna(df["tenure"] * df["monthly_charges"])

    # Decision 2: collapse redundant categories
    df[ADDON_COLS] = df[ADDON_COLS].replace("No internet service", "No")
    df["multiple_lines"] = df["multiple_lines"].replace("No phone service", "No")

    # New feature: how many add-on services the customer uses (engagement / lock-in)
    df["num_addon_services"] = (df[ADDON_COLS] == "Yes").sum(axis=1)

    # New feature: automatic payment (less effort to stay, less "decision moment" each month)
    df["is_auto_pay"] = (
        df["payment_method"].str.contains("automatic", case=False).astype(int)
    )

    # New feature: current bill vs. historical average bill.
    # Positive = customer now pays more than they used to (price increase signal).
    avg_monthly = np.where(
        df["tenure"] > 0,
        df["total_charges"] / df["tenure"].clip(lower=1),
        df["monthly_charges"],
    )
    df["charge_diff"] = df["monthly_charges"] - avg_monthly

    # Decision 3: drop sensitive / non-predictive columns
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def build_preprocessor(scale_numeric: bool = True) -> Pipeline:
    """Full preprocessing: engineering (stateless) + encoding/scaling (stateful).

    scale_numeric=False is useful for tree models, which don't need scaling.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    columns = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("ord", OrdinalEncoder(
                categories=[CONTRACT_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), ORDINAL_FEATURES),
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ), NOMINAL_FEATURES),
        ],
        remainder="drop",                 # anything not listed is dropped
        verbose_feature_names_out=False,  # clean names like "partner_Yes"
    )
    columns.set_output(transform="pandas")  # keep column names (useful for SHAP later)

    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features)),
        ("columns", columns),
    ])