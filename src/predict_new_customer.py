"""
src/predict_new_customer.py

Predict churn for new customer rows.

This script intentionally mirrors the preprocessing logic in `src/data_preprocessing.py`:
- Drop irrelevant columns
- Binary mapping for selected Yes/No columns
- One-hot encode selected categoricals using pandas.get_dummies(drop_first=True)
- Align columns to the training-time feature set (order matters!)
- Scale with the saved StandardScaler
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

SCALER_PATH = REPO_ROOT / "models/scaler.pkl"
FEATURE_COLUMNS_PATH = REPO_ROOT / "models/feature_columns.pkl"
NUMERIC_MEDIANS_PATH = REPO_ROOT / "models/numeric_medians.pkl"

# Model paths (we'll pick the first one that exists unless --model is provided)
RF_MODEL_PATH = REPO_ROOT / "models/random_forest_model.pkl"
LR_MODEL_PATH = REPO_ROOT / "models/logistic_model.pkl"
BEST_MODEL_PATH = REPO_ROOT / "models/best_model.pkl"


# Columns for preprocessing (should match `src/data_preprocessing.py`)
BINARY_COLS = [
    "Partner",
    "Dependents",
    "Senior Citizen",
    "Phone Service",
    "Multiple Lines",
    "Paperless Billing",
]

CATEGORICAL_COLS = [
    "Gender",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Payment Method",
    "State",
]

NUMERIC_COLS = ["Tenure Months", "Monthly Charges", "Total Charges"]

DROP_COLS = [
    "CustomerID",
    "Count",
    "Country",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

TARGET_COLS = ["Churn Value", "Churn Label"]


def _load_feature_columns() -> list[str]:
    """
    Preferred: load `models/feature_columns.pkl` saved during preprocessing.
    Fallback: derive from `data/processed/churn_processed.csv` if present.
    """
    if FEATURE_COLUMNS_PATH.exists():
        cols = joblib.load(str(FEATURE_COLUMNS_PATH))
        return list(cols)

    processed_path = REPO_ROOT / "data/processed/churn_processed.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        X = df.drop(TARGET_COLS, axis=1, errors="ignore")
        X = X.select_dtypes(include=[np.number])
        return list(X.columns)

    raise FileNotFoundError(
        "Could not find training feature columns. "
        f"Expected {FEATURE_COLUMNS_PATH} (preferred) or {processed_path} (fallback). "
        "Run `python3 src/data_preprocessing.py` first."
    )


def _load_numeric_medians() -> dict[str, float]:
    if NUMERIC_MEDIANS_PATH.exists():
        d = joblib.load(str(NUMERIC_MEDIANS_PATH))
        return {str(k): float(v) for k, v in dict(d).items()}
    return {}


def preprocess_new_customer(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()

    # Drop irrelevant columns (safe for both train-like and truly-new schemas)
    df = df.drop(DROP_COLS, axis=1, errors="ignore")

    # Binary mapping
    binary_map = {
        "Yes": 1,
        "No": 0,
        "No phone service": 0,
        "No internet service": 0,
        True: 1,
        False: 0,
        1: 1,
        0: 0,
    }
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].map(binary_map), errors="coerce").fillna(0)

    # One-hot encode categoricals using the same approach as training
    existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    # Drop target columns if present (e.g., if user passed processed/training data)
    df = df.drop(TARGET_COLS, axis=1, errors="ignore")

    # Coerce numeric columns and fill NaNs with training medians (if available)
    numeric_medians = _load_numeric_medians()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in numeric_medians:
                df[col] = df[col].fillna(numeric_medians[col])

    # Keep numeric only (matches training-time `select_dtypes`)
    X = df.select_dtypes(include=[np.number])
    X = X.fillna(0)

    # Align columns to training feature set (adds missing columns as 0, drops extra columns)
    feature_cols = _load_feature_columns()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    # Scale (order must match feature_cols)
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run `python3 src/data_preprocessing.py` first."
        )
    scaler = joblib.load(str(SCALER_PATH))
    # Pass a DataFrame to preserve feature names and avoid sklearn warnings.
    return scaler.transform(X)