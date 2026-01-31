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