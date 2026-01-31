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


