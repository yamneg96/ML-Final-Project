import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
# Paths
RAW_DATA_PATH = "data/raw/telecom_churn.xlsx"
PROCESSED_DATA_PATH = "data/processed/churn_processed.csv"
SCALER_PATH = "models/scaler.pkl"
# Ensure directories exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)