import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Paths
RAW_DATA_PATH = "data/raw/telecom_churn.xlsx"
PROCESSED_DATA_PATH = "data/processed/churn_processed.csv"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"
# Ensure directories exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)  

def load_data():
    """Load data from Excel file."""
    return pd.read_excel(RAW_DATA_PATH)

def preprocess_data(df):
    """Preprocess the telecom churn dataset based on column names."""

        # Drop irrelevant columns
    df = df.drop([
        "CustomerID", "Count", "Country", "City", "Zip Code",
        "Lat Long", "Latitude", "Longitude", "Churn Score", "CLTV", "Churn Reason"
    ], axis=1, errors='ignore')