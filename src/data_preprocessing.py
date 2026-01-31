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

        # Encode binary categorical columns
    binary_cols = ['Partner', 'Dependents', 'Senior Citizen', 'Phone Service', 'Multiple Lines', 
                   'Paperless Billing']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 1: 1, 0: 0})

        # Encode other categorical columns using one-hot encoding
    categorical_cols = [
        'Gender', 'Internet Service', 'Online Security', 'Online Backup',
        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
        'Contract', 'Payment Method'
    ]
    existing_cats = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

        # Target variable
    if 'Churn Value' in df.columns:
        y = df['Churn Value'].map({True: 1, False: 0})
    elif 'Churn Label' in df.columns:
        y = df['Churn Label'].map({'Yes': 1, 'No': 0})
    else:
        raise ValueError("No valid target column found. Use 'Churn Value' or 'Churn Label'.")
    
    X = df.drop(['Churn Value', 'Churn Label'], axis=1, errors='ignore')

        # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

        # Save scaler
    joblib.dump(scaler, SCALER_PATH)

        # Save processed data for inspection
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    return X_scaled, y