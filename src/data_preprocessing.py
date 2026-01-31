import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

# Paths
RAW_DATA_PATH = "data/raw/Telecom_churn.xlsx"
PROCESSED_DATA_PATH = "data/processed/churn_processed.csv"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"

# Ensure directories exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

def load_data():
    """Load dataset from Excel"""
    return pd.read_excel(RAW_DATA_PATH)

def preprocess_data(df):
    """Preprocess the telecom churn dataset"""

    # Drop irrelevant columns
    drop_cols = ["CustomerID", "Count", "Country", "City", "Zip Code",
                 "Lat Long", "Latitude", "Longitude", "Churn Score", "CLTV", "Churn Reason"]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Drop rows with missing target
    if 'Churn Value' in df.columns:
        df = df.dropna(subset=['Churn Value'])
    else:
        df = df.dropna(subset=['Churn Label'])

    # Binary encoding
    binary_cols = ['Partner', 'Dependents', 'Senior Citizen', 'Phone Service', 
                   'Multiple Lines', 'Paperless Billing']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1, 'No':0, 'Male':1, 'Female':0, 1:1, 0:0})

    # Handle numerical columns that might contain empty strings
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    for col in num_cols:
        if col in df.columns:
            # Replace empty strings or spaces with NaN
            df[col] = pd.to_numeric(df[col].replace(' ', np.nan), errors='coerce')
            # Fill NaN with column mean
            df[col] = df[col].fillna(df[col].mean())

    # One-hot encode categorical features
    categorical_cols = ['Gender', 'Internet Service', 'Online Security', 'Online Backup',
                        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
                        'Contract', 'Payment Method', 'State']
    existing_cats = [col for col in categorical_cols if col in df.columns]
    if existing_cats:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(df[existing_cats])
        joblib.dump(encoder, ENCODER_PATH)
    else:
        X_cat = np.array([]).reshape(len(df),0)

    # Numerical columns
    numerical_cols = df.drop(existing_cats + ['Churn Value', 'Churn Label'], axis=1, errors='ignore').columns
    X_num = df[numerical_cols].values

    # Combine numerical + categorical
    X_combined = np.hstack([X_num, X_cat]) if X_cat.size else X_num

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    joblib.dump(scaler, SCALER_PATH)

    # Target
    if 'Churn Value' in df.columns:
        y = df['Churn Value'].astype(int)
    else:
        y = df['Churn Label'].map({'Yes':1, 'No':0}).astype(int)

    # Save processed CSV for inspection
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    return X_scaled, y

def split_and_save(X, y):
    """Split data into train, validation, test sets"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(X, y)
    print("Data preprocessing complete.")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
