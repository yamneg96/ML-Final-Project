import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# Paths
RAW_DATA_PATH = "data/raw/Telecom_churn.xlsx"
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"

os.makedirs("models", exist_ok=True)

def preprocess_data(df):
    """Preprocess the dataset"""
    drop_cols = ["CustomerID", "Count", "Country", "City", "Zip Code",
                 "Lat Long", "Latitude", "Longitude", "Churn Score", "CLTV", "Churn Reason"]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Drop rows with missing target
    target_col = "Churn Value" if "Churn Value" in df.columns else "Churn Label"
    df = df.dropna(subset=[target_col])

    # Binary columns
    binary_cols = ['Partner', 'Dependents', 'Senior Citizen', 'Phone Service', 
                   'Multiple Lines', 'Paperless Billing']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1, 'No':0, 'Male':1, 'Female':0, 1:1, 0:0})

    # Numerical columns
    numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace(' ', np.nan), errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

    # Categorical columns
    categorical_cols = ['Gender', 'Internet Service', 'Online Security', 'Online Backup',
                        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
                        'Contract', 'Payment Method', 'State']
    existing_cats = [col for col in categorical_cols if col in df.columns]

    if existing_cats:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat = cat_imputer.fit_transform(df[existing_cats])
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = encoder.fit_transform(X_cat)
        joblib.dump(encoder, ENCODER_PATH)
    else:
        X_cat_encoded = np.array([]).reshape(len(df),0)

    # Numerical features
    X_num = df[numerical_cols].values

    # Combine features
    X = np.hstack([X_num, X_cat_encoded]) if X_cat_encoded.size else X_num

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    # Target
    y = df[target_col].map({'Yes':1, 'No':0}) if target_col == 'Churn Label' else df[target_col].astype(int)
    return X_scaled, y

def split_data(X, y):
    """Split into train, val, test"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, y_train, X_val, y_val):
    """Train Logistic Regression and Random Forest"""
    models = {}

    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_val_pred = lr.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"Logistic Regression Validation Accuracy: {acc:.4f}")
    models['lr'] = (lr, acc)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_val_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"Random Forest Validation Accuracy: {acc:.4f}")
    models['rf'] = (rf, acc)

    # Select best model
    best_model = max(models.items(), key=lambda x: x[1][1])[1][0]
    joblib.dump(best_model, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")
    return best_model

if __name__ == "__main__":
    df = pd.read_excel(RAW_DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    best_model = train_models(X_train, y_train, X_val, y_val)
    print("Pipeline finished successfully.")
