import pandas as pd
import joblib
import numpy as np
import os

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
INPUT_PATH  = os.path.join(BASE_DIR, "data", "raw", "new_customers.csv")  # your test CSV
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "new_customer_predictions.csv")

# Ensure results folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ---------------- Load saved objects ----------------
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
model  = joblib.load(MODEL_PATH)

# ---------------- Columns ----------------
binary_cols = [
    'Partner','Dependents','Senior Citizen',
    'Phone Service','Multiple Lines','Paperless Billing'
]

categorical_cols = [
    'Gender','Internet Service','Online Security','Online Backup',
    'Device Protection','Tech Support','Streaming TV','Streaming Movies',
    'Contract','Payment Method','State'
]

numerical_cols = [
    'Tenure Months','Monthly Charges','Total Charges'
]

def preprocess_new_customer(df):
    """Preprocess new customer data for prediction."""

    # --- Binary mapping ---
    for col in binary_cols:
        if col not in df.columns:
            df[col] = 0  # default 0 if missing
        df[col] = df[col].map({'Yes':1,'No':0,'Male':1,'Female':0,1:1,0:0})

    # --- Fill missing numerical columns ---
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0
    X_num = df[numerical_cols].fillna(0).values

    # --- Ensure all categorical columns exist ---
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'Unknown'  # default category

    X_cat = encoder.transform(df[categorical_cols])

    # --- Combine numeric + categorical ---
    X_combined = np.hstack([X_num, X_cat])

    # --- Scale ---
    X_scaled = scaler.transform(X_combined)
    return X_scaled

# ---------------- Main ----------------
if __name__ == "__main__":
    # Load new customer data
    new_data = pd.read_csv(INPUT_PATH)

    # Preprocess
    X_new = preprocess_new_customer(new_data)

    # Predict
    churn_pred = model.predict(X_new)
    churn_prob = model.predict_proba(X_new)[:,1] if hasattr(model,"predict_proba") else churn_pred

    # Save predictions
    new_data['Churn_Prediction'] = churn_pred
    new_data['Churn_Probability'] = churn_prob
    new_data.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Predictions saved to {OUTPUT_PATH}")
