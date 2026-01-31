"""
Telecom Churn Prediction Web App
Purpose: Let a user input customer data and get churn prediction + probability
Run: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")

# Load objects
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
model  = joblib.load(MODEL_PATH)

# Columns
binary_cols = ['Partner','Dependents','Senior Citizen',
               'Phone Service','Multiple Lines','Paperless Billing']

categorical_cols = ['Gender','Internet Service','Online Security','Online Backup',
                    'Device Protection','Tech Support','Streaming TV','Streaming Movies',
                    'Contract','Payment Method']

st.title("📊 Telecom Customer Churn Predictor")

st.markdown("""
Enter the customer information below to get a churn prediction and probability.
""")

# ---------------- Input form ----------------
with st.form("customer_form"):
    # Binary inputs
    inputs_binary = {col: st.selectbox(col, ["Yes", "No"]) for col in binary_cols}

    # Categorical inputs
    inputs_cat = {col: st.selectbox(col, ["Option 1", "Option 2", "Option 3"]) for col in categorical_cols}  
    # Note: replace with actual categories from your dataset

    # Numerical inputs (example)
    tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=5000.0, value=500.0)

    submit = st.form_submit_button("Predict Churn")

if submit:
    # Build dataframe for preprocessing
    data_dict = {**inputs_binary, **inputs_cat,
                 "Tenure Months": tenure,
                 "Monthly Charges": monthly_charges,
                 "Total Charges": total_charges}

    df_new = pd.DataFrame([data_dict])

    # Binary mapping
    for col in binary_cols:
        df_new[col] = df_new[col].map({'Yes':1,'No':0,'Male':1,'Female':0,1:1,0:0})

    # Encode categorical
    X_cat = encoder.transform(df_new[categorical_cols])

    # Numerical columns
    X_num = df_new[["Tenure Months","Monthly Charges","Total Charges"]].values

    # Combine
    X_combined = np.hstack([X_num, X_cat])

    # Scale
    X_scaled = scaler.transform(X_combined)

    # Predict
    churn_pred = model.predict(X_scaled)[0]
    churn_prob = model.predict_proba(X_scaled)[:,1][0] if hasattr(model,"predict_proba") else churn_pred

    st.subheader("Prediction Result")
    st.write("✅ Churn Prediction:", "Yes" if churn_pred==1 else "No")
    st.write("📈 Churn Probability:", round(churn_prob, 3))
