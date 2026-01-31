"""
Telecom Churn Prediction Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📊", layout="centered")

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCALER_PATH   = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH  = os.path.join(BASE_DIR, "models", "encoder.pkl")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "best_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

# ---------------- Load Objects ----------------
scaler   = joblib.load(SCALER_PATH)
encoder  = joblib.load(ENCODER_PATH)
model    = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# ---------------- Feature Groups ----------------
binary_cols = [
    'Partner','Dependents','Senior Citizen',
    'Phone Service','Multiple Lines','Paperless Billing'
]

categorical_cols = [
    'Gender','Internet Service','Online Security','Online Backup',
    'Device Protection','Tech Support','Streaming TV','Streaming Movies',
    'Contract','Payment Method'
]

numeric_cols = [
    "Tenure Months","Monthly Charges","Total Charges"
]

# ---------------- UI ----------------
st.title("📊 Telecom Customer Churn Predictor")
st.markdown("Fill in the customer details below:")
st.markdown("---")

# ---------------- Input Form ----------------
with st.form("customer_form"):

    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        inputs_binary = {
            col: st.selectbox(col, ["Yes", "No"])
            for col in binary_cols
        }

    with col2:
        inputs_cat = {}
        for i, col in enumerate(categorical_cols):
            options = list(encoder.categories_[i])
            inputs_cat[col] = st.selectbox(col, options)

    st.subheader("Billing & Usage")

    tenure = st.number_input("Tenure (Months)", 0, 120, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 1000.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)

    submit = st.form_submit_button("🚀 Predict Churn")

# ---------------- Prediction ----------------
if submit:

    row = {
        **inputs_binary,
        **inputs_cat,
        "Tenure Months": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }

    df_new = pd.DataFrame([row])

    # Map binary
    for col in binary_cols:
        df_new[col] = df_new[col].map({"Yes": 1, "No": 0})

    # Encode categorical
    X_cat = encoder.transform(df_new[categorical_cols])
    X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine numeric + encoded
    X_num = df_new[numeric_cols].reset_index(drop=True)
    X_all = pd.concat([X_num, X_cat], axis=1)

    # Ensure EXACT feature order
    X_all = X_all.reindex(columns=features, fill_value=0)

    # Scale
    X_scaled = scaler.transform(X_all)

    # Predict
    churn_pred = model.predict(X_scaled)[0]
    churn_prob = model.predict_proba(X_scaled)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if churn_pred == 1:
        st.error("⚠️ High Risk: Customer is likely to churn")
    else:
        st.success("✅ Low Risk: Customer is likely to stay")

    st.metric("Churn Probability", f"{churn_prob:.2%}")
