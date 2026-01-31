import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide"
)

# ---------------- Load Model Assets ----------------
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    encoder = joblib.load(os.path.join(BASE_DIR, "models", "encoder.pkl"))
    model  = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
    return scaler, encoder, model

try:
    scaler, encoder, model = load_assets()
except Exception as e:
    st.error(f"⚠️ Model Load Error: {e}")
    st.stop()

# ---------------- App Header ----------------
st.title("📡 Customer Retention Dashboard")

# ---------------- Input Form ----------------
with st.container():
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🛠 Services", "💰 Contract & Billing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior_citizen = col2.selectbox("Senior Citizen", ["No", "Yes"]) # Changed variable name to senior_citizen
        partner = col1.selectbox("Partner", ["No", "Yes"])
        dependents = col2.selectbox("Dependents", ["No", "Yes"])

    with tab2:
        col1, col2, col3 = st.columns(3)
        phone = col1.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = col1.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = col2.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = col2.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_bak = col2.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = col3.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = col3.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = col1.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_mov = col2.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with tab3:
        col1, col2 = st.columns(2)
        tenure = col1.slider("Tenure (Months)", 0, 72, 12)
        contract = col1.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = col1.selectbox("Paperless Billing", ["No", "Yes"])
        payment = col2.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = col2.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = col2.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

# ---------------- Prediction Logic ----------------
st.markdown("---")
if st.button("🚀 Run Risk Analysis", use_container_width=True):
    
    try:
        # 1. Manual Binary Mapping (The 6 Categorical Features)
        # This assumes your 9-feature scaler wants: 3 Numerics + 6 Binaries
        val_senior = 1 if senior_citizen == "Yes" else 0
        val_partner = 1 if partner == "Yes" else 0
        val_dependents = 1 if dependents == "Yes" else 0
        val_phone = 1 if phone == "Yes" else 0
        val_paperless = 1 if paperless == "Yes" else 0
        val_gender = 1 if gender == "Male" else 0

        cat_values = [val_senior, val_partner, val_dependents, val_phone, val_paperless, val_gender]
        
        # 2. Numerical Values
        num_values = [float(tenure), float(monthly_charges), float(total_charges)]

        # 3. Combine into the expected 9-feature array
        # Try Numeric first, then Categorical
        X_combined = np.array([num_values + cat_values]) 

        # 4. Scale and Predict
        # 
        X_scaled = scaler.transform(X_combined)
        prob = model.predict_proba(X_scaled)[:, 1][0]

        # ---------------- Results UI ----------------
        st.subheader("📊 Analysis Results")
        c1, c2 = st.columns(2)
        c1.metric("Churn Risk", "HIGH" if prob > 0.5 else "LOW")
        c2.metric("Probability", f"{prob*100:.1f}%")
        st.progress(int(prob * 100))

    except Exception as e:
        st.error(f"Transformation Error: {e}")
        st.info("If it still says 'expects 9 but got X', we need to swap the order of Numeric and Categorical features.")

st.markdown("---")
st.caption("Internal Telecom Tool • Model v1.0.7")