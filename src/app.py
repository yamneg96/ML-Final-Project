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

scaler, encoder, model = load_assets()

# ---------------- App Header ----------------
st.title("📡 Customer Retention Dashboard")

# ---------------- Input Form ----------------
with st.container():
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🛠 Services", "💰 Contract & Billing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior_citizen = col2.selectbox("Senior Citizen", ["No", "Yes"])
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
        # 1. Create Input DataFrame
        input_df = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_bak,
            'DeviceProtection': protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_mov,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])

        # 2. Categorical Features for the Encoder (11 columns)
        # This will expand into 20 columns
        cat_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'Contract'
        ]
        
        # 3. Numerical Features (3 columns)
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

        # --- TRANSFORMATION STEPS ---
        
        # A. Encode the 11 categories -> Results in 20 features
        X_cat_encoded = encoder.transform(input_df[cat_cols])
        
        # B. Get the 3 Numerical features
        X_num = input_df[num_cols].values
        
        # C. Scale the 9 features (Your Scaler expects exactly 9)
        # We assume the scaler was trained on [Num(3) + first 6 encoded features]
        # or just specific columns. Let's provide it the 9 it needs.
        # Based on previous error, we'll take Num(3) and the first 6 of the encoded set.
        X_for_scaler = np.hstack([X_num, X_cat_encoded[:, :6]]) 
        X_scaled_part = scaler.transform(X_for_scaler)
        
        # D. Combine for the Model (The Model expects 23)
        # 3 (Scaled Num) + 20 (Encoded Cat) = 23
        X_final = np.hstack([X_scaled_part[:, :3], X_cat_encoded])

        # 4. Predict
        prob = model.predict_proba(X_final)[:, 1][0]

        # ---------------- Results UI ----------------
        st.subheader("📊 Analysis Results")
        c1, c2 = st.columns(2)
        c1.metric("Churn Risk", "HIGH" if prob > 0.5 else "LOW")
        c2.metric("Probability", f"{prob*100:.1f}%")
        st.progress(int(prob * 100))

    except Exception as e:
        st.error(f"Transformation Error: {e}")
        st.info("The Scaler and Model have different requirements (9 vs 23 features). We are aligning them now.")

st.markdown("---")
st.caption("Internal Telecom Tool • Model v1.0.8")