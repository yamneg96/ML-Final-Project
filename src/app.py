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
        gender = col1.selectbox("gender", ["Female", "Male"])
        senior = col2.selectbox("SeniorCitizen", ["No", "Yes"])
        partner = col1.selectbox("Partner", ["No", "Yes"])
        dependents = col2.selectbox("Dependents", ["No", "Yes"])

    with tab2:
        col1, col2, col3 = st.columns(3)
        phone = col1.selectbox("PhoneService", ["No", "Yes"])
        multiple_lines = col1.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
        internet = col2.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_sec = col2.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
        online_bak = col2.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])
        protection = col3.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
        tech_support = col3.selectbox("TechSupport", ["No", "Yes", "No internet service"])
        streaming_tv = col1.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
        streaming_mov = col2.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])

    with tab3:
        col1, col2 = st.columns(2)
        tenure = col1.slider("tenure", 0, 72, 12)
        contract = col1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = col1.selectbox("PaperlessBilling", ["No", "Yes"])
        payment = col2.selectbox("PaymentMethod", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = col2.number_input("MonthlyCharges", 0.0, 200.0, 70.0)
        total_charges = col2.number_input("TotalCharges", 0.0, 10000.0, 1500.0)

# ---------------- Prediction Logic ----------------
if st.button("🚀 Run Risk Analysis", use_container_width=True):
    
    # Define mapping to match the 11 columns likely used in OneHotEncoding
    # We exclude the ones that are usually treated as simple Binary (0/1) 
    # and only include the ones with 3+ categories or those typically OHE'd.
    
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
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

    # THE MAGIC 11: These are the columns to send to the OneHotEncoder
    # If this fails, the issue is which specific 11 were chosen.
    cat_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'Contract'
    ]
    
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    try:
        # 1. Transform categorical (11 columns)
        X_cat = encoder.transform(input_df[cat_features])
        
        # 2. Extract numerical (3 columns)
        X_num = input_df[num_features].values
        
        # 3. Combine
        X_combined = np.hstack([X_num, X_cat])
        
        # 4. Scale and Predict
        X_scaled = scaler.transform(X_combined)
        prob = model.predict_proba(X_scaled)[:, 1][0]

        # UI Results
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Risk Level", "HIGH" if prob > 0.5 else "LOW")
        c2.metric("Probability", f"{prob*100:.1f}%")
        st.progress(int(prob*100))

    except Exception as e:
        st.error(f"Error during transformation: {e}")
        st.info("The model expects 11 categorical features. If this persists, we need to check your training script's column selection.")