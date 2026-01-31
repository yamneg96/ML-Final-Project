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
    # Relative pathing for Streamlit Cloud
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
st.markdown("Enter customer details below to calculate churn risk.")

# ---------------- Input Form ----------------
with st.container():
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🛠 Services", "💰 Contract & Billing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior = col2.selectbox("Senior Citizen", ["No", "Yes"])
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
    
    # Create the dataframe with exact names expected by the Encoder
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

    # THE ALIGNMENT:
    # 1. Your Encoder expects 11 columns to perform its logic.
    cat_features_for_encoder = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'Contract'
    ]
    
    # 2. Your Scaler expects exactly 9 features.
    # This usually means 3 Numerical + 6 specific encoded outputs.
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    try:
        # Step A: Transform Categorical
        X_cat_encoded = encoder.transform(input_df[cat_features_for_encoder])
        
        # Step B: Get Numerical
        X_num = input_df[num_features].values
        
        # Step C: Combine
        # If your encoder dropped columns, this will be 3 + 6 = 9.
        X_combined = np.hstack([X_num, X_cat_encoded])
        
        # Step D: Scale
        X_scaled = scaler.transform(X_combined)
        
        # Step E: Predict
        prob = model.predict_proba(X_scaled)[:, 1][0]
        is_churn = prob > 0.5

        # ---------------- Results UI ----------------
        st.subheader("📊 Analysis Results")
        c1, c2 = st.columns(2)
        
        with c1:
            label = "🚨 HIGH RISK" if is_churn else "✅ STABLE"
            st.metric("Customer Status", label)
        
        with c2:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
            
        st.progress(int(prob * 100))
        
        if is_churn:
            st.error("This customer shows high behavior patterns consistent with churn.")
        else:
            st.success("This customer is likely to stay with the current service plan.")

    except Exception as e:
        st.error(f"Transformation Error: {e}")
        st.info("Check if the order of [X_num, X_cat] matches your training script.")

st.markdown("---")
st.caption("Internal Telecom Tool • Model v1.0.6")