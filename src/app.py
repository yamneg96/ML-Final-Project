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

# ---------------- Custom Styling ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Load Model Assets ----------------
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Note: Ensure your paths match your folder structure
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    encoder = joblib.load(os.path.join(BASE_DIR, "models", "encoder.pkl"))
    model  = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
    return scaler, encoder, model

try:
    scaler, encoder, model = load_assets()
except Exception as e:
    st.error(f"Error loading model files. Please check paths. {e}")
    st.stop()

# ---------------- App Header ----------------
st.title("📡 Customer Retention Dashboard")
st.markdown("Enter customer details below to calculate churn risk using the ML Pipeline.")

# ---------------- Input Form ----------------
with st.container():
    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🛠 Services", "💰 Contract & Billing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["Male", "Female"])
        senior = col2.selectbox("Senior Citizen", ["No", "Yes"])
        partner = col1.selectbox("Partner", ["No", "Yes"])
        dependents = col2.selectbox("Dependents", ["No", "Yes"])

    with tab2:
        col1, col2, col3 = st.columns(3)
        phone = col1.selectbox("Phone Service", ["Yes", "No"])
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
        paperless = col1.selectbox("Paperless Billing", ["Yes", "No"])
        payment = col2.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = col2.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = col2.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

# ---------------- Prediction Logic ----------------
st.markdown("---")
if st.button("🚀 Run Risk Analysis", use_container_width=True):
    
    # Pre-processing mapping
    data = {
        'Gender': gender, 'Senior Citizen': senior, 'Partner': partner, 'Dependents': dependents,
        'Tenure Months': tenure, 'Phone Service': phone, 'Multiple Lines': multiple_lines,
        'Internet Service': internet, 'Online Security': online_sec, 'Online Backup': online_bak,
        'Device Protection': protection, 'Tech Support': tech_support, 'Streaming TV': streaming_tv,
        'Streaming Movies': streaming_mov, 'Contract': contract, 'Paperless Billing': paperless,
        'Payment Method': payment, 'Monthly Charges': monthly_charges, 'Total Charges': total_charges
    }
    
    df_new = pd.DataFrame([data])
    
    # Map binary features for the model
    binary_cols = ['Partner','Dependents','Senior Citizen','Phone Service','Paperless Billing']
    for col in binary_cols:
        df_new[col] = df_new[col].map({'Yes':1,'No':0})

    # Prepare features (Matching your project logic)
    categorical_cols = ['Gender','Internet Service','Online Security','Online Backup',
                        'Device Protection','Tech Support','Streaming TV','Streaming Movies',
                        'Contract','Payment Method']
    
    X_cat = encoder.transform(df_new[categorical_cols])
    X_num = df_new[["Tenure Months","Monthly Charges","Total Charges"]].values
    X_combined = np.hstack([X_num, X_cat])
    X_scaled = scaler.transform(X_combined)

    # Predict
    churn_prob = model.predict_proba(X_scaled)[:,1][0]
    churn_pred = churn_prob > 0.5

    # ---------------- Results UI ----------------
    st.subheader("📊 Analysis Results")
    res_col1, res_col2, res_col3 = st.columns([1, 1, 2])

    with res_col1:
        status = "HIGH RISK" if churn_pred else "STABLE"
        color = "inverse" if churn_pred else "normal"
        st.metric("Customer Status", status, delta=None)

    with res_col2:
        st.metric("Churn Probability", f"{churn_prob*100:.1f}%")

    with res_col3:
        # Visual Progress Bar
        st.write("Risk Confidence Level:")
        if churn_prob > 0.7:
            st.error(f"Critical Risk: {churn_prob*100:.1f}%")
        elif churn_prob > 0.4:
            st.warning(f"Moderate Risk: {churn_prob*100:.1f}%")
        else:
            st.success(f"Low Risk: {churn_prob*100:.1f}%")
        st.progress(int(churn_prob * 100))

st.markdown("---")
st.caption("Internal Telecom Tool • Model v1.0.4")