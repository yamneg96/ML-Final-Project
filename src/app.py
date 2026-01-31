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
    .main { background-color: #f5f7f9; }
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
st.markdown("Enter customer details below to calculate churn risk using the ML Pipeline.")

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
    
    # 1. Map inputs to a broad dictionary
    data = {
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
    }
    
    df_input = pd.DataFrame([data])

    # 2. THE 11-COLUMN CATEGORICAL LIST
    # These are the 11 columns most commonly encoded in Telco projects
    cat_features = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaymentMethod', 'PaperlessBilling'
    ]
    
    # 3. THE 3-COLUMN NUMERICAL LIST
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    try:
        # Preprocessing
        # Note: If your encoder complains about names, check the "Debug" output in the except block
        X_cat = encoder.transform(df_input[cat_features])
        X_num = df_input[num_features].values
        
        # Combine (Numerical + Categorical)
        X_combined = np.hstack([X_num, X_cat])
        X_scaled = scaler.transform(X_combined)

        # Predict
        churn_prob = model.predict_proba(X_scaled)[:, 1][0]
        churn_pred = churn_prob > 0.5

        # ---------------- Results UI ----------------
        st.subheader("📊 Analysis Results")
        res_col1, res_col2, res_col3 = st.columns([1, 1, 2])

        with res_col1:
            status = "HIGH RISK" if churn_pred else "STABLE"
            st.metric("Customer Status", status)

        with res_col2:
            st.metric("Churn Probability", f"{churn_prob*100:.1f}%")

        with res_col3:
            st.write("Risk Confidence Level:")
            if churn_prob > 0.7:
                st.error("Critical Risk: Immediate Action Needed")
            elif churn_prob > 0.4:
                st.warning("Moderate Risk: Monitor Closely")
            else:
                st.success("Low Risk: Customer is Stable")
            st.progress(int(churn_prob * 100))
            
    except ValueError as ve:
        st.error("🚨 **Feature Count Mismatch**")
        st.write(f"The encoder expects **{len(encoder.feature_names_in_)}** features.")
        st.write("Specifically, your encoder was trained on these columns:")
        st.code(list(encoder.feature_names_in_))
        st.info("Update the `cat_features` list in the code to match the names above exactly.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.caption("Internal Telecom Tool • Model v1.0.5")