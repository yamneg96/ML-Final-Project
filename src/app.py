import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📈",
    layout="wide"
)

# ---------------- Custom CSS for Professional Icons ----------------
st.markdown("""
<style>
    .icon-header {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        letter-spacing: 0.5px;
    }
    .risk-high {
        background-color: #dc3545;
        color: white;
    }
    .risk-medium {
        background-color: #ffc107;
        color: #212529;
    }
    .risk-low {
        background-color: #28a745;
        color: white;
    }
    .icon {
        font-size: 18px;
        margin-right: 6px;
    }
    .metric-container {
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
NUMERIC_MEDIANS_PATH = BASE_DIR / "models" / "numeric_medians.pkl"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

# ---------------- Load Model Assets ----------------
@st.cache_resource
def load_assets():
    scaler = joblib.load(str(SCALER_PATH))
    feature_columns = joblib.load(str(FEATURE_COLUMNS_PATH))
    numeric_medians = joblib.load(str(NUMERIC_MEDIANS_PATH)) if NUMERIC_MEDIANS_PATH.exists() else {}
    model = joblib.load(str(MODEL_PATH))
    return scaler, feature_columns, numeric_medians, model

try:
    scaler, feature_columns, numeric_medians, model = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# ---------------- App Header ----------------
st.markdown("""
<div class="icon-header">
    <span style="font-size: 28px; color: #1f77b4;">📈</span>
    <h1 style="display: inline; margin: 0;">Customer Retention Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ---------------- Input Form ----------------
with st.container():
    tab1, tab2, tab3 = st.tabs(["Customer Profile", "Services & Features", "Contract & Billing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior_citizen = col2.selectbox("Senior Citizen", ["No", "Yes"])
        partner = col1.selectbox("Partner", ["No", "Yes"])
        dependents = col2.selectbox("Dependents", ["No", "Yes"])
        state = col1.selectbox("State", ["California", "Texas", "Florida", "New York", "Illinois", "Other"])

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
if st.button("▶ Run Risk Analysis", use_container_width=True, type="primary"):
    
    try:
        # 1. Create Input DataFrame with correct column names
        input_df = pd.DataFrame([{
            'Gender': gender,
            'Senior Citizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'Tenure Months': tenure,
            'Phone Service': phone,
            'Multiple Lines': multiple_lines,
            'Internet Service': internet,
            'Online Security': online_sec,
            'Online Backup': online_bak,
            'Device Protection': protection,
            'Tech Support': tech_support,
            'Streaming TV': streaming_tv,
            'Streaming Movies': streaming_mov,
            'Contract': contract,
            'Paperless Billing': paperless,
            'Payment Method': payment,
            'Monthly Charges': monthly_charges,
            'Total Charges': total_charges,
            'State': state
        }])

        # 2. Binary mapping (matches data_preprocessing.py)
        binary_cols = ['Partner', 'Dependents', 'Senior Citizen', 'Phone Service', 'Multiple Lines', 'Paperless Billing']
        binary_map = {
            'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0,
            True: 1, False: 0, 1: 1, 0: 0, 'Male': 1, 'Female': 0
        }
        for col in binary_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col].map(binary_map), errors='coerce').fillna(0)

        # 3. One-hot encode categorical columns (matches data_preprocessing.py)
        categorical_cols = [
            'Gender', 'Internet Service', 'Online Security', 'Online Backup',
            'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
            'Contract', 'Payment Method', 'State'
        ]
        existing_cats = [col for col in categorical_cols if col in input_df.columns]
        input_df = pd.get_dummies(input_df, columns=existing_cats, drop_first=True)

        # 4. Convert numeric columns and handle NaNs
        numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                if col in numeric_medians:
                    input_df[col] = input_df[col].fillna(numeric_medians[col])

        # 5. Select only numeric columns
        X = input_df.select_dtypes(include=[np.number])
        X = X.fillna(0)

        # 6. Align columns to training feature set (CRITICAL!)
        # Add missing columns as 0, then reorder to match feature_columns exactly
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder to match training feature order exactly
        X = X.reindex(columns=feature_columns, fill_value=0)
        
        # Verify we have the right number of columns
        if X.shape[1] != len(feature_columns):
            st.error(f"Column alignment failed: Got {X.shape[1]} columns, expected {len(feature_columns)}")
            st.stop()

        # 7. Scale features - check for mismatches
        if scaler.n_features_in_ != X.shape[1]:
            st.error(f"⚠ Scaler mismatch: Scaler was trained on {scaler.n_features_in_} features, but we have {X.shape[1]} features")
            st.error("**Solution:** The scaler and feature_columns are out of sync. Regenerate preprocessing:")
            st.code("python src/data_preprocessing.py")
            st.stop()
        
        X_scaled = scaler.transform(X)
        
        # Verify shape matches model expectations  
        if X_scaled.shape[1] != model.n_features_in_:
            st.error(f"⚠ Model mismatch: Scaler output {X_scaled.shape[1]} features, but model expects {model.n_features_in_} features")
            st.error("**Root cause:** The scaler and model were trained with incompatible preprocessing.")
            st.error("**Solution:** Regenerate all files in this order:")
            st.code("python src/data_preprocessing.py\npython src/pipeline.py")
            st.write("This will ensure the scaler, feature_columns, and model all use the same preprocessing.")
            st.stop()

        # 8. Predict
        prob = model.predict_proba(X_scaled)[0, 1]

        # ---------------- Results UI ----------------
        st.subheader("Analysis Results")
        c1, c2 = st.columns(2)
        
        # Determine risk level with 3 categories
        if prob > 0.7:
            risk_level = "HIGH"
            risk_class = "risk-high"
            risk_icon = "▲"
        elif prob > 0.3:
            risk_level = "MEDIUM"
            risk_class = "risk-medium"
            risk_icon = "●"
        else:
            risk_level = "LOW"
            risk_class = "risk-low"
            risk_icon = "▼"
        
        # Display risk with professional badge
        risk_html = f'<div style="margin-top: 10px;"><span class="risk-badge {risk_class}">{risk_icon} {risk_level}</span></div>'
        c1.markdown(f"**Churn Risk**{risk_html}", unsafe_allow_html=True)
        c2.metric("Probability", f"{prob*100:.1f}%")
        
        # Progress bar with color coding
        if prob > 0.7:
            progress_color = "red"
        elif prob > 0.3:
            progress_color = "orange"
        else:
            progress_color = "green"
        
        st.markdown(f'<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
        st.progress(float(prob))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

st.markdown("---")
st.caption("Internal Telecom Tool • Model v1.0.8")
