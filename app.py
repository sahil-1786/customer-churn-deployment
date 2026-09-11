import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================
model = joblib.load("churn_pipeline.pkl")

FINAL_THRESHOLD = 0.35
HIGH_MONTHLY_CHARGE_THRESHOLD = 70.35   # replace if you used a different threshold


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Customer Churn Risk Analyzer",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.3rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        color: #111827;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    .section-box {
        background-color: #f8fafc;
        padding: 1.2rem 1.2rem 0.6rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .result-box-good {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 1rem;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    .result-box-risk {
        background-color: #fef2f2;
        border: 1px solid #ef4444;
        color: #991b1b;
        padding: 1rem;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.4rem;
    }

    .stButton > button {
        width: 100%;
        height: 3.1rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        background-color: #111827;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1f2937;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">Customer Churn Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Estimate churn probability for a telecom customer using the final deployed machine learning pipeline.</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# INPUT LAYOUT
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=800.0)

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)


with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Service Profile</div>', unsafe_allow_html=True)

    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
if tenure <= 12:
    tenure_group = "New"
elif tenure <= 24:
    tenure_group = "Developing"
elif tenure <= 48:
    tenure_group = "Established"
else:
    tenure_group = "Loyal"

is_new_customer = int(tenure <= 12)

avg_charge_per_month = total_charges / (tenure + 1)

service_columns = [
    phone_service,
    multiple_lines,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies
]

service_count = sum(service == "Yes" for service in service_columns)

high_monthly_charge = int(monthly_charges > HIGH_MONTHLY_CHARGE_THRESHOLD)

high_risk_customer = int(
    is_new_customer == 1 and
    high_monthly_charge == 1 and
    contract == "Month-to-month"
)

senior_citizen_numeric = 1 if senior_citizen == "Yes" else 0


# ============================================================
# MODEL INPUT
# ============================================================
customer_data = pd.DataFrame({
    "tenure": [tenure],
    "TenureGroup": [tenure_group],
    "IsNewCustomer": [is_new_customer],
    "Contract": [contract],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "AvgChargePerMonth": [avg_charge_per_month],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "ServiceCount": [service_count],
    "PaymentMethod": [payment_method],
    "PaperlessBilling": [paperless_billing],
    "SeniorCitizen": [senior_citizen_numeric],
    "Partner": [partner],
    "Dependents": [dependents],
    "HighRiskCustomer": [high_risk_customer]
})


# ============================================================
# PREDICT BUTTON
# ============================================================
if st.button("Predict Churn Risk"):

    probability = model.predict_proba(customer_data)[0, 1]
    prediction = int(probability >= FINAL_THRESHOLD)

    if probability < 0.20:
        risk_level = "Low"
    elif probability < 0.35:
        risk_level = "Moderate"
    elif probability < 0.60:
        risk_level = "High"
    else:
        risk_level = "Very High"

    st.markdown("---")
    st.markdown('<div class="section-title">Prediction Output</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Churn Probability", f"{probability:.2%}")

    with m2:
        st.metric("Decision Threshold", f"{FINAL_THRESHOLD:.2f}")

    with m3:
        st.metric("Risk Level", risk_level)

    if prediction == 1:
        st.markdown(
            '<div class="result-box-risk">Prediction: This customer is likely to churn.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box-good">Prediction: This customer is likely to stay.</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="small-note">This prediction is generated from the final deployed Logistic Regression pipeline using the locked threshold of 0.35.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Processed Input Used by the Model")
    st.dataframe(customer_data, use_container_width=True)
