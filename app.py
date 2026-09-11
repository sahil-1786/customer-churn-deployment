import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED PIPELINE
# ============================================================

model = joblib.load("churn_pipeline.pkl")

FINAL_THRESHOLD = 0.35


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate the probability of churn."
)


# ============================================================
# CUSTOMER INPUTS
# ============================================================

st.subheader("Customer Information")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

online_security = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

online_backup = st.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

device_protection = st.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

tech_support = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)


# ============================================================
# INPUTS NEEDED FOR ENGINEERED FEATURES
# ============================================================

st.subheader("Services")

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


# ============================================================
# ENGINEERED FEATURES
# ============================================================

# Tenure Group
if tenure <= 12:
    tenure_group = "New"

elif tenure <= 24:
    tenure_group = "Developing"

elif tenure <= 48:
    tenure_group = "Established"

else:
    tenure_group = "Loyal"


# New Customer Flag
is_new_customer = int(tenure <= 12)


# Average Charge per Month
avg_charge_per_month = (
    total_charges / (tenure + 1)
)


# Service Count
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

service_count = sum(
    service == "Yes"
    for service in service_columns
)


# ============================================================
# HIGH RISK CUSTOMER
# ============================================================

# IMPORTANT:
# Replace this value with the same MonthlyCharges threshold
# used when HighMonthlyCharge was originally created.

HIGH_MONTHLY_CHARGE_THRESHOLD = 70.35

high_monthly_charge = int(
    monthly_charges > HIGH_MONTHLY_CHARGE_THRESHOLD
)

long_term_contract = int(
    contract != "Month-to-month"
)

high_risk_customer = int(
    is_new_customer == 1
    and high_monthly_charge == 1
    and long_term_contract == 0
)


# ============================================================
# CONVERT YES/NO SENIOR CITIZEN TO MODEL FORMAT
# ============================================================

senior_citizen_numeric = (
    1 if senior_citizen == "Yes" else 0
)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

customer_data = pd.DataFrame({

    "tenure": [tenure],

    "TenureGroup": [tenure_group],

    "IsNewCustomer": [is_new_customer],

    "Contract": [contract],

    "MonthlyCharges": [monthly_charges],

    "TotalCharges": [total_charges],

    "AvgChargePerMonth": [
        avg_charge_per_month
    ],

    "InternetService": [
        internet_service
    ],

    "OnlineSecurity": [
        online_security
    ],

    "OnlineBackup": [
        online_backup
    ],

    "DeviceProtection": [
        device_protection
    ],

    "TechSupport": [
        tech_support
    ],

    "ServiceCount": [
        service_count
    ],

    "PaymentMethod": [
        payment_method
    ],

    "PaperlessBilling": [
        paperless_billing
    ],

    "SeniorCitizen": [
        senior_citizen_numeric
    ],

    "Partner": [
        partner
    ],

    "Dependents": [
        dependents
    ],

    "HighRiskCustomer": [
        high_risk_customer
    ]
})


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Churn Risk"):

    probability = model.predict_proba(
        customer_data
    )[0, 1]

    prediction = int(
        probability >= FINAL_THRESHOLD
    )

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn."
        )

    else:

        st.success(
            "✅ Customer is likely to stay."
        )

    st.write(
        f"Decision threshold: "
        f"{FINAL_THRESHOLD:.2f}"
    )


    # Optional risk band

    if probability < 0.20:

        risk = "Low"

    elif probability < 0.35:

        risk = "Moderate"

    elif probability < 0.60:

        risk = "High"

    else:

        risk = "Very High"

    st.write(
        f"Risk Level: **{risk}**"
    )


    # Show input table

    with st.expander(
        "View model input"
    ):

        st.dataframe(
            customer_data
        )
