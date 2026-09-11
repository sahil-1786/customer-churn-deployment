import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("churn_pipeline.pkl")

FINAL_THRESHOLD = 0.35

# IMPORTANT:
# Keep this value identical to the threshold used when
# HighMonthlyCharge was created during feature engineering.
HIGH_MONTHLY_CHARGE_THRESHOLD = 70.35


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }


    /* Hide Streamlit default decoration */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* Main heading */

    .main-title {
        font-size: 2.6rem;
        font-weight: 750;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.6rem;
    }


    /* Section headers */

    .section-title {
        font-size: 1.30rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }


    /* Info box */

    .info-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
        color: #475569;
        line-height: 1.6;
    }


    /* Model badge */

    .model-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 20px;
        padding: 0.35rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }


    /* Prediction cards */

    .risk-low {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        color: #166534;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 650;
        font-size: 1.05rem;
    }

    .risk-moderate {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        color: #92400E;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 650;
        font-size: 1.05rem;
    }

    .risk-high {
        background-color: #FFF7ED;
        border: 1px solid #FDBA74;
        color: #9A3412;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 650;
        font-size: 1.05rem;
    }

    .risk-very-high {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 650;
        font-size: 1.05rem;
    }


    /* Button */

    .stButton > button {

        width: 100%;
        height: 3.2rem;

        background-color: #0F172A;
        color: white;

        border: none;
        border-radius: 9px;

        font-size: 1rem;
        font-weight: 650;

    }

    .stButton > button:hover {

        background-color: #1E293B;
        color: white;

    }


    /* Metric cards */

    div[data-testid="stMetric"] {

        background-color: #F8FAFC;

        border: 1px solid #E2E8F0;

        padding: 1rem;

        border-radius: 12px;

    }


    /* Sidebar */

    section[data-testid="stSidebar"] {

        background-color: #F8FAFC;

        border-right: 1px solid #E2E8F0;

    }


    /* Small text */

    .small-text {

        color: #64748B;

        font-size: 0.86rem;

        line-height: 1.5;

    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Model Information")

    st.write("")

    st.markdown(
        """
        <span class="model-badge">
        Logistic Regression
        </span>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        """
        <div class="small-text">

        This application estimates the probability
        that a telecom customer will churn.

        <br><br>

        <b>Decision threshold:</b> 0.35

        <br><br>

        <b>Final test ROC-AUC:</b> 0.8414

        <br>

        <b>Final test PR-AUC:</b> 0.6532

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Prediction Flow")

    st.markdown(
        """
        Customer Data  
        ↓  
        Feature Engineering  
        ↓  
        Preprocessing  
        ↓  
        Logistic Regression  
        ↓  
        Churn Probability  
        ↓  
        Risk Classification
        """
    )

    st.markdown("---")

    st.caption(
        "Portfolio deployment demonstrating "
        "an end-to-end machine-learning prediction workflow."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Customer Churn Risk Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Interactive churn-propensity scoring using the final
    deployed machine-learning pipeline.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="info-box">

    Enter the customer's profile and service information.
    The system will generate a churn probability and compare
    it against the model's locked operating threshold.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Customer Information</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

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
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0,
        step=10.0
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

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

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [
            "No",
            "Yes"
        ]
    )

    partner = st.selectbox(
        "Partner",
        [
            "Yes",
            "No"
        ]
    )

    dependents = st.selectbox(
        "Dependents",
        [
            "Yes",
            "No"
        ]
    )


# ============================================================
# SERVICE DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">Service Portfolio</div>',
    unsafe_allow_html=True
)


s1, s2, s3, s4 = st.columns(4)


with s1:

    phone_service = st.selectbox(
        "Phone Service",
        [
            "Yes",
            "No"
        ]
    )


with s2:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )


with s3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with s4:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# ============================================================
# FEATURE ENGINEERING
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


# New customer

is_new_customer = int(
    tenure <= 12
)


# Average charge per month

avg_charge_per_month = (
    total_charges /
    (tenure + 1)
)


# Service count

services = [

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
    for service in services
)


# High Monthly Charge

high_monthly_charge = int(

    monthly_charges >
    HIGH_MONTHLY_CHARGE_THRESHOLD

)


# High Risk Customer

high_risk_customer = int(

    is_new_customer == 1

    and

    high_monthly_charge == 1

    and

    contract == "Month-to-month"

)


# Senior citizen numerical conversion

senior_citizen_numeric = (

    1
    if senior_citizen == "Yes"
    else 0

)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

customer_data = pd.DataFrame({

    "tenure": [
        tenure
    ],

    "TenureGroup": [
        tenure_group
    ],

    "IsNewCustomer": [
        is_new_customer
    ],

    "Contract": [
        contract
    ],

    "MonthlyCharges": [
        monthly_charges
    ],

    "TotalCharges": [
        total_charges
    ],

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
# PREDICT BUTTON
# ============================================================

st.write("")

predict_clicked = st.button(
    "Analyze Customer Churn Risk"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    probability = model.predict_proba(
        customer_data
    )[0, 1]


    prediction = int(
        probability >= FINAL_THRESHOLD
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability < 0.20:

        risk_level = "Low Risk"
        risk_class = "risk-low"

    elif probability < 0.35:

        risk_level = "Moderate Risk"
        risk_class = "risk-moderate"

    elif probability < 0.60:

        risk_level = "High Risk"
        risk_class = "risk-high"

    else:

        risk_level = "Very High Risk"
        risk_class = "risk-very-high"


    # ========================================================
    # OUTPUT
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Prediction Dashboard</div>',
        unsafe_allow_html=True
    )


    metric1, metric2, metric3, metric4 = st.columns(4)


    metric1.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )


    metric2.metric(
        "Risk Category",
        risk_level
    )


    metric3.metric(
        "Decision Threshold",
        f"{FINAL_THRESHOLD:.0%}"
    )


    metric4.metric(
        "Prediction",
        "Churn"
        if prediction == 1
        else "Stay"
    )


    # ========================================================
    # PROBABILITY GAUGE
    # ========================================================

    st.write("")

    st.markdown(
        "#### Churn Risk Score"
    )


    st.progress(
        float(probability)
    )


    gauge1, gauge2, gauge3 = st.columns(
        [1, 2, 1]
    )


    gauge1.caption(
        "0% - Low Risk"
    )

    gauge2.caption(
        f"Current score: {probability:.1%}"
    )

    gauge3.caption(
        "100% - Maximum Risk"
    )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    st.markdown(
        f"""
        <div class="{risk_class}">
        Risk Assessment: {risk_level}
        <br><br>
        The model estimates a churn probability of
        <b>{probability:.1%}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    # ========================================================
    # DECISION INTERPRETATION
    # ========================================================

    if prediction == 1:

        st.warning(
            """
            The predicted probability exceeds the
            model's operating threshold of 35%.

            This customer would therefore be flagged
            for potential retention intervention.
            """
        )

    else:

        st.success(
            """
            The predicted probability remains below
            the model's operating threshold of 35%.

            This customer would currently not be
            classified as a churn case.
            """
        )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    with st.expander(
        "View Customer Model Input"
    ):

        st.dataframe(
            customer_data,
            use_container_width=True
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "How is this prediction generated?"
    ):

        st.markdown(
            """
            The application follows the same modeling
            workflow used during model development:

            **1. Customer information**

            Raw customer characteristics are collected.

            **2. Feature engineering**

            Variables such as `TenureGroup`,
            `IsNewCustomer`, `AvgChargePerMonth`,
            `ServiceCount`, and `HighRiskCustomer`
            are generated.

            **3. Preprocessing**

            Numerical variables are standardized and
            categorical variables are one-hot encoded.

            **4. Logistic Regression**

            The fitted model generates the probability
            of churn.

            **5. Decision threshold**

            The probability is compared against the
            locked threshold of **0.35**.

            The threshold was selected from
            out-of-fold training predictions rather
            than from the final test set.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="small-text" style="text-align:center;">

    Customer Churn Propensity Modeling  
    Statistical Inference • Machine Learning • Explainability • Deployment

    <br>

    Developed by <b>Sahil Pathan</b>

    </div>
    """,
    unsafe_allow_html=True
)
