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
    :root {
        color-scheme: light dark;
        --app-bg: #f4f7fb;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --surface-raised: rgba(255, 255, 255, 0.88);
        --text: #111827;
        --muted: #5b6472;
        --border: #d9e2ef;
        --accent: #2563eb;
        --accent-strong: #1d4ed8;
        --accent-soft: #eaf1ff;
        --shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
        --input-bg: #ffffff;
        --success-bg: #ecfdf3;
        --success-border: #74d99f;
        --success-text: #14532d;
        --warning-bg: #fffbeb;
        --warning-border: #f2c94c;
        --warning-text: #78350f;
        --danger-bg: #fff1f2;
        --danger-border: #f59ca6;
        --danger-text: #881337;
        --orange-bg: #fff7ed;
        --orange-border: #fdba74;
        --orange-text: #9a3412;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg: #0d1117;
            --surface: #151b23;
            --surface-soft: #0f1620;
            --surface-raised: rgba(21, 27, 35, 0.90);
            --text: #f3f6fb;
            --muted: #a7b2c2;
            --border: #2b3543;
            --accent: #60a5fa;
            --accent-strong: #93c5fd;
            --accent-soft: rgba(96, 165, 250, 0.14);
            --shadow: 0 18px 45px rgba(0, 0, 0, 0.34);
            --input-bg: #111821;
            --success-bg: rgba(22, 101, 52, 0.20);
            --success-border: #2f9d5b;
            --success-text: #bbf7d0;
            --warning-bg: rgba(146, 64, 14, 0.22);
            --warning-border: #d99b24;
            --warning-text: #fde68a;
            --danger-bg: rgba(153, 27, 27, 0.22);
            --danger-border: #e87979;
            --danger-text: #fecaca;
            --orange-bg: rgba(154, 52, 18, 0.22);
            --orange-border: #fb923c;
            --orange-text: #fed7aa;
        }
    }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, var(--accent-soft), transparent 34rem),
            var(--app-bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1320px;
        padding: 2rem 2.25rem 3rem;
    }

    #MainMenu,
    footer,
    header[data-testid="stHeader"] {
        visibility: hidden;
    }

    h1, h2, h3, h4, h5, h6,
    p, label, span, div {
        letter-spacing: 0;
    }

    .hero-panel {
        background: var(--surface-raised);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
        padding: clamp(1.25rem, 3vw, 2rem);
        margin-bottom: 1.25rem;
    }

    .eyebrow {
        color: var(--accent-strong);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
    }

    .main-title {
        color: var(--text);
        font-size: clamp(2rem, 4vw, 3.35rem);
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.7rem;
    }

    .subtitle {
        color: var(--muted);
        font-size: clamp(1rem, 1.6vw, 1.16rem);
        line-height: 1.65;
        max-width: 760px;
        margin-bottom: 0;
    }

    .info-box {
        align-items: flex-start;
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        color: var(--muted);
        display: flex;
        gap: 0.8rem;
        line-height: 1.65;
        margin-bottom: 1.75rem;
        padding: 1rem 1.15rem;
    }

    .info-box b,
    .risk-low b,
    .risk-moderate b,
    .risk-high b,
    .risk-very-high b {
        color: inherit;
    }

    .section-title {
        color: var(--text);
        font-size: 1.18rem;
        font-weight: 800;
        margin: 1.4rem 0 0.9rem;
    }

    .section-title::after {
        background: linear-gradient(90deg, var(--accent), transparent);
        border-radius: 999px;
        content: "";
        display: block;
        height: 3px;
        margin-top: 0.5rem;
        width: 4rem;
    }

    .model-badge {
        background: var(--accent-soft);
        border: 1px solid color-mix(in srgb, var(--accent) 34%, transparent);
        border-radius: 999px;
        color: var(--accent-strong);
        display: inline-flex;
        font-size: 0.82rem;
        font-weight: 800;
        padding: 0.38rem 0.78rem;
    }

    .side-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
    }

    .flow-list {
        color: var(--muted);
        display: grid;
        gap: 0.4rem;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .flow-step {
        align-items: center;
        display: flex;
        gap: 0.5rem;
    }

    .flow-step::before {
        background: var(--accent);
        border-radius: 999px;
        content: "";
        height: 0.45rem;
        width: 0.45rem;
    }

    .risk-low,
    .risk-moderate,
    .risk-high,
    .risk-very-high {
        border-radius: 14px;
        font-size: 1.03rem;
        font-weight: 700;
        line-height: 1.55;
        margin-top: 1rem;
        padding: 1.15rem 1.25rem;
    }

    .risk-low {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        color: var(--success-text);
    }

    .risk-moderate {
        background: var(--warning-bg);
        border: 1px solid var(--warning-border);
        color: var(--warning-text);
    }

    .risk-high {
        background: var(--orange-bg);
        border: 1px solid var(--orange-border);
        color: var(--orange-text);
    }

    .risk-very-high {
        background: var(--danger-bg);
        border: 1px solid var(--danger-border);
        color: var(--danger-text);
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        border: 0;
        border-radius: 12px;
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.24);
        color: #ffffff;
        font-size: 1rem;
        font-weight: 800;
        height: 3.35rem;
        transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
        width: 100%;
    }

    .stButton > button:hover {
        box-shadow: 0 18px 34px rgba(37, 99, 235, 0.30);
        color: #ffffff;
        filter: brightness(1.04);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        padding: 1rem 1.05rem;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
        font-weight: 800;
    }

    div[data-testid="stSelectbox"] > div,
    div[data-testid="stNumberInput"] > div {
        background: transparent;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: var(--input-bg);
        border-color: var(--border);
        border-radius: 10px;
        color: var(--text);
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
    }

    label[data-testid="stWidgetLabel"] p {
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: var(--surface-soft);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.75rem;
    }

    .stAlert {
        border-radius: 12px;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
    }

    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border);
        margin: 1.6rem 0;
    }

    .small-text {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.6;
    }

    @media (max-width: 760px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero-panel {
            border-radius: 14px;
            padding: 1.2rem;
        }

        .info-box {
            display: block;
        }
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
        <div class="side-panel small-text">

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
        <div class="flow-list">
            <div class="flow-step">Customer Data</div>
            <div class="flow-step">Feature Engineering</div>
            <div class="flow-step">Preprocessing</div>
            <div class="flow-step">Logistic Regression</div>
            <div class="flow-step">Churn Probability</div>
            <div class="flow-step">Risk Classification</div>
        </div>
        """,
        unsafe_allow_html=True
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
    """
    <div class="hero-panel">
        <div class="eyebrow">Telecom retention intelligence</div>
        <div class="main-title">Customer Churn Risk Analyzer</div>
        <div class="subtitle">
        Interactive churn-propensity scoring using the final
        deployed machine-learning pipeline.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="info-box">
    <div><b>Ready to score.</b><br>
    Enter the customer's profile and service information.
    The system will generate a churn probability and compare
    it against the model's locked operating threshold.
    </div>
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
