<div align="center">

# 📊 Churn Propensity Modeling using Statistical Inference, ML Model Benchmarking & SHAP-Based Feature Attribution

### An end-to-end customer churn modeling project focused on statistical reasoning, model generalization, threshold optimization, explainability, and practical deployment

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Project-Complete-2EA44F?style=for-the-badge)

</div>

---

## 🎯 Project Overview

Customer churn is not only a classification problem.

A useful churn system should answer:

- Which customers are most likely to leave?
- How reliably can the model rank churn risk?
- Which features are genuinely informative?
- What threshold should convert probability into an actionable churn flag?
- Why did the model assign a customer high or low risk?
- How can the model be exposed through a simple application?

This project approaches churn from both a **statistical** and **machine learning** perspective.

The workflow combines:

**EDA → Statistical Inference → Feature Engineering → Feature Selection → Model Benchmarking → Cross-Validation → Hyperparameter Tuning → Threshold Optimization → Final Test Evaluation → SHAP Explainability → Deployment**

---

# 🧠 Core Objective

The goal is to estimate the probability that a telecom customer will churn and convert that probability into an interpretable and actionable risk signal.

Instead of focusing only on raw accuracy, the project emphasizes:

- churn probability
- ranking quality
- recall and precision trade-offs
- class imbalance
- feature redundancy
- generalization to unseen customers
- explainability
- business decision thresholds

---

# 🗺️ End-to-End Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Statistical Inference
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Train-Only Feature Diagnostics
        ↓
Feature Selection
        ↓
Preprocessing Pipeline
        ↓
Cross-Validated Model Benchmarking
        ↓
Hyperparameter Tuning
        ↓
OOF Probability Generation
        ↓
Threshold Optimization
        ↓
Lock Final Model
        ↓
Untouched Test Evaluation
        ↓
Coefficient + SHAP Interpretation
        ↓
Customer Risk Prediction
        ↓
Streamlit Deployment
