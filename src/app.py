import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# ── Load model ─────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'attrition_model.pkl')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Header ─────────────────────────────────────────────────────────
st.title("👥 Employee Attrition Predictor")
st.markdown("Enter employee details below to predict the likelihood of attrition and understand the key drivers.")
st.divider()

# ── Input form ─────────────────────────────────────────────────────
st.subheader("Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    Age              = st.slider("Age", 18, 60, 30)
    Gender           = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus    = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    DistanceFromHome = st.slider("Distance From Home (km)", 1, 29, 5)
    Education        = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                         format_func=lambda x: {1:"Below College", 2:"College", 3:"Bachelor", 4:"Master", 5:"Doctor"}[x])
    EducationField   = st.selectbox("Education Field",
                         ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])

with col2:
    st.markdown("**Job Info**")
    Department        = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    JobRole           = st.selectbox("Job Role", [
                            "Sales Executive", "Research Scientist", "Laboratory Technician",
                            "Manufacturing Director", "Healthcare Representative", "Manager",
                            "Sales Representative", "Research Director", "Human Resources"])
    JobLevel          = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    BusinessTravel    = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    OverTime          = st.selectbox("OverTime", ["Yes", "No"])
    MonthlyIncome     = st.number_input("Monthly Income ($)", min_value=1009, max_value=19999, value=5000, step=100)
    DailyRate         = st.slider("Daily Rate", 102, 1499, 800)
    HourlyRate        = st.slider("Hourly Rate", 30, 100, 65)
    MonthlyRate       = st.slider("Monthly Rate", 2094, 26999, 14000)
    PercentSalaryHike = st.slider("Percent Salary Hike (%)", 11, 25, 15)

with col3:
    st.markdown("**Satisfaction & Experience**")
    JobSatisfaction          = st.selectbox("Job Satisfaction", [1, 2, 3, 4],
                                 format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x])
    EnvironmentSatisfaction  = st.selectbox("Environment Satisfaction", [1, 2, 3, 4],
                                 format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x])
    WorkLifeBalance          = st.selectbox("Work Life Balance", [1, 2, 3, 4],
                                 format_func=lambda x: {1:"Bad", 2:"Good", 3:"Better", 4:"Best"}[x])
    JobInvolvement           = st.selectbox("Job Involvement", [1, 2, 3, 4],
                                 format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x])
    RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4],
                                 format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x])
    PerformanceRating        = st.selectbox("Performance Rating", [3, 4],
                                 format_func=lambda x: {3:"Excellent", 4:"Outstanding"}[x])
    StockOptionLevel         = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    NumCompaniesWorked       = st.slider("Num Companies Worked", 0, 9, 2)
    TotalWorkingYears        = st.slider("Total Working Years", 0, 40, 10)
    TrainingTimesLastYear    = st.slider("Training Times Last Year", 0, 6, 3)
    YearsAtCompany           = st.slider("Years At Company", 0, 40, 5)
    YearsInCurrentRole       = st.slider("Years In Current Role", 0, 18, 3)
    YearsSinceLastPromotion  = st.slider("Years Since Last Promotion", 0, 15, 2)
    YearsWithCurrManager     = st.slider("Years With Current Manager", 0, 17, 3)

st.divider()

# ── Predict ────────────────────────────────────────────────────────
if st.button("🔍 Predict Attrition Risk", use_container_width=True, type="primary"):

    input_df = pd.DataFrame([{
        "Age": Age, "BusinessTravel": BusinessTravel, "DailyRate": DailyRate,
        "Department": Department, "DistanceFromHome": DistanceFromHome,
        "Education": Education, "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction, "Gender": Gender,
        "HourlyRate": HourlyRate, "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel, "JobRole": JobRole, "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus, "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate, "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime, "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel, "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear, "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "WorkLifeBalance": WorkLifeBalance  # was missing
    }])

    prob       = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    if prob < 0.3:
        risk_label = 'Low'
    elif prob < 0.6:
        risk_label = 'Medium'
    else:
        risk_label = 'High'

    # ── Result ─────────────────────────────────────────────────────
    st.subheader("Prediction Result")
    r1, r2, r3 = st.columns(3)

    with r1:
        if prediction == 1:
            st.error("⚠️ High Attrition Risk")
        else:
            st.success("✅ Low Attrition Risk")
    with r2:
        st.metric("Attrition Probability", f"{prob*100:.1f}%")
    with r3:
        emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk_label]
        st.metric("Risk Level", f"{emoji} {risk_label}")

    st.divider()

    # ── SHAP ───────────────────────────────────────────────────────
    st.subheader("🔎 Why This Prediction? — SHAP Explanation")
    st.caption("Red bars push toward attrition. Blue bars push away from attrition.")

    transformer   = model.named_steps['transformer']
    gb_model      = model.named_steps['model']
    feature_names = transformer.get_feature_names_out().tolist()

    input_transformed = transformer.transform(input_df)
    explainer         = shap.TreeExplainer(gb_model)
    shap_values       = explainer.shap_values(input_transformed)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap.Explanation(
        values        = shap_values[0],
        base_values   = explainer.expected_value,
        data          = input_transformed[0],
        feature_names = feature_names
    ), show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()