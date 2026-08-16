# -*- coding: utf-8 -*-
import os
import streamlit as st
from utils import predict

# Set page config
st.set_page_config(
    page_title="Credit Risk Modeling & Scoring",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for polished styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏦 Credit Risk Assessment & Scoring Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Machine Learning Model for Loan Default Probability & Rating Estimation</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.write("""
    1. Enter customer financial parameters on the main panel.
    2. Review the automatically calculated **Loan-to-Income (LTI)** ratio.
    3. Click **Calculate Credit Risk & Score** to generate real-time risk predictions.
    """)
    img_path = os.path.join(os.path.dirname(__file__), "Lauki Finance.JPG")
    if os.path.exists(img_path):
        st.image(img_path, caption="Lauki Finance - Trusted Partner", use_column_width=True)

# Layout setup: 2 main columns (Inputs vs Results)
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.subheader("💼 Borrower Financial Profile")
    
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", min_value=18, max_value=100, value=28, help="Borrower age (18-100)")
    income = c2.number_input("Annual Income (₹)", min_value=1000, max_value=10000000, value=300000, step=25000)
    loan_amount = c3.number_input("Requested Loan Amount (₹)", min_value=1000, max_value=50000000, value=2500000, step=50000)

    lti = loan_amount / income if income > 0 else 0
    st.info(f"💡 **Loan-to-Income (LTI) Ratio:** `{lti:.2f}`")

    st.subheader("📑 Loan & Delinquency History")
    c4, c5, c6 = st.columns(3)
    loan_tenure_months = c4.slider("Loan Tenure (Months)", min_value=6, max_value=240, step=6, value=36)
    avg_dpd_per_dm = c5.number_input("Avg DPD (Days Past Due)", min_value=0, max_value=365, value=0, help="Average Delinquent Days across prior loans")
    dmtlm = c6.slider("DMTLM Ratio (%)", min_value=0, max_value=100, value=0, help="Delinquent Months to Loan Month Ratio")

    c7, c8 = st.columns(2)
    credit_utilization_ratio = c7.slider("Credit Utilization Ratio (%)", min_value=0, max_value=100, value=15)
    total_loan_months = c8.number_input("Total Prior Loan Tenure (Months)", min_value=0, value=12)

    st.subheader("🏠 Preferences & Property Status")
    c9, c10, c11 = st.columns(3)
    loan_purpose = c9.selectbox("Loan Purpose", ['Education', 'Home', 'Auto', 'Personal'])
    loan_type = c10.radio("Loan Type", ['Unsecured', 'Secured'])
    residence_type = c11.selectbox("Residence Type", ['Owned', 'Rented', 'Mortgage'])

    calc_btn = st.button("🚀 Calculate Credit Risk & Score", type="primary", use_container_width=True)

with right_col:
    st.subheader("🎯 Risk Assessment Report")
    
    if calc_btn:
        with st.spinner("Processing machine learning inference..."):
            prob, score, rating, badge = predict(
                age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income,
                loan_amount, loan_tenure_months, total_loan_months,
                loan_purpose, loan_type, residence_type
            )
        
        st.markdown(f"### {badge} Credit Rating: **{rating}**")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Credit Score", f"{score} / 900")
        col_res2.metric("Default Probability", f"{prob:.2%}")

        st.progress(max(0.0, min(1.0, (score - 300) / 600.0)))
        
        if rating in ['Poor', 'Average']:
            st.error("🚨 **High Credit Risk Detected!** The borrower exhibits indicators associated with higher default potential.")
        else:
            st.success("✅ **Low Credit Risk Profile!** Borrower meets key underwriting standards for approval.")

        with st.expander("🔍 Summary Breakdown"):
            st.write(f"- **Loan Purpose:** {loan_purpose}")
            st.write(f"- **Loan Type:** {loan_type}")
            st.write(f"- **Residence Status:** {residence_type}")
            st.write(f"- **Calculated LTI:** {lti:.2f}")
    else:
        st.info("👈 Enter borrower information on the left and click **Calculate Credit Risk & Score** to view detailed risk analytics.")
