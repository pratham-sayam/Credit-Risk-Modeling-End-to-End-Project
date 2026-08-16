# -*- coding: utf-8 -*-
"""
Streamlit Application for Credit Risk Assessment & Scoring Portal.
"""

from pathlib import Path
import streamlit as st
from utils import predict

# Set page config
st.set_page_config(
    page_title="Credit Risk Modeling & Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, theme-adaptive styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Credit Risk Assessment & Scoring Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Underwriting Model for Loan Default Probability & Rating Estimation</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.write("""
    1. Enter customer financial parameters on the main panel.
    2. Review the calculated **Loan-to-Income (LTI)** ratio.
    3. Click **Calculate Credit Risk & Score** to generate real-time risk predictions.
    """)
    
    img_path = Path(__file__).resolve().parent / "Lauki Finance.JPG"
    if img_path.exists():
        st.image(str(img_path), caption="Lauki Finance - Trusted Partner", use_container_width=True)
    
    st.divider()
    st.markdown("### 📊 Score Legend")
    st.markdown("""
    - **750 - 900**: 🟢 Excellent (Low Risk)
    - **650 - 749**: 🟡 Good (Standard Risk)
    - **500 - 649**: 🟠 Average (Moderate Risk)
    - **300 - 499**: 🔴 Poor (High Risk)
    """)

# Layout: 2 main columns (Inputs vs Results)
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.subheader("💼 Borrower Financial Profile")
    
    with st.form(key="credit_risk_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (Years)", min_value=18, max_value=100, value=28, help="Borrower age (18-100)")
        income = c2.number_input("Annual Income (₹)", min_value=1000, max_value=10000000, value=300000, step=25000, help="Annual verifiable gross income")
        loan_amount = c3.number_input("Requested Loan Amount (₹)", min_value=1000, max_value=50000000, value=2500000, step=50000, help="Total requested loan principal")

        st.subheader("📑 Loan & Delinquency History")
        c4, c5, c6 = st.columns(3)
        loan_tenure_months = c4.slider("Loan Tenure (Months)", min_value=6, max_value=240, step=6, value=36, help="Tenure of requested loan in months")
        avg_dpd_per_dm = c5.number_input("Avg DPD (Days Past Due)", min_value=0, max_value=365, value=0, help="Average Delinquent Days across prior loans")
        dmtlm = c6.slider("DMTLM Ratio (%)", min_value=0, max_value=100, value=0, help="Delinquent Months to Loan Month Ratio (%)")

        c7, c8 = st.columns(2)
        credit_utilization_ratio = c7.slider("Credit Utilization Ratio (%)", min_value=0, max_value=100, value=15, help="Percentage of revolving credit currently utilized")
        total_loan_months = c8.number_input("Total Prior Loan Tenure (Months)", min_value=0, value=12, help="Cumulative tenure across historical loans")

        st.subheader("🏠 Preferences & Property Status")
        c9, c10, c11 = st.columns(3)
        loan_purpose = c9.selectbox("Loan Purpose", ['Education', 'Home', 'Auto', 'Personal'], help="Purpose of the loan application")
        loan_type = c10.radio("Loan Type", ['Unsecured', 'Secured'], help="Secured (collateralized) vs Unsecured")
        residence_type = c11.selectbox("Residence Type", ['Owned', 'Rented', 'Mortgage'], help="Current housing ownership status")

        calc_btn = st.form_submit_button("🚀 Calculate Credit Risk & Score", type="primary", use_container_width=True)

    lti_val = loan_amount / income if income > 0 else 0.0
    if lti_val > 5.0:
        st.warning(f"⚠️ **High Loan-to-Income (LTI) Ratio detected (`{lti_val:.2f}`):** The loan amount is more than 5x annual income.")
    else:
        st.info(f"💡 **Current Loan-to-Income (LTI) Ratio:** `{lti_val:.2f}`")

with right_col:
    st.subheader("🎯 Risk Assessment Report")
    
    if calc_btn:
        with st.spinner("Executing machine learning scoring model..."):
            prob, score, rating, badge = predict(
                age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income,
                loan_amount, loan_tenure_months, total_loan_months,
                loan_purpose, loan_type, residence_type
            )
        
        st.markdown(f"### {badge} Credit Rating: **{rating}**")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Credit Score", f"{score} / 900", help="Derived standardized credit score")
        col_res2.metric("Default Probability", f"{prob:.2%}", delta=f"{-prob:.2%}", delta_color="inverse", help="Estimated likelihood of default")

        normalized_score = max(0.0, min(1.0, (score - 300) / 600.0))
        st.progress(normalized_score)
        st.caption(f"Score Scale Position: **{score}** (Base: 300 — Max: 900)")
        
        if rating in ['Poor', 'Average']:
            st.error("🚨 **High Credit Risk Detected!** The borrower exhibits indicators associated with higher default potential. Additional collateral or guarantor recommended.")
        else:
            st.success("✅ **Low Credit Risk Profile!** Borrower meets key underwriting standards for loan consideration.")

        with st.expander("🔍 Application Summary Breakdown", expanded=True):
            st.write(f"- **Loan Purpose:** {loan_purpose}")
            st.write(f"- **Loan Type:** {loan_type}")
            st.write(f"- **Residence Status:** {residence_type}")
            st.write(f"- **Calculated LTI Ratio:** {lti_val:.2f}")
            st.write(f"- **Credit Utilization:** {credit_utilization_ratio}%")
            st.write(f"- **Prior Delinquency Ratio (DMTLM):** {dmtlm}%")
    else:
        st.info("👈 Complete the borrower information form on the left and click **Calculate Credit Risk & Score** to view the real-time underwriting report.")

