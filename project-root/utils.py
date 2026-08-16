# -*- coding: utf-8 -*-
"""
Utils module for Credit Risk Modeling inference and scoring.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Resolve paths relative to this script directory
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "model" / "model_data.pkl"


def load_model_data():
    """Load serialized model artifacts from disk with caching if in Streamlit context."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# Load artifacts
_model_data = load_model_data()
model = _model_data['model']
scaler = _model_data['scaler']
features = _model_data['features']
columns_to_scale = _model_data['cols_to_scale']


def data_preparation(
    age: int,
    avg_dpd_per_dm: float,
    credit_utilization_ratio: float,
    dmtlm: float,
    income: float,
    loan_amount: float,
    loan_tenure_months: int,
    total_loan_months: int,
    loan_purpose: str,
    loan_type: str,
    residence_type: str,
) -> pd.DataFrame:
    """
    Prepare and scale user input features for model inference.
    """
    lti = (loan_amount / income) if income > 0 else 0.0

    data_input = {
        'age': float(age),
        'avg_dpd_per_dm': float(avg_dpd_per_dm),
        'credit_utilization_ratio': float(credit_utilization_ratio),
        'dmtlm': float(dmtlm),
        'income': float(income),
        'loan_amount': float(loan_amount),
        'lti': float(lti),
        'total_loan_months': float(total_loan_months),
        'loan_tenure_months': float(loan_tenure_months),
        'loan_purpose_Education': 1.0 if loan_purpose == 'Education' else 0.0,
        'loan_purpose_Home': 1.0 if loan_purpose == 'Home' else 0.0,
        'loan_purpose_Personal': 1.0 if loan_purpose == 'Personal' else 0.0,
        'loan_type_Unsecured': 1.0 if loan_type == 'Unsecured' else 0.0,
        'residence_type_Owned': 1.0 if residence_type == 'Owned' else 0.0,
        'residence_type_Rented': 1.0 if residence_type == 'Rented' else 0.0,
    }

    df = pd.DataFrame([data_input])
    # Apply standard scaling to numerical features
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    # Select columns in exact order expected by trained model
    df = df[features]

    return df


def calculate_credit_score(input_df: pd.DataFrame, base_score: int = 300, scale_length: int = 600):
    """
    Compute default probability, credit score (300-900), and categorical rating.
    """
    proba = model.predict_proba(input_df)
    default_probability = float(proba[0, 1])
    non_default_probability = 1.0 - default_probability

    # Calculate credit score (300 - 900)
    raw_score = base_score + (non_default_probability * scale_length)
    credit_score = int(round(raw_score))
    # Clamp score within valid bounds [300, 900]
    credit_score = max(300, min(900, credit_score))

    def get_rating(score: int):
        if 300 <= score < 500:
            return 'Poor', '🔴'
        elif 500 <= score < 650:
            return 'Average', '🟠'
        elif 650 <= score < 750:
            return 'Good', '🟡'
        elif 750 <= score <= 900:
            return 'Excellent', '🟢'
        else:
            return 'Undefined', '⚪'

    rating, badge = get_rating(credit_score)

    return default_probability, credit_score, rating, badge


def predict(
    age: int,
    avg_dpd_per_dm: float,
    credit_utilization_ratio: float,
    dmtlm: float,
    income: float,
    loan_amount: float,
    loan_tenure_months: int,
    total_loan_months: int,
    loan_purpose: str,
    loan_type: str,
    residence_type: str,
):
    """
    End-to-end prediction pipeline from raw inputs to evaluated credit score.
    """
    input_df = data_preparation(
        age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income,
        loan_amount, loan_tenure_months, total_loan_months,
        loan_purpose, loan_type, residence_type
    )

    probability, credit_score, rating, badge = calculate_credit_score(input_df)

    return probability, credit_score, rating, badge

