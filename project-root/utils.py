# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd

# Load model data safely with relative path resolving
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_data.pkl")

@st_cache_resource_or_load()
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def st_cache_resource_or_load():
    pass # Helper placeholder, real load below

model_data = load_model_data() if os.path.exists(MODEL_PATH) else joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
columns_to_scale = model_data['cols_to_scale']


def data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                     loan_amount, loan_tenure_months, total_loan_months, 
                     loan_purpose, loan_type, residence_type):
    data_input = {
        'age': age,
        'avg_dpd_per_dm': avg_dpd_per_dm,
        'credit_utilization_ratio': credit_utilization_ratio,
        'dmtlm': dmtlm,
        'income': income,
        'loan_amount': loan_amount,
        'lti': loan_amount / income if income > 0 else 0,
        'total_loan_months': total_loan_months,
        'loan_tenure_months': loan_tenure_months,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0
    }
    
    df = pd.DataFrame([data_input])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    df = df[features]
    
    return df


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    default_probability = model.predict_proba(input_df)[:, 1][0]  # Probability of default
    non_default_probability = 1.0 - default_probability

    # Calculate the credit score based on probabilities
    credit_score = int(base_score + non_default_probability * scale_length)
    
    def get_rating(score):
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

    return float(default_probability), credit_score, rating, badge


def predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
            loan_amount, loan_tenure_months, total_loan_months, 
            loan_purpose, loan_type, residence_type):

    input_df = data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                                 loan_amount, loan_tenure_months, total_loan_months, 
                                 loan_purpose, loan_type, residence_type)

    probability, credit_score, rating, badge = calculate_credit_score(input_df)

    return probability, credit_score, rating, badge
