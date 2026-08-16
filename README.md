# 📊 Credit Risk Modeling & Scoring System (End-to-End ML Pipeline)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning solution for credit risk evaluation, default probability estimation, and credit rating score generation. Built with **XGBoost**, **Scikit-Learn**, and deployed using an interactive **Streamlit** dashboard.

---

## 🌟 Key Features

- **Default Probability Estimation**: Uses trained XGBoost classifier algorithms to predict borrower default likelihood.
- **Credit Score Engine**: Maps non-default probabilities into a standardized credit score range (300 to 900).
- **Automated Rating Categorization**: Automatically groups applicants into **Poor**, **Average**, **Good**, or **Excellent** credit bands.
- **Dynamic LTI Ratio Calculator**: Real-time evaluation of Loan-to-Income impact on creditworthiness.
- **Interactive Streamlit Web Dashboard**: User-friendly UI with real-time feedback, visual metrics, and progress indicators.

---

## 📁 Repository Structure

```text
Credit-Risk-Modeling-End-to-End-Project/
├── project-root/
│   ├── main.py              # Streamlit Web Application entry point
│   ├── utils.py             # Inference pipeline & credit scoring helper functions
│   ├── requirements.txt     # Subfolder dependencies
│   ├── Lauki Finance.JPG    # UI Branding image
│   └── model/
│       ├── model_data.pkl   # Serialized model, scaler, feature list & metadata
│       └── tuned_hyperparameters.txt
├── notebooks/               # Jupyter notebooks for EDA & Model Development
├── requirements.txt         # Root requirements file for Cloud deployment
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pratham-sayam/Credit-Risk-Modeling-End-to-End-Project.git
cd Credit-Risk-Modeling-End-to-End-Project
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run project-root/main.py
```

---

## 📊 Credit Score Categorization Scale

| Credit Score Range | Rating Category | Default Risk Assessment |
| :--- | :--- | :--- |
| **300 - 499** | 🔴 **Poor** | High Risk - Default Likely |
| **500 - 649** | 🟠 **Average** | Moderate Risk - Extra Verification Needed |
| **650 - 749** | 🟡 **Good** | Low Risk - Standard Processing |
| **750 - 900** | 🟢 **Excellent** | Very Low Risk - Priority Approval |

---

## 🚀 Cloud Deployment (Streamlit Community Cloud)

1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"** and select this GitHub repository.
4. Set **Main file path** to: `project-root/main.py`
5. Click **Deploy!**

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
