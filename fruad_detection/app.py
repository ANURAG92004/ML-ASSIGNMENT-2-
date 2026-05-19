# ============================================================
#   CREDIT CARD FRAUD DETECTION - STREAMLIT WEB APP
#   Run: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
    
# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        color: #185FA5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 0.95rem;
    }
    .fraud-box {
        background: #FCEBEB;
        border: 2px solid #E24B4A;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .safe-box {
        background: #EAF3DE;
        border: 2px solid #639922;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-sub {
        font-size: 0.9rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Header
st.markdown('<div class="main-title">🔍 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ML-powered real-time transaction fraud analysis</div>', unsafe_allow_html=True)

st.divider()

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About This App")
    st.info("""
    This app uses a machine learning model trained on 284,807 real credit card transactions to detect fraud in real-time.

    **Model:** XGBoost Classifier
    **Dataset:** Kaggle ULB Credit Card Fraud
    **Accuracy:** ~99%
    """)
    st.header("📊 Dataset Info")
    st.metric("Total Transactions", "284,807")
    st.metric("Fraud Cases", "492 (0.17%)")
    st.metric("Features Used", "30")

# Input section
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("💰 Transaction Amount (₹)", min_value=0.0, value=150.0, step=10.0)
    time = st.number_input("⏱ Time (seconds since first transaction)", min_value=0.0, value=50000.0)

with col2:
    st.info("V1–V28 are PCA-transformed anonymized features from the bank. Use default values for demo.")

st.subheader("PCA Features (V1 – V28)")

# Create 4 columns for V features
cols = st.columns(4)
v_values = []
defaults = [
    -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
     0.36,  0.09, -0.55, -0.62, -0.99, -0.31, 1.47, 0.21,
     0.40,  0.03,  0.40, 0.25, -0.02, 0.28, -0.11, 0.07,
     0.13,  -0.19, 0.13, -0.02
]

for i in range(28):
    with cols[i % 4]:
        val = st.number_input(f"V{i+1}", value=defaults[i], format="%.4f", key=f"v{i+1}")
        v_values.append(val)

st.divider()

# Predict button
if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):
    features = np.array([[time, *v_values, amount]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    fraud_prob = round(probability[1] * 100, 2)
    legit_prob = round(probability[0] * 100, 2)

    if prediction == 1:
        st.markdown(f"""
        <div class="fraud-box">
            <div class="result-label" style="color:#A32D2D;">⚠️ FRAUD DETECTED</div>
            <div class="result-sub" style="color:#A32D2D;">This transaction is flagged as fraudulent</div>
            <br>
            <b>Fraud Probability: {fraud_prob}%</b>
        </div>
        """, unsafe_allow_html=True)
        st.error(f"🚨 Recommendation: Block this transaction immediately!")
    else:
        st.markdown(f"""
        <div class="safe-box">
            <div class="result-label" style="color:#3B6D11;">✅ LEGITIMATE</div>
            <div class="result-sub" style="color:#3B6D11;">This transaction appears to be safe</div>
            <br>
            <b>Legitimate Probability: {legit_prob}%</b>
        </div>
        """, unsafe_allow_html=True)
        st.success(f"✅ Recommendation: Transaction can proceed safely.")

    # Probability bar
    st.subheader("Confidence Breakdown")
    col1, col2 = st.columns(2)
    col1.metric("Legitimate", f"{legit_prob}%")
    col2.metric("Fraud", f"{fraud_prob}%")
    st.progress(fraud_prob / 100)

st.divider()
st.caption("ML Project | Credit Card Fraud Detection | Made with Streamlit")