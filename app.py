import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the combined model (includes classifier and scaler if you saved both in one file)
model = joblib.load("credit_card_fraud_model.pkl")

# Title and Image
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("🔒 Credit Card Fraud Detection App")
st.markdown("""Use this app to check if a credit card transaction is fraudulent based on 28 PCA components and  Time and Amount features.""")

st.markdown("""
<div style='background-color:#fff3cd; padding:15px; border-left:5px solid #ffecb5; border-radius:5px;'>
    <strong>📌 Important Note:</strong><br>
    <span style='color:#856404;'>
    <b>It contains only numerical input variables which are the result of a PCA transformation.</b><br>
    Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.<br>
    Features <b>V1, V2, … V28</b> are the principal components obtained with PCA, the only features which have not been transformed with PCA are <b>'Time'</b> and <b>'Amount'</b>.
    </span>
</div>
""", unsafe_allow_html=True)


image = Image.open("credit_card_fraud_banner.jpg")
st.image(image, use_container_width=True)



# Feature Input
st.header("🔎 Enter Transaction Details:")

def user_input():
    input_data = {}
    features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Scaled_Time', 'Scaled_Amount']

    for i, feature in enumerate(features):
        default_val = 0.0 if feature.startswith("V") else 0.5
        input_data[feature] = st.number_input(f"{feature}", value=default_val, step=0.1, format="%.4f")

    return pd.DataFrame([input_data])

input_df = user_input()

# Prediction
if st.button("🚀 Predict Fraud"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This transaction is predicted to be FRAUDULENT with probability {probability:.2f}.")
    else:
        st.success(f"✅ This transaction is predicted to be LEGITIMATE with probability {1 - probability:.2f}.")



