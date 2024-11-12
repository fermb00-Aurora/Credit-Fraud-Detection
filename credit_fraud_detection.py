# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt

# Set the page configuration with a banking/architecture theme
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load the pre-trained ANN model and the scaler
@st.cache_resource
def load_ann_model():
    return load_model("ann_model.h5")

@st.cache_resource
def load_scaler():
    return StandardScaler()

# Initialize model and scaler
model = load_ann_model()
scaler = load_scaler()

# Dashboard Header
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🔍 Credit Card Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>A seamless blend of architectural aesthetics and financial security.</h4>", unsafe_allow_html=True)

# Dashboard Overview with Key Statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions Analyzed", "284,807")
with col2:
    st.metric("Model Accuracy", "99.7%")
with col3:
    st.metric("Potential Savings per Fraudulent Transaction", "$5,000")

# Guided Input Form
st.markdown("## 📋 Enter Transaction Details")
st.write("Fill in the transaction details below for real-time fraud prediction.")

# Collect transaction details
time = st.number_input("Transaction Time (seconds)", min_value=0.0, max_value=172792.0, step=1.0, help="Time elapsed since the first transaction in the dataset.")
amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25691.16, step=0.1, help="Enter the amount of the transaction.")

# Display PCA features with brief explanations using tooltips
input_features = [time]
for i in range(1, 29):
    feature = st.number_input(f"V{i} (PCA Component)", min_value=-75.0, max_value=75.0, step=0.1, help=f"Principal Component Analysis (PCA) feature V{i}.")
    input_features.append(feature)

input_features.append(amount)
input_data = np.array(input_features).reshape(1, -1)
scaled_input = scaler.transform(input_data)

# Prediction and Confidence Indicator
prediction = model.predict(scaled_input)
predicted_class = int(prediction > 0.5)
confidence = round(float(prediction) * 100, 2)

# Display Prediction Result
st.markdown("## 🛡️ Prediction Result")
if predicted_class == 1:
    st.error("🚨 The transaction is predicted to be FRAUDULENT.", icon="🚨")
    st.markdown(f"<h3 style='color: #e74c3c;'>Confidence: {confidence}%</h3>", unsafe_allow_html=True)
else:
    st.success("✅ The transaction is predicted to be NON-FRAUDULENT.", icon="✅")
    st.markdown(f"<h3 style='color: #27ae60;'>Confidence: {confidence}%</h3>", unsafe_allow_html=True)

# Profitability Calculator
st.markdown("## 💰 Profitability Analysis")
savings = 5000 if predicted_class == 1 else 0
st.write(f"By flagging this transaction, the potential savings are estimated at **${savings:,}**.")

# Optional EDA Toggle
st.markdown("## 🔎 Exploratory Data Analysis (Optional)")
if st.checkbox("Show EDA"):
    fig, ax = plt.subplots()
    ax.hist(input_features[1:-1], bins=15, color="#3498db", alpha=0.7)
    ax.set_title("Distribution of PCA Features (V1 to V28)")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Downloadable Report
st.markdown("## 📄 Download Analysis Report")
report_data = f"""
Transaction Time: {time}
Transaction Amount: ${amount}
Prediction: {'Fraudulent' if predicted_class == 1 else 'Non-Fraudulent'}
Confidence: {confidence}%
Potential Savings: ${savings}
"""
st.download_button(label="Download Report", data=report_data, file_name="fraud_analysis_report.txt")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Developed by Fernando - Bridging Architecture and Finance with AI</h4>", unsafe_allow_html=True)
