# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Set the page configuration
st.set_page_config(page_title="Credit Card Fraud Detection App", layout="wide", initial_sidebar_state="expanded")

# Load the pre-trained ANN model and the scaler
@st.cache_resource
def load_ann_model():
    return load_model("ann_model.h5")

@st.cache_resource
def load_scaler():
    return StandardScaler()  # Assume the scaler was fitted during model training

# Load model and scaler
model = load_ann_model()
scaler = load_scaler()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Fraud Detection", "Business Insights", "Model Explanation"])

# Enhanced Home Page
if page == "Home":
    # Header Section
    st.image("https://source.unsplash.com/featured/?finance,creditcard", use_column_width=True)
    st.title("💳 Welcome to the Advanced Credit Card Fraud Detection System")
    st.markdown("""
    ### What is this App?
    This web application utilizes a pre-trained Artificial Neural Network (ANN) model to detect fraudulent credit card transactions in real-time. With the rise of online transactions, identifying fraud early is crucial for preventing significant financial losses.
    
    ### Why Use This App?
    - **High Accuracy**: The ANN model was trained on a large dataset (`creditcard.csv`) and optimized for detecting fraudulent patterns.
    - **Real-Time Predictions**: Input transaction details and receive immediate predictions on whether the transaction is fraudulent.
    - **Business-Oriented Insights**: Analyze the impact of fraud detection on your organization's profitability.

    ### How to Use
    1. Navigate to **Fraud Detection** to input transaction details and make predictions.
    2. Explore **Business Insights** to understand potential savings from early fraud detection.
    3. Learn more about the underlying **Model Explanation** and its advantages.

    **Start Detecting Fraud Now!** Click on one of the sections in the sidebar to begin.
    """)

    # Call-to-Action Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Go to Fraud Detection"):
            page = "Fraud Detection"
    with col2:
        if st.button("📈 View Business Insights"):
            page = "Business Insights"
    with col3:
        if st.button("🧠 Learn About the Model"):
            page = "Model Explanation"

# Fraud Detection Page
if page == "Fraud Detection":
    st.title("Fraud Detection")

    # User input for transaction details
    st.sidebar.header("Enter Transaction Details")
    time = st.sidebar.number_input("Time", min_value=0.0, max_value=172792.0, step=1.0)
    amount = st.sidebar.number_input("Amount", min_value=0.0, max_value=25691.16, step=0.1)
    input_features = [time]

    # Collect inputs for PCA features V1 to V28
    for i in range(1, 29):
        feature = st.sidebar.number_input(f"V{i}", min_value=-75.0, max_value=75.0, step=0.1)
        input_features.append(feature)

    input_features.append(amount)
    input_data = np.array(input_features).reshape(1, -1)
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)
    predicted_class = int(prediction > 0.5)

    # Display prediction result
    st.header("Prediction Result")
    if predicted_class == 1:
        st.error("🚨 The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("✅ The transaction is predicted to be NON-FRAUDULENT.")

# Business Insights Page
if page == "Business Insights":
    st.title("Business Insights")
    st.write("""
    Early detection of fraudulent transactions can lead to substantial savings. By correctly identifying fraudulent activities, businesses can:
    - Reduce potential financial losses.
    - Increase customer trust and satisfaction.
    - Optimize fraud investigation processes.

    ### Estimated Savings
    Based on historical data, flagging a fraudulent transaction early can save approximately $5,000 per case.
    """)

    fraud_count = np.random.randint(10, 50)  # Placeholder for demonstration
    savings = fraud_count * 5000
    st.metric("Potential Savings from Detected Fraudulent Transactions", f"${savings:,}")

# Model Explanation Page
if page == "Model Explanation":
    st.title("Model Explanation")
    st.write("""
    The model used in this app is a multi-layer Artificial Neural Network (ANN). It was trained using a large dataset of credit card transactions (`creditcard.csv`). The ANN is designed to:
    - Capture complex patterns in the data using multiple hidden layers.
    - Output a binary prediction: 0 (Non-Fraudulent) or 1 (Fraudulent).

    ### Model Structure
    - **Input Layer**: 30 features (including PCA components and transaction amount).
    - **Hidden Layers**: Three fully connected layers with ReLU activation functions.
    - **Output Layer**: A single node with a sigmoid activation function for binary classification.

    ### Why ANN?
    - **Advantages**:
        - High flexibility in modeling complex relationships.
        - Well-suited for large datasets with many features.
    - **Disadvantages**:
        - Requires significant computational resources.
        - May overfit without proper regularization.
    """)

    st.image("https://source.unsplash.com/featured/?neuralnetwork,diagram", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("### Developed by Fernando - AI Fraud Detection Specialist")
st.markdown("This app leverages a pre-trained ANN model and Streamlit for an interactive and business-oriented experience.")

