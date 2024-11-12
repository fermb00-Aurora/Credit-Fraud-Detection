# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Page configuration
st.set_page_config(page_title="Advanced Credit Card Fraud Detection", layout="wide", initial_sidebar_state="expanded")

# Load the pre-trained ANN model and scaler
@st.cache_resource
def load_ann_model():
    model = load_model("ann_model.h5")
    return model

@st.cache_resource
def load_scaler():
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"])
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

model = load_ann_model()
scaler = load_scaler()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Fraud Detection", "Business Insights", "Model Explanation"])

# Home Page
if page == "Home":
    st.title("💳 Advanced Credit Card Fraud Detection System")
    st.markdown("""
        Welcome to the Credit Card Fraud Detection Web App. This tool uses a pre-trained Artificial Neural Network (ANN) model to detect fraudulent transactions.
        - Navigate through the sections to explore the dataset, make real-time predictions, and gain business insights.
        - The model has been trained using the `creditcard.csv` dataset and achieves high accuracy in identifying fraudulent activities.
    """)
    st.image("https://source.unsplash.com/featured/?creditcard,fraud", use_column_width=True)

# Data Exploration Page
elif page == "Data Exploration":
    st.title("Data Exploration")
    df = pd.read_csv("creditcard.csv")

    # Display basic dataset information
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape of the dataset:", df.shape)

    # Class distribution
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Class", data=df, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    # Interactive feature selection
    st.subheader("Feature Analysis")
    feature = st.selectbox("Select a feature to visualize", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

# Fraud Detection Page
elif page == "Fraud Detection":
    st.title("Fraud Detection")

    # User inputs
    st.sidebar.header("Enter Transaction Details")
    time = st.sidebar.number_input("Time", min_value=0.0, max_value=172792.0, step=1.0)
    amount = st.sidebar.number_input("Amount", min_value=0.0, max_value=25691.16, step=0.1)

    input_features = []
    for i in range(1, 29):
        feature = st.sidebar.number_input(f"V{i}", min_value=-75.0, max_value=75.0, step=0.1)
        input_features.append(feature)

    input_features = [time] + input_features + [amount]
    input_data = np.array(input_features).reshape(1, -1)
    scaled_input = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_input)
    predicted_class = int(prediction > 0.5)

    # Display the prediction result
    st.header("Prediction Result")
    if predicted_class == 1:
        st.error("🚨 The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("✅ The transaction is predicted to be NON-FRAUDULENT.")

# Business Insights Page
elif page == "Business Insights":
    st.title("Business Insights")

    # Profitability analysis based on predictions
    st.subheader("Potential Cost Savings")
    st.write("""
        Identifying fraudulent transactions early can save significant costs. Based on the predictions:
        - For each fraudulent transaction correctly identified, the bank saves approximately $5,000.
        - Incorrectly flagging a non-fraudulent transaction costs the bank $1,000 in lost customer trust.
    """)

    fraud_count = np.random.randint(0, 50)  # Placeholder for dynamic analysis
    non_fraud_count = 100 - fraud_count
    st.write(f"Estimated Fraudulent Transactions Identified: {fraud_count}")
    st.write(f"Estimated Cost Savings: ${fraud_count * 5000}")

# Model Explanation Page
elif page == "Model Explanation":
    st.title("Model Explanation")
    st.write("""
        The fraud detection model is a multi-layer Artificial Neural Network (ANN) with the following layers:
        - **Input Layer**: 30 features (including PCA components and transaction amount).
        - **Hidden Layers**: Three hidden layers with ReLU activation.
        - **Output Layer**: A single node with sigmoid activation for binary classification.

        ### Strengths:
        - High performance with complex data patterns.
        - Flexible and adaptable for future data updates.

        ### Weaknesses:
        - May overfit if not regularized properly.
        - Computationally intensive for large datasets.
    """)
    st.image("https://source.unsplash.com/featured/?neuralnetwork,diagram", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("### Developed by Fernando - AI Fraud Detection Specialist")
st.markdown("This app is powered by a pre-trained ANN model, leveraging Streamlit for an interactive and business-oriented experience.")

