# app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Set the title of the app
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

# Load the dataset with caching for performance
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

# Load the pre-trained ANN model
@st.cache_resource
def load_ann_model():
    model = load_model("ann_model.h5")
    return model

# Load the scaler for feature standardization
@st.cache_resource
def load_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data.drop(columns=["Class"]))
    return scaler

# Data loading and preprocessing
df = load_data()
model = load_ann_model()
scaler = load_scaler(df)

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", ["Overview", "Business Insights", "Model Explanation", "Fraud Prediction"])

# Page 1: Data Overview
if options == "Overview":
    st.header("Data Overview")
    st.write("This dataset contains credit card transactions. The objective is to detect fraudulent transactions.")
    
    # Display basic statistics
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe().transpose())

    # Display class distribution
    st.subheader("Class Distribution")
    class_counts = df["Class"].value_counts()
    st.bar_chart(class_counts)
    st.write("Class 0: Non-Fraudulent, Class 1: Fraudulent")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Page 2: Business Insights
elif options == "Business Insights":
    st.header("Business Insights")

    # Analysis of transaction amount
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Amount"], kde=True, ax=ax)
    st.pyplot(fig)

    # Time analysis
    st.subheader("Transaction Count Over Time")
    fig, ax = plt.subplots()
    sns.lineplot(x="Time", y="Amount", data=df, ax=ax)
    st.pyplot(fig)

    # Analysis of fraudulent transactions
    st.subheader("Fraudulent vs. Non-Fraudulent Transaction Amounts")
    fig, ax = plt.subplots()
    sns.boxplot(x="Class", y="Amount", data=df, ax=ax)
    st.pyplot(fig)

    st.write("From the analysis, we can see that fraudulent transactions tend to have a lower amount compared to non-fraudulent ones. This insight can be used to set alerts for unusual low-value transactions.")

# Page 3: Model Explanation
elif options == "Model Explanation":
    st.header("Model Overview")

    st.write("""
    The model used is an Artificial Neural Network (ANN) with the following architecture:
    - Input Layer: 30 features after PCA.
    - Hidden Layers: 3 layers with ReLU activation.
    - Output Layer: Sigmoid activation for binary classification.

    ### Advantages:
    - Can capture complex non-linear relationships in the data.
    - High accuracy with low false positive rate, which is crucial for fraud detection.

    ### Disadvantages:
    - Requires a lot of data and computational resources.
    - Can be prone to overfitting if not properly regularized.
    """)

    st.subheader("Model Performance Metrics")
    y = df["Class"]
    X = scaler.transform(df.drop(columns=["Class"]))
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)

    # Display metrics
    st.write("**Accuracy:**", accuracy)
    st.write("**Precision:**", precision)
    st.write("**Recall:**", recall)
    st.write("**F1-Score:**", f1)
    st.write("**ROC-AUC Score:**", roc_auc)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# Page 4: Fraud Prediction
elif options == "Fraud Prediction":
    st.header("Fraud Detection Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(new_data.head())

        # Standardize the new data
        new_data_scaled = scaler.transform(new_data)

        # Make predictions
        predictions = model.predict(new_data_scaled)
        predictions = (predictions > 0.5).astype(int)

        st.subheader("Prediction Results")
        st.write("0 = Non-Fraudulent, 1 = Fraudulent")
        st.dataframe(predictions)

        # Summary
        fraud_count = np.sum(predictions)
        non_fraud_count = len(predictions) - fraud_count
        st.write(f"Fraudulent Transactions: {fraud_count}")
        st.write(f"Non-Fraudulent Transactions: {non_fraud_count}")
