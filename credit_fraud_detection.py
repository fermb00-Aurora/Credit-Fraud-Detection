# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Set the title of the Streamlit app
st.title("Fraud Detection System Using Pre-trained ANN Model")

# Load the pre-trained model
@st.cache_resource
def load_ann_model():
    model = load_model("ann_model.h5")
    return model

model = load_ann_model()

# Load the scaler
@st.cache_resource
def load_scaler():
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"])
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

scaler = load_scaler()

# User interface for uploading CSV file
uploaded_file = st.file_uploader("Upload a CSV file for prediction")

if uploaded_file is not None:
    # Read the uploaded CSV file
    st.subheader("Uploaded Data Preview")
    new_data = pd.read_csv(uploaded_file)
    st.dataframe(new_data.head())

    # Scale the input data
    st.write("Scaling the input data...")
    new_data_scaled = scaler.transform(new_data)

    # Make predictions
    st.write("Making predictions...")
    predictions = model.predict(new_data_scaled)
    predictions = (predictions > 0.5).astype(int)

    # Display results
    st.subheader("Prediction Results")
    st.write("0 = Non-Fraudulent, 1 = Fraudulent")
    st.dataframe(predictions)

    # Summary of predictions
    fraud_count = np.sum(predictions)
    non_fraud_count = len(predictions) - fraud_count
    st.write(f"Fraudulent Transactions: {fraud_count}")
    st.write(f"Non-Fraudulent Transactions: {non_fraud_count}")
