import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import os

# Check package versions to ensure compatibility
import tensorflow as tf
import ml_dtypes

st.write(f"TensorFlow Version: {tf.__version__}")
st.write(f"ML-Dtypes Version: {ml_dtypes.__version__}")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    scaler = StandardScaler()
    X = data.drop(columns=["Class"])
    y = data["Class"]
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])
    return X, y

X, y = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Build ANN model
ann_model = Sequential([
    Dense(32, input_dim=X_train_res.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train ANN model
history = ann_model.fit(X_train_res, y_train_res, validation_data=(X_test, y_test),
                        epochs=50, batch_size=32, callbacks=[early_stopping])

# Save ANN model
os.makedirs("models", exist_ok=True)
ann_model.save("models/ann_model.h5")

# Streamlit App
st.title("Credit Card Fraud Detection")
st.write("This app uses a Deep Learning model (ANN) to detect fraudulent transactions.")

# User Input
st.sidebar.header("User Input")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=50.0)
time = st.sidebar.number_input("Transaction Time", min_value=0.0, value=10000.0)
features = np.array([time, amount] + [0] * (X.shape[1] - 2)).reshape(1, -1)

# Load ANN model
ann_model = load_model("models/ann_model.h5")

# Make prediction with ANN
ann_pred = ann_model.predict(features)

# Display predictions
st.subheader("Prediction")
st.write(f"ANN Prediction: {'Fraud' if ann_pred[0][0] > 0.5 else 'Not Fraud'}")

# Evaluation metrics for ANN
def plot_metrics():
    st.subheader("Model Performance")
    y_pred_ann = (ann_model.predict(X_test) > 0.5).astype(int)

    st.write("Confusion Matrix (ANN):")
    st.write(confusion_matrix(y_test, y_pred_ann))

    st.write("ROC AUC Score (ANN):", roc_auc_score(y_test, y_pred_ann))

if st.sidebar.button("Evaluate ANN Model"):
    plot_metrics()

# Plot ANN training history
if st.sidebar.checkbox("Show ANN Training History"):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('ANN Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)
