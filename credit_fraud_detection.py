import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
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

# Train ML models
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_res, y_train_res)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_res, y_train_res)

# Save models
os.makedirs("models", exist_ok=True)
pickle.dump(log_reg, open("models/logistic_regression.pkl", "wb"))
pickle.dump(rf_clf, open("models/random_forest.pkl", "wb"))

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
ann_model.save("models/ann_model.h5")

# Streamlit App
st.title("Credit Card Fraud Detection")
st.write("This app uses Machine Learning and Deep Learning models to detect fraudulent transactions.")

# User Input
st.sidebar.header("User Input")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=50.0)
time = st.sidebar.number_input("Transaction Time", min_value=0.0, value=10000.0)
features = np.array([time, amount] + [0] * (X.shape[1] - 2)).reshape(1, -1)

# Load models
log_model = pickle.load(open("models/logistic_regression.pkl", "rb"))
rf_model = pickle.load(open("models/random_forest.pkl", "rb"))
ann_model = load_model("models/ann_model.h5")

# Make predictions
log_pred = log_model.predict(features)
rf_pred = rf_model.predict(features)
ann_pred = ann_model.predict(features)

# Display predictions
st.subheader("Predictions")
st.write(f"Logistic Regression Prediction: {'Fraud' if log_pred[0] == 1 else 'Not Fraud'}")
st.write(f"Random Forest Prediction: {'Fraud' if rf_pred[0] == 1 else 'Not Fraud'}")
st.write(f"ANN Prediction: {'Fraud' if ann_pred[0][0] > 0.5 else 'Not Fraud'}")

# Evaluation metrics
def plot_metrics():
    st.subheader("Model Performance")
    st.write("Evaluating Logistic Regression and Random Forest on the test set.")

    y_pred_log = log_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    st.write("Confusion Matrix (Logistic Regression):")
    st.write(confusion_matrix(y_test, y_pred_log))

    st.write("Confusion Matrix (Random Forest):")
    st.write(confusion_matrix(y_test, y_pred_rf))

    st.write("ROC AUC Score (Logistic Regression):", roc_auc_score(y_test, y_pred_log))
    st.write("ROC AUC Score (Random Forest):", roc_auc_score(y_test, y_pred_rf))

if st.sidebar.button("Evaluate Models"):
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
