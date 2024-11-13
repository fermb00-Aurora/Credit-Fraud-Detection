import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import warnings
import streamlit as st
import joblib
import os
from fpdf import FPDF
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, roc_auc_score, precision_recall_curve

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit App Title and Sidebar
st.title('💳 Credit Card Fraud Detection Dashboard')
st.sidebar.header("Navigation")
page_selection = st.sidebar.radio("Navigate:", ["Introduction", "Data Overview", "EDA", "Feature Importance", "Model Evaluation", "Real-Time Prediction", "Download Report", "Feedback"])

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Introduction
if page_selection == "Introduction":
    st.header("📘 Introduction")
    st.write("""
    Welcome to the enhanced Credit Card Fraud Detection Dashboard! This app provides:
    - Comprehensive data analysis and visualization.
    - Evaluation of pre-trained machine learning models (Logistic Regression, kNN, Random Forest, Extra Trees).
    - Business insights, cost-benefit analysis, and a detailed report.
    """)

# Data Overview
if page_selection == "Data Overview":
    st.header("🔍 Data Overview")
    if st.sidebar.checkbox('Show DataFrame Sample'):
        st.dataframe(df.head(100))

    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]
    outlier_percentage = (len(fraud) / len(valid)) * 100

    st.write(f"Fraudulent transactions: **{outlier_percentage:.3f}%**")
    st.write(f"Fraud Cases: **{len(fraud)}**, Valid Cases: **{len(valid)}**")

# Exploratory Data Analysis (EDA)
if page_selection == "EDA":
    st.header("📊 Exploratory Data Analysis")
    fig = px.histogram(df, x="Amount", color="Class", marginal="box", title="Transaction Amount Distribution")
    st.plotly_chart(fig)

    corr = df.corr()
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    st.pyplot(fig_corr)

# Feature Importance
if page_selection == "Feature Importance":
    st.header("🔍 Feature Importance")
    model_files = {
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl'
    }

    selected_model = st.sidebar.selectbox("Choose Model for Feature Importance", list(model_files.keys()))
    model_path = os.path.join(os.path.dirname(__file__), model_files[selected_model])
    model = joblib.load(model_path)

    feature_importances = model.feature_importances_
    features = df.drop(columns=['Class']).columns
    fig_imp = px.bar(x=features, y=feature_importances, title="Feature Importance")
    st.plotly_chart(fig_imp)

# Model Evaluation
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")
    model_choices = {
        'Logistic Regression': 'logistic_regression.pkl',
        'k-Nearest Neighbors (kNN)': 'knn.pkl',
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl'
    }

    classifier = st.sidebar.selectbox("Select Model", list(model_choices.keys()))
    model_file = model_choices[classifier]
    model_path = os.path.join(os.path.dirname(__file__), model_file)
    model = joblib.load(model_path)

    test_size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {classifier}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📈 Additional Metrics")
    roc_auc = roc_auc_score(y_test, y_pred)
    st.write(f"ROC-AUC Score: **{roc_auc:.3f}**")

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    fig_pr = px.area(x=recall, y=precision, title="Precision-Recall Curve")
    st.plotly_chart(fig_pr)

# Real-Time Prediction
if page_selection == "Real-Time Prediction":
    st.header("🔍 Real-Time Prediction")
    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        predictions = model.predict(new_data)
        new_data['Predictions'] = predictions
        st.write("Predictions:")
        st.dataframe(new_data)

# Download Report
if page_selection == "Download Report":
    st.header("📄 Generate PDF Report")

    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')
        pdf.multi_cell(0, 10, classification_report(y_test, y_pred))
        report_file = "fraud_detection_report.pdf"
        pdf.output(report_file)
        with open(report_file, "rb") as file:
            st.download_button("Download Report", file, file_name=report_file)

    st.button("Generate Report", on_click=generate_report)

# Feedback
if page_selection == "Feedback":
    st.header("💬 Feedback")
    feedback = st.text_area("Provide your feedback here:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

