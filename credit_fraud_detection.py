import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import streamlit as st
import joblib
import os
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit App Title and Sidebar
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title('💳 Credit Card Fraud Detection Dashboard')
st.sidebar.header("Menu")
page_selection = st.sidebar.radio("Navigate:", [
    "Introduction", "Exploratory Data Analysis", "Feature Selection", 
    "Model Evaluation", "Real-Time Prediction", "Download Report", "Feedback"
])

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Introduction
if page_selection == "Introduction":
    st.header("📘 Executive Summary")
    st.write("""
    Welcome to the Credit Card Fraud Detection Dashboard, tailored for executives in the banking sector.
    This app offers comprehensive analysis and evaluation of machine learning models for fraud detection.
    Key Features:
    - Detailed Exploratory Data Analysis (EDA)
    - Advanced Feature Selection for model optimization
    - Evaluation of pre-trained models like Logistic Regression, kNN, Random Forest, and Extra Trees
    """)

# Feature Selection
if page_selection == "Feature Selection":
    st.header("🔍 Feature Selection")

    # Dropdown to select the model for feature importance
    model_choices = {
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl'
    }
    feature_model = st.sidebar.selectbox("Select model for feature importance:", list(model_choices.keys()))
    model_path = os.path.join(os.path.dirname(__file__), model_choices[feature_model])
    model = joblib.load(model_path)

    # Feature importance
    feature_importances = model.feature_importances_
    features = df.drop(columns=['Class']).columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Checkbox to show plot of feature importance
    show_plot = st.sidebar.checkbox("Show plot of feature importance")

    # Slider for selecting the number of top features
    num_top_features = st.sidebar.slider("Number of top features", min_value=5, max_value=20, value=15)

    # Checkbox to display selected top features
    show_selected_features = st.sidebar.checkbox("Show selected top features")

    if show_plot:
        fig_imp = px.bar(importance_df.head(num_top_features), x='Importance', y='Feature', orientation='h',
                         title="Top Features by Importance")
        st.plotly_chart(fig_imp)

    if show_selected_features:
        st.write("Selected Top Features:")
        st.write(importance_df.head(num_top_features))

# Model Evaluation
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")
    classifier = st.sidebar.selectbox("Select Model for Evaluation", list(model_choices.keys()))
    model_file = model_choices[classifier]
    model_path = os.path.join(os.path.dirname(__file__), model_file)
    model = joblib.load(model_path)

    test_size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr')
    plt.title(f"Confusion Matrix for {classifier}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig_cm)

    # Enhanced Classification Report
    st.subheader("📋 Enhanced Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

    # Additional Metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.write(f"**F1-Score**: {f1:.3f}")
    st.write(f"**Accuracy**: {accuracy:.3f}")
    st.write(f"**Matthews Correlation Coefficient (MCC)**: {mcc:.3f}")

# Download Report
if page_selection == "Download Report":
    st.header("📄 Generate PDF Report")

    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')
        pdf.multi_cell(0, 10, txt=str(report_df))
        pdf.cell(200, 10, txt=f"F1-Score: {f1:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"MCC: {mcc:.3f}", ln=True)
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

