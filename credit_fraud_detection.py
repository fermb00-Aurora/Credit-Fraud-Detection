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

# Custom CSS for Light Sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f0f0;
        color: #333333;
    }
    [data-testid="stSidebar"] h2 {
        color: #333333;
    }
    [data-testid="stSidebar"] label {
        color: #333333;
    }
    .css-1v3fvcr:hover {
        background-color: #e0e0e0;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Title and Sidebar
st.title('💳 Credit Card Fraud Detection Dashboard')
st.sidebar.header("Menu")
page_selection = st.sidebar.radio("Navigate:", ["Introduction", "Data Overview", "Exploratory Data Analysis", "Feature Importance", "Model Evaluation", "Real-Time Prediction", "Download Report", "Feedback"])

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

# Exploratory Data Analysis
if page_selection == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Correlation Heatmap")

    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="YlOrRd",
        hoverongaps=False
    ))
    fig.update_layout(title='Interactive Correlation Heatmap', height=700)
    st.plotly_chart(fig)

# Feature Importance
if page_selection == "Feature Importance":
    st.header("🔍 Feature Importance")
    model_filename = 'random_forest.pkl'
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    model = joblib.load(model_path)

    feature_importances = model.feature_importances_
    features = df.drop(columns=['Class']).columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    st.subheader("Top 3 Most and Least Important Features")
    st.write("These features have the highest and lowest impact on predicting fraud cases.")

    # Top 3 Most Important Features
    for i in range(3):
        st.write(f"🏅 **{i+1}. {importance_df.iloc[i]['Feature']}** - Importance: **{importance_df.iloc[i]['Importance']:.4f}**")

    # Top 3 Least Important Features
    for i in range(1, 4):
        st.write(f"🥉 **{4-i}. {importance_df.iloc[-i]['Feature']}** - Importance: **{importance_df.iloc[-i]['Importance']:.4f}**")

    # Feature Importance Bar Plot
    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
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

    # Enhanced Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr')
    plt.title(f"Confusion Matrix for {classifier}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig_cm)

    # Enhanced Classification Report
    st.subheader("📋 Enhanced Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

# Report Generation
if page_selection == "Download Report":
    st.header("📄 Generate PDF Report")

    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')
        pdf.multi_cell(0, 10, classification_report(y_test, y_pred))
        pdf.cell(200, 10, txt=f"F1-Score: {f1:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"MCC: {mcc:.3f}", ln=True)
        report_file = "fraud_detection_report.pdf"
        pdf.output(report_file)
        with open(report_file, "rb") as file:
            st.download_button("Download Report", file, file_name=report_file)

    st.button("Generate Report", on_click=generate_report)

# Feedback Section
if page_selection == "Feedback":
    st.header("💬 Feedback")
    feedback = st.text_area("Provide your feedback here:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")




