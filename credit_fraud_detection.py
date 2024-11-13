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
st.title('💳 Credit Card Fraud Detection Dashboard')
st.sidebar.header("Navigation")
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
    
    # Correlation Heatmap with Plotly
    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="YlOrRd",
        hoverongaps=False
    ))
    fig.update_layout(
        title='Interactive Correlation Heatmap',
        xaxis_nticks=36,
        height=700
    )
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

    st.subheader("Top 3 Most Important Features")
    for i in range(3):
        st.write(f"{i+1}. **{importance_df.iloc[i]['Feature']}** with importance score: **{importance_df.iloc[i]['Importance']:.4f}**")

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

    # Confusion Matrix with Seaborn Heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr')
    plt.title(f"Confusion Matrix for {classifier}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig_cm)

    # Enhanced Classification Report
    st.subheader("📋 Enhanced Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df.style.background_gradient(cmap='coolwarm'))

    # Additional Metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.write(f"**F1-Score**: {f1:.3f}")
    st.write(f"**Accuracy**: {accuracy:.3f}")
    st.write(f"**Matthews Correlation Coefficient (MCC)**: {mcc:.3f}")

    # Business-Relevant Graph: Fraud vs. Amount
    fig_amount = px.histogram(df, x='Amount', color='Class', title='Transaction Amount vs. Fraud Detection')
    st.plotly_chart(fig_amount)

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


