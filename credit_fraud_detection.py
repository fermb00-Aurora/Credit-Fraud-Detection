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

# Custom Sidebar Design
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #f0f2f6 !important;
        }
        .css-qbe2hs {
            color: #1c1e21 !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.header("Menu")
page_selection = st.sidebar.radio("Navigate:", ["Executive Summary", "Data Overview", "Exploratory Data Analysis", "Feature Importance", "Model Evaluation", "Real-Time Prediction", "Download Report", "Feedback"])

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Executive Summary
if page_selection == "Executive Summary":
    st.header("🏦 Executive Summary")
    st.markdown("""
    ## Welcome to the Credit Card Fraud Detection Dashboard

    This interactive web application is designed for financial executives and decision-makers to quickly identify and analyze fraudulent transactions. It provides:
    - **In-depth Data Analysis**: Explore transaction data and detect patterns of fraudulent behavior.
    - **Model Evaluation**: Assess the performance of pre-trained models (Logistic Regression, kNN, Random Forest, Extra Trees) for fraud detection.
    - **Business Insights**: Actionable insights to help mitigate financial risks associated with credit card fraud.
    - **Automated PDF Reporting**: Generate detailed, shareable reports for decision-making.

    Navigate through the menu to explore data, evaluate model performance, and gain actionable insights.
    """)

# Data Overview
if page_selection == "Data Overview":
    st.header("🔍 Data Overview")

    # Data Sample
    if st.sidebar.checkbox('Show DataFrame Sample'):
        st.dataframe(df.head(100))

    # Data Summary Table
    st.subheader("Data Summary")
    data_summary = df.describe().T
    st.dataframe(data_summary.style.background_gradient(cmap='coolwarm'))

    # Missing Values Table
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    missing_table = pd.DataFrame(missing_values[missing_values > 0], columns=["Missing Values"])
    if not missing_table.empty:
        st.dataframe(missing_table)
    else:
        st.write("No missing values in the dataset.")

    # Fraud Ratio Pie Chart
    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]
    outlier_percentage = (len(fraud) / len(valid)) * 100

    st.subheader("Fraud Ratio")
    fig_pie = px.pie(
        names=['Valid Transactions', 'Fraudulent Transactions'],
        values=[len(valid), len(fraud)],
        title="Proportion of Fraudulent vs. Valid Transactions",
        color_discrete_sequence=['#1f77b4', '#ff7f0e']
    )
    st.plotly_chart(fig_pie)

    # Distribution of Transaction Amount
    st.subheader("Transaction Amount Distribution")
    fig_amount = px.histogram(
        df, x='Amount', color='Class',
        title='Transaction Amount Distribution by Class',
        marginal='box',
        color_discrete_sequence=['#2ca02c', '#d62728']
    )
    st.plotly_chart(fig_amount)

    # Top 5 Most Frequent Values for Amount
    st.subheader("Top 5 Most Frequent Transaction Amounts")
    top_amounts = df['Amount'].value_counts().head(5)
    st.dataframe(top_amounts.to_frame().rename(columns={'Amount': 'Frequency'}))

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

# Feature Importance and Model Evaluation remain unchanged as per previous request.

# Feedback Section remains unchanged.
