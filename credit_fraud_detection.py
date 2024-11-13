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
    """)

# Data Overview
if page_selection == "Data Overview":
    st.header("🔍 Data Overview")
    if st.sidebar.checkbox('Show DataFrame Sample'):
        st.dataframe(df.head(100))

    st.subheader("Dataset Summary")
    st.dataframe(df.describe().T.style.background_gradient(cmap='coolwarm'))

# Exploratory Data Analysis
if page_selection == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    # Fraud Ratio Pie Chart
    st.subheader("Fraud Ratio Analysis")
    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]
    fraud_ratio = (len(fraud) / len(valid)) * 100

    fig_pie = px.pie(
        names=['Valid Transactions', 'Fraudulent Transactions'],
        values=[len(valid), len(fraud)],
        title="Proportion of Fraudulent vs. Valid Transactions",
        color_discrete_sequence=['#1f77b4', '#ff7f0e']
    )
    st.plotly_chart(fig_pie)

    st.write(f"**Fraudulent transactions represent**: {fraud_ratio:.3f}% of all transactions.")

    # Transaction Amount Distribution
    st.subheader("Transaction Amount Distribution by Class")
    fig_amount = px.histogram(
        df, x='Amount', color='Class',
        title='Transaction Amount Distribution by Fraud Class',
        marginal='box',
        color_discrete_sequence=['#2ca02c', '#d62728']
    )
    st.plotly_chart(fig_amount)

    # Top 5 Most Frequent Amount Ranges
    st.subheader("Top 5 Most Frequent Transaction Amount Ranges")
    df['AmountRange'] = pd.cut(df['Amount'], bins=[0, 10, 50, 100, 500, 1000, 5000, 10000, 20000], right=False)
    amount_range_counts = df['AmountRange'].value_counts().sort_index()
    
    fig_range = px.bar(
        amount_range_counts,
        x=amount_range_counts.index.astype(str),
        y=amount_range_counts.values,
        labels={'x': 'Amount Range', 'y': 'Frequency'},
        title="Frequency of Transaction Amount Ranges"
    )
    st.plotly_chart(fig_range)

    st.write("The above chart shows the frequency of transactions within specific amount ranges. This information helps identify common transaction values and potential risk thresholds.")

# Feature Importance and Model Evaluation sections remain unchanged.

# Feedback Section remains unchanged.

