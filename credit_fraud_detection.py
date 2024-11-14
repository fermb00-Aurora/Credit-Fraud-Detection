# fraud_detection_app.py

# Import necessary libraries
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score
)
import tempfile

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Streamlit App Configuration
st.set_page_config(
    page_title='💳 Credit Card Fraud Detection Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Title and Sidebar Menu
st.title('💳 Credit Card Fraud Detection Dashboard')
st.sidebar.header("Navigation Menu")
page_selection = st.sidebar.radio("Go to", [
    "Introduction",
    "Data Overview",
    "Exploratory Data Analysis",
    "Feature Importance",
    "Model Evaluation",
    "Simulator",
    "Download Report",
    "Feedback"
])

# Function to load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

# Load the dataset
df = load_data()

# Initialize session state for model evaluation results
if 'model_evaluation' not in st.session_state:
    st.session_state['model_evaluation'] = {}

# Dictionary of all available models
all_models = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Random Forest': 'random_forest.pkl',
    'Extra Trees': 'extra_trees.pkl',
    'k-Nearest Neighbors': 'knn.pkl'
}

# Ensure that the dataset is loaded before proceeding
if df is not None:
    # Introduction Page
    if page_selection == "Introduction":
        st.header("📘 Executive Summary")
        st.markdown("""
        **Objective:**  
        Empower financial executives with advanced tools to detect and analyze fraudulent credit card transactions. By leveraging sophisticated machine learning models, this platform provides actionable insights to mitigate financial losses and enhance security measures.

        **Key Highlights:**
        - **Comprehensive Data Analysis:** In-depth exploration of transaction data to identify patterns and anomalies.
        - **Advanced Machine Learning Models:** Evaluation of pre-trained models including Logistic Regression, Random Forest, and Extra Trees for accurate fraud detection.
        - **Interactive Visualizations:** Dynamic charts and graphs that facilitate intuitive understanding of data trends and model performances.
        - **Actionable Insights:** Detailed reports and metrics that support strategic decision-making and risk management.
        - **Customizable Reports:** Generate and download tailored PDF reports to share findings with stakeholders.

        **Business Implications:**
        - **Risk Mitigation:** Early detection of fraudulent activities reduces financial losses and safeguards customer trust.
        - **Operational Efficiency:** Streamlined analysis processes save time and resources, enabling focus on critical tasks.
        - **Strategic Decision-Making:** Data-driven insights inform policies and strategies to enhance security protocols and customer satisfaction.
        """)

    # Data Overview Page
    elif page_selection == "Data Overview":
        st.header("🔍 Data Overview")

        # Display the first few rows of the dataset
        st.subheader("📂 Dataset Preview")
        st.dataframe(df.head(10).style.highlight_max(axis=0))

        st.markdown("---")

        # Data Summary Statistics
        st.subheader("📊 Data Summary")
        st.dataframe(df.describe().T.style.background_gradient(cmap='YlGnBu'))

        st.markdown("""
        **Discussion:**
        - **Total Transactions:** The dataset comprises **{:,}** transactions.
        - **Class Distribution:** Out of these, **{:,}** are labeled as fraudulent (**{:.4f}%**) and the remaining **{:,}** as valid.
        - **Feature Details:**
            - **V1 to V28:** Result of a PCA transformation to ensure anonymity and reduce dimensionality.
            - **Time:** Seconds elapsed since the first transaction in the dataset, providing temporal context.
            - **Amount:** Transaction amount in US dollars.
        - **Data Imbalance:** The significant imbalance between fraudulent and valid transactions underscores the challenge in fraud detection, necessitating specialized modeling techniques.
        """.format(
            len(df),
            df['Class'].sum(),
            (df['Class'].sum() / len(df)) * 100,
            len(df) - df['Class'].sum()
        ))

    # Exploratory Data Analysis Page
    elif page_selection == "Exploratory Data Analysis":
        st.header("📊 Exploratory Data Analysis")
        # [Content remains the same as in your existing code]
        # ...

    # Feature Importance Page
    elif page_selection == "Feature Importance":
        st.header("🔍 Feature Importance Analysis")
        # [Content remains the same as in your existing code]
        # ...

    # Model Evaluation Page
    elif page_selection == "Model Evaluation":
        st.header("🧠 Model Evaluation")
        # [Content remains the same as in your existing code]
        # ...

    # Simulator Page
    elif page_selection == "Simulator":
        st.header("🚀 Simulator")
        st.markdown("""
        **Simulate and Predict Fraudulent Transactions:**
        Enter transaction details to receive an immediate prediction on whether the transaction is fraudulent.
        """)

        # Check if a model has been evaluated and stored in session state
        eval_data = st.session_state.get('model_evaluation', {})
        if 'classifier' in eval_data:
            # Use the model selected in Model Evaluation
            classifier = eval_data['classifier']
            model_file = all_models[classifier]
            st.info(f"Using the model selected in Model Evaluation: **{classifier}**")
        else:
            # Allow the user to select a model
            st.warning("No model selected in Model Evaluation. Please select a model for simulation.")
            classifier = st.selectbox("Select Model for Simulation:", list(all_models.keys()))
            model_file = all_models[classifier]

        model_path = os.path.join(os.path.dirname(__file__), model_file)

        try:
            model_sim = joblib.load(model_path)

            # Input transaction details
            st.subheader("🔍 Enter Transaction Details")
            col1, col2 = st.columns(2)

            with col1:
                V_features = {}
                for i in range(1, 29):
                    V_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.5f", key=f'Sim_V{i}')

            with col2:
                Time = st.number_input('Time (seconds since first transaction)', min_value=0, value=0, step=1, key='Sim_Time')
                Amount = st.number_input('Transaction Amount ($)', min_value=0.0, value=0.0, format="%.2f", key='Sim_Amount')

            # Predict button
            if st.button("Simulate"):
                input_data = pd.DataFrame({
                    **V_features,
                    'Time': [Time],
                    'Amount': [Amount]
                })

                prediction = model_sim.predict(input_data)[0]
                if hasattr(model_sim, "predict_proba"):
                    prediction_proba = model_sim.predict_proba(input_data)[0][1]
                    fraud_probability = f"{prediction_proba:.2%}"
                else:
                    fraud_probability = "N/A"

                if prediction == 1:
                    st.error(f"⚠️ **Fraudulent Transaction Detected!** Probability of Fraud: {fraud_probability}")
                else:
                    st.success(f"✅ **Valid Transaction.** Probability of Fraud: {fraud_probability}")
        except Exception as e:
            st.error(f"Error loading model '{model_file}': {e}")

    # Download Report Page
    elif page_selection == "Download Report":
        st.header("📄 Download Report")
        # [Content remains the same as in your existing code]
        # ...

    # Feedback Page
    elif page_selection == "Feedback":
        st.header("💬 Feedback")
        # [Content remains the same as in your existing code]
        # ...

    else:
        st.error("Page not found.")
