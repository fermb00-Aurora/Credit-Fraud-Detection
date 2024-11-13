# fraud_detection_app.py

# Import necessary libraries
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
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
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
    cohen_kappa_score,
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
    try:
        # Load the dataset from the current directory
        data_path = 'creditcard.csv'
        
        # Check if the file exists
        if not os.path.exists(data_path):
            st.error(f"The file 'creditcard.csv' was not found in the current directory.")
            # Optional: List files in the directory for debugging
            files = os.listdir('.')
            st.info("Files in the current directory:")
            for file in files:
                st.write(f"- {file}")
            return None
        
        # Load the dataset
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to load models with caching
@st.cache_resource
def load_model(model_filename):
    try:
        # Load the model from the current directory
        model_path = model_filename
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_filename}' not found in the current directory.")
            # Optional: List files in the directory for debugging
            files = os.listdir('.')
            st.info("Files in the current directory:")
            for file in files:
                st.write(f"- {file}")
            return None
        
        # Load the model
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model '{model_filename}': {e}")
        return None

# Initialize session state for model evaluation results
if 'model_evaluation' not in st.session_state:
    st.session_state['model_evaluation'] = {}

# Load the dataset
df = load_data()

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
        # [Code remains the same as provided]
        # Ensure that 'df' is used consistently throughout this section.
        # No changes needed here.

    # Feature Importance Page
    elif page_selection == "Feature Importance":
        st.header("🔍 Feature Importance Analysis")
        st.markdown("""
        **Understanding Feature Impact:**
        Identifying which features significantly influence model predictions is paramount in credit card fraud detection. This section delves into the importance of various features across different machine learning models, providing clarity on what drives fraud detection decisions.

        **Models Analyzed:**
        - **Random Forest:** Utilizes ensemble learning to provide feature importance based on the mean decrease in impurity.
        - **Extra Trees:** Similar to Random Forest but with more randomness, offering robust feature importance metrics.
        - **Logistic Regression:** Assesses feature importance through the magnitude of coefficients, indicating the strength and direction of influence.
        """)

        # Dictionary of models supporting feature importance
        feature_importance_models = {
            'Random Forest': 'random_forest.pkl',
            'Extra Trees': 'extra_trees.pkl',
            'Logistic Regression': 'logistic_regression.pkl'
        }

        selected_model = st.selectbox("Select a Model for Feature Importance:", list(feature_importance_models.keys()))
        model_filename = feature_importance_models[selected_model]
        model = load_model(model_filename)

        if model:
            # Extract feature names
            features = df.drop(columns=['Class']).columns

            # Determine feature importances based on model type
            if selected_model in ['Random Forest', 'Extra Trees']:
                importances = model.feature_importances_
            elif selected_model == 'Logistic Regression':
                importances = np.abs(model.coef_[0])
            else:
                st.error("Selected model does not support feature importance.")
                importances = None

            if importances is not None:
                # Create a DataFrame for feature importances
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                # [Rest of the code remains the same]

    # Model Evaluation Page
    elif page_selection == "Model Evaluation":
        # [Code remains the same as provided]
        # Ensure that 'df' is used to prepare 'X' and 'y' for evaluation.
        # The models are loaded using 'load_model', and predictions are made.

    # Simulator Page
    elif page_selection == "Simulator":
        st.header("🚀 Simulator")
        st.markdown("""
        **Simulate and Predict Fraudulent Transactions:**
        Enter transaction details to receive an immediate prediction on whether the transaction is fraudulent.
        """)

        # Load the model
        default_model_filename = 'random_forest.pkl'
        model_sim = load_model(default_model_filename)

        if model_sim:
            # Input transaction details
            st.subheader("🔍 Enter Transaction Details")
            col1, col2 = st.columns(2)

            with col1:
                V_features = {}
                for i in range(1, 29):
                    V_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.5f", key=f'V{i}')

            with col2:
                Time = st.number_input('Time (seconds since first transaction)', min_value=0, value=0, step=1, key='Time')
                Amount = st.number_input('Transaction Amount ($)', min_value=0.0, value=0.0, format="%.2f", key='Amount')

            # Predict button
            if st.button("Simulate"):
                input_data = pd.DataFrame({
                    **V_features,
                    'Time': [Time],
                    'Amount': [Amount]
                })

                # Ensure that the columns in 'input_data' are in the same order as expected by the model
                input_data = input_data[model_sim.feature_names_in_]

                prediction = model_sim.predict(input_data)[0]
                prediction_proba = model_sim.predict_proba(input_data)[0][1]

                if prediction == 1:
                    st.error(f"⚠️ **Fraudulent Transaction Detected!** Probability: {prediction_proba:.2%}")
                else:
                    st.success(f"✅ **Valid Transaction.** Probability of Fraud: {prediction_proba:.2%}")
        else:
            st.error("Simulator model could not be loaded.")

    # Download Report Page
    elif page_selection == "Download Report":
        # [Code remains the same as provided]
        # Ensure that any references to 'df' or models are correct.

    # Feedback Page
    elif page_selection == "Feedback":
        # [Code remains the same as provided]

    else:
        st.error("Page not found.")

else:
    st.error("Data could not be loaded. Please ensure 'creditcard.csv' is in the current directory.")


