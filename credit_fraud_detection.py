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
st.sidebar.header("Menu")
page_selection = st.sidebar.radio("Navigate:", [
    "Introduction", "Data Overview", "Exploratory Data Analysis",
    "Feature Importance", "Model Evaluation", "Real-Time Prediction",
    "Download Report", "Feedback"
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
    Welcome to the enhanced Credit Card Fraud Detection Dashboard, designed for financial executives to analyze and detect fraudulent transactions.
    This application offers:
    - **Comprehensive data analysis and visualization**
    - **Evaluation of pre-trained machine learning models** (Logistic Regression, Random Forest, Extra Trees)
    - **Actionable insights** with business implications
    - **Customizable PDF reports** for decision-making support
    """)

# Data Overview
if page_selection == "Data Overview":
    st.header("🔍 Data Overview")
    st.dataframe(df.describe())

    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]
    outlier_percentage = (len(fraud) / len(valid)) * 100

    st.write(f"Fraudulent transactions: **{outlier_percentage:.3f}%**")
    st.write(f"Fraud Cases: **{len(fraud)}**, Valid Cases: **{len(valid)}**")

# Exploratory Data Analysis
if page_selection == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="YlOrBr",
        hoverongaps=False
    ))
    fig_corr.update_layout(title='Interactive Correlation Heatmap', height=700)
    st.plotly_chart(fig_corr)

    # Transaction Amount Distribution by Class
    st.subheader("Transaction Amount Distribution by Class")
    fig_amount = px.histogram(df, x='Amount', color='Class', title="Distribution of Transaction Amounts by Class",
                              barmode='overlay', log_y=True)
    st.plotly_chart(fig_amount)

# Feature Importance
if page_selection == "Feature Importance":
    st.header("🔍 Feature Importance Analysis")
    st.markdown("""
        In this section, we analyze the impact of different features on the predictions of models that support feature importance:
        
        - **Random Forest** and **Extra Trees**: Tree-based models that provide importance scores based on node impurity reduction.
        - **Logistic Regression**: Uses absolute values of coefficients as a proxy for feature importance (only valid if features are standardized).
        
        Models like **k-Nearest Neighbors (kNN)** and **Support Vector Machine (SVM)** are excluded because they do not offer meaningful feature importance scores.
    """)

    # Models supporting feature importance
    feature_importance_models = {
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl',
        'Logistic Regression': 'logistic_regression.pkl'
    }

    selected_model = st.sidebar.selectbox("Select a model for feature importance:", list(feature_importance_models.keys()))
    model_filename = feature_importance_models[selected_model]
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    try:
        model = joblib.load(model_path)
        features = df.drop(columns=['Class']).columns

        if selected_model in ['Random Forest', 'Extra Trees']:
            importances = model.feature_importances_
        elif selected_model == 'Logistic Regression':
            importances = np.abs(model.coef_[0])

        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

        st.subheader("Top 3 Most Important Features")
        for i in range(3):
            st.write(f"🏅 **{i + 1}. {importance_df.iloc[i]['Feature']}** - Importance: **{importance_df.iloc[i]['Importance']:.4f}**")

        st.subheader("Top 3 Least Important Features")
        for i in range(1, 4):
            st.write(f"🥉 **{4 - i}. {importance_df.iloc[-i]['Feature']}** - Importance: **{importance_df.iloc[-i]['Importance']:.4f}**")

        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title=f"Feature Importance for {selected_model}")
        st.plotly_chart(fig_imp)

    except Exception as e:
        st.error(f"Error loading model: {e}")

# Model Evaluation
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")
    model_choices = feature_importance_models
    classifier = st.sidebar.selectbox("Select Model", list(model_choices.keys()))
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

    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.write(f"**F1-Score**: {f1:.3f}")
    st.write(f"**Accuracy**: {accuracy:.3f}")
    st.write(f"**MCC**: {mcc:.3f}")

# Feedback Section
if page_selection == "Feedback":
    st.header("💬 Feedback")
    feedback = st.text_area("Provide your feedback here:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")


