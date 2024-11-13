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
page_selection = st.sidebar.radio("Navigate:", ["Introduction", "Data Overview", "Exploratory Data Analysis", "Feature Importance", "Model Evaluation", "Real-Time Prediction", "Download Report", "Feedback"])

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Global variables for report generation
st.session_state["y_pred"] = None
st.session_state["f1"] = None
st.session_state["accuracy"] = None
st.session_state["mcc"] = None
st.session_state["classifier"] = None

# Introduction
if page_selection == "Introduction":
    st.header("📘 Executive Summary")
    st.write("""
    Welcome to the Credit Card Fraud Detection Dashboard! This application provides:
    - In-depth data analysis and visualization.
    - Evaluation of pre-trained machine learning models (Logistic Regression, Random Forest, Extra Trees).
    - Business insights and a detailed PDF report for download.
    """)

# Exploratory Data Analysis
if page_selection == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    # Correlation Heatmap
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

    # Transaction Amount Distribution
    st.subheader("Transaction Amount Distribution by Class")
    fig_amount = px.histogram(df, x='Amount', color='Class', title='Transaction Amount Distribution (Fraud vs. Valid)',
                              marginal='box', hover_data=df.columns)
    st.plotly_chart(fig_amount)

# Feature Importance
if page_selection == "Feature Importance":
    st.header("🔍 Feature Importance Analysis")
    st.markdown("""
        This section shows feature importance from models that support it:
        - **Random Forest** and **Extra Trees** use impurity reduction.
        - **Logistic Regression** uses absolute coefficient values.
    """)

    feature_importance_models = {
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl',
        'Logistic Regression': 'logistic_regression.pkl'
    }

    selected_model = st.sidebar.selectbox("Select a model for feature importance:", list(feature_importance_models.keys()))
    model_path = os.path.join(os.path.dirname(__file__), feature_importance_models[selected_model])
    model = joblib.load(model_path)

    # Feature importance calculation
    features = df.drop(columns=['Class']).columns
    if selected_model in ['Random Forest', 'Extra Trees']:
        importances = model.feature_importances_
    elif selected_model == 'Logistic Regression':
        importances = np.abs(model.coef_[0])

    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    num_features = st.sidebar.slider("Select number of features to display:", min_value=5, max_value=len(features), value=10)

    # Feature Importance Plot
    fig_imp = px.bar(importance_df.head(num_features), x='Importance', y='Feature', orientation='h', title=f"Feature Importance for {selected_model}")
    st.plotly_chart(fig_imp)

# Model Evaluation
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")
    model_choices = ['Logistic Regression', 'Random Forest', 'Extra Trees']
    classifier = st.sidebar.selectbox("Select Model", model_choices)
    model_path = os.path.join(os.path.dirname(__file__), feature_importance_models[classifier])
    model = joblib.load(model_path)

    test_size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_pred = model.predict(X_test)
    st.session_state["y_pred"] = y_pred
    st.session_state["classifier"] = classifier

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr')
    plt.title(f"Confusion Matrix for {classifier}")
    st.pyplot(fig_cm)

    # Dynamic Explanation
    st.subheader("Confusion Matrix Analysis")
    tn, fp, fn, tp = cm.ravel()
    st.write(f"**True Positives (TP):** {tp} - Correctly identified fraud cases.")
    st.write(f"**True Negatives (TN):** {tn} - Correctly identified valid transactions.")
    st.write(f"**False Positives (FP):** {fp} - Misclassified valid transactions as fraud.")
    st.write(f"**False Negatives (FN):** {fn} - Missed fraud cases.")

    # Enhanced Classification Report
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

    # Store metrics for report generation
    st.session_state["f1"] = f1_score(y_test, y_pred)
    st.session_state["accuracy"] = accuracy_score(y_test, y_pred)
    st.session_state["mcc"] = matthews_corrcoef(y_test, y_pred)

# Download Report
if page_selection == "Download Report":
    st.header("📄 Generate PDF Report")

    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Model: {st.session_state['classifier']}", ln=True)
        pdf.multi_cell(0, 10, str(report_df))
        pdf.cell(200, 10, txt=f"F1-Score: {st.session_state['f1']:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"Accuracy: {st.session_state['accuracy']:.3f}", ln=True)
        pdf.cell(200, 10, txt=f"MCC: {st.session_state['mcc']:.3f}", ln=True)
        pdf.output("fraud_detection_report.pdf")
        with open("fraud_detection_report.pdf", "rb") as file:
            st.download_button("Download Report", file, file_name="fraud_detection_report.pdf")

    st.button("Generate Report", on_click=generate_report)
