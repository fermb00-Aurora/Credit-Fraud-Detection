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

# Streamlit App Title
st.title('💳 Credit Card Fraud Detection Dashboard')
st.sidebar.header("Navigation")
page_selection = st.sidebar.radio("Navigate:", ["Introduction", "Data Overview", "Exploratory Data Analysis", "Feature Importance", "Model Evaluation", "Download Report", "Feedback"])

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Global variables for report generation
y_test_global = None
y_pred_global = None
classifier_global = None

# Model Evaluation Section
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

    # Store global variables for report generation
    y_test_global = y_test
    y_pred_global = y_pred
    classifier_global = classifier

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

    # Additional Metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.write(f"**F1-Score**: {f1:.3f}")
    st.write(f"**Accuracy**: {accuracy:.3f}")
    st.write(f"**Matthews Correlation Coefficient (MCC)**: {mcc:.3f}")

# Download Report Section
if page_selection == "Download Report":
    st.header("📄 Generate PDF Report")

    def generate_report():
        if y_test_global is None or y_pred_global is None:
            st.error("No evaluation data available. Please run model evaluation first.")
            return

        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Selected Model: {classifier_global}", ln=True)
            pdf.multi_cell(0, 10, classification_report(y_test_global, y_pred_global))
            pdf.cell(200, 10, txt=f"F1-Score: {f1_score(y_test_global, y_pred_global):.3f}", ln=True)
            pdf.cell(200, 10, txt=f"Accuracy: {accuracy_score(y_test_global, y_pred_global):.3f}", ln=True)
            pdf.cell(200, 10, txt=f"MCC: {matthews_corrcoef(y_test_global, y_pred_global):.3f}", ln=True)

            report_file = "fraud_detection_report.pdf"
            pdf.output(report_file)
            with open(report_file, "rb") as file:
                st.download_button("Download Report", file, file_name=report_file)

        except Exception as e:
            st.error(f"Failed to generate report: {e}")

    st.button("Generate Report", on_click=generate_report)

# Feedback Section
if page_selection == "Feedback":
    st.header("💬 Feedback")
    feedback = st.text_area("Provide your feedback here:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")



