import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import streamlit as st
import joblib
import os
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit App Title
st.title('💳 Credit Card Fraud Detection Dashboard')

st.sidebar.header("Navigation")
page_selection = st.sidebar.radio("Go to:", ["Introduction", "Data Overview", "Model Evaluation", "Download Report"])

# Load the dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Introduction
if page_selection == "Introduction":
    st.header("Introduction")
    st.write("""
    Welcome to the Credit Card Fraud Detection Dashboard. This app provides an in-depth analysis of fraud detection using various machine learning models.
    Use the navigation on the left to explore different sections:
    - Data Overview
    - Model Evaluation
    - Download a detailed report
    """)

# Data Overview
if page_selection == "Data Overview":
    st.header("🔍 Data Overview")

    if st.sidebar.checkbox('Show the first 100 rows of the dataframe'):
        st.dataframe(df.head(100))

    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]
    outlier_percentage = (len(fraud) / len(valid)) * 100

    st.write(f"**Fraudulent transactions represent**: {outlier_percentage:.3f}% of all transactions.")
    st.write(f"**Total Fraud Cases**: {len(fraud)}")
    st.write(f"**Total Valid Cases**: {len(valid)}")

# Splitting the features and labels
X = df.drop(columns=['Class'])
y = df['Class']

size = st.sidebar.slider('Select Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

# Model Evaluation
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")

    # List of pre-trained models
    model_filenames = {
        'Logistic Regression': 'logistic_regression.pkl',
        'k-Nearest Neighbors (kNN)': 'knn.pkl',
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl'
    }

    classifier = st.sidebar.selectbox("Select a model to evaluate:", list(model_filenames.keys()))
    model_filename = model_filenames[classifier]

    try:
        # Load the selected model
        model_path = os.path.join(os.path.dirname(__file__), model_filename)
        model = joblib.load(model_path)

        st.write(f"Evaluating {classifier}...")
        y_pred_test = model.predict(X_test)

        # Confusion Matrix with Heatmap
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix for {classifier}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot()

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred_test))

        # Additional Metrics
        mcc = matthews_corrcoef(y_test, y_pred_test)
        st.write(f"Matthews Correlation Coefficient (MCC): {mcc:.3f}")

        # Specific Visualizations for Random Forest
        if classifier == 'Random Forest':
            st.subheader("Random Forest Tree Visualization (Top Estimator)")
            from sklearn.tree import export_graphviz
            from io import StringIO
            import pydot
            from IPython.display import Image

            estimator = model.estimators_[0]
            dot_data = StringIO()
            export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True, special_characters=True)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            st.image(graph[0].create_png(), caption="Decision Tree Visualization")

    except Exception as e:
        st.error(f"Error loading model: {e}")

# Download Report
if page_selection == "Download Report":
    st.header("📄 Generate PDF Report")

    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')
        pdf.cell(200, 10, txt="Summary of Model Evaluation", ln=True)

        pdf.cell(200, 10, txt=f"Selected Model: {classifier}", ln=True)
        pdf.multi_cell(0, 10, classification_report(y_test, y_pred_test))
        pdf.cell(200, 10, txt=f"Matthews Correlation Coefficient: {mcc:.3f}", ln=True)

        report_filename = "fraud_detection_report.pdf"
        pdf.output(report_filename)
        st.success("Report generated successfully.")
        with open(report_filename, "rb") as file:
            st.download_button("Download Report", file, file_name=report_filename)

    if st.button("Generate Report"):
        generate_report()
