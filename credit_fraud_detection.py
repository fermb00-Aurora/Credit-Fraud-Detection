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
    df = pd.read_csv('creditcard.csv')
    return df

# Load the dataset
df = load_data()

# Initialize session state for model evaluation results
if 'model_evaluation' not in st.session_state:
    st.session_state['model_evaluation'] = {}

# Ensure that the dataset is loaded before proceeding
if df is not None:
    # ... (Other pages code remains unchanged) ...

    # Download Report Page
    elif page_selection == "Download Report":
        st.header("📄 Download Report")
        st.markdown("""
        **Generate and Download a Comprehensive PDF Report:**
        Compile your analysis and model evaluation results into a downloadable PDF report for offline review and sharing with stakeholders.
        """)

        # Models supporting feature importance plus additional models
        all_models = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'Extra Trees': 'extra_trees.pkl',
            'Support Vector Machine': 'svm.pkl',
            'k-Nearest Neighbors': 'knn.pkl'
        }

        # Allow the user to select model and test size
        classifier = st.selectbox("Select Model for Report:", list(all_models.keys()))
        model_file = all_models[classifier]
        model_path = os.path.join(os.path.dirname(__file__), model_file)

        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Test set size slider
        test_size = st.slider('Test Set Size (%) for Report', min_value=10, max_value=50, value=30, step=5)
        test_size_fraction = test_size / 100

        X = df.drop(columns=['Class'])
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_fraction, random_state=42, stratify=y
        )

        # Make predictions
        y_pred = model.predict(X_test)

        # Compute evaluation metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)

        metrics = {
            'f1_score': f1,
            'accuracy': accuracy,
            'mcc': mcc,
            'precision': precision,
            'recall': recall,
            'f2_score': f2
        }

        # Check if model supports predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            y_proba = None
            roc_auc = "N/A"

        # Button to generate report
        if st.button("Generate Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Initialize PDF
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)

                    # Title Page
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, "Credit Card Fraud Detection Report", ln=True, align='C')
                    pdf.ln(10)

                    # Executive Summary
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Executive Summary", ln=True)
                    pdf.set_font("Arial", '', 12)
                    exec_summary = (
                        "This report provides a comprehensive analysis of credit card transactions to identify and detect fraudulent activities. "
                        "It encompasses data overview, exploratory data analysis, feature importance, model evaluations, and actionable insights to support strategic decision-making and risk management."
                    )
                    pdf.multi_cell(0, 10, exec_summary)
                    pdf.ln(5)

                    # Data Overview
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Data Overview", ln=True)
                    pdf.set_font("Arial", '', 12)
                    data_overview = (
                        f"- **Total Transactions:** {len(df):,}\n"
                        f"- **Fraudulent Transactions:** {df['Class'].sum():,} ({(df['Class'].sum() / len(df)) * 100:.4f}%)\n"
                        f"- **Valid Transactions:** {len(df) - df['Class'].sum():,} ({100 - (df['Class'].sum() / len(df)) * 100:.4f}%)\n"
                        "- **Feature Details:** V1 to V28 are PCA-transformed features ensuring anonymity and reduced dimensionality. 'Time' indicates time since the first transaction, and 'Amount' represents transaction value in USD.\n"
                        "- **Data Imbalance:** The dataset is highly imbalanced, with fraudulent transactions constituting a small fraction, posing challenges for effective fraud detection."
                    )
                    pdf.multi_cell(0, 10, data_overview)
                    pdf.ln(5)

                    # Model Evaluation Summary
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Model Evaluation Summary", ln=True)
                    pdf.set_font("Arial", '', 12)
                    model_evaluation_summary = (
                        f"- **Model:** {classifier}\n"
                        f"- **Test Set Size:** {test_size}%\n"
                        f"- **Total Test Samples:** {len(y_test)}\n"
                        f"- **Fraudulent Transactions in Test Set:** {y_test.sum()} ({(y_test.sum() / len(y_test)) * 100:.4f}%)\n"
                        f"- **Valid Transactions in Test Set:** {len(y_test) - y_test.sum()} ({100 - (y_test.sum() / len(y_test)) * 100:.4f}%)\n"
                        f"- **Accuracy:** {metrics['accuracy']:.4f}\n"
                        f"- **F1-Score:** {metrics['f1_score']:.4f}\n"
                        f"- **Matthews Correlation Coefficient (MCC):** {metrics['mcc']:.4f}\n"
                        f"- **Precision:** {metrics['precision']:.4f}\n"
                        f"- **Recall:** {metrics['recall']:.4f}\n"
                        f"- **F2-Score:** {metrics['f2_score']:.4f}\n"
                        f"- **ROC-AUC:** {roc_auc if roc_auc != 'N/A' else 'N/A'}\n"
                    )
                    pdf.multi_cell(0, 10, model_evaluation_summary)
                    pdf.ln(5)

                    # Confusion Matrix Visualization
                    # Save the confusion matrix plot as a temporary file
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                    sns.heatmap(
                        confusion_matrix(y_test, y_pred),
                        annot=True,
                        fmt='d',
                        cmap='YlOrBr',
                        xticklabels=['Valid', 'Fraud'],
                        yticklabels=['Valid', 'Fraud'],
                        ax=ax_cm
                    )
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    ax_cm.set_title(f"Confusion Matrix for {classifier}")
                    plt.tight_layout()
                    cm_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                    plt.savefig(cm_image_path, dpi=300)
                    plt.close(fig_cm)

                    # Add Confusion Matrix to PDF
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Confusion Matrix", ln=True, align='C')
                    pdf.image(cm_image_path, x=30, y=30, w=150)
                    pdf.ln(100)  # Adjust as per image size
                    os.remove(cm_image_path)  # Delete the temporary file

                    # ROC Curve Visualization (if applicable)
                    if roc_auc != "N/A" and y_proba is not None:
                        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        roc_auc_val = auc(fpr, tpr)
                        sns.lineplot(x=fpr, y=tpr, label=f'ROC Curve (AUC = {roc_auc_val:.4f})', ax=ax_roc)
                        sns.lineplot([0, 1], [0, 1], linestyle='--', color='grey', ax=ax_roc)
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title(f"ROC Curve for {classifier}")
                        ax_roc.legend(loc='lower right')
                        plt.tight_layout()
                        roc_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                        plt.savefig(roc_image_path, dpi=300)
                        plt.close(fig_roc)

                        # Add ROC Curve to PDF
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "ROC Curve", ln=True, align='C')
                        pdf.image(roc_image_path, x=30, y=30, w=150)
                        pdf.ln(100)  # Adjust as per image size
                        os.remove(roc_image_path)  # Delete the temporary file

                    # Finalize and Save the PDF
                    report_path = "fraud_detection_report.pdf"
                    pdf.output(report_path)

                    # Provide download button
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=file,
                            file_name=report_path,
                            mime="application/pdf"
                        )
                    st.success("Report generated and ready for download!")

                    # Clean up the temporary PDF file
                    os.remove(report_path)

                except Exception as e:
                    st.error(f"Error generating report: {e}")

    # Feedback Page
    elif page_selection == "Feedback":
        st.header("💬 Feedback")
        st.markdown("""
        **We Value Your Feedback:**
        Help us improve the Credit Card Fraud Detection Dashboard by providing your valuable feedback and suggestions.
        """)

        # Feedback input
        feedback = st.text_area("Provide your feedback here:")

        # Submit feedback button
        if st.button("Submit Feedback"):
            if feedback.strip() == "":
                st.warning("Please enter your feedback before submitting.")
            else:
                # Placeholder for feedback storage (e.g., database or email)
                # Implement actual storage mechanism as needed
                st.success("Thank you for your feedback!")

    else:
        st.error("Page not found.")
