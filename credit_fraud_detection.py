import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import streamlit as st
import joblib
import os
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, roc_auc_score, accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit App Title
st.title('💳 Credit Card Fraud Detection Dashboard')

st.sidebar.header("Navigation")
page_selection = st.sidebar.radio("Go to:", ["Introduction", "Exploratory Data Analysis", "Feature Importance", "Model Evaluation", "Download Report"])

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
    Welcome to the Credit Card Fraud Detection Dashboard. This app provides an in-depth analysis of fraud detection using various pre-trained machine learning models.
    Explore the data, analyze feature importance, evaluate the models, and generate a detailed report. Detecting fraudulent transactions can help businesses prevent significant financial losses.
    """)

# Exploratory Data Analysis
if page_selection == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, color_continuous_scale='thermal', title='Interactive Correlation Heatmap')
    st.plotly_chart(fig)

# Feature Importance
if page_selection == "Feature Importance":
    st.header("🔍 Feature Importance Analysis")

    model_path_rf = os.path.join(os.path.dirname(__file__), 'random_forest.pkl')
    model_rf = joblib.load(model_path_rf)

    feature_importances = model_rf.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]

    top_features = pd.DataFrame({
        'Feature': df.drop(columns=['Class']).columns[sorted_idx[:3]],
        'Importance': feature_importances[sorted_idx[:3]]
    })
    least_features = pd.DataFrame({
        'Feature': df.drop(columns=['Class']).columns[sorted_idx[-3:]],
        'Importance': feature_importances[sorted_idx[-3:]]
    })

    st.subheader("Top 3 Most Important Features 🥇🥈🥉")
    st.write(top_features)

    st.subheader("Top 3 Least Important Features 🥉🥈🥇")
    st.write(least_features)

    st.markdown("The most important features indicate variables that have the highest predictive power for detecting fraud. The least important ones contribute the least.")

# Model Evaluation
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")
    size = st.sidebar.slider('Select Test Set Size', min_value=0.2, max_value=0.4)
    selected_features = df.drop(columns=['Class']).columns.tolist()
    X = df[selected_features]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

    model_choices = {
        'Logistic Regression': 'logistic_regression.pkl',
        'k-Nearest Neighbors (kNN)': 'knn.pkl',
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl'
    }

    classifier = st.sidebar.selectbox("Select a model to evaluate:", list(model_choices.keys()))
    model_file = model_choices[classifier]

    try:
        model_path = os.path.join(os.path.dirname(__file__), model_file)
        model = joblib.load(model_path)

        st.write(f"Evaluating {classifier}...")
        y_pred_test = model.predict(X_test)

        # Confusion Matrix with Heatmap
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=False, ax=ax)
        ax.set_title(f"Confusion Matrix for {classifier}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        report = classification_report(y_test, y_pred_test, output_dict=True)
        st.subheader("Detailed Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        # Additional Metrics
        roc_auc = roc_auc_score(y_test, y_pred_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        mcc = matthews_corrcoef(y_test, y_pred_test)

        st.write(f"ROC AUC Score: {roc_auc:.3f}")
        st.write(f"Accuracy: {accuracy:.3f}")
        st.write(f"Matthews Correlation Coefficient: {mcc:.3f}")

        # Business Relevance Plot
        st.subheader("Fraud Rate by Transaction Amount")
        fraud_df = df[df['Class'] == 1]
        fig = px.histogram(fraud_df, x='Amount', title="Distribution of Fraudulent Transactions by Amount", color_discrete_sequence=['darkred'])
        st.plotly_chart(fig)

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
        pdf.cell(200, 10, txt="Model Evaluation Summary", ln=True)

        pdf.cell(200, 10, txt=f"Selected Model: {classifier}", ln=True)
        pdf.multi_cell(0, 10, f"ROC AUC Score: {roc_auc:.3f}\nAccuracy: {accuracy:.3f}\nMCC: {mcc:.3f}")

        report_filename = "fraud_detection_report.pdf"
        pdf.output(report_filename)
        st.success("Report generated successfully.")
        with open(report_filename, "rb") as file:
            st.download_button("Download Report", file, file_name=report_filename)

    if st.button("Generate Report"):
        generate_report()



