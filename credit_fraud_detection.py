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

# Directory for models
models_dir = "models"

# Streamlit App Title
st.title('📊 Credit Card Fraud Detection - Advanced Web App')

# Load the dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Display DataFrame details
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ', df.shape)
    st.write('Data description:', df.describe())

# Fraud and Valid Transaction Analysis
fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
outlier_percentage = (len(fraud) / len(valid)) * 100

if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write(f'Fraudulent transactions are: {outlier_percentage:.3f}%')
    st.write('Fraud Cases:', len(fraud))
    st.write('Valid Cases:', len(valid))

# Splitting features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Train-test split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

# Load models
model_filenames = {
    'Logistic Regression': 'logistic_regression.pkl',
    'kNN': 'knn.pkl',
    'Random Forest': 'random_forest.pkl',
    'Extra Trees': 'extra_trees.pkl'
}

# Sidebar model selection
classifier = st.sidebar.selectbox('Select the classifier for evaluation', list(model_filenames.keys()))
model_path = os.path.join(models_dir, model_filenames[classifier])

# Load the selected model
st.write(f"Loading the pre-trained model '{classifier}'...")
try:
    model = joblib.load(model_path)
    st.success(f"Model '{classifier}' loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Feature Importance
@st.cache_data
def get_feature_importance(_model, X_train, y_train):
    _model.fit(X_train, y_train)
    return _model.feature_importances_

# Feature Importance Plot
if classifier in ['Random Forest', 'Extra Trees']:
    if st.sidebar.checkbox('Show plot of feature importance'):
        importance = get_feature_importance(model, X_train, y_train)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=X_train.columns)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        st.pyplot()

# Model Evaluation
st.write(f"Evaluating {classifier}...")

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Confusion Matrix with Heatmap
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot()

# Display metrics
st.write("Train Set Metrics:")
plot_confusion_matrix(y_train, y_pred_train, "Confusion Matrix (Train Set)")
st.text(classification_report(y_train, y_pred_train))

st.write("Test Set Metrics:")
plot_confusion_matrix(y_test, y_pred_test, "Confusion Matrix (Test Set)")
st.text(classification_report(y_test, y_pred_test))

mcc = matthews_corrcoef(y_test, y_pred_test)
st.write(f'Matthews Correlation Coefficient (MCC): {mcc:.3f}')

# Generate PDF Report
def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')

    # Dataset Info
    pdf.cell(200, 10, txt=f"Fraudulent Transactions: {outlier_percentage:.3f}%", ln=True)
    pdf.cell(200, 10, txt=f"Fraud Cases: {len(fraud)} | Valid Cases: {len(valid)}", ln=True)

    # Model Info
    pdf.cell(200, 10, txt=f"Selected Model: {classifier}", ln=True)

    # Metrics
    pdf.cell(200, 10, txt="Classification Report (Test Set):", ln=True)
    pdf.multi_cell(0, 10, classification_report(y_test, y_pred_test))

    pdf.cell(200, 10, txt=f"Matthews Correlation Coefficient (MCC): {mcc:.3f}", ln=True)

    # Save PDF
    report_filename = "fraud_detection_report.pdf"
    pdf.output(report_filename)
    st.success(f"Report generated: {report_filename}")
    with open(report_filename, "rb") as file:
        st.download_button("Download Report", file, file_name=report_filename)

# Button to download report
if st.sidebar.button("Generate and Download Report"):
    generate_report()

