import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import streamlit as st
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split
from fpdf import FPDF

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit App Title
st.title('Credit Card Fraud Detection - Business-Oriented WebApp')

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('creditcard.csv')
        st.success("Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = load_data()

# Display DataFrame details
if st.sidebar.checkbox('Show DataFrame Details'):
    st.write(df.head(100))
    st.write('Shape of the DataFrame:', df.shape)
    st.write('Data Description:', df.describe())

# Fraud and Valid Transaction Analysis
fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
fraud_percentage = (len(fraud) / len(df)) * 100

if st.sidebar.checkbox('Show Fraud and Valid Transaction Details'):
    st.write(f'Fraudulent Transactions: {fraud_percentage:.2f}%')
    st.write('Number of Fraud Cases:', len(fraud))
    st.write('Number of Valid Cases:', len(valid))

# Splitting the features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Train-test split
test_size = st.sidebar.slider('Select Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Model filenames
models_dir = "."
model_filenames = {
    'Logistic Regression': 'logistic_regression.pkl',
    'kNN': 'knn.pkl',
    'Random Forest': 'random_forest.pkl',
    'Extra Trees': 'extra_trees.pkl'
}

# Sidebar selection for the classifier
classifier = st.sidebar.selectbox('Select Classifier', list(model_filenames.keys()))

# Load the selected model
try:
    model_path = os.path.join(models_dir, model_filenames[classifier])
    model = joblib.load(model_path)
    st.success(f"Loaded model: {classifier}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Predictions and Evaluation
st.write(f"Evaluating {classifier} on the test set...")
y_pred = model.predict(X_test)

# Confusion Matrix and Heatmap
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
st.write('Classification Report:')
st.json(report)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, y_pred)
st.write(f'Matthews Correlation Coefficient: {mcc:.3f}')

# Downloadable PDF Report
def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, f'{classifier} Model Evaluation Report', ln=True, align='C')

    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Fraudulent Transactions: {fraud_percentage:.2f}%', ln=True)
    pdf.cell(0, 10, f'Matthews Correlation Coefficient: {mcc:.3f}', ln=True)

    # Add Classification Report
    pdf.cell(0, 10, 'Classification Report:', ln=True)
    for key, value in report.items():
        pdf.cell(0, 10, f'{key}: {value}', ln=True)

    # Save PDF
    report_filename = f'{classifier}_evaluation_report.pdf'
    pdf.output(report_filename)
    return report_filename

if st.sidebar.button('Download Report'):
    report_file = generate_report()
    st.success(f'Report generated: {report_file}')
    with open(report_file, "rb") as file:
        btn = st.download_button(
            label="Download Report",
            data=file,
            file_name=report_file,
            mime="application/pdf"
        )
