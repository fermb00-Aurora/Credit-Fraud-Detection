import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydot
from fpdf import FPDF

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Streamlit App Title
st.title('Credit Card Fraud Detection - Using Pre-trained Models')

# Directory containing the saved models
models_dir = os.path.dirname(__file__)

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

# Splitting the features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Train-test split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

# List of pre-trained models and their filenames
model_filenames = {
    'Logistic Regression': 'logistic_regression.pkl',
    'kNN': 'knn.pkl',
    'Random Forest': 'random_forest.pkl',
    'Extra Trees': 'extra_trees.pkl'
}

# Sidebar selection for the classifier
classifier = st.sidebar.selectbox('Select the classifier for evaluation', list(model_filenames.keys()))

# Load the selected model
model_filename = model_filenames[classifier]
st.write(f"Loading the pre-trained model '{model_filename}'...")
try:
    model = joblib.load(model_filename)
    st.write(f"Model '{classifier}' loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Model evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    st.write(f"Evaluating {classifier}...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Confusion matrix for training and test sets
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    # Heatmap for confusion matrices
    st.subheader("Confusion Matrix - Training Set")
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
    st.pyplot()

    st.subheader("Confusion Matrix - Test Set")
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
    st.pyplot()

    # Classification report
    st.write("Classification Report - Test Set")
    st.text(classification_report(y_test, y_pred_test))

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_test, y_pred_test)
    st.write(f'Matthews Correlation Coefficient: {mcc:.3f}')

    # Tree visualization for Random Forest or Extra Trees
    if classifier in ['Random Forest', 'Extra Trees']:
        st.subheader(f"{classifier} Tree Visualization")
        features = list(X.columns)
        dot_data = StringIO()
        export_graphviz(model.estimators_[0], out_file=dot_data, feature_names=features,
                        filled=True, rounded=True, special_characters=True)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        st.image(graph[0].create_png())

# Call the evaluation function
evaluate_model(model, X_train, X_test, y_train, y_test)

# PDF Report Generation
def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add content to the PDF
    pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Model Used: {classifier}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Matthews Correlation Coefficient: {mcc:.3f}", ln=True, align='L')
    pdf.cell(200, 10, txt="Classification Report:", ln=True, align='L')

    # Write classification report
    report = classification_report(y_test, y_pred_test, output_dict=False)
    pdf.multi_cell(0, 10, txt=report)

    # Save the PDF
    report_filename = f"{classifier}_report.pdf"
    pdf.output(report_filename)
    return report_filename

# Download Report
if st.sidebar.button('Download Report'):
    report_file = generate_report()
    with open(report_file, "rb") as file:
        btn = st.download_button(
            label="Download PDF Report",
            data=file,
            file_name=report_file,
            mime="application/pdf"
        )


