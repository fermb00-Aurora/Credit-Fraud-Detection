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
import pydot
from io import StringIO
from PIL import Image
import plotly.express as px

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the directory for the saved models
models_dir = "."
st.title('Credit Card Fraud Detection App - Professional Design')

# Load the dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

df = load_data()

# Display dataset information
if st.sidebar.checkbox('Show DataFrame Overview'):
    st.dataframe(df.head())
    st.write('Data Shape:', df.shape)
    st.write('Data Description:', df.describe())

# Fraud analysis
fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
outlier_percentage = (len(fraud) / len(valid)) * 100

if st.sidebar.checkbox('Show Fraud Analysis'):
    st.write(f"Fraudulent Transactions: {outlier_percentage:.2f}%")
    st.write(f"Fraud Cases: {len(fraud)}, Valid Cases: {len(valid)}")

# Split the data
X = df.drop(columns=['Class'])
y = df['Class']
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

# Define model filenames
model_filenames = {
    'Logistic Regression': 'logistic_regression.pkl',
    'kNN': 'knn.pkl',
    'Random Forest': 'random_forest.pkl',
    'Extra Trees': 'extra_trees.pkl'
}

# Model selection
classifier = st.sidebar.selectbox('Choose a Classifier', list(model_filenames.keys()))
model_filename = model_filenames[classifier]

# Load the pre-trained model
st.write(f"Loading model: {model_filename}")
try:
    model = joblib.load(model_filename)
    st.success(f"Loaded {classifier} model successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Model evaluation and visualizations
def evaluate_model(model, X_test, y_test, model_name):
    st.write(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot()

    # Classification Report
    st.text('Classification Report:')
    st.text(classification_report(y_test, y_pred))

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    st.write(f"Matthews Correlation Coefficient: {mcc:.3f}")

    # Additional metrics for Random Forest and Extra Trees
    if model_name in ['Random Forest', 'Extra Trees']:
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        fig = px.bar(importance_df, x='Importance', y='Feature', title='Feature Importances')
        st.plotly_chart(fig)

        # Decision tree visualization (for Random Forest or Extra Trees)
        dot_data = StringIO()
        export_graphviz(model.estimators_[0], out_file=dot_data,
                        feature_names=X_test.columns, filled=True, rounded=True)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        img_path = f"{model_name}_tree.png"
        graph[0].write_png(img_path)
        st.image(Image.open(img_path), caption=f'{model_name} Decision Tree')

# Evaluate the selected model
evaluate_model(model, X_test, y_test, classifier)

# Generate a report for download
def generate_report(model_name, cm, classification_rep, mcc):
    report_text = f"""
    Credit Card Fraud Detection Report - {model_name}
    ===============================================
    Confusion Matrix:
    {cm}

    Classification Report:
    {classification_rep}

    Matthews Correlation Coefficient: {mcc:.3f}
    """
    return report_text

if st.sidebar.button('Download Report'):
    report = generate_report(classifier, confusion_matrix(y_test, model.predict(X_test)),
                             classification_report(y_test, model.predict(X_test)), 
                             matthews_corrcoef(y_test, model.predict(X_test)))
    with open("fraud_detection_report.txt", "w") as file:
        file.write(report)
    st.success("Report generated successfully! Click the link below to download.")
    st.download_button("Download Report", data=report, file_name="fraud_detection_report.txt")

# Design updates and enhancements
st.markdown("""
<style>
    body {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
    }
    .stSidebar {
        background-color: #f8f9fc;
    }
</style>
""", unsafe_allow_html=True)


