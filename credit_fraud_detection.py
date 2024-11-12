import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import streamlit as st
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef

# Suppress warnings
warnings.filterwarnings("ignore")

# Directory containing the saved models
models_dir = "models"

# Streamlit App Title
st.title('Credit Card Fraud Detection - Using Pre-trained Models')

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
    'Logistic Regression': os.path.join(models_dir, 'logistic_regression.pkl'),
    'kNN': os.path.join(models_dir, 'knn.pkl'),
    'Random Forest': os.path.join(models_dir, 'random_forest.pkl'),
    'Extra Trees': os.path.join(models_dir, 'extra_trees.pkl')
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

# Model evaluation
st.write(f"Evaluating {classifier}...")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
st.write('Confusion Matrix:', cm)
st.write('Classification Report:', classification_report(y_test, y_pred))
mcc = matthews_corrcoef(y_test, y_pred)
st.write(f'Matthews Correlation Coefficient: {mcc:.3f}')

# Plot Confusion Matrix
if st.sidebar.checkbox('Show plot of confusion matrix'):
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Valid', 'Fraud'])
    plt.yticks([0, 1], ['Valid', 'Fraud'])
    plt.show()
    st.pyplot()
