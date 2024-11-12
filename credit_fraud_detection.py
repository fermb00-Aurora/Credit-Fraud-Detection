import timeit
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
import warnings
import streamlit as st
import joblib
import os
from fpdf import FPDF
import pydot
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, roc_curve, auc

# Suppress warnings
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Advanced Credit Card Fraud Detection", page_icon="💳")

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

df = load_data()

# Sidebar options
if st.sidebar.checkbox("Show dataset overview"):
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Description:", df.describe())

# Fraud analysis
fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
outlier_percentage = (len(fraud) / len(valid)) * 100

if st.sidebar.checkbox("Show fraud analysis"):
    st.metric("Fraud Cases", len(fraud), delta=f"{outlier_percentage:.2f}% of total")
    st.metric("Valid Cases", len(valid))

# Splitting features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Train-test split
size = st.sidebar.slider("Select test size", 0.2, 0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

# Load pre-trained models
models = {
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "kNN": joblib.load("knn.pkl"),
    "Random Forest": joblib.load("random_forest.pkl"),
    "Extra Trees": joblib.load("extra_trees.pkl")
}

# Model selection
selected_model = st.sidebar.selectbox("Choose model for evaluation", list(models.keys()))
model = models[selected_model]

# Evaluation
st.subheader(f"Evaluating {selected_model}")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Confusion Matrix and Metrics
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

# Display Confusion Matrix Heatmaps
st.subheader("Confusion Matrix - Training Set")
fig, ax = plt.subplots()
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title("Confusion Matrix - Training Set")
st.pyplot(fig)

st.subheader("Confusion Matrix - Test Set")
fig, ax = plt.subplots()
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title("Confusion Matrix - Test Set")
st.pyplot(fig)

# Display Classification Report
st.text("Classification Report - Test Set")
st.text(classification_report(y_test, y_pred_test))
mcc = matthews_corrcoef(y_test, y_pred_test)
st.metric("Matthews Correlation Coefficient", f"{mcc:.3f}")

# Model-Specific Evaluation
if selected_model == "Random Forest":
    st.subheader("Random Forest Specific Metrics")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("ROC Curve - Random Forest")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Tree Visualization
    st.subheader("Random Forest Tree Visualization")
    dot_data = StringIO()
    export_graphviz(model.estimators_[0], out_file=dot_data,
                    feature_names=X.columns, filled=True, rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    st.image(graph[0].create_png())

elif selected_model == "Logistic Regression":
    st.subheader("Logistic Regression Specific Metrics")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("ROC Curve - Logistic Regression")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

elif selected_model == "kNN":
    st.subheader("kNN Specific Metrics")
    st.write("No additional specific metrics for kNN.")

elif selected_model == "Extra Trees":
    st.subheader("Extra Trees Specific Metrics")
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(X.columns[sorted_idx])
    ax.set_title("Feature Importances - Extra Trees")
    st.pyplot(fig)

# Generate PDF Report
def create_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align="C")
    pdf.cell(0, 10, txt=f"Model: {selected_model}", ln=True)
    pdf.cell(0, 10, txt="Metrics:", ln=True)
    pdf.cell(0, 10, txt=f"Matthews Correlation Coefficient: {mcc:.3f}", ln=True)
    pdf.output("report.pdf")
    return "report.pdf"

if st.sidebar.button("Download PDF Report"):
    report = create_pdf_report()
    with open(report, "rb") as file:
        st.download_button("Download Report", data=file, file_name="report.pdf", mime="application/pdf")
