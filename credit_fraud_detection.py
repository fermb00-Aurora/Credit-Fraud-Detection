# fraud_detection_app.py

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
    precision_score,
    recall_score,
    fbeta_score,
    precision_recall_curve,
    average_precision_score,
)
import tempfile
from io import BytesIO
import base64

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
    # Introduction Page
    if page_selection == "Introduction":
        # [Introduction code remains unchanged]
        pass

    # Data Overview Page
    elif page_selection == "Data Overview":
        # [Data Overview code remains unchanged]
        pass

    # Exploratory Data Analysis Page
    elif page_selection == "Exploratory Data Analysis":
        # [EDA code remains unchanged]
        pass

    # Feature Importance Page
    elif page_selection == "Feature Importance":
        # [Feature Importance code remains unchanged]
        pass

    # Model Evaluation Page
    elif page_selection == "Model Evaluation":
        st.header("🧠 Model Evaluation")

        st.markdown("""
            **Comprehensive Model Assessment:**
            This section provides an in-depth evaluation of various machine learning models used for fraud detection. By analyzing key performance metrics and visualizations, executives can understand each model's effectiveness and suitability for deployment.
        """)

        # Models supporting feature importance plus additional models
        all_models = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'Extra Trees': 'extra_trees.pkl',
            'k-Nearest Neighbors': 'knn.pkl'
        }

        classifier = st.selectbox("Select Model for Evaluation:", list(all_models.keys()))
        model_file = all_models[classifier]
        model_path = os.path.join(os.path.dirname(__file__), model_file)

        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Test set size slider
        test_size = st.slider('Test Set Size (%)', min_value=10, max_value=50, value=30, step=5)
        test_size_fraction = test_size / 100

        # Prepare data and perform train/test split
        X = df.drop(columns=['Class'])
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_fraction, random_state=42, stratify=y
        )

        # Make predictions
        y_pred = model.predict(X_test)

        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
                    xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {classifier}")
        st.pyplot(fig_cm)

        # Save confusion matrix to buffer for report
        buf_cm = BytesIO()
        plt.savefig(buf_cm, format='png')
        buf_cm.seek(0)
        cm_base64 = base64.b64encode(buf_cm.read()).decode('utf-8')
        buf_cm.close()
        plt.close(fig_cm)

        # Performance Metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        cls_report = classification_report(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("🔹 F1-Score", f"{f1:.4f}")
        col2.metric("🔹 Precision", f"{precision:.4f}")
        col3.metric("🔹 Recall", f"{recall:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("🔹 Accuracy", f"{accuracy:.4f}")
        col5.metric("🔹 F2-Score", f"{f2:.4f}")
        col6.metric("🔹 Matthews Corr. Coef.", f"{mcc:.4f}")

        # Precision-Recall Curve
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Normalize

        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
        average_precision = average_precision_score(y_test, y_proba)

        st.subheader("📈 Precision-Recall Curve")
        fig_pr = px.area(
            x=recall_vals, y=precision_vals,
            title=f"Precision-Recall Curve (AP = {average_precision:.4f}) for {classifier}",
            labels={'x': 'Recall', 'y': 'Precision'},
            width=700, height=500
        )
        fig_pr.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        fig_pr.update_yaxes(range=[0, 1], constrain='domain')
        fig_pr.update_xaxes(range=[0, 1], constrain='domain')
        st.plotly_chart(fig_pr, use_container_width=True)

        # Save PR curve to buffer for report
        buf_pr = BytesIO()
        fig_pr.write_image(buf_pr, format='png')
        buf_pr.seek(0)
        pr_base64 = base64.b64encode(buf_pr.read()).decode('utf-8')
        buf_pr.close()

        # Threshold vs. F1 Score Plot
        st.subheader("📉 Threshold vs. F1 Score")
        f1_scores = []
        thresholds_list = np.linspace(0, 1, 100)
        for thresh in thresholds_list:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_test, y_pred_thresh))
        fig_thresh = px.line(
            x=thresholds_list, y=f1_scores,
            labels={'x': 'Threshold', 'y': 'F1 Score'},
            title='F1 Score vs. Decision Threshold'
        )
        st.plotly_chart(fig_thresh, use_container_width=True)

        # Save Threshold vs F1 Score plot to buffer for report
        buf_thresh = BytesIO()
        fig_thresh.write_image(buf_thresh, format='png')
        buf_thresh.seek(0)
        thresh_base64 = base64.b64encode(buf_thresh.read()).decode('utf-8')
        buf_thresh.close()

        st.markdown("""
        **Insights:**
        - **Precision-Recall Curve:** Illustrates the trade-off between precision and recall for different threshold settings. It's particularly useful for imbalanced datasets.
        - **Threshold Analysis:** The F1 Score vs. Threshold plot helps in selecting an optimal decision threshold that balances precision and recall according to business needs.
        """)

        # Store evaluation results in session state
        st.session_state['model_evaluation'] = {
            'classifier': classifier,
            'model': model,
            'test_size': test_size,
            'X_test': X_test,
            'y_test': y_test,
            'f1': f1,
            'accuracy': accuracy,
            'mcc': mcc,
            'precision': precision,
            'recall': recall,
            'f2': f2,
            'classification_report': cls_report,
            'confusion_matrix_base64': cm_base64,
            'pr_curve_base64': pr_base64,
            'thresh_curve_base64': thresh_base64
        }

    # Simulator Page
    elif page_selection == "Simulator":
        st.header("🚀 Simulator")
        st.markdown("""
        **Simulate and Predict Fraudulent Transactions:**
        Enter transaction details to receive an immediate prediction on whether the transaction is fraudulent.
        """)

        # Retrieve model and data from session state
        if 'model_evaluation' in st.session_state and st.session_state['model_evaluation']:
            model_sim = st.session_state['model_evaluation']['model']
            classifier = st.session_state['model_evaluation']['classifier']
            X_test = st.session_state['model_evaluation']['X_test']
            st.info(f"Using the {classifier} model from Model Evaluation.")
        else:
            st.warning("Please select a model in the Model Evaluation section first.")
            st.stop()

        # Input transaction details
        st.subheader("🔍 Enter Transaction Details")
        col1, col2 = st.columns(2)

        with col1:
            V_features = {}
            for i in range(1, 29):
                V_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.5f", key=f'V{i}')

        with col2:
            Time = st.number_input('Time (seconds since first transaction)', min_value=0, value=0, step=1, key='Time')
            Amount = st.number_input('Transaction Amount ($)', min_value=0.0, value=0.0, format="%.2f", key='Amount')

        # Predict button
        if st.button("Simulate"):
            input_data = pd.DataFrame({
                **V_features,
                'Time': [Time],
                'Amount': [Amount]
            })

            prediction = model_sim.predict(input_data)[0]
            if hasattr(model_sim, "predict_proba"):
                prediction_proba = model_sim.predict_proba(input_data)[0][1]
            else:
                prediction_proba = model_sim.decision_function(input_data)[0]
                # Normalize the decision function output
                df_scores = model_sim.decision_function(X_test)
                prediction_proba = (prediction_proba - df_scores.min()) / (df_scores.max() - df_scores.min())

            if prediction == 1:
                st.error(f"⚠️ **Fraudulent Transaction Detected!** Probability: {prediction_proba:.2%}")
            else:
                st.success(f"✅ **Valid Transaction.** Probability of Fraud: {prediction_proba:.2%}")

    # Download Report Page
    elif page_selection == "Download Report":
        st.header("📄 Download Report")
        st.markdown("""
        **Generate and download a comprehensive report of the fraud detection analysis.**
        """)

        # Retrieve evaluation results from session state
        if 'model_evaluation' in st.session_state and st.session_state['model_evaluation']:
            eval_results = st.session_state['model_evaluation']
        else:
            st.warning("Please perform model evaluation first in the Model Evaluation section.")
            st.stop()

        # Generate PDF Report
        def generate_report():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Title
            pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')

            # Dataset Info
            total_transactions = len(df)
            total_fraudulent = df['Class'].sum()
            fraud_percentage = (total_fraudulent / total_transactions) * 100
            pdf.cell(200, 10, txt=f"Total Transactions: {total_transactions}", ln=True)
            pdf.cell(200, 10, txt=f"Fraudulent Transactions: {total_fraudulent} ({fraud_percentage:.4f}%)", ln=True)

            # Model Info
            pdf.cell(200, 10, txt=f"Selected Model: {eval_results['classifier']}", ln=True)
            pdf.cell(200, 10, txt=f"Test Set Size: {eval_results['test_size']}%", ln=True)

            # Metrics
            pdf.cell(200, 10, txt="Performance Metrics:", ln=True)
            pdf.cell(200, 10, txt=f"Accuracy: {eval_results['accuracy']:.4f}", ln=True)
            pdf.cell(200, 10, txt=f"Precision: {eval_results['precision']:.4f}", ln=True)
            pdf.cell(200, 10, txt=f"Recall: {eval_results['recall']:.4f}", ln=True)
            pdf.cell(200, 10, txt=f"F1-Score: {eval_results['f1']:.4f}", ln=True)
            pdf.cell(200, 10, txt=f"F2-Score: {eval_results['f2']:.4f}", ln=True)
            pdf.cell(200, 10, txt=f"Matthews Corr. Coefficient (MCC): {eval_results['mcc']:.4f}", ln=True)

            # Classification Report
            pdf.cell(200, 10, txt="Classification Report:", ln=True)
            pdf.set_font("Courier", size=10)
            pdf.multi_cell(0, 5, eval_results['classification_report'])
            pdf.set_font("Arial", size=12)

            # Confusion Matrix Image
            pdf.cell(200, 10, txt="Confusion Matrix:", ln=True)
            pdf.image(BytesIO(base64.b64decode(eval_results['confusion_matrix_base64'])), w=100)

            # Precision-Recall Curve Image
            pdf.cell(200, 10, txt="Precision-Recall Curve:", ln=True)
            pdf.image(BytesIO(base64.b64decode(eval_results['pr_curve_base64'])), w=150)

            # Threshold vs F1 Score Plot Image
            pdf.cell(200, 10, txt="F1 Score vs. Decision Threshold:", ln=True)
            pdf.image(BytesIO(base64.b64decode(eval_results['thresh_curve_base64'])), w=150)

            # Save PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                pdf.output(tmpfile.name)
                return tmpfile.name

        # Button to generate and download report
        if st.button("Generate and Download Report"):
            report_file = generate_report()
            with open(report_file, "rb") as file:
                st.download_button("Download Report", file, file_name="fraud_detection_report.pdf")
            os.remove(report_file)

    # Feedback Page
    elif page_selection == "Feedback":
        # [Feedback code remains unchanged]
        pass

    else:
        st.error("Page not found.")
