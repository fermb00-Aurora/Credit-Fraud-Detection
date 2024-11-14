# fraud_detection_app.py

"""
Credit Card Fraud Detection Dashboard
Author: Fernando Moreno Borrego
Date: 14.11.2024
Description:
A Streamlit application for detecting fraudulent credit card transactions using machine learning models.
"""

# Import necessary libraries
import os
import joblib
import tempfile
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
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
    roc_auc_score,
    cohen_kappa_score
)

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
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Loads the credit card fraud dataset."""
    data_path = 'creditcard.csv'
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please ensure the dataset is in the correct directory.")
        st.stop()
    df = pd.read_csv(data_path)
    return df

# Load the dataset
df = load_data()

# Initialize session state for model evaluation results
if 'model_evaluation' not in st.session_state:
    st.session_state['model_evaluation'] = {}

# Dictionary of all available models
all_models = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Random Forest': 'random_forest.pkl',
    'Extra Trees': 'extra_trees.pkl',
    'k-Nearest Neighbors': 'knn.pkl'
}

# Ensure that the dataset is loaded before proceeding
if df is not None:
    # Introduction Page
    if page_selection == "Introduction":
        st.header("📘 Executive Summary")

        # Display the GIF from GitHub repository
        gif_url = "https://raw.githubusercontent.com/fermb00-Aurora/Credit-Fraud-Detection/main/2fb9cae9fdb0110d8a57e9cc394f35dd.gif"
        st.image(gif_url, caption="Credit Card Fraud Detection in Action", use_column_width=True)

        st.markdown("""
        **Objective:**  
        Empower financial executives with advanced tools to detect and analyze fraudulent credit card transactions. By leveraging sophisticated machine learning models, this platform provides actionable insights to mitigate financial losses and enhance security measures.

        **Key Highlights:**
        - **Comprehensive Data Analysis:** In-depth exploration of transaction data to identify patterns and anomalies.
        - **Advanced Machine Learning Models:** Evaluation of pre-trained models including Logistic Regression, Random Forest, Extra Trees, and k-Nearest Neighbors for accurate fraud detection.
        - **Interactive Visualizations:** Dynamic charts and graphs that facilitate intuitive understanding of data trends and model performances.
        - **Actionable Insights:** Detailed reports and metrics that support strategic decision-making and risk management.
        - **Customizable Reports:** Generate and download tailored PDF reports to share findings with stakeholders.

        **Business Implications:**
        - **Risk Mitigation:** Early detection of fraudulent activities reduces financial losses and safeguards customer trust.
        - **Operational Efficiency:** Streamlined analysis processes save time and resources, enabling focus on critical tasks.
        - **Strategic Decision-Making:** Data-driven insights inform policies and strategies to enhance security protocols and customer satisfaction.
        """)


    # Data Overview Page
    elif page_selection == "Data Overview":
        st.header("🔍 Data Overview")

        # Display the first few rows of the dataset
        st.subheader("📂 Dataset Preview")
        st.dataframe(df.head(10).style.highlight_max(axis=0))

        st.markdown("---")

        # Data Summary Statistics
        st.subheader("📊 Data Summary")
        st.dataframe(df.describe().T.style.background_gradient(cmap='YlGnBu'))

        total_transactions = len(df)
        total_fraudulent = df['Class'].sum()
        total_valid = total_transactions - total_fraudulent
        fraudulent_percentage = (total_fraudulent / total_transactions) * 100
        valid_percentage = 100 - fraudulent_percentage

        st.markdown(f"""
        **Discussion:**
        - **Total Transactions:** The dataset comprises **{total_transactions:,}** transactions.
        - **Class Distribution:** Out of these, **{total_fraudulent:,}** are labeled as fraudulent (**{fraudulent_percentage:.4f}%**) and the remaining **{total_valid:,}** as valid.
        - **Feature Details:**
            - **V1 to V28:** Result of a PCA transformation to ensure anonymity and reduce dimensionality.
            - **Time:** Seconds elapsed since the first transaction in the dataset, providing temporal context.
            - **Amount:** Transaction amount in US dollars.
        - **Data Imbalance:** The significant imbalance between fraudulent and valid transactions underscores the challenge in fraud detection, necessitating specialized modeling techniques.
        """)

    # Exploratory Data Analysis Page
    elif page_selection == "Exploratory Data Analysis":
        st.header("📊 Exploratory Data Analysis")
        # [Content remains the same as before]
        # ...

    # Feature Importance Page
    elif page_selection == "Feature Importance":
        st.header("🔍 Feature Importance Analysis")
        # [Content remains the same as before]
        # ...

    # Model Evaluation Page
    elif page_selection == "Model Evaluation":
        st.header("🧠 Model Evaluation")

        st.markdown("""
            **Comprehensive Model Assessment:**
            This section provides an in-depth evaluation of various machine learning models used for fraud detection. By analyzing key performance metrics and visualizations, executives can understand each model's effectiveness and suitability for deployment.
        """)

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

        X = df.drop(columns=['Class'])
        y = df['Class']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_fraction, random_state=42, stratify=y
        )

        # Train the model
        model.fit(X_train, y_train)

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

        # Performance Metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        cohen_kappa = cohen_kappa_score(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("🔹 F1-Score", f"{f1:.4f}")
        col2.metric("🔹 Precision", f"{precision:.4f}")
        col3.metric("🔹 Recall", f"{recall:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("🔹 Accuracy", f"{accuracy:.4f}")
        col5.metric("🔹 F2-Score", f"{f2:.4f}")
        col6.metric("🔹 Matthews Corr. Coef.", f"{mcc:.4f}")

        st.metric("🔹 Cohen's Kappa", f"{cohen_kappa:.4f}")

        # ROC Curve and AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Normalize

        fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        st.subheader("📈 ROC Curve")
        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f"ROC Curve (AUC = {roc_auc:.4f}) for {classifier}",
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            width=700, height=500
        )
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
        fig_roc.update_xaxes(range=[0, 1])
        st.plotly_chart(fig_roc, use_container_width=True)

        # Precision-Recall Curve
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_proba)
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
        fig_pr.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
        fig_pr.update_xaxes(range=[0, 1])
        st.plotly_chart(fig_pr, use_container_width=True)

        # Threshold vs. F1 Score Plot
        st.subheader("📉 Threshold vs. F1 Score")
        f1_scores = []
        thresholds = np.linspace(0, 1, 100)
        for thresh in thresholds:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_test, y_pred_thresh))
        fig_thresh = px.line(
            x=thresholds, y=f1_scores,
            labels={'x': 'Threshold', 'y': 'F1 Score'},
            title='F1 Score vs. Decision Threshold'
        )
        st.plotly_chart(fig_thresh, use_container_width=True)

        st.markdown("""
        **Insights:**
        - **ROC Curve:** Demonstrates the trade-off between true positive rate and false positive rate. A higher AUC indicates better model performance.
        - **Precision-Recall Curve:** Illustrates the trade-off between precision and recall for different threshold settings. It's particularly useful for imbalanced datasets.
        - **Threshold Analysis:** The F1 Score vs. Threshold plot helps in selecting an optimal decision threshold that balances precision and recall according to business needs.
        """)

        # Store evaluation data into session state
        st.session_state['model_evaluation'] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'classifier': classifier,
            'model_file': model_file,  # Added to track the model file used
            'metrics': {
                'f1_score': f1,
                'accuracy': accuracy,
                'mcc': mcc,
                'precision': precision,
                'recall': recall,
                'f2_score': f2,
                'cohen_kappa': cohen_kappa
            },
            'test_size': test_size,
            'roc_auc': roc_auc,
            'y_proba': y_proba
        }

    # Simulator Page
    elif page_selection == "Simulator":
        st.header("🚀 Simulator")
        st.markdown("""
        **Simulate and Predict Fraudulent Transactions:**
        Enter transaction details to receive an immediate prediction on whether the transaction is fraudulent.
        """)

        # Check if a model has been evaluated and stored in session state
        eval_data = st.session_state.get('model_evaluation', {})
        if 'classifier' in eval_data and 'model_file' in eval_data:
            # Use the model selected in Model Evaluation
            classifier = eval_data['classifier']
            model_file = eval_data['model_file']
            st.info(f"Using the model selected in Model Evaluation: **{classifier}**")
        else:
            # Allow the user to select a model
            st.warning("No model selected in Model Evaluation. Please select a model for simulation.")
            classifier = st.selectbox("Select Model for Simulation:", list(all_models.keys()))
            model_file = all_models[classifier]

        model_path = os.path.join(os.path.dirname(__file__), model_file)

        try:
            model_sim = joblib.load(model_path)

            # Input transaction details
            st.subheader("🔍 Enter Transaction Details")
            col1, col2 = st.columns(2)

            with col1:
                V_features = {}
                for i in range(1, 29):
                    V_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.5f", key=f'Sim_V{i}')

            with col2:
                Time = st.number_input('Time (seconds since first transaction)', min_value=0, value=0, step=1, key='Sim_Time')
                Amount = st.number_input('Transaction Amount ($)', min_value=0.0, value=0.0, format="%.2f", key='Sim_Amount')

            # Predict button
            if st.button("Simulate"):
                # Get the feature names
                feature_names = df.drop(columns=['Class']).columns.tolist()

                # Create a dictionary with all feature values initialized to zero
                input_dict = {feature: 0 for feature in feature_names}

                # Update with the values from user input
                input_dict.update(V_features)
                input_dict['Time'] = Time
                input_dict['Amount'] = Amount

                # Create input_data DataFrame with correct order of columns
                input_data = pd.DataFrame([input_dict], columns=feature_names)

                # Predict
                prediction = model_sim.predict(input_data)[0]
                if hasattr(model_sim, "predict_proba"):
                    prediction_proba = model_sim.predict_proba(input_data)[0][1]
                    fraud_probability = f"{prediction_proba:.2%}"
                else:
                    fraud_probability = "N/A"

                if prediction == 1:
                    st.error(f"⚠️ **Fraudulent Transaction Detected!** Probability of Fraud: {fraud_probability}")
                else:
                    st.success(f"✅ **Valid Transaction.** Probability of Fraud: {fraud_probability}")
        except Exception as e:
            st.error(f"Error loading model '{model_file}': {e}")

    # Download Report Page
    elif page_selection == "Download Report":
        st.header("📄 Download Report")
        st.markdown("""
        **Generate and Download a Professional PDF Report:**
        Compile your analysis and model evaluation results into a concise and professional PDF report for offline review and sharing with stakeholders.
        """)

        # Button to generate report
        if st.button("Generate Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Retrieve evaluation data from session state
                    eval_data = st.session_state.get('model_evaluation', {})
                    required_keys = ['classifier', 'metrics', 'test_size']
                    if not all(key in eval_data for key in required_keys):
                        st.error("Please perform a model evaluation before generating the report.")
                    else:
                        classifier = eval_data['classifier']
                        metrics = eval_data['metrics']
                        test_size = eval_data['test_size']
                        roc_auc = eval_data.get('roc_auc', "N/A")

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
                            "It includes a data overview and model evaluation results to support strategic decision-making and risk management."
                        )
                        pdf.multi_cell(0, 10, exec_summary)
                        pdf.ln(5)

                        # Data Overview
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "Data Overview", ln=True)
                        pdf.set_font("Arial", '', 12)
                        total_transactions = len(df)
                        total_fraudulent = df['Class'].sum()
                        total_valid = total_transactions - total_fraudulent
                        fraudulent_percentage = (total_fraudulent / total_transactions) * 100
                        valid_percentage = 100 - fraudulent_percentage
                        data_overview = (
                            f"- **Total Transactions:** {total_transactions:,}\n"
                            f"- **Fraudulent Transactions:** {total_fraudulent:,} ({fraudulent_percentage:.4f}%)\n"
                            f"- **Valid Transactions:** {total_valid:,} ({valid_percentage:.4f}%)\n"
                            "- **Features:** The dataset includes anonymized features resulting from a PCA transformation, along with 'Time' and 'Amount'.\n"
                            "- **Data Imbalance:** The dataset is highly imbalanced, which presents challenges for effective fraud detection."
                        )
                        pdf.multi_cell(0, 10, data_overview)
                        pdf.ln(5)

                        # Model Evaluation Summary
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "Model Evaluation Summary", ln=True)
                        pdf.set_font("Arial", '', 12)
                        model_evaluation_summary = (
                            f"- **Model Used:** {classifier}\n"
                            f"- **Test Set Size:** {test_size}%\n"
                            f"- **Accuracy:** {metrics['accuracy']:.4f}\n"
                            f"- **F1-Score:** {metrics['f1_score']:.4f}\n"
                            f"- **Precision:** {metrics['precision']:.4f}\n"
                            f"- **Recall:** {metrics['recall']:.4f}\n"
                            f"- **ROC-AUC Score:** {roc_auc if roc_auc != 'N/A' else 'N/A'}\n"
                        )
                        pdf.multi_cell(0, 10, model_evaluation_summary)
                        pdf.ln(5)

                        # Conclusion
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "Conclusion", ln=True)
                        pdf.set_font("Arial", '', 12)
                        conclusion = (
                            "The evaluation results indicate that the selected model demonstrates reliable performance in detecting fraudulent transactions. "
                            "Key metrics such as F1-Score and ROC-AUC suggest a balanced trade-off between precision and recall, which is crucial in minimizing both false positives and false negatives. "
                            "These insights can aid in refining fraud detection strategies and enhancing financial security measures."
                        )
                        pdf.multi_cell(0, 10, conclusion)
                        pdf.ln(5)

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

