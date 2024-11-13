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
    # Introduction Page
    if page_selection == "Introduction":
        st.header("📘 Executive Summary")
        st.markdown("""
        **Objective:**  
        Empower financial executives with advanced tools to detect and analyze fraudulent credit card transactions. By leveraging sophisticated machine learning models, this platform provides actionable insights to mitigate financial losses and enhance security measures.

        **Key Highlights:**
        - **Comprehensive Data Analysis:** In-depth exploration of transaction data to identify patterns and anomalies.
        - **Advanced Machine Learning Models:** Evaluation of pre-trained models including Logistic Regression, Random Forest, and Extra Trees for accurate fraud detection.
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

        st.markdown("""
        **Discussion:**
        - **Total Transactions:** The dataset comprises **{:,}** transactions.
        - **Class Distribution:** Out of these, **{:,}** are labeled as fraudulent (**{:.4f}%**) and the remaining **{:,}** as valid.
        - **Feature Details:**
            - **V1 to V28:** Result of a PCA transformation to ensure anonymity and reduce dimensionality.
            - **Time:** Seconds elapsed since the first transaction in the dataset, providing temporal context.
            - **Amount:** Transaction amount in US dollars.
        - **Data Imbalance:** The significant imbalance between fraudulent and valid transactions underscores the challenge in fraud detection, necessitating specialized modeling techniques.
        """.format(
            len(df),
            df['Class'].sum(),
            (df['Class'].sum() / len(df)) * 100,
            len(df) - df['Class'].sum()
        ))

    # Exploratory Data Analysis Page
    elif page_selection == "Exploratory Data Analysis":
        st.header("📊 Exploratory Data Analysis")
        # Your original code for this section remains unchanged

    # Feature Importance Page
    elif page_selection == "Feature Importance":
        st.header("🔍 Feature Importance Analysis")
        st.markdown("""
        **Understanding Feature Impact:**
        Identifying which features significantly influence model predictions is paramount in credit card fraud detection. This section delves into the importance of various features across different machine learning models, providing clarity on what drives fraud detection decisions.

        **Models Analyzed:**
        - **Random Forest:** Utilizes ensemble learning to provide feature importance based on the mean decrease in impurity.
        - **Extra Trees:** Similar to Random Forest but with more randomness, offering robust feature importance metrics.
        - **Logistic Regression:** Assesses feature importance through the magnitude of coefficients, indicating the strength and direction of influence.
        """)

        # Dictionary of models supporting feature importance
        feature_importance_models = {
            'Random Forest': 'random_forest.pkl',
            'Extra Trees': 'extra_trees.pkl',
            'Logistic Regression': 'logistic_regression.pkl'
        }

        selected_model = st.selectbox("Select a model for feature importance:", list(feature_importance_models.keys()))
        model_filename = feature_importance_models[selected_model]
        model_path = os.path.join(os.path.dirname(__file__), model_filename)

        try:
            model = joblib.load(model_path)
            features = df.drop(columns=['Class']).columns

            # Determine feature importances based on model type
            if selected_model in ['Random Forest', 'Extra Trees']:
                importances = model.feature_importances_
            elif selected_model == 'Logistic Regression':
                importances = np.abs(model.coef_[0])
            else:
                st.error("Selected model does not support feature importance.")
                importances = None

            if importances is not None:
                # Create a DataFrame for feature importances
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                # Slider for selecting number of top features to display
                top_n = st.slider("Select Number of Top Features to Display:", min_value=5, max_value=20, value=10, step=1)

                # Display Top N Important Features
                st.subheader(f"📌 Top {top_n} Most Important Features")
                fig_imp_top = px.bar(
                    importance_df.head(top_n),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top {top_n} Feature Importances for {selected_model}",
                    labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                    color='Importance',
                    color_continuous_scale='YlOrRd'
                )
                st.plotly_chart(fig_imp_top, use_container_width=True)

                # Display Top N Least Important Features
                st.subheader(f"📉 Top {top_n} Least Important Features")
                fig_imp_bottom = px.bar(
                    importance_df.tail(top_n),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top {top_n} Least Important Features for {selected_model}",
                    labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_imp_bottom, use_container_width=True)

                st.markdown("""
                **Strategic Insights:**
                - **High-Impact Features:** Understanding which features most significantly influence fraud detection enables targeted enhancements in data collection and monitoring processes.
                - **Low-Impact Features:** Identifying features with minimal influence can streamline data preprocessing and reduce computational overhead without compromising model performance.
                - **Model Selection:** Different models may prioritize different features, offering diverse perspectives on what drives fraudulent activities.
                """)

                # Optional: Display Feature Importance Table
                st.subheader("📄 Feature Importance Table")
                st.dataframe(importance_df.style.background_gradient(cmap='YlOrRd'))
            else:
                st.error("Unable to extract feature importances for the selected model.")
        except Exception as e:
            st.error(f"Error loading model '{model_filename}': {e}")

    # Model Evaluation Page
    elif page_selection == "Model Evaluation":
        st.header("🧠 Model Evaluation")
        st.markdown("""
        **Comprehensive Model Assessment:**
        This section provides an in-depth evaluation of various machine learning models used for fraud detection. By analyzing key performance metrics and interactive visualizations, executives can understand each model's effectiveness and suitability for deployment.
        """)

        # Dictionary of all available models for evaluation
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
            # Slider for selecting test set size
            test_size = st.slider('Select Test Set Size (%)', min_value=10, max_value=50, value=30, step=5)
            test_size_fraction = test_size / 100

            # Prepare feature matrix X and target vector y
            X = df.drop(columns=['Class'])
            y = df['Class']

            # Split the data with stratification to maintain class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_fraction,
                random_state=42,
                stratify=y
            )

            # Make predictions
            y_pred = model.predict(X_test)

            # Store evaluation results in session state
            st.session_state['model_evaluation']['y_test'] = y_test
            st.session_state['model_evaluation']['y_pred'] = y_pred
            st.session_state['model_evaluation']['classifier'] = classifier
            st.session_state['model_evaluation']['metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f2_score': fbeta_score(y_test, y_pred, beta=2),
                'cohen_kappa': cohen_kappa_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) if hasattr(model, "predict_proba") else "N/A"
            }
            st.session_state['model_evaluation']['test_size'] = test_size  # Store test_size

            # Rest of your original code for Model Evaluation continues here

        except Exception as e:
            st.error(f"Error loading model '{model_file}': {e}")

    # Simulator Page
    elif page_selection == "Simulator":
        st.header("🚀 Simulator")
        st.markdown("""
        **Simulate and Predict Fraudulent Transactions:**
        Enter transaction details to receive an immediate prediction on whether the transaction is fraudulent.
        """)

        # Load the model
        default_model_filename = 'random_forest.pkl'
        model_path = os.path.join(os.path.dirname(__file__), default_model_filename)

        try:
            model_sim = joblib.load(model_path)

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
                prediction_proba = model_sim.predict_proba(input_data)[0][1]

                if prediction == 1:
                    st.error(f"⚠️ **Fraudulent Transaction Detected!** Probability: {prediction_proba:.2%}")
                else:
                    st.success(f"✅ **Valid Transaction.** Probability of Fraud: {prediction_proba:.2%}")
        except Exception as e:
            st.error(f"Error loading model '{default_model_filename}': {e}")

 # Download Report Page
    elif page_selection == "Download Report":
        st.header("📄 Download Report")
        st.markdown("""
        **Generate and Download a Comprehensive PDF Report:**
        Compile your analysis and model evaluation results into a downloadable PDF report for offline review and sharing with stakeholders.
        """)

        # Button to generate report
        if st.button("Generate Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Retrieve evaluation data from session state
                    eval_data = st.session_state['model_evaluation']
                    required_keys = ['y_test', 'y_pred', 'classifier', 'metrics', 'test_size']
                    if not all(key in eval_data for key in required_keys):
                        st.error("Please perform a model evaluation before generating the report.")
                    else:
                        y_test = eval_data['y_test']
                        y_pred = eval_data['y_pred']
                        classifier = eval_data['classifier']
                        metrics = eval_data['metrics']
                        test_size = eval_data['test_size']
                        roc_auc = eval_data.get('roc_auc', "N/A")
                        y_proba = eval_data.get('y_proba', None)

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
                            f"- **Total Transactions:** {len(y_test) + (len(df) - len(y_test)):,}\n"
                            f"- **Fraudulent Transactions:** {y_test.sum():,} ({(y_test.sum() / len(y_test)) * 100:.4f}%)\n"
                            f"- **Valid Transactions:** {len(y_test) - y_test.sum():,} ({100 - (y_test.sum() / len(y_test)) * 100:.4f}%)\n"
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
                        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlOrBr',
                                    xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'], ax=ax_cm)
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
