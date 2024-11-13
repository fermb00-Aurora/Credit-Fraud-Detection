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
    auc
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
    "Real-Time Prediction",
    "Download Report",
    "Feedback"
])

# Function to load data with caching
@st.cache_data
def load_data(filepath='creditcard.csv'):
    df = pd.read_csv(filepath)
    return df

# Function to load models with caching
@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    if not os.path.exists(model_path):
        st.error(f"Model file {model_filename} not found in the directory.")
        return None
    model = joblib.load(model_path)
    return model

# Load the dataset
df = load_data()

# Introduction Page
if page_selection == "Introduction":
    st.header("📘 Executive Summary")
    st.markdown("""
    **Objective:**  
    Empower financial executives with advanced tools to detect and analyze fraudulent credit card transactions. This dashboard leverages sophisticated machine learning models to provide actionable insights, thereby mitigating financial losses and enhancing security measures.

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
if page_selection == "Data Overview":
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
if page_selection == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    # Correlation Heatmap
    st.subheader("🔗 Feature Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='YlOrBr',
        title='Correlation Heatmap of Features',
        aspect="auto",
        labels=dict(color="Correlation")
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Key Observations:**
    - **High Correlation Among V* Features:** Features V1 to V28, which are the result of a PCA transformation, exhibit high inter-correlation, indicating potential multicollinearity.
    - **Amount Feature:** The 'Amount' feature shows some correlation with other features, suggesting its significance in distinguishing between transaction classes.
    """)

    # Transaction Amount Over Time
    st.subheader("⏰ Transaction Amount Over Time")
    fig_time = px.scatter(
        df.sample(n=5000, random_state=42),  # Sampling for performance
        x='Time',
        y='Amount',
        color='Class',
        labels={
            'Time': 'Seconds Since First Transaction',
            'Amount': 'Transaction Amount ($)',
            'Class': 'Transaction Class'
        },
        title="Transaction Amounts Over Time",
        opacity=0.5,
        color_discrete_map={'0': 'green', '1': 'red'},
        hover_data={'Time': True, 'Amount': True, 'Class': True}
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Density Plot of Transaction Amounts
    st.subheader("📈 Density Plot of Transaction Amounts")
    fig_density = px.histogram(
        df,
        x='Amount',
        color='Class',
        nbins=50,
        histnorm='density',
        title="Density of Transaction Amounts by Class",
        labels={'Amount': 'Transaction Amount ($)', 'density': 'Density'},
        color_discrete_map={'0': 'green', '1': 'red'},
        opacity=0.6
    )
    st.plotly_chart(fig_density, use_container_width=True)

    # Transactions Over Time by Hour
    st.subheader("📅 Transactions Over Time")
    # Convert 'Time' from seconds to hours
    df['Hour'] = (df['Time'] // 3600) % 24
    transactions_per_hour = df.groupby(['Hour', 'Class']).size().reset_index(name='Counts')
    fig_hour = px.bar(
        transactions_per_hour,
        x='Hour',
        y='Counts',
        color='Class',
        labels={
            'Hour': 'Hour of Day',
            'Counts': 'Number of Transactions',
            'Class': 'Transaction Class'
        },
        title="Number of Transactions per Hour",
        color_discrete_map={'0': 'green', '1': 'red'},
        barmode='group'
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("""
    **In-Depth Analysis:**
    - **Temporal Patterns:** The distribution of transactions across different hours indicates peak periods of activity, which can be critical for monitoring and deploying fraud detection mechanisms during high-risk times.
    - **Transaction Density:** The density plots reveal the concentration of transaction amounts, providing insights into typical spending behaviors and potential outliers.
    - **Feature Relationships:** The correlation heatmap highlights interdependencies between features, informing feature selection and engineering processes for model training.
    """)

# Feature Importance Page
if page_selection == "Feature Importance":
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

    selected_model = st.selectbox("Select a Model for Feature Importance:", list(feature_importance_models.keys()))
    model_filename = feature_importance_models[selected_model]
    model = load_model(model_filename)

    if model:
        # Extract feature names
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

            # Display Top 10 Important Features
            st.subheader("📌 Top 10 Most Important Features")
            fig_imp_top = px.bar(
                importance_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 10 Feature Importances for {selected_model}",
                labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                color='Importance',
                color_continuous_scale='YlOrRd'
            )
            st.plotly_chart(fig_imp_top, use_container_width=True)

            # Display Top 10 Least Important Features
            st.subheader("📉 Top 10 Least Important Features")
            fig_imp_bottom = px.bar(
                importance_df.tail(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 10 Least Important Features for {selected_model}",
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
    else:
        st.error("Failed to load the selected model.")

# Model Evaluation Page
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")
    st.markdown("""
    **Comprehensive Model Assessment:**
    This section provides an in-depth evaluation of various machine learning models used for fraud detection. By analyzing key performance metrics and visualizations, executives can understand each model's effectiveness and suitability for deployment.
    """)

    # Dictionary of all available models for evaluation
    all_models = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl',
        'Support Vector Machine': 'svm.pkl',
        'k-Nearest Neighbors': 'knn.pkl'
    }

    classifier = st.selectbox("Select Model for Evaluation:", list(all_models.keys()))
    model_file = all_models[classifier]
    model = load_model(model_file)

    if model:
        # Slider for selecting test set size
        test_size = st.slider('Select Test Set Size (%)', min_value=10, max_value=50, value=30, step=5)
        test_size_fraction = test_size / 100

        # Prepare feature matrix X and target vector y
        X = df.drop(columns=['Class'])
        y = df['Class']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size_fraction,
            random_state=42,
            stratify=y
        )

        # Make predictions
        y_pred = model.predict(X_test)

        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"Confusion Matrix for {classifier}")
        st.pyplot(fig_cm)

        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

        # Performance Metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("🔹 F1-Score", f"{f1:.4f}")
        col2.metric("🔹 Accuracy", f"{accuracy:.4f}")
        col3.metric("🔹 Matthews Corr. Coef.", f"{mcc:.4f}")

        # ROC Curve
        st.subheader("📈 Receiver Operating Characteristic (ROC) Curve")
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
            # Normalize decision function scores
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f"ROC Curve (AUC = {roc_auc:.4f}) for {classifier}",
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            width=700, height=500
        )
        fig_roc.add_shape(
            type='line',
            line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_yaxes(scale=1.05)
        fig_roc.update_xaxes(scale=1.05)
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("""
        **Comprehensive Metrics:**
        - **F1-Score:** Balances precision and recall, providing a single metric that considers both false positives and false negatives.
        - **Accuracy:** Measures the proportion of correct predictions, though it can be misleading in imbalanced datasets.
        - **Matthews Correlation Coefficient (MCC):** Accounts for true and false positives and negatives, offering a balanced measure even in imbalanced scenarios.
        - **ROC-AUC:** Evaluates the trade-off between true positive rate and false positive rate, with higher values indicating better model performance.
        """)

    else:
        st.error("Failed to load the selected model.")

# Real-Time Prediction Page
if page_selection == "Real-Time Prediction":
    st.header("🚀 Real-Time Prediction")
    st.markdown("""
    **Simulate and Predict Fraudulent Transactions:**
    Enter transaction details to receive an immediate prediction on whether the transaction is fraudulent.
    """)

    # Default model for real-time prediction
    default_model_filename = 'random_forest.pkl'
    model_rt = load_model(default_model_filename)

    if model_rt:
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
        if st.button("Predict"):
            input_data = pd.DataFrame({
                **V_features,
                'Time': [Time],
                'Amount': [Amount]
            })

            prediction = model_rt.predict(input_data)[0]
            prediction_proba = model_rt.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.error(f"⚠️ **Fraudulent Transaction Detected!** Probability: {prediction_proba:.2%}")
            else:
                st.success(f"✅ **Valid Transaction.** Probability of Fraud: {prediction_proba:.2%}")
    else:
        st.error("Real-Time Prediction model could not be loaded.")

# Download Report Page
if page_selection == "Download Report":
    st.header("📄 Download Report")
    st.markdown("""
    **Generate and Download Comprehensive PDF Reports:**
    Compile all your analyses, visualizations, and insights into a downloadable PDF report for offline review and sharing with stakeholders.
    """)

    if st.button("Generate Report"):
        with st.spinner("Generating PDF report..."):
            try:
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
                    f"- **Total Transactions:** {len(df):,}\n"
                    f"- **Fraudulent Transactions:** {df['Class'].sum():,} ({(df['Class'].sum() / len(df)) * 100:.4f}%)\n"
                    f"- **Valid Transactions:** {len(df) - df['Class'].sum():,} ({100 - (df['Class'].sum() / len(df)) * 100:.4f}%)\n"
                    "- **Feature Details:** V1 to V28 are PCA-transformed features ensuring anonymity and reduced dimensionality. 'Time' indicates seconds since the first transaction, and 'Amount' represents transaction value in USD.\n"
                    "- **Data Imbalance:** The dataset is highly imbalanced, with fraudulent transactions constituting a small fraction, posing challenges for effective fraud detection."
                )
                pdf.multi_cell(0, 10, data_overview)
                pdf.ln(5)

                # Exploratory Data Analysis
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Exploratory Data Analysis", ln=True)
                pdf.set_font("Arial", '', 12)
                eda_summary = (
                    "- **Feature Correlation:** High inter-correlation among V1 to V28 indicates potential multicollinearity. The 'Amount' feature shows moderate correlation with other features.\n"
                    "- **Transaction Patterns:** Peak transaction times and distribution of transaction amounts provide insights into typical and anomalous behaviors."
                )
                pdf.multi_cell(0, 10, eda_summary)
                pdf.ln(5)

                # Feature Importance
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Feature Importance", ln=True)
                pdf.set_font("Arial", '', 12)
                feature_importance_summary = (
                    "- **High-Impact Features:** Identifying key features that significantly influence fraud detection aids in enhancing data collection and monitoring processes.\n"
                    "- **Low-Impact Features:** Recognizing features with minimal influence can streamline data preprocessing and reduce computational overhead."
                )
                pdf.multi_cell(0, 10, feature_importance_summary)
                pdf.ln(5)

                # Model Evaluation
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Model Evaluation", ln=True)
                pdf.set_font("Arial", '', 12)
                model_evaluation_summary = (
                    "- **Performance Metrics:** Models are evaluated based on F1-Score, Accuracy, Matthews Correlation Coefficient (MCC), and ROC-AUC.\n"
                    "- **ROC Curve Analysis:** The ROC-AUC provides insights into the trade-off between true positive and false positive rates, indicating model effectiveness."
                )
                pdf.multi_cell(0, 10, model_evaluation_summary)
                pdf.ln(5)

                # Save the PDF temporarily
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
if page_selection == "Feedback":
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





