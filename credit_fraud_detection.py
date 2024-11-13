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
    precision_score,
    recall_score,
    fbeta_score
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

# Initialize session state for model evaluation results
if 'model_evaluation' not in st.session_state:
    st.session_state['model_evaluation'] = {}

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
    # Sample the data for performance
    sampled_df = df.sample(n=5000, random_state=42) if len(df) > 5000 else df
    fig_time = px.scatter(
        sampled_df,
        x='Time',
        y='Amount',
        color='Class',
        labels={
            'Time': 'Time',
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

    # Additional Insightful Visualizations for Business
    st.subheader("📊 Additional Business Insights")

    # Average Transaction Amount per Hour
    st.markdown("### 📈 Average Transaction Amount per Hour")
    avg_amount_hour = df.groupby(['Hour', 'Class'])['Amount'].mean().reset_index()
    fig_avg_amount = px.line(
        avg_amount_hour,
        x='Hour',
        y='Amount',
        color='Class',
        labels={
            'Hour': 'Hour of Day',
            'Amount': 'Average Transaction Amount ($)',
            'Class': 'Transaction Class'
        },
        title="Average Transaction Amount per Hour",
        color_discrete_map={'0': 'green', '1': 'red'},
        markers=True
    )
    st.plotly_chart(fig_avg_amount, use_container_width=True)

    # Fraud Rate by Hour
    st.markdown("### 📉 Fraud Rate by Hour")
    fraud_rate_hour = df.groupby('Hour')['Class'].mean().reset_index()
    fig_fraud_rate = px.bar(
        fraud_rate_hour,
        x='Hour',
        y='Class',
        labels={
            'Hour': 'Hour of Day',
            'Class': 'Fraud Rate',
        },
        title="Fraud Rate by Hour of Day",
        color='Class',
        color_continuous_scale='Reds',
        range_y=[0, fraud_rate_hour['Class'].max() + 0.01]
    )
    st.plotly_chart(fig_fraud_rate, use_container_width=True)

    st.markdown("""
    **In-Depth Analysis:**
    - **Temporal Patterns:** The distribution of transactions across different hours indicates peak periods of activity, which can be critical for monitoring and deploying fraud detection mechanisms during high-risk times.
    - **Transaction Density:** The density plots reveal the concentration of transaction amounts, providing insights into typical spending behaviors and potential outliers.
    - **Average Transaction Amount:** Understanding average transaction amounts per hour can help identify unusual spikes that may signify fraudulent activities.
    - **Fraud Rate Analysis:** Monitoring fraud rates across different hours helps in allocating resources effectively and enhancing surveillance during high-risk periods.
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
    else:
        st.error("Failed to load the selected model.")

# Model Evaluation Page
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")
    st.markdown("""
    **Comprehensive Model Assessment:**
    This section provides an in-depth evaluation of various machine learning models used for fraud detection. By analyzing key performance metrics and visualizations, executives can understand each model's effectiveness and suitability for deployment.
    """)

    # Dictionary of all available models for evaluation (excluding SVM)
    all_models = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl',
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
            'f2_score': fbeta_score(y_test, y_pred, beta=2)
        }

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
        # Enhance the classification report with better formatting
        st.dataframe(report_df.style.applymap(lambda x: 'background-color: #FDEBD0' if isinstance(x, float) and x < 0.5 else '').background_gradient(cmap='coolwarm'))

        # Performance Metrics
        metrics = st.session_state['model_evaluation']['metrics']
        col1, col2, col3 = st.columns(3)
        col1.metric("🔹 Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("🔹 F1-Score", f"{metrics['f1_score']:.4f}")
        col3.metric("🔹 Matthews Corr. Coef.", f"{metrics['mcc']:.4f}")
        col4, col5 = st.columns(2)
        col4.metric("🔹 Precision", f"{metrics['precision']:.4f}")
        col5.metric("🔹 Recall", f"{metrics['recall']:.4f}")
        st.metric("🔹 F2-Score", f"{metrics['f2_score']:.4f}")

        # ROC Curve - Only for models that support it
        st.subheader("📈 Receiver Operating Characteristic (ROC) Curve")
        roc_available = False
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_available = True
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            # Normalize decision function scores
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            roc_available = True

        if roc_available:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            st.session_state['model_evaluation']['roc_auc'] = roc_auc

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
        else:
            st.info("ROC Curve is not available for the selected model.")

        # Additional Evaluation Metrics
        st.subheader("📈 Additional Evaluation Metrics")
        st.markdown("""
        **Precision:** Measures the proportion of positive identifications that were actually correct. High precision indicates a low false positive rate.

        **Recall (Sensitivity):** Measures the proportion of actual positives that were identified correctly. High recall indicates a low false negative rate.

        **F2-Score:** Places more emphasis on recall than precision, useful when false negatives are more critical than false positives.
        """)

        st.markdown("""
        **Comprehensive Metrics:**
        - **Accuracy:** Proportion of correct predictions.
        - **F1-Score:** Harmonic mean of precision and recall.
        - **Matthews Correlation Coefficient (MCC):** Balanced measure even if classes are of very different sizes.
        - **Precision:** Proportion of positive identifications that were actually correct.
        - **Recall:** Proportion of actual positives that were identified correctly.
        - **F2-Score:** Focuses more on recall than precision.
        """)

        # Personalized and Cool Report Enhancements
        st.subheader("📄 Personalized Model Evaluation Summary")
        roc_auc_text = f"{roc_auc:.4f}" if roc_available else "N/A"
        st.markdown(f"""
        **Model:** {classifier}  
        **Test Set Size:** {test_size}%  
        **Total Test Samples:** {len(y_test)}  
        **Fraudulent Transactions in Test Set:** {y_test.sum()} ({(y_test.sum() / len(y_test)) * 100:.4f}%)  
        **Valid Transactions in Test Set:** {len(y_test) - y_test.sum()} ({100 - (y_test.sum() / len(y_test)) * 100:.4f}%)  
        
        **Performance Overview:**
        - **Accuracy:** {metrics['accuracy']:.4f}
        - **F1-Score:** {metrics['f1_score']:.4f}
        - **Matthews Corr. Coef.:** {metrics['mcc']:.4f}
        - **Precision:** {metrics['precision']:.4f}
        - **Recall:** {metrics['recall']:.4f}
        - **F2-Score:** {metrics['f2_score']:.4f}
        - **ROC-AUC:** {roc_auc_text}
        """)

        # Dynamic Insights with Relevant Number Data
        st.markdown("""
        **Dynamic Insights:**
        - The **Accuracy** of {model_accuracy:.2f}% indicates that the model correctly predicted {correct_preds} out of {total_preds} transactions.
        - With an **F1-Score** of {model_f1:.4f}, the model balances precision and recall effectively.
        - The **Matthews Correlation Coefficient (MCC)** of {model_mcc:.4f} suggests a strong correlation between the observed and predicted classifications.
        - **Precision** and **Recall** scores of {model_precision:.4f} and {model_recall:.4f} respectively highlight the model's capability to minimize false positives and false negatives.
        - An **F2-Score** of {model_f2:.4f} emphasizes the model's focus on recall, ensuring that most fraudulent transactions are detected.
        - The **ROC-AUC** of {model_roc_auc} demonstrates the model's ability to distinguish between fraudulent and valid transactions.
        """.format(
            model_accuracy=metrics['accuracy'] * 100,
            correct_preds=int(metrics['accuracy'] * len(y_test)),
            total_preds=len(y_test),
            model_f1=metrics['f1_score'],
            model_mcc=metrics['mcc'],
            model_precision=metrics['precision'],
            model_recall=metrics['recall'],
            model_f2=metrics['f2_score'],
            model_roc_auc=metrics['roc_auc'] if roc_available else "N/A"
        ))

    else:
        st.error("Failed to load the selected model.")

# Simulator Page
if page_selection == "Simulator":
    st.header("🚀 Simulator")
    st.markdown("""
    **Simulate and Predict Fraudulent Transactions:**
    Enter transaction details to receive an immediate prediction on whether the transaction is fraudulent.
    """)

    # Default model for simulation
    default_model_filename = 'random_forest.pkl'
    model_sim = load_model(default_model_filename)

    if model_sim:
        # Input transaction details
        st.subheader("🔍 Enter Transaction Details")
        col1, col2 = st.columns(2)

        with col1:
            V_features = {}
            for i in range(1, 29):
                V_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.5f", key=f'V{i}')

        with col2:
            Time = st.number_input('Time', min_value=0, value=0, step=1, key='Time')
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
    else:
        st.error("Simulator model could not be loaded.")

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
                # Check if model evaluation data is available
                if 'y_test' not in st.session_state['model_evaluation'] or \
                   'y_pred' not in st.session_state['model_evaluation'] or \
                   'classifier' not in st.session_state['model_evaluation']:
                    st.error("Please perform a model evaluation before generating the report.")
                else:
                    # Retrieve evaluation data
                    y_test = st.session_state['model_evaluation']['y_test']
                    y_pred = st.session_state['model_evaluation']['y_pred']
                    classifier = st.session_state['model_evaluation']['classifier']
                    metrics = st.session_state['model_evaluation']['metrics']
                    roc_auc = st.session_state['model_evaluation'].get('roc_auc', "N/A")

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
                        "- **Feature Details:** V1 to V28 are PCA-transformed features ensuring anonymity and reduced dimensionality. 'Time' indicates time since the first transaction, and 'Amount' represents transaction value in USD.\n"
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
                        "- **Transaction Patterns:** Peak transaction times and distribution of transaction amounts provide insights into typical and anomalous behaviors.\n"
                        "- **Average Transaction Amount:** Understanding average transaction amounts per hour can help identify unusual spikes that may signify fraudulent activities.\n"
                        "- **Fraud Rate Analysis:** Monitoring fraud rates across different hours helps in allocating resources effectively and enhancing surveillance during high-risk periods."
                    )
                    pdf.multi_cell(0, 10, eda_summary)
                    pdf.ln(5)

                    # Feature Importance
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Feature Importance", ln=True)
                    pdf.set_font("Arial", '', 12)
                    feature_importance_summary = (
                        "- **High-Impact Features:** Identifying key features that significantly influence fraud detection aids in enhancing data collection and monitoring processes.\n"
                        "- **Low-Impact Features:** Recognizing features with minimal influence can streamline data preprocessing and reduce computational overhead without compromising model performance.\n"
                        "- **Model Selection:** Different models may prioritize different features, offering diverse perspectives on what drives fraudulent activities."
                    )
                    pdf.multi_cell(0, 10, feature_importance_summary)
                    pdf.ln(5)

                    # Model Evaluation
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Model Evaluation", ln=True)
                    pdf.set_font("Arial", '', 12)
                    model_evaluation_summary = (
                        "- **Performance Metrics:** Models are evaluated based on Accuracy, F1-Score, Matthews Correlation Coefficient (MCC), Precision, Recall, and F2-Score.\n"
                        "- **ROC Curve Analysis:** The ROC-AUC provides insights into the trade-off between true positive and false positive rates, indicating model effectiveness.\n"
                        "- **Personalized Metrics:** Detailed performance metrics tailored to the specific model and dataset configuration offer a clear understanding of model strengths and weaknesses."
                    )
                    pdf.multi_cell(0, 10, model_evaluation_summary)
                    pdf.ln(5)

                    # Adding Visualizations to PDF
                    # Correlation Heatmap
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    sns.heatmap(df.corr(), cmap='YlOrBr', ax=ax_corr)
                    plt.title('Correlation Heatmap of Features')
                    plt.tight_layout()
                    corr_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    plt.savefig(corr_image.name, dpi=300)
                    plt.close(fig_corr)
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Correlation Heatmap of Features", ln=True, align='C')
                    pdf.image(corr_image.name, x=10, y=20, w=190)
                    os.unlink(corr_image.name)  # Delete the temporary file
                    pdf.ln(100)  # Adjust as per image size

                    # Confusion Matrix
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlOrBr',
                                xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'], ax=ax_cm)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title(f"Confusion Matrix for {classifier}")
                    cm_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    plt.savefig(cm_image.name, dpi=300)
                    plt.close(fig_cm)
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, f"Confusion Matrix for {classifier}", ln=True, align='C')
                    pdf.image(cm_image.name, x=10, y=20, w=190)
                    os.unlink(cm_image.name)  # Delete the temporary file
                    pdf.ln(100)  # Adjust as per image size

                    # ROC Curve (if applicable)
                    if roc_auc != "N/A":
                        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test)[:, 1]
                        else:
                            y_proba = model.decision_function(X_test)
                            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        sns.lineplot(x=fpr, y=tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', ax=ax_roc)
                        sns.lineplot([0, 1], [0, 1], linestyle='--', color='grey', ax=ax_roc)
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title(f"ROC Curve for {classifier}")
                        ax_roc.legend(loc='lower right')
                        plt.tight_layout()
                        roc_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        plt.savefig(roc_image.name, dpi=300)
                        plt.close(fig_roc)
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, f"ROC Curve for {classifier}", ln=True, align='C')
                        pdf.image(roc_image.name, x=10, y=20, w=190)
                        os.unlink(roc_image.name)  # Delete the temporary file
                        pdf.ln(100)  # Adjust as per image size

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




