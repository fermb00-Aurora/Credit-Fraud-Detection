import timeit
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
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit App Configuration
st.set_page_config(page_title='💳 Credit Card Fraud Detection Dashboard',
                   layout='wide',
                   initial_sidebar_state='expanded')

# Streamlit App Title and Sidebar
st.title('💳 Credit Card Fraud Detection Dashboard')
st.sidebar.header("Menu")
page_selection = st.sidebar.radio("Navigate:", [
    "Introduction", "Data Overview", "Exploratory Data Analysis",
    "Feature Importance", "Model Evaluation", "Real-Time Prediction",
    "Download Report", "Feedback"
])

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# Introduction
if page_selection == "Introduction":
    st.header("📘 Executive Summary")
    st.markdown("""
    **Objective:**  
    The primary goal of this dashboard is to empower financial executives with tools to detect and analyze fraudulent credit card transactions. By leveraging advanced machine learning models, this platform provides actionable insights to mitigate financial losses and enhance security measures.

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

# Data Overview
if page_selection == "Data Overview":
    st.header("🔍 Data Overview")
    st.subheader("Dataset Summary")
    st.dataframe(df.describe().T.style.background_gradient(cmap='YlGnBu'))

    st.markdown("---")

    # Key Metrics
    total_transactions = len(df)
    fraud_transactions = len(df[df['Class'] == 1])
    valid_transactions = total_transactions - fraud_transactions
    fraud_percentage = (fraud_transactions / total_transactions) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraudulent Transactions", f"{fraud_transactions:,}", f"{fraud_percentage:.4f}%")
    col3.metric("Valid Transactions", f"{valid_transactions:,}", f"{100 - fraud_percentage:.4f}%")

    st.markdown("---")

    # Visualization 1: Transaction Class Distribution
    st.subheader("💼 Transaction Class Distribution")
    fig_pie = px.pie(
        names=['Valid', 'Fraudulent'],
        values=[valid_transactions, fraud_transactions],
        hole=0.4,
        color=['Valid', 'Fraudulent'],
        color_discrete_map={'Valid': 'green', 'Fraudulent': 'red'},
        labels={'names': 'Transaction Type', 'values': 'Count'},
        title="Proportion of Valid vs. Fraudulent Transactions"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Visualization 2: Transaction Amount Distribution
    st.subheader("💸 Transaction Amount Distribution")
    fig_box = px.box(
        df, x='Class', y='Amount',
        labels={'Class': 'Transaction Class', 'Amount': 'Transaction Amount ($)'},
        title="Box Plot of Transaction Amounts by Class",
        color='Class',
        color_discrete_map={'0': 'green', '1': 'red'}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
    **Insights:**
    - **Imbalanced Dataset:** The dataset is highly imbalanced with fraudulent transactions constituting a mere 0.172% of the total, highlighting the challenge in detecting fraud.
    - **Transaction Amounts:** There is a noticeable difference in the distribution of transaction amounts between valid and fraudulent transactions, which can be leveraged by machine learning models for better accuracy.
    """)

# Exploratory Data Analysis
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
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Key Observations:**
    - **High Correlation Among V* Features:** Features V1 to V28, which are the result of a PCA transformation, exhibit high inter-correlation, indicating potential multicollinearity.
    - **Amount Feature:** The 'Amount' feature shows some correlation with other features, suggesting its significance in distinguishing between transaction classes.
    """)

    # Visualization 2: Time vs. Transaction Amount
    st.subheader("⏰ Transaction Amount Over Time")
    fig_time = px.scatter(
        df, x='Time', y='Amount',
        color='Class',
        labels={'Time': 'Seconds Elapsed Since First Transaction', 'Amount': 'Transaction Amount ($)', 'Class': 'Transaction Class'},
        title="Transaction Amounts Over Time",
        opacity=0.5,
        color_discrete_map={'0': 'green', '1': 'red'}
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Visualization 3: Density Plot of Transaction Amounts
    st.subheader("📈 Density Plot of Transaction Amounts")
    fig_density = px.histogram(
        df, x='Amount', color='Class',
        nbins=50, histnorm='density',
        title="Density of Transaction Amounts by Class",
        labels={'Amount': 'Transaction Amount ($)', 'density': 'Density'},
        color_discrete_map={'0': 'green', '1': 'red'},
        opacity=0.6
    )
    st.plotly_chart(fig_density, use_container_width=True)

    # Visualization 4: Count of Transactions Over Time
    st.subheader("📅 Transactions Over Time")
    # Assuming 'Time' is in seconds, convert to hours for better readability
    df['Hour'] = (df['Time'] // 3600) % 24
    transactions_per_hour = df.groupby(['Hour', 'Class']).size().reset_index(name='Counts')
    fig_hour = px.bar(
        transactions_per_hour, x='Hour', y='Counts', color='Class',
        labels={'Hour': 'Hour of Day', 'Counts': 'Number of Transactions', 'Class': 'Transaction Class'},
        title="Number of Transactions per Hour",
        color_discrete_map={'0': 'green', '1': 'red'},
        barmode='group'
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("""
    **In-Depth Analysis:**
    - **Temporal Patterns:** The distribution of transactions across different hours indicates peak periods of activity, which can be critical for monitoring and deploying fraud detection mechanisms during high-risk times.
    - **Transaction Density:** The density plots reveal the concentration of transaction amounts, providing insights into typical spending behaviors and potential outliers.
    """)

# Feature Importance
if page_selection == "Feature Importance":
    st.header("🔍 Feature Importance Analysis")
    st.markdown("""
        **Understanding Feature Impact:**
        In the realm of credit card fraud detection, identifying which features significantly influence model predictions is paramount. This section delves into the importance of various features across different machine learning models, providing clarity on what drives fraud detection decisions.

        **Models Analyzed:**
        - **Random Forest:** Utilizes ensemble learning to provide feature importance based on the mean decrease in impurity.
        - **Extra Trees:** Similar to Random Forest but with more randomness, offering robust feature importance metrics.
        - **Logistic Regression:** Assesses feature importance through the magnitude of coefficients, indicating the strength and direction of influence.
    """)

    # Models supporting feature importance
    feature_importance_models = {
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl',
        'Logistic Regression': 'logistic_regression.pkl'
    }

    selected_model = st.selectbox("Select a Model for Feature Importance:", list(feature_importance_models.keys()))
    model_filename = feature_importance_models[selected_model]
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    try:
        model = joblib.load(model_path)
        features = df.drop(columns=['Class']).columns

        if selected_model in ['Random Forest', 'Extra Trees']:
            importances = model.feature_importances_
        elif selected_model == 'Logistic Regression':
            importances = np.abs(model.coef_[0])

        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

        st.subheader("📌 Top 10 Most Important Features")
        fig_imp_top = px.bar(
            importance_df.head(10),
            x='Importance', y='Feature',
            orientation='h',
            title=f"Top 10 Feature Importances for {selected_model}",
            labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
            color='Importance',
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig_imp_top, use_container_width=True)

        st.subheader("📉 Top 10 Least Important Features")
        fig_imp_bottom = px.bar(
            importance_df.tail(10),
            x='Importance', y='Feature',
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

    except Exception as e:
        st.error(f"Error loading model: {e}")

# Model Evaluation
if page_selection == "Model Evaluation":
    st.header("🧠 Model Evaluation")

    st.markdown("""
        **Comprehensive Model Assessment:**
        This section provides an in-depth evaluation of various machine learning models used for fraud detection. By analyzing key performance metrics and visualizations, executives can understand each model's effectiveness and suitability for deployment.
    """)

    # Models supporting feature importance
    all_models = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl',
        'Support Vector Machine': 'svm.pkl',
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

    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42, stratify=y)

    # Make predictions
    y_pred = model.predict(X_test)

    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {classifier}")
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
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Normalize

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    st.subheader("📈 Receiver Operating Characteristic (ROC) Curve")
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

# Real-Time Prediction
if page_selection == "Real-Time Prediction":
    st.header("🚀 Real-Time Prediction")
    st.markdown("""
        **Simulate and Predict Fraudulent Transactions:**
        Enter transaction details to receive an immediate prediction on whether the transaction is fraudulent.
    """)

    # Load the model
    model_file_rt = 'random_forest.pkl'  # Default model
    model_path_rt = os.path.join(os.path.dirname(__file__), model_file_rt)

    try:
        model_rt = joblib.load(model_path_rt)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Input features (assuming features V1 to V28, Time, Amount)
    st.subheader("🔍 Enter Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        V_features = {}
        for i in range(1, 29):
            V_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.5f")

    with col2:
        Time = st.number_input('Time (seconds since first transaction)', min_value=0, value=0, step=1)
        Amount = st.number_input('Transaction Amount ($)', min_value=0.0, value=0.0, format="%.2f")

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

# Download Report
if page_selection == "Download Report":
    st.header("📄 Download Report")
    st.markdown("""
        **Generate and Download Comprehensive PDF Reports:**
        Compile all your analyses, visualizations, and insights into a downloadable PDF report for offline review and sharing with stakeholders.
    """)

    if st.button("Generate Report"):
        with st.spinner("Generating PDF report..."):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Credit Card Fraud Detection Report", ln=True, align='C')

                # Executive Summary
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Executive Summary", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, """
                This report provides a comprehensive analysis of credit card transactions to identify and detect fraudulent activities. It encompasses data overview, exploratory data analysis, feature importance, model evaluations, and actionable insights to support strategic decision-making and risk management.
                """)

                # Data Overview
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Data Overview", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, f"""
                - **Total Transactions:** {len(df):,}
                - **Fraudulent Transactions:** {fraud_transactions:,} ({fraud_percentage:.4f}%)
                - **Valid Transactions:** {valid_transactions:,} ({100 - fraud_percentage:.4f}%)
                """)

                # Save PDF
                report_path = "fraud_detection_report.pdf"
                pdf.output(report_path)

                with open(report_path, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download PDF Report",
                        data=file,
                        file_name=report_path,
                        mime="application/pdf"
                    )
                if btn:
                    st.success("Report downloaded successfully!")

                # Clean up
                os.remove(report_path)

            except Exception as e:
                st.error(f"Error generating report: {e}")

# Feedback Section
if page_selection == "Feedback":
    st.header("💬 Feedback")
    st.markdown("""
        **We Value Your Feedback:**
        Help us improve the Credit Card Fraud Detection Dashboard by providing your valuable feedback and suggestions.
    """)
    feedback = st.text_area("Provide your feedback here:")
    if st.button("Submit Feedback"):
        if feedback.strip() == "":
            st.warning("Please enter your feedback before submitting.")
        else:
            # Here you can implement functionality to save feedback, e.g., to a database or send via email
            # For demonstration, we'll just acknowledge the submission
            st.success("Thank you for your feedback!")




