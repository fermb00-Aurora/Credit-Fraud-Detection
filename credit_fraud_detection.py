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
        # Convert 'Time' from seconds to hours for better readability
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

        # Heatmap of Fraud Rate by Hour and Amount Bracket
        st.markdown("### 🔥 Fraud Rate by Hour and Transaction Amount Bracket")
        # Create amount brackets
        df['Amount_Bracket'] = pd.qcut(df['Amount'], q=4, labels=["Low", "Medium", "High", "Very High"])
        fraud_rate_heatmap = df.groupby(['Hour', 'Amount_Bracket'])['Class'].mean().reset_index()
        pivot_heatmap = fraud_rate_heatmap.pivot(index='Hour', columns='Amount_Bracket', values='Class')
        fig_heatmap = px.imshow(
            pivot_heatmap,
            labels=dict(x="Amount Bracket", y="Hour of Day", color="Fraud Rate"),
            x=pivot_heatmap.columns,
            y=pivot_heatmap.index,
            color_continuous_scale='Reds',
            title="Fraud Rate by Hour and Transaction Amount Bracket",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("""
        **In-Depth Analysis:**
        - **Temporal Patterns:** The distribution of transactions across different hours indicates peak periods of activity, which can be critical for monitoring and deploying fraud detection mechanisms during high-risk times.
        - **Transaction Density:** The density plots reveal the concentration of transaction amounts, providing insights into typical spending behaviors and potential outliers.
        - **Average Transaction Amount:** Understanding average transaction amounts per hour can help identify unusual spikes that may signify fraudulent activities.
        - **Fraud Rate Analysis:** Monitoring fraud rates across different hours helps in allocating resources effectively and enhancing surveillance during high-risk periods.
        - **Fraud Rate by Amount Bracket:** Analyzing fraud rates across transaction amount brackets can identify high-risk spending behaviors, enabling targeted fraud prevention strategies.
        """)

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

        # Performance Metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)

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

        from sklearn.metrics import precision_recall_curve, average_precision_score

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
        - **Precision-Recall Curve:** Illustrates the trade-off between precision and recall for different threshold settings. It's particularly useful for imbalanced datasets.
        - **Threshold Analysis:** The F1 Score vs. Threshold plot helps in selecting an optimal decision threshold that balances precision and recall according to business needs.
        """)

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

def generate_report(classifier, y_test, y_pred_test, mcc):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Credit Card Fraud Detection Report", ln=True, align='C')

    # Dataset Info
    total_transactions = len(df)
    fraud_cases = df['Class'].sum()
    valid_cases = total_transactions - fraud_cases
    fraud_percentage = (fraud_cases / total_transactions) * 100

    pdf.cell(200, 10, txt=f"Fraudulent Transactions: {fraud_percentage:.3f}%", ln=True)
    pdf.cell(200, 10, txt=f"Fraud Cases: {fraud_cases} | Valid Cases: {valid_cases}", ln=True)

    # Model Info
    pdf.cell(200, 10, txt=f"Selected Model: {classifier}", ln=True)

    # Metrics
    pdf.cell(200, 10, txt="Classification Report (Test Set):", ln=True)
    pdf.multi_cell(0, 10, classification_report(y_test, y_pred_test))
    pdf.cell(200, 10, txt=f"Matthews Correlation Coefficient (MCC): {mcc:.3f}", ln=True)

    # Save PDF
    report_filename = "fraud_detection_report.pdf"
    pdf.output(report_filename)
    st.success(f"Report generated: {report_filename}")
    with open(report_filename, "rb") as file:
        st.download_button("Download Report", file, file_name=report_filename)

# Button to generate and download the report
if page_selection == "Download Report":
    st.header("📄 Generate PDF Report")
    if st.button("Generate Report"):
        # Pass necessary arguments from the evaluation context
        generate_report(classifier, y_test, y_pred, mcc)



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
