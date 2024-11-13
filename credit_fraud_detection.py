import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
import io
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. Data Loading
# ---------------------------
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data.csv')  # Ensure the CSV file is in the correct path
        st.success("Data loaded successfully!")
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data.csv' is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")

data = load_data()

# Proceed only if data is loaded successfully
if data is not None:
    st.title("Credit Card Fraud Detection")

    # ---------------------------
    # 2. Data Overview
    # ---------------------------
    st.header("Data Overview")
    
    st.write("### First 5 Rows of the Dataset")
    st.dataframe(data.head())

    st.write("### Dataset Statistics")
    st.write(data.describe())

    st.write("### Class Distribution")
    class_counts = data['target'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    st.pyplot(fig)

    # ---------------------------
    # 3. Feature and Target Separation
    # ---------------------------
    try:
        X = data.drop('target', axis=1)  # Replace 'target' with your actual target column
        y = data['target']
    except KeyError:
        st.error("The target column 'target' does not exist in the dataset.")
        st.stop()

    # ---------------------------
    # 4. Train-Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write(f"### Training Set Size: {X_train.shape[0]} samples")
    st.write(f"### Testing Set Size: {X_test.shape[0]} samples")

    # ---------------------------
    # 5. Model Loading
    # ---------------------------
    @st.cache_resource
    def load_model():
        model_path = 'fraud_model.joblib'
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                st.success("Model loaded successfully!")
                return model
            except Exception as e:
                st.error(f"An error occurred while loading the model: {e}")
                return None
        else:
            st.error(f"Model file '{model_path}' not found. Please ensure the model is trained and saved as '{model_path}'.")
            return None

    model = load_model()

    # Proceed only if the model is loaded successfully
    if model is not None:
        # ---------------------------
        # 6. Feature Importance
        # ---------------------------
        st.header("Feature Importance")

        def plot_feature_importance(model, feature_names):
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis', ax=ax)
                    ax.set_title('Feature Importance')
                    ax.set_xlabel('Importance Score')
                    ax.set_ylabel('Features')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.write("The model does not support feature importances.")
            except Exception as e:
                st.error(f"An error occurred while plotting feature importance: {e}")

        plot_feature_importance(model, X_train.columns)

        # ---------------------------
        # 7. Model Evaluation
        # ---------------------------
        st.header("Model Evaluation")

        def evaluate_model(model, X_test, y_test):
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                                     columns=['Predicted Negative', 'Predicted Positive'])
                fig, ax = plt.subplots()
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                ax.set_title('Confusion Matrix')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("Classification Report")
                cr = classification_report(y_test, y_pred, output_dict=True)
                cr_df = pd.DataFrame(cr).transpose()
                st.dataframe(cr_df)
                
                if y_pred_proba is not None:
                    st.subheader("ROC Curve")
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax.legend(loc='lower right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add ROC AUC score
                    st.write(f"**ROC AUC Score:** {roc_auc:.2f}")
                else:
                    st.write("ROC Curve cannot be displayed because the model does not support probability estimates.")
            except Exception as e:
                st.error(f"An error occurred during model evaluation: {e}")

        evaluate_model(model, X_test, y_test)

        # ---------------------------
        # 8. Fraud Detection Simulator
        # ---------------------------
        st.header("Fraud Detection Simulator")

        def fraud_simulator(model, feature_names):
            try:
                st.write("### Input Feature Values")
                user_input = {}
                for feature in feature_names:
                    if pd.api.types.is_numeric_dtype(X[feature]):
                        user_input[feature] = st.number_input(
                            f"{feature}",
                            value=float(X_train[feature].mean()),
                            format="%.5f"
                        )
                    else:
                        # Handle categorical features if any
                        unique_vals = X[feature].unique().tolist()
                        user_input[feature] = st.selectbox(f"{feature}", unique_vals)
                
                if st.button("Predict Fraud"):
                    input_df = pd.DataFrame([user_input])
                    prediction = model.predict(input_df)[0]
                    prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
                    
                    st.write("### Prediction Results")
                    st.write(f"**Prediction:** {'Fraud' if prediction == 1 else 'Not Fraud'}")
                    if prediction_proba is not None:
                        st.write(f"**Probability of Fraud:** {prediction_proba:.2f}")
                    else:
                        st.write("Probability estimation not available for this model.")
            except Exception as e:
                st.error(f"An error occurred in the simulator: {e}")

        fraud_simulator(model, X_train.columns)

        # ---------------------------
        # 9. Report Generator
        # ---------------------------
        st.header("Report Generator")

        def generate_report(model, X_test, y_test):
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                                     columns=['Predicted Negative', 'Predicted Positive'])

                # Classification Report
                cr = classification_report(y_test, y_pred, output_dict=True)
                cr_df = pd.DataFrame(cr).transpose()

                # ROC Curve Data
                if y_pred_proba is not None:
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    roc_df = pd.DataFrame({
                        'False Positive Rate': fpr,
                        'True Positive Rate': tpr,
                        'Thresholds': thresholds
                    })
                else:
                    roc_df = pd.DataFrame()

                # Create a BytesIO buffer to save the report
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    cm_df.to_excel(writer, sheet_name='Confusion Matrix')
                    cr_df.to_excel(writer, sheet_name='Classification Report')
                    if not roc_df.empty:
                        roc_df.to_excel(writer, sheet_name='ROC Curve Data')
                buffer.seek(0)

                st.download_button(
                    label="Download Evaluation Report",
                    data=buffer,
                    file_name="model_evaluation_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"An error occurred while generating the report: {e}")

        generate_report(model, X_test, y_test)

        # ---------------------------
        # 10. Feedback Section
        # ---------------------------
        st.header("Feedback")

        def feedback_section():
            try:
                st.write("### Provide Your Feedback")
                with st.form("feedback_form"):
                    transaction_id = st.text_input("Transaction ID")
                    feedback = st.radio("Was the prediction correct?", ("Yes", "No"))
                    comments = st.text_area("Additional Comments (optional)")
                    submitted = st.form_submit_button("Submit Feedback")
                
                if submitted:
                    if not transaction_id:
                        st.error("Please provide a Transaction ID.")
                    else:
                        # Load existing feedback or create a new DataFrame
                        feedback_file = 'feedback.csv'
                        if os.path.exists(feedback_file):
                            try:
                                feedback_df = pd.read_csv(feedback_file)
                            except Exception as e:
                                st.error(f"An error occurred while loading existing feedback: {e}")
                                return
                        else:
                            feedback_df = pd.DataFrame(columns=['Transaction ID', 'Feedback', 'Comments'])
                        
                        # Append new feedback
                        new_feedback = {
                            'Transaction ID': transaction_id,
                            'Feedback': feedback,
                            'Comments': comments
                        }
                        feedback_df = feedback_df.append(new_feedback, ignore_index=True)
                        
                        # Save back to CSV
                        try:
                            feedback_df.to_csv(feedback_file, index=False)
                            st.success("Thank you for your feedback!")
                        except Exception as e:
                            st.error(f"An error occurred while saving feedback: {e}")
            except Exception as e:
                st.error(f"An error occurred in the feedback section: {e}")

        feedback_section()
