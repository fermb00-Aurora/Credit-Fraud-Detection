# Download Report Page
elif page_selection == "Download Report":
    st.header("📄 Download Report")
    st.markdown("""
    **Generate and Download a Comprehensive PDF Report:**
    Compile your analysis and model evaluation results into a downloadable PDF report for offline review and sharing with stakeholders.
    """)

    # Check if evaluation data is available in session state
    if 'model_evaluation' not in st.session_state or not st.session_state['model_evaluation']:
        st.error("Please perform a model evaluation before generating the report.")
    else:
        # Button to generate report
        if st.button("Generate Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Retrieve evaluation data from session state
                    eval_data = st.session_state['model_evaluation']
                    model = eval_data['model']
                    X_test = eval_data['X_test']
                    y_test = eval_data['y_test']
                    y_pred = eval_data['y_pred']
                    classifier = eval_data['classifier']
                    metrics = eval_data['metrics']
                    test_size = eval_data['test_size']
                    y_proba = eval_data.get('y_proba', None)
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

                    # Model Evaluation Summary
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Model Evaluation Summary", ln=True)
                    pdf.set_font("Arial", '', 12)
                    model_evaluation_summary = (
                        f"- **Model:** {classifier}\n"
                        f"- **Test Set Size:** {test_size}%\n"
                        f"- **Total Test Samples:** {len(y_test)}\n"
                        f"- **Fraudulent Transactions in Test Set:** {sum(y_test)} ({(sum(y_test) / len(y_test)) * 100:.4f}%)\n"
                        f"- **Valid Transactions in Test Set:** {len(y_test) - sum(y_test)} ({100 - (sum(y_test) / len(y_test)) * 100:.4f}%)\n"
                        f"- **Accuracy:** {metrics['accuracy']:.4f}\n"
                        f"- **F1-Score:** {metrics['f1_score']:.4f}\n"
                        f"- **Precision:** {metrics['precision']:.4f}\n"
                        f"- **Recall:** {metrics['recall']:.4f}\n"
                        f"- **F2-Score:** {metrics['f2_score']:.4f}\n"
                        f"- **Matthews Corr. Coef.:** {metrics['mcc']:.4f}\n"
                        f"- **ROC-AUC:** {roc_auc if roc_auc != 'N/A' else 'N/A'}\n"
                    )
                    pdf.multi_cell(0, 10, model_evaluation_summary)
                    pdf.ln(5)

                    # Confusion Matrix Visualization
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
                    pdf.ln(100)
                    os.remove(cm_image_path)

                    # ROC Curve Visualization (if applicable)
                    if roc_auc != "N/A" and y_proba is not None:
                        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        roc_auc_val = auc(fpr, tpr)
                        ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
                        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='grey')
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
                        pdf.ln(100)
                        os.remove(roc_image_path)

                    # Feature Importance (if applicable)
                    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                        # Compute feature importances
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                        elif hasattr(model, 'coef_'):
                            importances = np.abs(model.coef_[0])
                        else:
                            importances = None

                        if importances is not None:
                            features = X_test.columns
                            importance_df = pd.DataFrame({
                                'Feature': features,
                                'Importance': importances
                            }).sort_values(by='Importance', ascending=False)

                            # Plot Top 10 Features
                            top_n = 10
                            fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                            sns.barplot(
                                x='Importance', y='Feature',
                                data=importance_df.head(top_n),
                                palette='YlOrRd', ax=ax_imp
                            )
                            ax_imp.set_title(f"Top {top_n} Feature Importances for {classifier}")
                            plt.tight_layout()
                            imp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                            plt.savefig(imp_image_path, dpi=300)
                            plt.close(fig_imp)

                            # Add Feature Importance to PDF
                            pdf.add_page()
                            pdf.set_font("Arial", 'B', 12)
                            pdf.cell(0, 10, "Feature Importance", ln=True, align='C')
                            pdf.image(imp_image_path, x=30, y=30, w=150)
                            pdf.ln(100)
                            os.remove(imp_image_path)

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
