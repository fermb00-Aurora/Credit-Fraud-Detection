# Feature Selection
if page_selection == "Feature Selection":
    st.header("🔍 Feature Selection")

    # Dropdown to select the model for feature importance
    model_choices = {
        'Logistic Regression': 'logistic_regression.pkl',
        'k-Nearest Neighbors (kNN)': 'knn.pkl',
        'Random Forest': 'random_forest.pkl',
        'Extra Trees': 'extra_trees.pkl'
    }
    feature_model = st.sidebar.selectbox("Select model for feature importance:", list(model_choices.keys()))
    model_path = os.path.join(os.path.dirname(__file__), model_choices[feature_model])
    model = joblib.load(model_path)

    # Handle feature importance differently based on the model type
    if feature_model in ['Random Forest', 'Extra Trees']:
        feature_importances = model.feature_importances_
        importance_method = "Feature Importances"
    elif feature_model == 'Logistic Regression':
        if hasattr(model, "coef_"):
            feature_importances = np.abs(model.coef_[0])
            importance_method = "Model Coefficients (Absolute Value)"
        else:
            st.warning("Logistic Regression model does not have coefficients.")
            feature_importances = np.zeros(len(df.drop(columns=['Class']).columns))
    elif feature_model == 'k-Nearest Neighbors (kNN)':
        st.warning("kNN does not support feature importance analysis.")
        feature_importances = np.zeros(len(df.drop(columns=['Class']).columns))
        importance_method = "Not Available"

    # Display feature importance information
    if feature_importances is not None and feature_importances.any():
        features = df.drop(columns=['Class']).columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        st.subheader(f"Top Features Based on {importance_method}")
        st.write("The following features have the highest impact on the model's decision-making:")

        # Top 3 Most Important Features
        for i in range(3):
            st.write(f"🏅 **{i+1}. {importance_df.iloc[i]['Feature']}** - Importance: **{importance_df.iloc[i]['Importance']:.4f}**")

        # Top 3 Least Important Features
        for i in range(1, 4):
            st.write(f"🥉 **{4-i}. {importance_df.iloc[-i]['Feature']}** - Importance: **{importance_df.iloc[-i]['Importance']:.4f}**")

        # Feature Importance Bar Plot
        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title=f"Feature Importance ({feature_model})")
        st.plotly_chart(fig_imp)
    else:
        st.info(f"Feature importance is not available for the selected model '{feature_model}'.")



