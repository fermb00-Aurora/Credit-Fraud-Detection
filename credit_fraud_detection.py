# Exploratory Data Analysis
if page_selection == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    # Fraud Ratio Pie Chart
    st.subheader("Fraud Ratio Analysis")
    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]
    fraud_ratio = (len(fraud) / len(valid)) * 100

    fig_pie = px.pie(
        names=['Valid Transactions', 'Fraudulent Transactions'],
        values=[len(valid), len(fraud)],
        title="Proportion of Fraudulent vs. Valid Transactions",
        color_discrete_sequence=['#1f77b4', '#ff7f0e']
    )
    st.plotly_chart(fig_pie)

    st.write(f"**Fraudulent transactions represent**: {fraud_ratio:.3f}% of all transactions.")

    # Enhanced Transaction Amount Distribution by Class
    st.subheader("Enhanced Transaction Amount Distribution by Class")

    fig_amount = px.strip(
        df, x='Class', y='Amount', color='Class',
        title='Transaction Amount Distribution (Swarm Plot with Box Plot Overlay)',
        hover_data=['Amount'],
        color_discrete_map={0: '#2ca02c', 1: '#d62728'}
    )
    fig_amount.update_layout(
        yaxis_type="log",
        yaxis_title="Transaction Amount (Log Scale)",
        xaxis_title="Class (0 = Valid, 1 = Fraud)",
        boxmode='overlay'
    )

    # Overlay a Box Plot for additional insights
    fig_box = px.box(
        df, x='Class', y='Amount', color='Class',
        color_discrete_map={0: '#2ca02c', 1: '#d62728'},
        title="Box Plot Overlay for Amount Distribution"
    )
    fig_box.update_traces(marker=dict(opacity=0.5))

    # Combine both plots
    fig_amount.add_traces(fig_box.data)
    st.plotly_chart(fig_amount)

    st.write("""
    The above visualization combines a **swarm plot** and a **box plot** to show the distribution of transaction amounts.
    - The **swarm plot** displays individual transaction points, highlighting outliers and the spread of data.
    - The **box plot** overlay provides statistical insights like the median, quartiles, and outliers.
    - This combined view helps identify typical transaction ranges and unusual activities.
    """)

