st.set_page_config(page_title="CPG Decision Agent")

st.title("Smart CPG Decision Support Agent")

agent = CPGDecisionAgent()

query = st.text_input("Ask a business question:")

if st.button("Analyze"):
    if query:

        result = agent.run(query)

        # ---------------- TREND ----------------
        if result["type"] == "trend":

            st.subheader("📈 Business Summary")

            summary = result["summary"]
            st.metric("Total Revenue", summary["total_revenue"])
            st.metric("Total Units Sold", summary["total_units"])
            st.metric("Top Category", summary["top_category"])

            st.subheader("📊 Monthly Revenue Trends")

            monthly = result["monthly"]

            chart_data = monthly.pivot(
                index="month",
                columns="category",
                values="revenue"
            )

            st.line_chart(chart_data)

        # ---------------- ANOMALY ----------------
        elif result["type"] == "anomaly":

            st.subheader("🚨 Detected Anomalies")
            st.metric("Lower Bound", summary["lower_bound"])
            st.metric("Upper Bound", summary["upper_bound"])
            st.metric("Total anomalies found are", summary["total anomalies found"])
            st.dataframe(summary["anomalies_below_lower_bound"])
            st.dataframe(summary["anomalies_above_upper_bound"])

        # ---------------- PRICE ----------------
        elif result["type"] == "price":

            impact = result["impact"]

            st.subheader("💰 Price Impact Simulation")
            st.metric("Baseline Revenue", impact["baseline_revenue"])
            st.metric("Projected Revenue", impact["projected_revenue"])
            st.metric("Revenue Change %", f"{impact["revenue_change_pct"]}%")
            st.metric("Elasticity Coefficient Used", impact["elasticity_coefficient_used"])

        # ---------------- PROMO ----------------
        elif result["type"] == "promo":

            impact = result["impact"]

            st.metric("Baseline Revenue", impact["baseline_revenue"])
            st.metric("Projected Revenue", impact["projected_revenue"])
            st.metric("Revenue Change %", f"{impact["revenue_change_pct"]}%")
            st.metric("Discount % Used", impact["discount_pct_used"])
            st.metric("Conversion rate Used", impact["conversion_rate"])

        # ---------------- LLM ----------------
        else:
            st.subheader("📌 Strategy Memo")
            st.write(result["text"])
