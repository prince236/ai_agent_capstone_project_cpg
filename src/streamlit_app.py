import streamlit as st
from agent import CPGDecisionAgent
from utils import generate_final_response

# 1. Initialize the CPGDecisionAgent inside Streamlit session state if it doesn't exist
if "agent" not in st.session_state:
    st.session_state.agent = CPGDecisionAgent()

# Initialize conversational UI chat logs
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "Hello! Ask me any retail business query regarding trends, anomalies, or simulations."}
    ]

# --- STREAMLIT UI LAYOUT ---
st.title("📊 CPG Analytics Decision Hub")
st.caption("Interactive Strategy & Simulation Agent Loop")

# Sidebar utilities for cleaning history state
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Reset Agent Session", type="primary"):
        # Re-initialize both agent instance and message queues
        st.session_state.agent = CPGDecisionAgent()
        st.session_state.chat_history = [
            {"role": "assistant",
             "content": "Hello! Ask me any retail business query regarding trends, anomalies, or simulations."}
        ]
        st.rerun()

# 2. Render historical chat turns from state so they persist on screen
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Streamlit replacement for the terminal `while True / input()` loop
if query := st.chat_input("Ask a business question..."):

    # Render and store user question
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate Assistant response bubble
    with st.chat_message("assistant"):
        with st.spinner("Agent routing logic executing..."):

            # CRITICAL FIX: Pull the exact ongoing agent instance from session state
            current_agent = st.session_state.agent

            # Run the agent to detect intent and execute the tool
            result = current_agent.run(query)
            response_text = ""

            # ---------------- TREND ANALYSIS ----------------
            if result["type"] == "trend_analysis":
                summary = result["summary"]
                response_text = generate_final_response(summary)
                st.markdown(response_text)
                st.subheader("📊 Monthly Revenue Trends")

                monthly = result["monthly"]

                chart_data = monthly.pivot(
                    index="month",
                    columns="category",
                    values="revenue"
                )

                st.line_chart(chart_data)

            # ---------------- ANOMALY DETECTION ----------------
            elif result["type"] == "anomaly_detection":
                summary = result["summary"]
                response_text = generate_final_response(summary)
                st.markdown(response_text)

            # ---------------- PRICE CHANGE SIMULATION ----------------
            elif result["type"] == "price_change_simulation":
                impact = result["impact"]
                response_text = generate_final_response(impact)
                st.markdown(response_text)

            # ---------------- PROMO CAMPAIGN SIMULATION ----------------
            elif result["type"] == "promo_campaign_simulation":
                impact = result["impact"]
                response_text = generate_final_response(impact)
                st.markdown(response_text)

            # ---------------- STRATEGY MEMO GENERATION ----------------
            elif result["type"] == "strategy_memo":
                response_text = result["text"]
                st.markdown(response_text)

            # ---------------- OTHERS / FALLBACK ----------------
            else:
                response_text = result["text"]
                st.markdown(response_text)

            # Append the full turn to memory so subsequent strategy memos can read it!
            if result["type"] != "strategy_memo":
                current_agent.memory.add("User", query)
                current_agent.memory.add("Assistant", response_text)

            # Log back to chat state history for presentation rendering
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
