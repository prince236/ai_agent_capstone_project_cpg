from agent import CPGDecisionAgent
from utils import generate_final_response

def main():
    agent = CPGDecisionAgent()

    while True:
        query = input("Ask a business question (or type 'exit'): ")
        if not query or query.lower() == 'exit':
            break

        # Run the agent to detect intent and execute the tool
        result = agent.run(query)

        # ---------------- TREND ANALYSIS ----------------
        if result["type"] == "trend_analysis":

            summary = result["summary"]
            print("summary: ", summary)

            response_text = generate_final_response(summary)
            print("Final Response: ", response_text)

        # ---------------- ANOMALY DETECTION ----------------
        elif result["type"] == "anomaly_detection":
            summary = result["summary"]
            response_text = generate_final_response(summary)
            print("Final Response: ", response_text)

        # ---------------- PRICE CHANGE SIMULATION ----------------
        elif result["type"] == "price_change_simulation":
            impact = result["impact"]
            response_text = generate_final_response(impact)
            print("Final Response: ", response_text)

        # ---------------- PROMO CAMPAIGN SIMULATION ----------------
        elif result["type"] == "promo_campaign_simulation":
            impact = result["impact"]
            response_text = generate_final_response(impact)
            print("Final Response: ", response_text)

        # ---------------- STRATEGY MEMO GENERATION ----------------
        elif result["type"] == "strategy_memo":
            response_text = result["text"]
            print("Final Response: ", response_text)

        # ---------------- OTHERS / FALLBACK ----------------
        else:
            response_text = result["text"]
            print("Final Response: ", response_text)

        # Append the full turn to memory so the next 'strategy_memo' call can read it!
        if result["type"] != "strategy_memo":
            agent.memory.add("User", query)
            agent.memory.add("Assistant", response_text)

if __name__ == "__main__":
    main()