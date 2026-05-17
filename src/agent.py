from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import json

from data_loader import load_data
from tools import analyze_trends, detect_anomalies, simulate_price_change, simulate_promo_campaign
from memory import Memory

load_dotenv()

class CPGDecisionAgent:
    def __init__(self):
        self.df = load_data()
        self.memory = Memory()

        # LLM initialization
        # self.llm = ChatOpenAI(
        #     model="gpt-4o-mini",
        #     temperature=0.0
        # )
        self.llm = ChatOllama(
            model="gpt-oss",
            temperature=0.0
        )

    # -----------------------------
    # Main Agent Execution
    # -----------------------------
    def run(self, query: str):

        detect_intent_prompt = f"""
        You are an intelligent intent detection and parameter extraction assistant for a CPG analytics platform. 
        Analyze the user's query, select the correct tool ("trend_analysis", "anomaly_detection", "price_change_simulation", "promo_campaign_simulation", "strategy_memo" and "other"), and extract parameters.

        Output Format (Strict JSON only, no markdown wrappers):
        {{
            "tool": "tool_name",
            "parameters": {{
                "target_regions": ["Region1", "Region2"] or null,
                "target_categories": ["Category1"] or null,
                "anomaly_col": target column for anomaly detection query or null (if not provided),
                "price_change_pct": float or null,
                "elasticity_coefficient": elasticity coefficient for price change simulation or null (if not provided),
                "promo_type"= "FlashSale" or "Discount" or "BuyOneGetOne" or null (if not provided),
                "discount_pct"= For promo campaign simulation query or null (if not provided),
                "conversion_rate" = For promo campaign simulation query or null (if not provided),
                "lift_factor" = For promo campaign simulation query or null (if not provided)
            }}
        }}

        Business query: {query}
        Output:
        """

        # Get response from LLM
        response = self.llm.invoke(detect_intent_prompt).content.strip()

        print("llm response for detect_intent_prompt: ", response)

        try:
            intent_data = json.loads(response)
            tool = intent_data.get("tool")
            params = intent_data.get("parameters", {})
        except Exception as e:
            # Fallback to safe defaults if JSON parsing fails
            print(f"Failed to parse LLM response: {e}")
            return {"type": "other", "text": "Sorry, I couldn't process that query structurally."}

        # ---------------- DYNAMIC TOOL ROUTING ----------------
        if tool == "trend_analysis":
            # Pass extracted parameters dynamically instead of hardcoding
            regions = params.get("target_regions")
            categories = params.get("target_categories")

            monthly, summary = analyze_trends(self.df, target_regions=regions, target_categories=categories)
            return {"type": "trend_analysis", "monthly": monthly, "summary": summary}

        elif tool == "anomaly_detection":
            col = params.get("anomaly_col") if params.get("anomaly_col") is not None else "units_sold"
            regions = params.get("target_regions")
            categories = params.get("target_categories")

            summary = detect_anomalies(self.df, target_col=col, target_regions=regions, target_categories=categories)
            return {"type": "anomaly_detection", "summary": summary}

        elif tool == "price_change_simulation":
            # Handle default percentage if not extracted, or let function handle it
            pct = params.get("price_change_pct") if params.get("price_change_pct") is not None else 0.15
            categories = params.get("target_categories")

            impact = simulate_price_change(self.df, target_categories=categories, price_change_pct=pct)
            return {"type": "price_change_simulation", "impact": impact}

        elif tool == "promo_campaign_simulation":
            # Handle default percentage if not extracted, or let function handle it
            discount_pct = params.get("discount_pct") if params.get("discount_pct") is not None else 0.2
            categories = params.get("target_categories")
            promo_type = params.get("promo_type") if params.get("promo_type") is not None else "FlashSale"
            conversion_rate = params.get("conversion_rate") if params.get("conversion_rate") is not None else 0.3
            lift_factor = params.get("lift_factor") if params.get("lift_factor") is not None else 1.5

            impact = simulate_promo_campaign(self.df, target_categories=categories, discount_pct=discount_pct,
                                             promo_type=promo_type, conversion_rate=conversion_rate,
                                             lift_factor=lift_factor)
            return {"type": "promo_campaign_simulation", "impact": impact}

        elif tool == "strategy_memo":
            memory_text = self.memory.get()

            prompt = f"""
                You are a Senior CPG Strategy Consultant.

                Business Question:
                {query}

                Previous Conversation:
                {memory_text}

                Generate:
                1. Key Insight
                2. Business Impact
                3. Recommendation

                Keep it executive-level and concise.
                """

            print("memory_text: ", memory_text)
            answer = self.llm.invoke(prompt).content.strip()

            # Save to memory and return
            self.memory.add("User", query)
            self.memory.add("Assistant", answer)
            return {
                "type": "strategy_memo",
                "text": answer
            }
        else:
            return {
                "type": "other",
                "text": "Unable to detect intent or the intent not related to CPG business query."
            }
