from explain_transfer import explain_transfer
from explain_manufacturing import explain_manufacturing
from explain_scenario import explain_scenario
from llm_refiner import refine_with_llm
import traceback

def detect_intent(query: str):
    q = query.lower()
    
    why_keywords = ["why", "reason", "cause", "justification", "logic"]
    if any(k in q for k in why_keywords):
        return "why"
    
    impact_keywords = ["impact", "cost", "price", "financial", "money", "budget", "saving"]
    if any(k in q for k in impact_keywords):
        return "impact"
        
    summary_keywords = ["summary", "overview", "what happened", "describe", "explain", "review"]
    if any(k in q for k in summary_keywords):
        return "summary"
        
    return "default"


def handle_query(query, data):
    try:
        if not data:
            return "Reference data is missing. Please check the system inputs."
            
        intent = detect_intent(query)
        text = ""

        if intent == "why":
            if "transfer" in data and data["transfer"]:
                text = explain_transfer(data["transfer"])
            elif "manufacturing" in data and data["manufacturing"]:
                text = explain_manufacturing(data["manufacturing"])
            else:
                text = "I cannot find specific transfer or manufacturing details to explain."

        elif intent == "impact":
            if "transfer" in data and data["transfer"]:
                 # reuse logic that contains cost info
                text = explain_transfer(data["transfer"])
            elif "scenario" in data and data["scenario"]:
                text = explain_scenario(data["scenario"])
            else:
                text = "I couldn't generate a specific cost impact answer."

        elif intent == "summary" or intent == "default":
            if "scenario" in data and data["scenario"]:
                text = explain_scenario(data["scenario"])
            else:
                text = "Scenario data is unavailable."
        
        else:
             if "scenario" in data and data["scenario"]:
                text = explain_scenario(data["scenario"])
             else:
                text = "I'm not sure how to answer that."

        return refine_with_llm(text)
        
    except Exception as e:
        print(f"Error handling query: {e}")
        traceback.print_exc()
        return "I encountered an error processing your request. Please try again."
