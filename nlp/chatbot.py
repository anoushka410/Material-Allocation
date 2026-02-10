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
        
    summary_keywords = ["summary", "overview", "what happened", "describe", "explain", "review", "total", "how many", "count"]
    if any(k in q for k in summary_keywords):
        return "summary"
        
    return "default"

def summarize_transfers(transfers_list):
    total_qty = sum(t['quantity'] for t in transfers_list)
    return (
        f"There are {len(transfers_list)} recommended transfers totaling {total_qty} units. "
        "These transfers aim to balance inventory levels across stores and reduce stockouts."
    )

def summarize_manufacturing(manufacturing_list):
    total_qty = sum(m['manufacture_quantity'] for m in manufacturing_list)
    return (
        f"There are {len(manufacturing_list)} manufacturing orders recommended, producing a total of {total_qty} units "
        "to meet aggregate demand and address capacity constraints."
    )

def handle_query(query, data):
    try:
        if not data:
            return "Reference data is missing. Please check the system inputs."
            
        intent = detect_intent(query)
        text = ""

        # Check if we have specific selected items in context
        has_specific_transfer = "transfer" in data and data["transfer"] and not isinstance(data["transfer"], list)
        has_specific_manufacture = "manufacturing" in data and data["manufacturing"] and not isinstance(data["manufacturing"], list)
        
        # Check if we have full lists
        has_transfer_list = "all_transfers" in data and isinstance(data["all_transfers"], list)
        has_manufacturing_list = "all_manufacturing" in data and isinstance(data["all_manufacturing"], list)

        if intent == "why":
            if has_specific_transfer:
                text = explain_transfer(data["transfer"])
            elif has_specific_manufacture:
                text = explain_manufacturing(data["manufacturing"])
            else:
                # Fallback to general explanation if no specific item selected
                text = "Please select a specific recommendation to get a detailed reason, or ask for a summary of all actions."

        elif intent == "impact":
            if has_specific_transfer:
                text = explain_transfer(data["transfer"]) # has cost info
            elif has_specific_manufacture:
                text = explain_manufacturing(data["manufacturing"]) # has cost info
            elif "scenario" in data and data["scenario"]:
                text = explain_scenario(data["scenario"])
            else:
                text = "I couldn't generate a specific cost impact answer."

        elif intent == "summary" or intent == "default":
            parts = []
            if "scenario" in data and data["scenario"]:
                parts.append(explain_scenario(data["scenario"]))
            
            # If user asks for summary and we have lists, provide high-level stats
            if has_transfer_list:
                parts.append(summarize_transfers(data["all_transfers"]))
            if has_manufacturing_list:
                parts.append(summarize_manufacturing(data["all_manufacturing"]))
                
            if parts:
                text = "\n\n".join(parts)
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
