from explain_transfer import explain_transfer
from explain_manufacturing import explain_manufacturing
from explain_scenario import explain_scenario
from llm_refiner import refine_with_llm


def detect_intent(query: str):
    q = query.lower()
    if "why" in q:
        return "why"
    if "impact" in q or "cost" in q:
        return "impact"
    if "summary" in q or "overview" in q:
        return "summary"
    return "default"


def handle_query(query, data):
    intent = detect_intent(query)

    if intent == "why":
        if "transfer" in data:
            text = explain_transfer(data["transfer"])
        else:
            text = explain_manufacturing(data["manufacturing"])

    elif intent == "summary":
        text = explain_scenario(data["scenario"])

    else:
        text = explain_scenario(data["scenario"])

    return refine_with_llm(text)
