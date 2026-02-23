import re
from nlp.llm_client import call_llm

ALLOWED_INTENTS = {
    "explain_transfer",
    "explain_manufacturing",
    "scenario_summary",
    "impact_analysis",
    "out_of_scope",
    "greeting",
}

_GREETING_KEYWORDS = {
    "hi", "hello", "hey", "howdy", "greetings", "sup", "what's up",
    "good morning", "good afternoon", "good evening", "hiya", "yo",
}

SYSTEM_PROMPT = (
    "You are an intent classifier for a supply chain NLP assistant. "
    "Given a user message, return exactly one label from this list:\n"
    "  explain_transfer\n"
    "  explain_manufacturing\n"
    "  scenario_summary\n"
    "  impact_analysis\n"
    "  out_of_scope\n\n"
    "Label definitions:\n"
    "  explain_transfer      → anything about moving/transferring inventory between stores, "
    "why a transfer was made, transfer recommendations\n"
    "  explain_manufacturing → anything about producing/manufacturing goods, "
    "why items were manufactured, production decisions\n"
    "  scenario_summary      → high-level overview of a scenario, baseline vs optimized comparison\n"
    "  impact_analysis       → cost savings, stockout reductions, financial impact of decisions\n"
    "  out_of_scope          → unrelated to supply chain (e.g. weather, cooking, general chat)\n\n"
    "Examples:\n"
    "  'why should I transfer?' → explain_transfer\n"
    "  'what are my recommendations?' → explain_transfer\n"
    "  'explain the transfers' → explain_transfer\n"
    "  'why was product manufactured?' → explain_manufacturing\n"
    "  'how did the scenario perform?' → scenario_summary\n"
    "  'what was the cost impact?' → impact_analysis\n"
    "  'what is the weather today?' → out_of_scope\n\n"
    "Rules:\n"
    "- Return ONLY the label. No punctuation, no explanation, no extra text.\n"
    "- If the message relates to supply chain decisions in any way, do NOT return out_of_scope.\n"
    "- Only return out_of_scope if the message is clearly unrelated to supply chain."
)

_KEYWORD_MAP = {
    "explain_transfer": [
        "transfer", "move inventory", "reroute", "shift stock", "send inventory",
        "from store", "to store", "inter-store", "why transfer", "should i transfer",
        "recommend", "recommendation", "what should", "allocation decision",
        "why move", "store to store",
    ],
    "explain_manufacturing": [
        "manufactur", "produce", "production", "make more", "fabricat",
        "assembly", "build more", "why manufactur", "manufacturing decision",
        "manufacture", "production decision",
    ],
    "scenario_summary": [
        "scenario", "summary", "overview", "high risk", "low risk",
        "optimized", "baseline", "what happened", "performance", "compare",
        "how did", "results",
    ],
    "impact_analysis": [
        "impact", "effect", "cost change", "cost impact", "how much",
        "savings", "stockout reduction", "net change", "financial", "benefit",
    ],
}


def _keyword_classify(text: str) -> str:
    lower = text.lower()
    for intent, keywords in _KEYWORD_MAP.items():
        if any(kw in lower for kw in keywords):
            return intent
    return "out_of_scope"


def extract_parameters(text: str) -> dict:
    lower = text.lower()
    # Normalize "store 223" -> "store_223" and "product 892" -> "product_892"
    normalized = re.sub(r'\b(store|product)\s+(\d+)\b', r'\1_\2', lower)
    # Extract IDs
    t_ids = [m.replace("_", "").upper() for m in re.findall(r'\bt_?\d{3}\b', normalized)]
    m_ids = [m.replace("_", "").upper() for m in re.findall(r'\bm_?\d{3}\b', normalized)]
    
    return {
        "transfer_id": t_ids,
        "manufacturing_id": m_ids,
        "product_id": re.findall(r'\bproduct_\d+\b', normalized),
        "store_id": re.findall(r'\bstore_\d+\b', normalized),
        "is_all": any(w in lower for w in ["all", "every", "overview", "list", "total", "everything"]),
    }


def classify_intent(user_message: str) -> str:
    lower = user_message.strip().lower()
    if lower in _GREETING_KEYWORDS or any(lower.startswith(g) for g in _GREETING_KEYWORDS):
        return "greeting"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    
    def _fallback_with_params(intent: str) -> str:
        if intent == "out_of_scope":
            params = extract_parameters(user_message)
            if params.get("store_id") or params.get("transfer_id"):
                return "explain_transfer"
            if params.get("manufacturing_id"):
                return "explain_manufacturing"
            if params.get("product_id"):
                return "explain_transfer" # defaults to showing product transfers
        return intent

    try:
        raw = call_llm(messages).strip().lower()
        label = raw.split()[0] if raw else ""
        if label in ALLOWED_INTENTS:
            return _fallback_with_params(label)
        return _fallback_with_params(_keyword_classify(user_message))
    except Exception:
        return _fallback_with_params(_keyword_classify(user_message))
