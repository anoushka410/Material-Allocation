import re
from nlp.llm_client import call_llm

ALLOWED_INTENTS = {
    "explain_transfer",
    "explain_manufacturing",
    "scenario_summary",
    "impact_analysis",
    "total_counts",
    "top_transfers",
    "top_manufacturing",
    "urgent_transfers",
    "high_cost_actions",
    "reason_analysis",
    "store_activity",
    "product_recommendations",
    "cost_breakdown",
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
    "  total_counts\n"
    "  top_transfers\n"
    "  top_manufacturing\n"
    "  urgent_transfers\n"
    "  high_cost_actions\n"
    "  reason_analysis\n"
    "  store_activity\n"
    "  product_recommendations\n"
    "  cost_breakdown\n"
    "  out_of_scope\n\n"
    "Label definitions:\n"
    "  explain_transfer           → detailed transfer recommendations and routing\n"
    "  explain_manufacturing      → manufacturing decisions and production actions\n"
    "  scenario_summary           → high-level scenario overview\n"
    "  impact_analysis            → cost impact, financial metrics\n"
    "  total_counts               → summary counts and statistics\n"
    "  top_transfers              → prioritized transfers by quantity or cost\n"
    "  top_manufacturing          → prioritized manufacturing actions by cost or quantity\n"
    "  urgent_transfers           → transfers marked with high urgency/priority (stockout prevention)\n"
    "  high_cost_actions          → expensive transfers or manufacturing decisions\n"
    "  reason_analysis            → analysis of why decisions were made (reason codes)\n"
    "  store_activity             → store-specific involvement in transfers\n"
    "  product_recommendations    → product-specific decisions and impact\n"
    "  cost_breakdown             → detailed cost composition analysis\n"
    "  out_of_scope               → unrelated to supply chain\n\n"
    "Examples:\n"
    "  'what are the top transfers' → top_transfers\n"
    "  'most expensive manufacturing' → high_cost_actions\n"
    "  'why are these transfers needed' → reason_analysis\n"
    "  'what stores are most active' → store_activity\n"
    "  'how much does manufacturing cost' → cost_breakdown\n"
    "  'urgent transfers' → urgent_transfers\n"
    "  'product 489 details' → product_recommendations\n\n"
    "Rules:\n"
    "- Return ONLY the label. No punctuation, no explanation.\n"
    "- If ambiguous between two intents, prefer the more specific one.\n"
    "- If about costs/expenses, default to high_cost_actions or cost_breakdown."
)

_KEYWORD_MAP = {
    "explain_transfer": [
        "transfer", "move inventory", "reroute", "shift stock", "send inventory",
        "from store", "to store", "inter-store", "why transfer", "should i transfer",
        "transfer recommend", "transfer recommendation", "what should",
        "allocation decision", "why move", "store to store", "explain transfer",
    ],
    "explain_manufacturing": [
        "manufactur", "produce", "production", "make more", "fabricat",
        "assembly", "build more", "why manufactur", "manufacturing decision",
        "manufacture", "production decision", "explain manufactur",
    ],
    "scenario_summary": [
        "scenario", "summary", "overview", "high risk", "low risk",
        "optimized", "baseline", "what happened", "performance", "compare",
        "how did", "results", "scenario summary",
    ],
    "impact_analysis": [
        "impact", "effect", "cost change", "cost impact", "how much",
        "savings", "stockout reduction", "net change", "financial", "benefit",
        "analysis", "compare cost",
    ],
    "total_counts": [
        "how many", "count", "total number", "amount of", "number of",
        "total count", "total recommendations",
    ],
    "top_transfers": [
        "top transfer", "highest transfer", "largest transfer", "biggest transfer",
        "priority transfer", "most expensive transfer", "most costly transfer",
        "ranking transfer", "top routes", "transfer priority",
    ],
    "top_manufacturing": [
        "top manufactur", "highest manufactur", "largest manufactur", "biggest manufactur",
        "most expensive manufactur", "most costly manufactur", "priority manufactur",
        "ranking manufactur", "highest cost manufactur", "manufacturing recommendation",
        "manufactur recommend",
    ],
    "urgent_transfers": [
        "urgent", "priority", "critical", "emergency", "rush", "asap", "immediate",
        "high priority transfer", "urgent transfer", "critical transfer", "stockout prevention",
    ],
    "high_cost_actions": [
        "expensive", "costly", "high cost", "most expensive", "highest cost",
        "expensive transfer", "expensive manufactur", "cost intensive",
        "budget impact", "expensive action",
    ],
    "reason_analysis": [
        "why", "reason", "cause", "explain why", "reasoning", "justification",
        "what's the reason", "reason code", "decision reason", "why was", "why this",
    ],
    "store_activity": [
        "store", "facility", "location", "which store", "which stores",
        "store involvement", "store activity", "active store", "store network",
    ],
    "product_recommendations": [
        "product", "which product", "which products", "product detail",
        "product specific", "product involvement", "product action",
    ],
    "cost_breakdown": [
        "cost breakdown", "cost composition", "cost structure", "cost component",
        "how much cost", "cost details", "cost analysis", "where is cost",
    ],
}


def _keyword_classify(text: str) -> str:
    lower = text.lower()
    
    # Check specific/table intents FIRST (they are more specific and should take precedence)
    # This ensures "top manufacturing" matches top_manufacturing, not explain_manufacturing
    priority_intents = [
        "total_counts",
        "top_transfers",
        "top_manufacturing",
        "urgent_transfers",
        "high_cost_actions",
        "reason_analysis",
        "store_activity",
        "product_recommendations",
        "cost_breakdown",
    ]

    for intent in priority_intents:
        if intent in _KEYWORD_MAP:
            if any(kw in lower for kw in _KEYWORD_MAP[intent]):
                return intent

    # Then check remaining intents (general ones)
    for intent, keywords in _KEYWORD_MAP.items():
        if intent in priority_intents:
            continue  # already checked
        if any(kw in lower for kw in keywords):
            return intent
            
    return "out_of_scope"


import re
from nlp.llm_client import call_llm


class StructuredParameterExtractor:
    """Extract all query parameters (limit, sort, filter, etc.) from user input."""

    # Patterns for extracting numeric limits
    LIMIT_PATTERNS = [
        r'\btop\s+(\d+)',           # "top 3", "top 10"
        r'(\d+)\s+(?:top|best)',    # "3 top", "5 best"
        r'(?:give|show|list)\s+(?:me\s+)?(?:the\s+)?(?:top\s+)?(\d+)', # "give me 5"
        r'(?:first|next)\s+(\d+)',  # "first 5", "next 10"
        r'(?:limit|max)\s+(?:to\s+)?(\d+)',  # "limit to 5"
    ]

    # Patterns for sorting/ordering
    SORT_PATTERNS = {
        'cost': [r'by\s+cost', r'by\s+price', r'expensive', r'cost.*high', r'highest.*cost'],
        'quantity': [r'by\s+quantity', r'by\s+qty', r'by\s+units', r'by\s+volume'],
        'impact': [r'by\s+impact', r'by\s+savings', r'effect'],
    }

    # Patterns for filtering/scoping
    FILTER_PATTERNS = {
        'urgent': [r'\burgeent\b', r'\bcritical\b', r'\bemergency\b', r'stockout'],
        'risky': [r'\brisky\b', r'\brisk\b', r'uncertainty', r'uncertain'],
        'high_impact': [r'high\s+impact', r'most\s+impact', r'significant'],
        'low_cost': [r'cheap', r'low\s+cost', r'economical'],
    }

    @classmethod
    def extract(cls, text: str) -> dict:
        """Extract all parameters from user query."""
        lower = text.lower()
        normalized = re.sub(r'\b(store|product)\s+(\d+)\b', r'\1_\2', lower)

        return {
            # Entities
            "product_id": re.findall(r'\bproduct_\d+\b', normalized),
            "store_id": re.findall(r'\bstore_\d+\b', normalized),

            # Numeric limit (default: all/10 depending on context)
            "limit": cls._extract_limit(lower),

            # Sorting criteria
            "sort_by": cls._extract_sort(lower),

            # Filters
            "filters": cls._extract_filters(lower),

            # Flags
            "is_all": any(w in lower for w in ["all", "every", "overview", "list", "total", "everything"]),
            "is_detailed": any(w in lower for w in ["detailed", "explain", "detail", "why"]),
        }

    @classmethod
    def _extract_limit(cls, text: str) -> int:
        """Extract numeric limit from text. Returns None if not specified.

        Supports digits ("top 3") and written numbers/ordinals ("top three", "top 3rd").
        """
        # First try digit patterns
        for pattern in cls.LIMIT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    pass

        # Try to find written numbers/ordinals (e.g., 'three', 'third')
        # Simple mapping for common words (0- ninety)
        WORD_NUM = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100
        }

        # Ordinal words map
        ORDINAL_MAP = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
            'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14, 'fifteenth': 15,
            'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19, 'twentieth': 20
        }

        # Check for ordinals first
        for ord_word, val in ORDINAL_MAP.items():
            if re.search(r'\b' + re.escape(ord_word) + r'\b', text):
                return val

        # Check for simple written numbers up to 99 (handles 'twenty three')
        tokens = re.findall(r"[a-zA-Z]+", text)
        total = 0
        found = False
        i = 0
        while i < len(tokens):
            w = tokens[i].lower()
            if w in WORD_NUM:
                val = WORD_NUM[w]
                # handle composite like 'twenty three'
                if val >= 20 and i + 1 < len(tokens):
                    nxt = tokens[i + 1].lower()
                    if nxt in WORD_NUM and WORD_NUM[nxt] < 10:
                        val += WORD_NUM[nxt]
                        i += 1
                total += val
                found = True
            i += 1

        if found and total > 0:
            return total

        return None  # No limit specified; caller should use default

    @classmethod
    def _extract_sort(cls, text: str) -> str:
        """Extract sorting criterion. Returns primary match or None."""
        for criterion, patterns in cls.SORT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return criterion
        return None

    @classmethod
    def _extract_filters(cls, text: str) -> list:
        """Extract filter tags. Returns list of applicable filters."""
        filters = []
        for filter_tag, patterns in cls.FILTER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    filters.append(filter_tag)
                    break  # Only add once per filter
        return filters


def extract_parameters(user_message: str) -> dict:
    """Extract structured parameters from user input."""
    return StructuredParameterExtractor.extract(user_message)


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
            if params.get("store_id") or params.get("product_id"):
                return "explain_transfer"
        return intent

    try:
        raw = call_llm(messages).strip().lower()
        label = raw.split()[0] if raw else ""
        if label in ALLOWED_INTENTS:
            return _fallback_with_params(label)
        return _fallback_with_params(_keyword_classify(user_message))
    except Exception:
        return _fallback_with_params(_keyword_classify(user_message))
