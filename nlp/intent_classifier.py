import re
from nlp.llm_client import call_llm

ALLOWED_INTENTS = {
    # ── New pipeline intents ───────────────────────────────────────────────
    "inventory_status",           # Show current vs target inventory levels
    "manufacturing_plan",         # Manufacturing decisions and production actions
    "transfer_recommendations",   # Transfer recommendations and routing
    "scenario_summary",           # High-level scenario metrics
    "top_transfers_by_cost",      # Transfers ranked by transport cost (descending)
    "top_transfers_by_quantity",  # Transfers ranked by quantity (descending)
    "top_manufacturing_items",    # Manufacturing actions ranked by cost or quantity
    "scenario_comparison",        # Compare two scenarios side-by-side
    # ── Legacy intents kept for backward compatibility ─────────────────────
    "explain_transfer",
    "explain_manufacturing",
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
    # ── Always present ─────────────────────────────────────────────────────
    "out_of_scope",
    "greeting",
}

_GREETING_KEYWORDS = {
    "hi", "hello", "hey", "howdy", "greetings", "sup", "what's up",
    "good morning", "good afternoon", "good evening", "hiya", "yo",
}

SYSTEM_PROMPT = (
    "You are an intent classifier for a supply chain optimization NLP assistant. "
    "Given a user message, return exactly one label from this list:\n"
    "  inventory_status\n"
    "  manufacturing_plan\n"
    "  transfer_recommendations\n"
    "  scenario_summary\n"
    "  top_transfers_by_cost\n"
    "  top_transfers_by_quantity\n"
    "  top_manufacturing_items\n"
    "  scenario_comparison\n"
    "  explain_transfer\n"
    "  explain_manufacturing\n"
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
    "  inventory_status           → current vs target inventory levels for stores or products\n"
    "  manufacturing_plan         → manufacturing decisions and production actions\n"
    "  transfer_recommendations   → transfer recommendations and inter-store routing\n"
    "  scenario_summary           → high-level scenario metrics and overview\n"
    "  top_transfers_by_cost      → transfers ranked by transport cost (highest first)\n"
    "  top_transfers_by_quantity  → transfers ranked by quantity (highest first)\n"
    "  top_manufacturing_items    → manufacturing actions ranked by cost or quantity\n"
    "  scenario_comparison        → comparing two scenarios side-by-side\n"
    "  explain_transfer           → detailed narrative for a specific transfer\n"
    "  explain_manufacturing      → detailed narrative for a specific manufacturing decision\n"
    "  impact_analysis            → cost impact, financial metrics, savings\n"
    "  total_counts               → summary counts and statistics\n"
    "  top_transfers              → prioritized transfers (generic)\n"
    "  top_manufacturing          → prioritized manufacturing actions (generic)\n"
    "  urgent_transfers           → stockout-prevention transfers\n"
    "  high_cost_actions          → most expensive transfers or manufacturing decisions\n"
    "  reason_analysis            → analysis of decision reason codes\n"
    "  store_activity             → store-specific involvement in transfers\n"
    "  product_recommendations    → product-specific actions and impact\n"
    "  cost_breakdown             → detailed cost composition analysis\n"
    "  out_of_scope               → unrelated to supply chain optimization\n\n"
    "Examples:\n"
    "  'show inventory status' → inventory_status\n"
    "  'what is the manufacturing plan' → manufacturing_plan\n"
    "  'show transfer recommendations' → transfer_recommendations\n"
    "  'top 5 transfers by cost' → top_transfers_by_cost\n"
    "  'top transfers by quantity' → top_transfers_by_quantity\n"
    "  'top manufacturing items' → top_manufacturing_items\n"
    "  'compare base case vs high disruption' → scenario_comparison\n"
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
    "- If the query asks to rank/sort transfers by cost → top_transfers_by_cost.\n"
    "- If the query asks to rank/sort transfers by quantity → top_transfers_by_quantity.\n"
    "- If about costs/expenses, default to high_cost_actions or cost_breakdown."
)

_KEYWORD_MAP = {
    # ── New pipeline intents ───────────────────────────────────────────────
    "inventory_status": [
        "inventory", "stock level", "current stock", "inventory level",
        "inventory status", "stock status", "on hand", "final stock",
        "target stock", "current inventory", "inventory position",
    ],
    "manufacturing_plan": [
        "manufacturing plan", "production plan", "manufacture plan",
        "what should be manufactured", "planned manufacturing",
        "production schedule", "manufacturing schedule",
    ],
    "transfer_recommendations": [
        "show transfer recommendations", "list transfer recommendations",
        "show all transfers", "list all transfers", "transfer overview",
        "all transfer recommendations",
    ],
    "top_transfers_by_cost": [
        "top transfers by cost", "transfers by cost", "transfers ranked by cost",
        "transfers sorted by cost", "transfer cost ranking",
    ],
    "top_transfers_by_quantity": [
        "top transfers by quantity", "transfers by quantity", "most units transferred",
        "highest quantity transfers", "transfers sorted by quantity", "transfer by volume",
    ],
    "top_manufacturing_items": [
        "top manufacturing items", "manufacturing items by cost",
        "manufacturing ranked by cost", "manufacturing sorted by cost",
        "ranked manufacturing items",
    ],
    "scenario_comparison": [
        "compare scenario", "scenario vs", "vs scenario", "scenario comparison",
        "compare base", "compare high disruption", "compare fuel shock",
        "compare demand spike", "compare lead time", "scenario difference",
        "which scenario", "compare two scenario",
    ],
    # ── Legacy intents ─────────────────────────────────────────────────────
    "explain_transfer": [
        "transfer", "move inventory", "reroute", "shift stock", "send inventory",
        "from store", "to store", "inter-store", "why transfer", "should i transfer",
        "transfer recommend", "allocation decision", "why move", "store to store",
        "explain transfer",
    ],
    "explain_manufacturing": [
        "manufactur", "produce", "production", "make more", "fabricat",
        "assembly", "build more", "why manufactur", "manufacturing decision",
        "manufacture", "production decision", "explain manufactur",
    ],
    "scenario_summary": [
        "scenario", "summary", "overview", "high risk", "low risk",
        "optimized", "baseline", "what happened", "performance",
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
        "priority transfer", "most costly transfer",
        "ranking transfer", "top routes", "transfer priority",
    ],
    "top_manufacturing": [
        "top manufactur", "highest manufactur", "largest manufactur", "biggest manufactur",
        "most expensive manufactur", "most costly manufactur", "priority manufactur",
        "ranking manufactur", "highest cost manufactur", "manufactur recommend",
    ],
    "urgent_transfers": [
        "urgent", "priority", "critical", "emergency", "rush", "asap", "immediate",
        "high priority transfer", "urgent transfer", "critical transfer", "stockout prevention",
    ],
    "high_cost_actions": [
        "expensive", "costly", "high cost", "highest cost",
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
    priority_intents = [
        "inventory_status",
        "top_transfers_by_cost",
        "top_transfers_by_quantity",
        "top_manufacturing_items",
        "scenario_comparison",
        "manufacturing_plan",
        "transfer_recommendations",
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


def extract_list_parameters(user_message: str) -> dict:
    """Extract list parameters (limit, sort, filter, etc.) from user input.

    Alias for extract_parameters for backward compatibility.
    """
    return extract_parameters(user_message)


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


def parse_result_limit(user_message: str, default: int = 10, max_limit: int = 10) -> int:
    """Parse the requested result limit from a user message.

    Rules enforced by the pipeline:
    - Default limit is 10 when not specified.
    - Maximum limit is 10 (capped even if the user requests more).

    Parameters
    ----------
    user_message:
        Raw user query string.
    default:
        Limit to use when no number is found in the query.
    max_limit:
        Hard upper bound; returned value will never exceed this.

    Returns
    -------
    int
        Effective limit, clamped to [1, max_limit].
    """
    extracted = StructuredParameterExtractor._extract_limit(user_message.lower())
    if extracted is None:
        return default
    # Enforce maximum — never return more than max_limit results
    return max(1, min(extracted, max_limit))
