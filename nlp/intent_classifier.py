import re
from nlp.llm_client import call_llm

# ── Intent registry ────────────────────────────────────────────────────────────

ALLOWED_INTENTS = {
    # ── Required pipeline intents ──────────────────────────────────────────
    "transfer_recommendations",   # detailed transfer routing decisions
    "manufacturing_plan",         # manufacturing / production decisions
    "inventory_status",           # overall inventory health summary
    "scenario_summary",           # high-level scenario overview & KPIs
    "top_transfers_by_cost",      # top N transfers ranked by cost
    "top_transfers_by_quantity",  # top N transfers ranked by quantity
    "top_manufacturing_items",    # top N manufacturing actions by cost
    "scenario_comparison",        # compare two scenarios side-by-side
    "out_of_scope",               # unrelated to supply chain
    # ── Additional logical intents for supply chain managers ───────────────
    "greeting",
    "impact_analysis",            # cost savings / financial impact
    "total_counts",               # summary counts and statistics
    "urgent_transfers",           # stockout-prevention transfers
    "high_cost_actions",          # most expensive actions across categories
    "reason_analysis",            # why were decisions made (reason codes)
    "store_activity",             # store-level transfer involvement
    "product_recommendations",    # product-specific decisions
    "cost_breakdown",             # detailed cost composition
    "inventory_gaps",             # inventory items below safety-stock target
}

_GREETING_KEYWORDS = {
    "hi", "hello", "hey", "howdy", "greetings", "sup", "what's up",
    "good morning", "good afternoon", "good evening", "hiya", "yo",
}

SYSTEM_PROMPT = (
    "You are an intent classifier for a supply chain NLP assistant. "
    "Given a user message, return exactly one label from this list:\n"
    "  transfer_recommendations\n"
    "  manufacturing_plan\n"
    "  inventory_status\n"
    "  scenario_summary\n"
    "  top_transfers_by_cost\n"
    "  top_transfers_by_quantity\n"
    "  top_manufacturing_items\n"
    "  scenario_comparison\n"
    "  impact_analysis\n"
    "  total_counts\n"
    "  urgent_transfers\n"
    "  high_cost_actions\n"
    "  reason_analysis\n"
    "  store_activity\n"
    "  product_recommendations\n"
    "  cost_breakdown\n"
    "  inventory_gaps\n"
    "  out_of_scope\n\n"
    "Label definitions:\n"
    "  transfer_recommendations   → detailed transfer recommendations and routing\n"
    "  manufacturing_plan         → manufacturing decisions and production actions\n"
    "  inventory_status           → overall inventory health and stock levels\n"
    "  scenario_summary           → high-level scenario overview and KPIs\n"
    "  top_transfers_by_cost      → top transfers sorted/ranked by cost\n"
    "  top_transfers_by_quantity  → top transfers sorted/ranked by quantity\n"
    "  top_manufacturing_items    → top manufacturing actions by cost or quantity\n"
    "  scenario_comparison        → compare two optimization scenarios\n"
    "  impact_analysis            → cost impact, financial metrics, savings\n"
    "  total_counts               → summary counts and statistics\n"
    "  urgent_transfers           → transfers for stockout prevention (high urgency)\n"
    "  high_cost_actions          → most expensive transfers or manufacturing decisions\n"
    "  reason_analysis            → analysis of decision reason codes\n"
    "  store_activity             → store-specific involvement in transfers\n"
    "  product_recommendations    → product-specific decisions and impact\n"
    "  cost_breakdown             → detailed cost composition analysis\n"
    "  inventory_gaps             → inventory items below safety-stock or target levels\n"
    "  out_of_scope               → unrelated to supply chain\n\n"
    "Examples:\n"
    "  'what are the top transfers by cost' → top_transfers_by_cost\n"
    "  'top 5 transfers by quantity' → top_transfers_by_quantity\n"
    "  'most expensive manufacturing' → high_cost_actions\n"
    "  'why are these transfers needed' → reason_analysis\n"
    "  'what stores are most active' → store_activity\n"
    "  'how much does manufacturing cost' → cost_breakdown\n"
    "  'urgent transfers' → urgent_transfers\n"
    "  'product 489 details' → product_recommendations\n"
    "  'compare high disruption vs baseline' → scenario_comparison\n"
    "  'inventory health' → inventory_status\n"
    "  'inventory below target' → inventory_gaps\n\n"
    "Rules:\n"
    "- Return ONLY the label. No punctuation, no explanation.\n"
    "- If ambiguous between two intents, prefer the more specific one.\n"
    "- If about costs/expenses, default to high_cost_actions or cost_breakdown.\n"
    "- If about top/ranking with cost, use top_transfers_by_cost.\n"
    "- If about top/ranking with quantity/units, use top_transfers_by_quantity."
)

# ── Keyword map ────────────────────────────────────────────────────────────────

_KEYWORD_MAP = {
    "transfer_recommendations": [
        "transfer", "move inventory", "reroute", "shift stock", "send inventory",
        "from store", "to store", "inter-store", "why transfer", "should i transfer",
        "transfer recommend", "transfer recommendation", "what should",
        "allocation decision", "why move", "store to store", "explain transfer",
    ],
    "manufacturing_plan": [
        "manufactur", "produce", "production", "make more", "fabricat",
        "assembly", "build more", "why manufactur", "manufacturing decision",
        "manufacture", "production decision", "explain manufactur",
        "manufacturing plan", "what to produce",
    ],
    "inventory_status": [
        "inventory status", "stock level", "inventory health", "current stock",
        "stock summary", "inventory overview", "how much stock", "on hand",
        "inventory report", "stock position", "what is the inventory",
        "inventory for product", "inventory for store", "current inventory",
    ],
    "scenario_summary": [
        "scenario", "summary", "overview", "high risk", "low risk",
        "optimized", "baseline", "what happened", "performance",
        "how did", "results", "scenario summary",
    ],
    "top_transfers_by_cost": [
        "top transfer", "top transfers", "highest transfer", "most expensive transfer",
        "most costly transfer", "transfer by cost", "transfers by cost", "transfers ranked by cost",
        "top cost transfer", "largest cost transfer", "priority transfer by cost",
        "transfer ranked", "transfer sorted by cost", "transfer recommendation by cost",
        "transfer recommendations by cost", "top transfer recommendation",
    ],
    "top_transfers_by_quantity": [
        "top transfer by quantity", "top transfers by quantity", "transfer by quantity",
        "transfers by quantity", "most units transferred",
        "largest transfer by quantity", "highest quantity transfer",
        "transfer by volume", "biggest transfer by units", "by units transferred",
        "ranked by quantity", "sorted by quantity", "ordered by quantity",
    ],
    "top_manufacturing_items": [
        "top manufactur", "highest manufactur", "largest manufactur",
        "most expensive manufactur", "most costly manufactur",
        "priority manufactur", "ranking manufactur",
        "highest cost manufactur", "manufacturing recommendation",
        "manufactur recommend", "top production",
    ],
    "scenario_comparison": [
        "compare scenario", "scenario vs", "vs scenario", "compare scenarios",
        "scenario comparison", "baseline vs", "vs baseline",
        "compare optimization", "difference between scenario",
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
    "urgent_transfers": [
        "urgent", "critical", "emergency", "rush", "asap", "immediate",
        "high priority transfer", "urgent transfer", "critical transfer",
        "stockout prevention",
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
        "which store", "which stores", "store involvement",
        "store activity", "active store", "store network",
    ],
    "product_recommendations": [
        "product", "which product", "which products", "product detail",
        "product specific", "product involvement", "product action",
    ],
    "cost_breakdown": [
        "cost breakdown", "cost composition", "cost structure", "cost component",
        "how much cost", "cost details", "cost analysis", "where is cost",
    ],
    "inventory_gaps": [
        "inventory gap", "below target", "below safety stock", "stockout risk",
        "low inventory", "inventory shortage", "inventory deficit",
        "under target", "short inventory", "inventory below",
    ],
}


def _keyword_classify(text: str) -> str:
    """Fallback keyword-based intent classifier.

    Checks high-specificity intents first to avoid ambiguous matches
    (e.g. 'top manufacturing' should map to top_manufacturing_items, not
    manufacturing_plan).
    """
    lower = text.lower()

    # Check specific/table intents FIRST (they are more specific and should take precedence)
    priority_intents = [
        "total_counts",
        "top_transfers_by_quantity",   # more specific → checked before top_transfers_by_cost
        "top_transfers_by_cost",
        "top_manufacturing_items",
        "urgent_transfers",
        "high_cost_actions",
        "reason_analysis",
        "store_activity",
        "product_recommendations",
        "cost_breakdown",
        "inventory_status",
        "inventory_gaps",
        "scenario_comparison",
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


# Maximum number of results to return (safety cap)
MAX_RESULTS = 10


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
        'urgent': [r'\burgent\b', r'\bcritical\b', r'\bemergency\b', r'stockout'],
        'risky': [r'\brisky\b', r'\brisk\b', r'uncertainty', r'uncertain'],
        'high_impact': [r'high\s+impact', r'most\s+impact', r'significant'],
        'low_cost': [r'cheap', r'low\s+cost', r'economical'],
    }

    @classmethod
    def extract(cls, text: str) -> dict:
        """Extract all parameters from user query."""
        lower = text.lower()
        normalized = re.sub(r'\b(store|product)\s+(\d+)\b', r'\1_\2', lower)

        raw_limit = cls._extract_limit(lower)
        # Enforce the global maximum result cap — never return more than MAX_RESULTS
        capped_limit = min(raw_limit, MAX_RESULTS) if raw_limit is not None else None

        return {
            # Entities
            "product_id": re.findall(r'\bproduct_\d+\b', normalized),
            "store_id": re.findall(r'\bstore_\d+\b', normalized),

            # Numeric limit — capped at MAX_RESULTS
            "limit": capped_limit,

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
    """Classify user query into one of the supported supply chain intents.

    Uses the local LLM (via call_llm) as the primary classifier, with a
    keyword-based fallback when the LLM is unavailable or returns an
    unrecognised label.

    Returns one of the labels defined in ALLOWED_INTENTS.
    """
    lower = user_message.strip().lower()
    if lower in _GREETING_KEYWORDS or any(lower.startswith(g) for g in _GREETING_KEYWORDS):
        return "greeting"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    def _is_direct_inventory_lookup(message: str, params: dict) -> bool:
        lower_msg = message.lower()
        has_inventory_word = any(w in lower_msg for w in ["inventory", "stock", "on hand", "current"])
        has_entity = bool(params.get("store_id") or params.get("product_id"))
        gap_markers = [
            "gap", "below", "shortage", "deficit", "under target", "stockout", "low inventory",
        ]
        is_gap_style = any(m in lower_msg for m in gap_markers)
        return has_inventory_word and has_entity and not is_gap_style

    def _fallback_with_params(intent: str) -> str:
        params = extract_parameters(user_message)

        # Direct inventory lookups with product/store should return status values,
        # not transfer actions or gap-only summaries.
        if _is_direct_inventory_lookup(user_message, params):
            return "inventory_status"

        # If LLM/keyword gives out_of_scope but user mentioned a specific
        # store or product, keep legacy behavior unless inventory lookup pattern matched above.
        if intent == "out_of_scope":
            if params.get("store_id") or params.get("product_id"):
                return "transfer_recommendations"
        return intent

    try:
        raw = call_llm(messages).strip().lower()
        label = raw.split()[0] if raw else ""
        if label in ALLOWED_INTENTS:
            return _fallback_with_params(label)
        return _fallback_with_params(_keyword_classify(user_message))
    except Exception:
        return _fallback_with_params(_keyword_classify(user_message))
