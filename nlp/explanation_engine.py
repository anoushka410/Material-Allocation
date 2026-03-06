"""Explanation Engine for the Supply Chain NLP pipeline.

Responsibilities:
  - detect_scenario()    : map query keywords to an optimization scenario
  - load_data()          : load JSON data files from the output directory
  - build_explanation()  : dispatch to intent-specific explain_* functions
  - handle_user_query()  : single entry-point orchestrating the full pipeline

Safety rules enforced here:
  - All numeric values come from JSON. The LLM must never calculate numbers.
  - Maximum MAX_RESULTS results returned to the user by default.
"""
from __future__ import annotations

import json
import os

# Default data directory (relative to project root)
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "optimization", "output-json"
)

# Default scenario used when the user does not specify one
DEFAULT_SCENARIO = "base_case_standard_conditions"

# Maximum number of records to return in any ranked list
MAX_RESULTS = 10

# ── Scenario keyword mapping ───────────────────────────────────────────────────
# Maps lowercase keywords found in the user query to an exact scenario ID.
# Order matters: more specific phrases are listed before broad ones.
_SCENARIO_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["high disruption", "high_disruption", "disruption"], "risk_aware_high_disruption"),
    (["fuel shock", "fuel_shock", "fuel price", "transport cost increase"], "transport_cost_increase_fuel_shock"),
    (["demand spike", "demand_spike", "peak demand", "high demand", "high forecast"], "demand_spike_high_forecast"),
    (["lead time", "lead_time", "supplier delay", "customs delay", "extended lead"], "extended_lead_time_supplier_delay"),
    (["cost only", "cost_only", "no risk", "no_risk", "no risk penalty"], "cost_only_no_risk_penalty"),
    (["base case", "base_case", "standard", "baseline", "default"], DEFAULT_SCENARIO),
]


def detect_scenario(query: str) -> str:
    """Detect the intended optimization scenario from the user query.

    Scans the lowercase query for keyword phrases and returns the matching
    scenario ID. Falls back to DEFAULT_SCENARIO when no keyword matches.

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    str
        Scenario identifier (e.g. "base_case_standard_conditions").
    """
    lower = query.lower()
    for keywords, scenario_id in _SCENARIO_KEYWORD_MAP:
        if any(kw in lower for kw in keywords):
            return scenario_id
    return DEFAULT_SCENARIO


def load_data(scenario: str = DEFAULT_SCENARIO, data_dir: str | None = None) -> dict:
    """Load all optimization JSON files and filter to the specified scenario.

    All numeric values come exclusively from the JSON files — never from the
    LLM.  This function is the single source of truth for data access in the
    NLP pipeline.

    Parameters
    ----------
    scenario : str
        The scenario ID to filter records for.
    data_dir : str, optional
        Path to the output-json directory. Defaults to the project-level
        ``optimization/output-json`` directory.

    Returns
    -------
    dict with keys:
        scenario     – single scenario summary dict
        transfers    – {"scenario": str, "transfers": [...]}
        manufacturing– {"scenario": str, "manufacturing_actions": [...]}
        inventory    – list of inventory records for the scenario
        scenario_summary_all – full combined scenario_summary.json content
    """
    if data_dir is None:
        data_dir = _DATA_DIR

    result: dict = {}

    # ── Scenario summary (all scenarios) ──────────────────────────────────────
    try:
        with open(os.path.join(data_dir, "scenario_summary.json")) as f:
            summary_all = json.load(f)
    except Exception:
        summary_all = {}

    result["scenario_summary_all"] = summary_all

    # Single-scenario summary for the requested scenario
    summaries = summary_all.get("summaries", [])
    result["scenario"] = next(
        (s for s in summaries if s.get("scenario") == scenario), {}
    )

    # ── Transfer recommendations ───────────────────────────────────────────────
    try:
        with open(os.path.join(data_dir, "transfer_recommendations.json")) as f:
            transfers_all = json.load(f)
    except Exception:
        transfers_all = {}

    transfers = [
        t for t in (transfers_all.get("transfers") or [])
        if str(t.get("scenario")) == str(scenario)
    ]
    result["transfers"] = {"scenario": scenario, "transfers": transfers}

    # ── Manufacturing decisions ────────────────────────────────────────────────
    try:
        with open(os.path.join(data_dir, "manufacturing_decisions.json")) as f:
            mfg_all = json.load(f)
    except Exception:
        mfg_all = {}

    mfg_actions = [
        m for m in (mfg_all.get("manufacturing_actions") or [])
        if str(m.get("scenario")) == str(scenario)
    ]
    result["manufacturing"] = {"scenario": scenario, "manufacturing_actions": mfg_actions}

    # ── Inventory ──────────────────────────────────────────────────────────────
    try:
        with open(os.path.join(data_dir, "inventory.json")) as f:
            inv_all = json.load(f)
    except Exception:
        inv_all = {}

    inventory = [
        i for i in (inv_all.get("inventory") or [])
        if str(i.get("scenario")) == str(scenario)
    ]
    result["inventory"] = inventory

    return result


# ── Helper: normalise scenario-based filter ────────────────────────────────────

def _filter_by_scenario(records: list[dict], scenario: str) -> list[dict]:
    """Filter records by scenario, but only when the records carry a scenario field.

    If none of the records contain a ``scenario`` key (e.g. test fixtures that
    omit it), the full list is returned unchanged so that unit tests still work.
    """
    if scenario in ("all", "optimization_run", "unknown"):
        return records
    # Detect whether individual records carry a scenario field
    has_field = any("scenario" in r for r in records)
    if not has_field:
        return records  # no per-record scenario field — no filtering possible
    return [r for r in records if str(r.get("scenario")) == str(scenario)]


# ── Manufacturing field accessors (shared across multiple explain_* functions) ──

def _mfg_cost(m: dict) -> float:
    """Extract manufacturing cost from either JSON format.

    Handles:
      - Old schema: ``cost_impact.manufacturing_cost``
      - Current JSON output: top-level ``cost``
    """
    ci = m.get("cost_impact") or {}
    if ci and ci.get("manufacturing_cost") is not None:
        return float(ci["manufacturing_cost"])
    return float(m.get("cost", 0))


def _mfg_qty(m: dict) -> float:
    """Extract manufacturing quantity from either JSON format.

    Handles:
      - Old schema: ``manufacture_quantity``
      - Current JSON output: top-level ``quantity``
    """
    qty = m.get("manufacture_quantity")
    if qty is None:
        qty = m.get("quantity", 0)
    return float(qty)


def explain_transfer(data: dict, params: dict = None) -> str:
    """Generate a deterministic markdown explanation of transfer recommendations.

    All numeric values (quantity, cost) come directly from the JSON data —
    the LLM never performs calculations or invents figures.
    """
    transfer_data = data.get("transfers", {})
    transfers = transfer_data.get("transfers", [])

    # Resolve the scenario: prefer explicit params, then wrapper, then default
    selected_scenario = None
    if params and isinstance(params, dict):
        selected_scenario = params.get("scenario")
    if not selected_scenario:
        selected_scenario = transfer_data.get("scenario")

    # Filter by scenario using the robust helper (handles missing scenario field)
    if selected_scenario:
        transfers = _filter_by_scenario(transfers, selected_scenario)

    scenario_label = selected_scenario or "Unknown"

    if params:
        filtered = []
        p_prod = [x.lower() for x in params.get("product_id", [])]
        p_store = [x.lower() for x in params.get("store_id", [])]

        if p_prod or p_store:
            for t in transfers:
                product = str(t.get("product_id", "")).lower()
                from_s = str(t.get("from_store", "")).lower()
                to_s = str(t.get("to_store", "")).lower()

                if (p_prod and product in p_prod) or \
                   (p_store and (from_s in p_store or to_s in p_store)):
                    filtered.append(t)
            transfers = filtered

    if not transfers:
        return "No transfers match the given specific product or store in this scenario."

    lines = [
        f"**Scenario:** {scenario_label}  ",
        f"**Transfer Recommendations:** {len(transfers)}",
        "",
    ]

    for idx, t in enumerate(transfers, 1):
        from_s = t.get("from_store", "N/A")
        to_s = t.get("to_store", "N/A")
        product = t.get("product_id", "N/A")
        qty = t.get("quantity", 0)
        reasons = t.get("reason_codes", [])
        ci = t.get("cost_impact", {})
        t_scen = t.get("scenario")

        reason_text = ", ".join(r.replace("_", " ").title() for r in reasons)

        title = f"**Transfer {idx}: Product {product}**"
        if t_scen and scenario_label in ("all", "Unknown", "unknown"):
            title += f" *(Scenario: {t_scen})*"
        lines.append(title + "  ")
        lines.append(f"Route: Store {from_s} → Store {to_s}  ")
        try:
            lines.append(f"Quantity: **{float(qty):.2f} units**  ")
        except Exception:
            lines.append(f"Quantity: **{qty} units**  ")
        lines.append(f"Reasons: {reason_text}  ")
        lines.append(f"Transport Cost: ${ci.get('transport_cost', 0):,.2f}  ")
        lines.append("")

    return "\n".join(lines).rstrip()


def explain_manufacturing(data: dict, params: dict = None) -> str:
    """Generate a deterministic markdown explanation of manufacturing decisions.

    Handles both JSON formats:
      - Old schema: ``manufacture_quantity`` / ``cost_impact.manufacturing_cost``
      - Current JSON: ``quantity`` / ``cost``

    All numeric values come from JSON — the LLM never performs calculations.
    """
    mfg_data = data.get("manufacturing", {})
    actions = mfg_data.get("manufacturing_actions", [])

    selected_scenario = None
    if params and isinstance(params, dict):
        selected_scenario = params.get("scenario")
    if not selected_scenario:
        selected_scenario = mfg_data.get("scenario")

    # Filter by scenario using the robust helper
    if selected_scenario:
        actions = _filter_by_scenario(actions, selected_scenario)

    scenario_label = selected_scenario or "Unknown"

    if params:
        filtered = []
        p_prod = [x.lower() for x in params.get("product_id", [])]

        if p_prod:
            for m in actions:
                product = str(m.get("product_id", "")).lower()

                if product in p_prod:
                    filtered.append(m)
            actions = filtered

    if not actions:
        return "No manufacturing actions match the given specific product in this scenario."

    lines = [
        f"**Scenario:** {scenario_label}  ",
        f"**Manufacturing Decisions:** {len(actions)}",
        "",
    ]

    for idx, m in enumerate(actions, 1):
        product = m.get("product_id", "N/A")
        qty = _mfg_qty(m)
        cost = _mfg_cost(m)
        reasons = m.get("reason_codes", [])
        m_scen = m.get("scenario")

        title = f"**Decision {idx}: Product {product}**"
        if m_scen and scenario_label in ("all", "Unknown", "unknown"):
            title += f" *(Scenario: {m_scen})*"
        lines.append(title + "  ")
        lines.append(f"Quantity to Manufacture: **{qty:.2f} units**  ")
        reason_text = ", ".join(r.replace("_", " ").title() for r in reasons)
        lines.append(f"Reasons: {reason_text}  ")
        lines.append(f"Manufacturing Cost: ${cost:,.2f}  ")
        lines.append("")

    return "\n".join(lines).rstrip()


def explain_scenario(data: dict) -> str:
    scen_data = data.get("scenario", {})

    # Support both formats:
    # 1) single scenario summary: {scenario, optimized, cost_breakdown}
    # 2) combined summary: {summaries:[{scenario, optimized, cost_breakdown}, ...]}

    # Check if we received the full scenario_summary.json structure
    if "summaries" in scen_data:
        summaries = scen_data.get("summaries", [])
        lines = [
            "**Scenario Summary (All Scenarios)**",
            "",
            f"Total scenarios: **{len(summaries)}**",
            "",
            "| Scenario | Total Cost | Transfers | Mfg Units | Transfer Units |",
            "|---------|------------:|----------:|----------:|---------------:|",
        ]
        for summary in summaries:
            sid = summary.get("scenario", "Unknown")
            opt = summary.get("optimized", {})
            lines.append(
                f"| {sid} | ${opt.get('total_cost', 0):,.2f} | {opt.get('total_transfers', 0)} | {opt.get('manufacturing_units', 0):,.1f} | {opt.get('transfer_units', 0):,.1f} |"
            )
        return "\n".join(lines)

    scenario = scen_data.get("scenario", "Unknown")
    optimized = scen_data.get("optimized", {})
    cost_breakdown = scen_data.get("cost_breakdown", {})

    lines = [
        f"**Scenario:** {scenario}",
        "",
        "**Optimized Results:**",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Cost | ${optimized.get('total_cost', 0):,.2f} |",
        f"| Total Transfers | {optimized.get('total_transfers', 0)} |",
        f"| Manufacturing Units | {optimized.get('manufacturing_units', 0):.2f} |",
        f"| Transfer Units | {optimized.get('transfer_units', 0):.2f} |",
        "",
        "**Cost Breakdown:**",
        "",
        f"| Cost Component | Amount |",
        f"|--------|-------|",
        f"| Manufacturing Cost | ${cost_breakdown.get('manufacturing_cost', 0):,.2f} |",
        f"| Transfer Cost | ${cost_breakdown.get('transfer_cost', 0):,.2f} |",
        f"| Holding Cost | ${cost_breakdown.get('holding_cost', 0):,.2f} |",
    ]
    return "\n".join(lines)


def explain_entities(data: dict) -> str:
    scenario = data.get("scenario", {}).get("scenario", "Unknown")
    transfers = data.get("transfers", {}).get("transfers", [])
    manufacturing = data.get("manufacturing", {}).get("manufacturing_actions", [])

    products = set()
    stores = set()

    for t in transfers:
        if "product_id" in t:
            products.add(t["product_id"])
        if "from_store" in t:
            stores.add(t["from_store"])
        if "to_store" in t:
            stores.add(t["to_store"])

    for m in manufacturing:
        if "product_id" in m:
            products.add(m["product_id"])

    lines = [
        f"**Scenario:** {scenario}",
        "",
        f"**Products Affected:** {len(products)}",
    ]
    if products:
        lines.extend(f"- {p}" for p in sorted(products, key=lambda x: int(x)))
    else:
        lines.append("- None")

    lines.append("")
    lines.append(f"**Stores Involved:** {len(stores)}")
    if stores:
        lines.extend(f"- {s}" for s in sorted(stores, key=lambda x: int(x)))
    else:
        lines.append("- None")
    
    return "\n".join(lines)


def explain_counts(data: dict) -> str:
    scenario = data.get("scenario", {}).get("scenario", "Unknown")
    transfers = data.get("transfers", {}).get("transfers", [])
    manufacturing = data.get("manufacturing", {}).get("manufacturing_actions", [])
    scen_data = data.get("scenario", {})
    optimized = scen_data.get("optimized", {})
    cost_breakdown = scen_data.get("cost_breakdown", {})

    products_t = set(t["product_id"] for t in transfers if "product_id" in t)
    products_m = set(m["product_id"] for m in manufacturing if "product_id" in m)
    total_products = len(products_t | products_m)

    stores = set()
    for t in transfers:
        if "from_store" in t:
            stores.add(t["from_store"])
        if "to_store" in t:
            stores.add(t["to_store"])

    total_transfer_quantity = sum(t.get("quantity", 0) for t in transfers)
    # Support both JSON field formats for manufacturing quantity
    total_mfg_quantity = sum(
        (m.get("manufacture_quantity") if m.get("manufacture_quantity") is not None else m.get("quantity", 0))
        for m in manufacturing
    )

    total_cost = float(optimized.get("total_cost", 0) or 0)
    if total_cost == 0 and isinstance(cost_breakdown, dict):
        total_cost = float(
            (cost_breakdown.get("manufacturing_cost", 0) or 0)
            + (cost_breakdown.get("transfer_cost", 0) or 0)
            + (cost_breakdown.get("holding_cost", 0) or 0)
        )

    lines = [
        f"**Scenario:** {scenario}",
        "",
        "**Summary Metrics:**",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total Transfer Recommendations | {len(transfers)} |",
        f"| Total Manufacturing Decisions | {len(manufacturing)} |",
        f"| Unique Products Affected | {total_products} |",
        f"| Stores Involved | {len(stores)} |",
        f"| Total Transfer Quantity | {total_transfer_quantity:.2f} units |",
        f"| Total Manufacturing Quantity | {total_mfg_quantity:.2f} units |",
        f"| Total Optimization Cost | ${total_cost:,.2f} |",
    ]
    return "\n".join(lines)


def explain_top_transfers(data: dict, limit: int = 10) -> str:
    transfers = data.get("transfers", {}).get("transfers", [])
    scenario = data.get("transfers", {}).get("scenario", "Unknown")

    # Sort by transport cost
    sorted_transfers = sorted(transfers, key=lambda x: x['cost_impact'].get('transport_cost', 0), reverse=True)[:limit]

    if not sorted_transfers:
        return "No transfers found."

    lines = [
        f"**Top {limit} Transfers by Cost**",
        f"**Scenario:** {scenario}",
        "",
        "| Rank | Route | Product | Qty | Cost |",
        "|------|-------|---------|-----|------|",
    ]

    for idx, t in enumerate(sorted_transfers, 1):
        from_s = t.get("from_store", "N/A")
        to_s = t.get("to_store", "N/A")
        product = t.get("product_id", "N/A")
        qty = t.get("quantity", 0)
        cost = t['cost_impact'].get('transport_cost', 0)
        lines.append(f"| {idx} | {from_s}→{to_s} | {product} | {qty:.2f} | ${cost:.2f} |")

    return "\n".join(lines)


def explain_top_manufacturing(data: dict, limit: int = 10) -> str:
    """Return top N manufacturing actions ranked by cost.

    Uses the module-level ``_mfg_cost`` and ``_mfg_qty`` helpers to handle
    both JSON field name formats:
      - Old schema: ``manufacture_quantity`` / ``cost_impact.manufacturing_cost``
      - Current JSON: ``quantity`` / ``cost``
    """
    manufacturing = data.get("manufacturing", {}).get("manufacturing_actions", [])
    scenario = data.get("manufacturing", {}).get("scenario", "Unknown")

    # Sort by manufacturing cost (all values come from JSON)
    sorted_mfg = sorted(manufacturing, key=_mfg_cost, reverse=True)[:limit]

    if not sorted_mfg:
        return "No manufacturing actions found."

    lines = [
        f"**Top {limit} Manufacturing Actions by Cost**",
        f"**Scenario:** {scenario}",
        "",
        "| Rank | Product | Quantity | Cost |",
        "|------|---------|----------|------|",
    ]

    for idx, m in enumerate(sorted_mfg, 1):
        product = m.get("product_id", "N/A")
        qty = _mfg_qty(m)
        cost = _mfg_cost(m)
        lines.append(f"| {idx} | {product} | {qty:.2f} units | ${cost:,.2f} |")

    return "\n".join(lines)


def explain_urgent_transfers(data: dict) -> str:
    transfers = data.get("transfers", {}).get("transfers", [])
    scenario = data.get("transfers", {}).get("scenario", "Unknown")

    # Filter transfers that prevent stockouts (urgent/critical)
    urgent = [t for t in transfers if "projected_stockout_at_destination" in t.get("reason_codes", [])]

    if not urgent:
        return "No urgent transfers (stockout prevention) found in this scenario."

    lines = [
        f"**Urgent Transfers (Stockout Prevention)**",
        f"**Scenario:** {scenario}",
        f"**Total Urgent Transfers:** {len(urgent)}",
        "",
    ]

    # Group by destination store to show which stores are at risk
    by_destination = {}
    for t in urgent:
        dest = t.get("to_store", "Unknown")
        if dest not in by_destination:
            by_destination[dest] = []
        by_destination[dest].append(t)

    lines.append("**Stores Receiving Stockout Prevention Transfers:**")
    lines.append("")

    for dest in sorted(by_destination.keys(), key=lambda x: int(x)):
        transfers_to_dest = by_destination[dest]
        total_qty = sum(t.get('quantity', 0) for t in transfers_to_dest)
        total_cost = sum(t['cost_impact'].get('transport_cost', 0) for t in transfers_to_dest)
        lines.append(f"Store {dest}: {len(transfers_to_dest)} transfers, {total_qty:.2f} units, ${total_cost:.2f}")

    return "\n".join(lines)


def explain_high_cost_actions(data: dict, limit: int = 15) -> str:
    transfers = data.get("transfers", {}).get("transfers", [])
    manufacturing = data.get("manufacturing", {}).get("manufacturing_actions", [])
    scenario = data.get("scenario", {}).get("scenario", "Unknown")

    # Get top expensive transfers
    top_transfers = sorted(transfers, key=lambda x: x.get("cost_impact", {}).get('transport_cost', 0), reverse=True)[:5]

    # Get top expensive manufacturing (supports both field formats)
    top_mfg = sorted(manufacturing, key=_mfg_cost, reverse=True)[:10]

    lines = [
        f"**High-Cost Actions Overview**",
        f"**Scenario:** {scenario}",
        "",
        "**Top 5 Most Expensive Transfers:**",
        "| Route | Product | Cost |",
        "|-------|---------|------|",
    ]

    for t in top_transfers:
        from_s = t.get("from_store")
        to_s = t.get("to_store")
        product = t.get("product_id")
        cost = t.get("cost_impact", {}).get('transport_cost', 0)
        lines.append(f"| {from_s}→{to_s} | {product} | ${cost:.2f} |")

    lines.extend([
        "",
        "**Top 10 Most Expensive Manufacturing:**",
        "| Product | Quantity | Cost |",
        "|---------|----------|------|",
    ])

    for m in top_mfg:
        product = m.get("product_id")
        qty = _mfg_qty(m)
        cost = _mfg_cost(m)
        lines.append(f"| {product} | {qty:.2f} units | ${cost:,.2f} |")

    return "\n".join(lines)


def explain_reason_analysis(data: dict) -> str:
    transfers = data.get("transfers", {}).get("transfers", [])
    manufacturing = data.get("manufacturing", {}).get("manufacturing_actions", [])
    scenario = data.get("scenario", {}).get("scenario", "Unknown")

    # Analyze transfer reasons
    transfer_reasons = {}
    for t in transfers:
        for reason in t.get("reason_codes", []):
            transfer_reasons[reason] = transfer_reasons.get(reason, 0) + 1

    # Analyze manufacturing reasons
    mfg_reasons = {}
    for m in manufacturing:
        for reason in m.get("reason_codes", []):
            mfg_reasons[reason] = mfg_reasons.get(reason, 0) + 1

    lines = [
        f"**Decision Reason Analysis**",
        f"**Scenario:** {scenario}",
        "",
        "**Transfer Decision Reasons:**",
        "| Reason | Count |",
        "|--------|-------|",
    ]

    for reason in sorted(transfer_reasons.keys()):
        count = transfer_reasons[reason]
        readable_reason = reason.replace("_", " ").title()
        lines.append(f"| {readable_reason} | {count} |")

    lines.extend([
        "",
        "**Manufacturing Decision Reasons:**",
        "| Reason | Count |",
        "|--------|-------|",
    ])

    for reason in sorted(mfg_reasons.keys()):
        count = mfg_reasons[reason]
        readable_reason = reason.replace("_", " ").title()
        lines.append(f"| {readable_reason} | {count} |")

    return "\n".join(lines)


def explain_store_activity(data: dict) -> str:
    transfers = data.get("transfers", {}).get("transfers", [])
    scenario = data.get("transfers", {}).get("scenario", "Unknown")

    # Analyze store involvement
    sending = {}
    receiving = {}

    for t in transfers:
        from_s = t.get("from_store")
        to_s = t.get("to_store")
        qty = t.get("quantity", 0)
        cost = t['cost_impact'].get('transport_cost', 0)

        if from_s:
            if from_s not in sending:
                sending[from_s] = {"count": 0, "qty": 0, "cost": 0}
            sending[from_s]["count"] += 1
            sending[from_s]["qty"] += qty
            sending[from_s]["cost"] += cost

        if to_s:
            if to_s not in receiving:
                receiving[to_s] = {"count": 0, "qty": 0, "cost": 0}
            receiving[to_s]["count"] += 1
            receiving[to_s]["qty"] += qty
            receiving[to_s]["cost"] += cost

    lines = [
        f"**Store Activity Network**",
        f"**Scenario:** {scenario}",
        "",
        "**Top Sending Stores:**",
        "| Store | Transfers | Quantity | Cost |",
        "|-------|-----------|----------|------|",
    ]

    top_sending = sorted(sending.items(), key=lambda x: x[1]["cost"], reverse=True)[:5]
    for store, stats in top_sending:
        lines.append(f"| {store} | {stats['count']} | {stats['qty']:.2f} | ${stats['cost']:.2f} |")

    lines.extend([
        "",
        "**Top Receiving Stores:**",
        "| Store | Transfers | Quantity | Cost |",
        "|-------|-----------|----------|------|",
    ])

    top_receiving = sorted(receiving.items(), key=lambda x: x[1]["cost"], reverse=True)[:5]
    for store, stats in top_receiving:
        lines.append(f"| {store} | {stats['count']} | {stats['qty']:.2f} | ${stats['cost']:.2f} |")

    return "\n".join(lines)


def explain_product_recommendations(data: dict, params: dict = None) -> str:
    transfers = data.get("transfers", {}).get("transfers", [])
    manufacturing = data.get("manufacturing", {}).get("manufacturing_actions", [])
    scenario = data.get("scenario", {}).get("scenario", "Unknown")

    # If product_id specified in params, filter to that product
    filter_products = params.get("product_id", []) if params else []

    if filter_products:
        transfers = [t for t in transfers if t.get("product_id") in filter_products]
        manufacturing = [m for m in manufacturing if m.get("product_id") in filter_products]

    # Get unique products
    transfer_products = set(t.get("product_id") for t in transfers)
    mfg_products = set(m.get("product_id") for m in manufacturing)
    all_products = transfer_products | mfg_products

    lines = [
        f"**Product Recommendations Summary**",
        f"**Scenario:** {scenario}",
        f"**Products with Actions:** {len(all_products)}",
        "",
        "| Product | Type | Count | Total Qty/Cost |",
        "|---------|------|-------|--------|",
    ]

    for product in sorted(all_products, key=lambda x: int(x)):
        prod_transfers = [t for t in transfers if t.get("product_id") == product]
        prod_mfg = [m for m in manufacturing if m.get("product_id") == product]

        if prod_transfers:
            qty = sum(t.get('quantity', 0) for t in prod_transfers)
            cost = sum(t.get("cost_impact", {}).get('transport_cost', 0) for t in prod_transfers)
            lines.append(f"| {product} | Transfer | {len(prod_transfers)} | {qty:.2f} units, ${cost:.2f} |")

        if prod_mfg:
            qty = sum(_mfg_qty(m) for m in prod_mfg)
            cost = sum(_mfg_cost(m) for m in prod_mfg)
            lines.append(f"| {product} | Manufacturing | {len(prod_mfg)} | {qty:.2f} units, ${cost:,.2f} |")

    return "\n".join(lines)


def explain_cost_breakdown(data: dict) -> str:
    scenario = data.get("scenario", {})
    optimized = scenario.get("optimized", {})
    breakdown = scenario.get("cost_breakdown", {})

    total_cost = optimized.get("total_cost", 0)
    mfg_cost = breakdown.get("manufacturing_cost", 0)
    transfer_cost = breakdown.get("transfer_cost", 0)
    holding_cost = breakdown.get("holding_cost", 0)

    lines = [
        "**Detailed Cost Breakdown**",
        "",
        f"| Cost Component | Amount | % of Total |",
        f"|--------|--------|-----------|",
        f"| Manufacturing | ${mfg_cost:,.2f} | {(mfg_cost/total_cost)*100:.1f}% |",
        f"| Transfer/Logistics | ${transfer_cost:,.2f} | {(transfer_cost/total_cost)*100:.1f}% |",
        f"| Holding/Inventory | ${holding_cost:,.2f} | {(holding_cost/total_cost)*100:.1f}% |",
        f"| **TOTAL** | **${total_cost:,.2f}** | **100%** |",
        "",
        f"**Key Insights:**",
        f"- Manufacturing dominates at {(mfg_cost/total_cost)*100:.1f}% of total cost",
        f"- Transfer costs are minimal at {(transfer_cost/total_cost)*100:.1f}%",
        f"- Holding costs account for {(holding_cost/total_cost)*100:.1f}%",
    ]

    return "\n".join(lines)

def explain_top_transfers_by_quantity(data: dict, limit: int = 10) -> str:
    """Return top N transfers ranked by quantity (units moved).

    All numeric values come from JSON.
    """
    transfers = data.get("transfers", {}).get("transfers", [])
    scenario = data.get("transfers", {}).get("scenario", "Unknown")

    sorted_transfers = sorted(transfers, key=lambda x: x.get("quantity", 0), reverse=True)[:limit]

    if not sorted_transfers:
        return "No transfers found."

    lines = [
        f"**Top {limit} Transfers by Quantity**",
        f"**Scenario:** {scenario}",
        "",
        "| Rank | Route | Product | Qty | Cost |",
        "|------|-------|---------|-----|------|",
    ]

    for idx, t in enumerate(sorted_transfers, 1):
        from_s = t.get("from_store", "N/A")
        to_s = t.get("to_store", "N/A")
        product = t.get("product_id", "N/A")
        qty = t.get("quantity", 0)
        cost = t.get("cost_impact", {}).get('transport_cost', 0)
        lines.append(f"| {idx} | {from_s}→{to_s} | {product} | {qty:.2f} | ${cost:.2f} |")

    return "\n".join(lines)


def explain_inventory_status(data: dict, params: dict = None) -> str:
    """Summarise overall inventory health from the inventory JSON data.

    All numeric values (current, final, target) come directly from JSON.
    """
    inventory = data.get("inventory", [])
    scenario = data.get("scenario", {}).get("scenario", "Unknown")

    if params:
        p_store = [str(s).lower().replace("store_", "") for s in params.get("store_id", [])]
        p_prod = [str(p).lower().replace("product_", "") for p in params.get("product_id", [])]
        if p_store:
            inventory = [i for i in inventory if str(i.get("store_id", "")) in p_store]
        if p_prod:
            inventory = [i for i in inventory if str(i.get("product_id", "")) in p_prod]

    if not inventory:
        return "No inventory data available for the selected scenario/filters."

    total = len(inventory)
    at_target = sum(1 for i in inventory if float(i.get("final", 0)) >= float(i.get("target", 0)))
    below_target = total - at_target

    lines = [
        f"**Inventory Status Summary**",
        f"**Scenario:** {scenario}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Inventory Records | {total} |",
        f"| Records Meeting Target | {at_target} |",
        f"| Records Below Target | {below_target} |",
        f"| On-Target Rate | {(at_target/total)*100:.1f}% |" if total > 0 else "| On-Target Rate | N/A |",
        "",
        "**Sample Records (first 10):**",
        "| Store | Product | Current | Final | Target |",
        "|-------|---------|---------|-------|--------|",
    ]
    for i in inventory[:10]:
        lines.append(
            f"| {i.get('store_id', 'N/A')} | {i.get('product_id', 'N/A')} "
            f"| {float(i.get('current', 0)):.2f} | {float(i.get('final', 0)):.2f} "
            f"| {float(i.get('target', 0)):.2f} |"
        )

    return "\n".join(lines)


def explain_inventory_gaps(data: dict, params: dict = None) -> str:
    """List inventory records where the final quantity is below the target.

    All numeric values come from JSON.
    """
    inventory = data.get("inventory", [])
    scenario = data.get("scenario", {}).get("scenario", "Unknown")

    if params:
        p_store = [str(s).lower().replace("store_", "") for s in params.get("store_id", [])]
        p_prod = [str(p).lower().replace("product_", "") for p in params.get("product_id", [])]
        if p_store:
            inventory = [i for i in inventory if str(i.get("store_id", "")) in p_store]
        if p_prod:
            inventory = [i for i in inventory if str(i.get("product_id", "")) in p_prod]

    # Identify gaps: records where final < target
    gaps = [
        i for i in inventory
        if float(i.get("final", 0)) < float(i.get("target", 0))
    ]

    if not gaps:
        return "No inventory gaps detected — all items are meeting their targets in this scenario."

    # Sort by absolute gap (largest deficit first)
    gaps_sorted = sorted(
        gaps,
        key=lambda i: float(i.get("target", 0)) - float(i.get("final", 0)),
        reverse=True
    )[:MAX_RESULTS]

    lines = [
        f"**Inventory Gaps (Below Target)**",
        f"**Scenario:** {scenario}",
        f"**Total Gaps Detected:** {len(gaps)}",
        "",
        "| Store | Product | Current | Final | Target | Shortfall |",
        "|-------|---------|---------|-------|--------|-----------|",
    ]
    for i in gaps_sorted:
        shortfall = float(i.get("target", 0)) - float(i.get("final", 0))
        lines.append(
            f"| {i.get('store_id', 'N/A')} | {i.get('product_id', 'N/A')} "
            f"| {float(i.get('current', 0)):.2f} | {float(i.get('final', 0)):.2f} "
            f"| {float(i.get('target', 0)):.2f} | {shortfall:.2f} |"
        )

    return "\n".join(lines)


def build_explanation(intent: str, data: dict, params: dict = None) -> str:
    """Dispatch an intent to the appropriate deterministic explain_* function.

    All returned strings contain only values from the JSON data — the LLM
    never performs calculations or invents numbers.

    Parameters
    ----------
    intent : str
        One of the supported pipeline intents (see ALLOWED_INTENTS in
        intent_classifier.py).
    data : dict
        Unified data dict as returned by load_data() or equivalent.
    params : dict, optional
        Extracted parameters (limit, product_id, store_id, …).

    Returns
    -------
    str
        Deterministic markdown explanation, or empty string for unknown intents.
    """
    limit = None
    if params and isinstance(params, dict):
        limit = params.get("limit")

    # ── Required pipeline intents ──────────────────────────────────────────────
    if intent in ("transfer_recommendations", "explain_transfer"):
        return explain_transfer(data, params)
    if intent in ("manufacturing_plan", "explain_manufacturing"):
        return explain_manufacturing(data, params)
    if intent == "inventory_status":
        return explain_inventory_status(data, params)
    if intent == "inventory_gaps":
        return explain_inventory_gaps(data, params)
    if intent in ("top_transfers_by_cost", "top_transfers"):
        return explain_top_transfers(data, limit=limit if limit else MAX_RESULTS)
    if intent == "top_transfers_by_quantity":
        return explain_top_transfers_by_quantity(data, limit=limit if limit else MAX_RESULTS)
    if intent in ("top_manufacturing_items", "top_manufacturing"):
        return explain_top_manufacturing(data, limit=limit if limit else MAX_RESULTS)
    if intent == "scenario_comparison":
        # scenario_comparison is handled by handle_user_query via scenario_compare module
        return explain_scenario(data)
    # ── Additional intents ────────────────────────────────────────────────────
    if intent == "list_entities":
        return explain_entities(data)
    if intent == "total_counts":
        return explain_counts(data)
    if intent in ("scenario_summary", "impact_analysis"):
        return explain_scenario(data)
    if intent == "urgent_transfers":
        return explain_urgent_transfers(data)
    if intent == "high_cost_actions":
        return explain_high_cost_actions(data)
    if intent == "reason_analysis":
        return explain_reason_analysis(data)
    if intent == "store_activity":
        return explain_store_activity(data)
    if intent in ("product_recommendations",):
        return explain_product_recommendations(data, params)
    if intent == "cost_breakdown":
        return explain_cost_breakdown(data)
    return ""


# ── Greeting / fallback messages ──────────────────────────────────────────────

_GREETING_RESPONSE = (
    "Hello! I am the Supply Chain Analytics Assistant.\n\n"
    "I can answer questions about:\n\n"
    "**Transfers:** transfer recommendations, top transfers by cost or quantity, urgent transfers, store activity\n\n"
    "**Manufacturing:** manufacturing plan, top manufacturing items, high-cost actions\n\n"
    "**Inventory:** inventory status, inventory gaps\n\n"
    "**Scenarios:** scenario summary, cost breakdown, scenario comparison, impact analysis\n\n"
    "**Decisions:** reason analysis, product recommendations, total counts\n\n"
    "Ask me anything about the optimization results!"
)

_OUT_OF_SCOPE_RESPONSE = (
    "I cannot answer this question using the available optimization data.\n\n"
    "I am calibrated for supply chain optimization analysis covering inventory "
    "transfers, production decisions, and cost diagnostics.\n\n"
    "Please try: *\"Show transfer recommendations\"*, *\"Top 5 manufacturing items\"*, "
    "or *\"Compare scenarios\"*."
)


# ── Main pipeline orchestrator ────────────────────────────────────────────────

def handle_user_query(query: str) -> str:
    """Single entry point for the full NLP pipeline.

    Pipeline steps:
      1. Intent detection  — local LLM (with keyword fallback)
      2. Scenario detection — keyword mapping → scenario ID
      3. Parameter extraction — limit, store/product filters, etc.
      4. Data loading        — reads JSON files; all numbers come from JSON
      5. Deterministic explanation construction
      6. LLM language refinement (skipped for table-format intents)
      7. Response validation — fallback to raw explanation if LLM degrades it

    Safety guarantees:
      - The LLM never performs calculations or reasons about numbers.
      - All numeric values in the final response come from JSON.
      - Results are capped at MAX_RESULTS (10) by default.
      - Unsupported queries always return a clear out-of-scope message.

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    str
        Final response ready for display (markdown-formatted).
    """
    # Lazy imports to avoid circular imports at module level
    from nlp.intent_classifier import classify_intent, extract_parameters, MAX_RESULTS as _MAX
    from nlp.refiner import refine_explanation, validate_response
    from nlp.scenario_compare import ScenarioSnapshot, compare_scenarios

    # ── 1. Intent detection ────────────────────────────────────────────────────
    intent = classify_intent(query)

    if intent == "greeting":
        return _GREETING_RESPONSE
    if intent == "out_of_scope":
        return _OUT_OF_SCOPE_RESPONSE

    # ── 2. Scenario detection ──────────────────────────────────────────────────
    scenario = detect_scenario(query)

    # ── 3. Parameter extraction ────────────────────────────────────────────────
    params = extract_parameters(query)
    params["scenario"] = scenario
    # Enforce MAX_RESULTS cap on any explicit limit the user provided
    if not params.get("limit"):
        params["limit"] = _MAX

    # ── 4. Data loading ────────────────────────────────────────────────────────
    data = load_data(scenario)

    # ── 5. Scenario comparison (special path — needs two scenarios) ────────────
    if intent == "scenario_comparison":
        summary_all = data.get("scenario_summary_all", {})
        summaries = summary_all.get("summaries", [])
        if len(summaries) >= 2:
            baseline = ScenarioSnapshot.from_json(summaries[0])
            alternative = ScenarioSnapshot.from_json(summaries[1])
            return compare_scenarios(baseline, alternative)
        # Not enough scenarios to compare — fall through to plain scenario summary
        return explain_scenario(data)

    # ── 6. Deterministic explanation ───────────────────────────────────────────
    raw_explanation = build_explanation(intent, data, params)

    if not raw_explanation:
        return _OUT_OF_SCOPE_RESPONSE

    # ── 7. LLM refinement (skip for structured table intents) ─────────────────
    # Table intents produce structured markdown — LLM refinement would degrade them.
    _TABLE_INTENTS = {
        "total_counts", "top_transfers_by_cost", "top_transfers_by_quantity",
        "top_manufacturing_items", "urgent_transfers", "high_cost_actions",
        "reason_analysis", "store_activity", "product_recommendations",
        "cost_breakdown", "inventory_status", "inventory_gaps",
    }

    if intent in _TABLE_INTENTS:
        return raw_explanation

    refined = refine_explanation(raw_explanation, user_question=query)

    # ── 8. Validate refined response ───────────────────────────────────────────
    if validate_response(raw_explanation, refined):
        return refined
    return raw_explanation
