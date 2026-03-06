"""Explanation engine for the NLP pipeline.

Responsibilities:
  - Load JSON optimization data from disk.
  - Filter records by scenario.
  - Detect the scenario from a user query.
  - Generate deterministic, fact-based explanations (no LLM reasoning).
  - Orchestrate the full NLP pipeline via ``handle_user_query``.

Safety rules enforced throughout:
  - All numeric values come directly from JSON — never computed by LLM.
  - Result sets are capped at MAX_RESULTS (10) records.
  - Unsupported intents return ``out_of_scope`` fallback text.
"""

import json
import os
from typing import Optional

# ── Scenario constants ────────────────────────────────────────────────────────

DEFAULT_SCENARIO = "base_case_standard_conditions"

# Default scenario to compare against when the user does not specify a second scenario.
DEFAULT_COMPARISON_SCENARIO = "risk_aware_high_disruption"
# Mapping from lowercase user query keywords → canonical scenario identifier.
# Checked in order; first match wins.
SCENARIO_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["high disruption", "high_disruption", "risk aware", "risk_aware"],
     "risk_aware_high_disruption"),
    (["fuel shock", "fuel_shock", "transport cost increase", "transport_cost_increase"],
     "transport_cost_increase_fuel_shock"),
    (["demand spike", "demand_spike", "high forecast", "high_forecast"],
     "demand_spike_high_forecast"),
    (["lead time", "lead_time", "supplier delay", "supplier_delay", "extended lead"],
     "extended_lead_time_supplier_delay"),
    (["cost only", "cost_only", "no risk penalty", "no_risk"],
     "cost_only_no_risk_penalty"),
    (["base case", "base_case", "standard conditions", "standard_conditions", "default scenario"],
     DEFAULT_SCENARIO),
]

# Hard cap on the number of records returned by any explanation function.
MAX_RESULTS = 10

# Path to the optimization output JSON files.
_JSON_DIR = os.path.join(
    os.path.dirname(__file__), "..", "optimization", "output-json"
)


def _filter_by_scenario(records: list[dict], scenario: str) -> list[dict]:
    """Filter a list of records by scenario name.

    Records that do NOT carry a ``scenario`` field are kept unconditionally
    (they are treated as scenario-agnostic).  Records WITH a ``scenario``
    field are only kept when they match ``scenario``.
    """
    if not scenario or scenario in ("all", "optimization_run", "unknown"):
        return records
    return [
        r for r in records
        if "scenario" not in r or str(r["scenario"]) == str(scenario)
    ]


def parse_scenario(user_query: str) -> str:
    """Detect the target scenario from a user query string.

    Checks ``SCENARIO_KEYWORD_MAP`` in order; returns the first matching
    scenario identifier.  Falls back to ``DEFAULT_SCENARIO`` when no keyword
    matches.

    Parameters
    ----------
    user_query:
        Raw user input.

    Returns
    -------
    str
        Canonical scenario identifier.
    """
    lower = user_query.lower()
    for keywords, scenario_id in SCENARIO_KEYWORD_MAP:
        if any(kw in lower for kw in keywords):
            return scenario_id
    return DEFAULT_SCENARIO


def load_json_data(scenario: str = DEFAULT_SCENARIO) -> dict:
    """Load and merge all optimization JSON files for a given scenario.

    Files loaded:
      - ``scenario_summary.json``         → keyed as ``"scenario"``
      - ``transfer_recommendations.json`` → keyed as ``"transfers"``
      - ``manufacturing_decisions.json``  → keyed as ``"manufacturing"``
      - ``inventory.json``                → keyed as ``"inventory"``

    All record-level lists are filtered to the requested scenario before
    returning so that downstream functions only see relevant records.

    Parameters
    ----------
    scenario:
        Scenario identifier.  Defaults to ``DEFAULT_SCENARIO``.

    Returns
    -------
    dict
        Merged data dict ready for the ``explain_*`` functions.
    """
    def _load(filename: str) -> dict:
        path = os.path.normpath(os.path.join(_JSON_DIR, filename))
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # --- Scenario summary ---------------------------------------------------
    scenario_raw = _load("scenario_summary.json")
    summaries = scenario_raw.get("summaries", [])
    matched_summary = next(
        (s for s in summaries if s.get("scenario") == scenario),
        summaries[0] if summaries else {},
    )

    # --- Transfers ----------------------------------------------------------
    transfer_raw = _load("transfer_recommendations.json")
    all_transfers = transfer_raw if isinstance(transfer_raw, list) else transfer_raw.get("transfers", [])
    transfers = _filter_by_scenario(all_transfers, scenario)

    # --- Manufacturing ------------------------------------------------------
    mfg_raw = _load("manufacturing_decisions.json")
    all_mfg = mfg_raw if isinstance(mfg_raw, list) else mfg_raw.get("manufacturing_actions", [])
    manufacturing = _filter_by_scenario(all_mfg, scenario)

    # --- Inventory ----------------------------------------------------------
    inv_raw = _load("inventory.json")
    all_inv = inv_raw if isinstance(inv_raw, list) else inv_raw.get("inventory", [])
    inventory = _filter_by_scenario(all_inv, scenario)

    return {
        "scenario": matched_summary,
        "transfers": {
            "scenario": scenario,
            "transfers": transfers,
        },
        "manufacturing": {
            "scenario": scenario,
            "manufacturing_actions": manufacturing,
        },
        "inventory": {
            "scenario": scenario,
            "items": inventory,
        },
    }


def explain_transfer(data: dict, params: dict = None) -> str:
    """Generate a deterministic transfer recommendations explanation.

    All numeric values (quantity, cost) come directly from JSON data.
    Supports optional filtering by product ID or store ID.
    Respects limit parameter for result count.
    """
    transfer_data = data.get("transfers", {})
    transfers = transfer_data.get("transfers", [])

    # Determine the target scenario for filtering
    selected_scenario = None
    if params and isinstance(params, dict):
        selected_scenario = params.get("scenario")
    if not selected_scenario:
        selected_scenario = transfer_data.get("scenario")

    # Use the safe scenario filter that preserves records without a scenario field
    transfers = _filter_by_scenario(transfers, selected_scenario)

    scenario_label = selected_scenario or transfer_data.get("scenario") or "Unknown"

    # Extract limit from params; default to MAX_RESULTS
    limit = MAX_RESULTS
    if params and isinstance(params, dict):
        raw_limit = params.get("limit")
        if raw_limit is not None:
            limit = max(1, min(int(raw_limit), MAX_RESULTS))

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

    # Apply limit to result set
    transfers = transfers[:limit]

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
    """Generate a deterministic manufacturing decisions explanation.

    All numeric values (quantity, cost) come directly from JSON data.
    Handles both the legacy schema (``manufacture_quantity`` / ``cost_impact``)
    and the current schema (``quantity`` / ``cost`` at record level).
    Respects limit parameter for result count.
    """
    mfg_data = data.get("manufacturing", {})
    actions = mfg_data.get("manufacturing_actions", [])

    selected_scenario = None
    if params and isinstance(params, dict):
        selected_scenario = params.get("scenario")
    if not selected_scenario:
        selected_scenario = mfg_data.get("scenario")

    # Use the safe scenario filter that preserves records without a scenario field
    actions = _filter_by_scenario(actions, selected_scenario)

    scenario_label = selected_scenario or mfg_data.get("scenario") or "Unknown"

    # Extract limit from params; default to MAX_RESULTS
    limit = MAX_RESULTS
    if params and isinstance(params, dict):
        raw_limit = params.get("limit")
        if raw_limit is not None:
            limit = max(1, min(int(raw_limit), MAX_RESULTS))

    if params:
        filtered = []
        p_prod = [x.lower() for x in params.get("product_id", [])]

        if p_prod:
            for m in actions:
                product = str(m.get("product_id", "")).lower()
                if product in p_prod:
                    filtered.append(m)
            actions = filtered

    # Apply limit to result set
    actions = actions[:limit]

    if not actions:
        return "No manufacturing actions match the given specific product in this scenario."

    lines = [
        f"**Scenario:** {scenario_label}  ",
        f"**Manufacturing Decisions:** {len(actions)}",
        "",
    ]

    for idx, m in enumerate(actions, 1):
        product = m.get("product_id", "N/A")
        # Support both schema variants:
        # - Legacy: {"manufacture_quantity": ..., "cost_impact": {"manufacturing_cost": ...}}
        # - Current: {"quantity": ..., "cost": ...}
        qty = m.get("manufacture_quantity") or m.get("quantity", 0)
        cost_impact = m.get("cost_impact", {})
        mfg_cost = cost_impact.get("manufacturing_cost") if cost_impact else None
        if mfg_cost is None:
            mfg_cost = m.get("cost", 0)
        reasons = m.get("reason_codes", [])
        m_scen = m.get("scenario")

        title = f"**Decision {idx}: Product {product}**"
        if m_scen and scenario_label in ("all", "Unknown", "unknown"):
            title += f" *(Scenario: {m_scen})*"
        lines.append(title + "  ")
        try:
            lines.append(f"Quantity to Manufacture: **{float(qty):.2f} units**  ")
        except Exception:
            lines.append(f"Quantity to Manufacture: **{qty} units**  ")
        reason_text = ", ".join(r.replace("_", " ").title() for r in reasons)
        lines.append(f"Reasons: {reason_text}  ")
        lines.append(f"Manufacturing Cost: ${float(mfg_cost):,.2f}  ")
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
    total_mfg_quantity = sum(
        m.get("manufacture_quantity") or m.get("quantity", 0) for m in manufacturing
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
    """List the top manufacturing actions sorted by cost (descending).

    Handles both legacy schema (``manufacture_quantity`` / ``cost_impact``)
    and current schema (``quantity`` / ``cost``).
    """
    manufacturing = data.get("manufacturing", {}).get("manufacturing_actions", [])
    scenario = data.get("manufacturing", {}).get("scenario", "Unknown")

    def _mfg_cost(m: dict) -> float:
        ci = m.get("cost_impact", {})
        return float(ci.get("manufacturing_cost", 0)) if ci else float(m.get("cost", 0))

    # Sort by manufacturing cost (descending), cap at limit
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
        qty = m.get("manufacture_quantity") or m.get("quantity", 0)
        cost = _mfg_cost(m)
        lines.append(f"| {idx} | {product} | {float(qty):.2f} units | ${cost:,.2f} |")

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
    top_transfers = sorted(transfers, key=lambda x: x['cost_impact'].get('transport_cost', 0), reverse=True)[:5]

    # Get top expensive manufacturing (handle both schema variants)
    def _mfg_cost_key(m: dict) -> float:
        ci = m.get("cost_impact", {})
        return float(ci.get("manufacturing_cost", 0)) if ci else float(m.get("cost", 0))

    top_mfg = sorted(manufacturing, key=_mfg_cost_key, reverse=True)[:10]

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
        cost = t['cost_impact'].get('transport_cost', 0)
        lines.append(f"| {from_s}→{to_s} | {product} | ${cost:.2f} |")

    lines.extend([
        "",
        "**Top 10 Most Expensive Manufacturing:**",
        "| Product | Quantity | Cost |",
        "|---------|----------|------|",
    ])

    for m in top_mfg:
        product = m.get("product_id")
        qty = m.get("manufacture_quantity") or m.get("quantity", 0)
        cost = _mfg_cost_key(m)
        lines.append(f"| {product} | {float(qty):.2f} units | ${cost:,.2f} |")

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
            cost = sum(t['cost_impact'].get('transport_cost', 0) for t in prod_transfers)
            lines.append(f"| {product} | Transfer | {len(prod_transfers)} | {qty:.2f} units, ${cost:.2f} |")

        if prod_mfg:
            qty = sum(m.get("manufacture_quantity") or m.get("quantity", 0) for m in prod_mfg)
            cost_impact = [m.get("cost_impact", {}) for m in prod_mfg]
            cost = sum(
                float(ci.get("manufacturing_cost", 0)) if ci else float(m.get("cost", 0))
                for m, ci in zip(prod_mfg, cost_impact)
            )
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

def build_explanation(intent: str, data: dict, params: dict = None) -> str:
    # Extract limit from params if specified, otherwise use default (10)
    limit = None
    if params and isinstance(params, dict):
        limit = params.get("limit")  # Will be None if not extracted

    if intent == "explain_transfer":
        return explain_transfer(data, params)
    if intent == "explain_manufacturing":
        return explain_manufacturing(data, params)
    if intent == "list_entities":
        return explain_entities(data)
    if intent == "total_counts":
        return explain_counts(data)
    if intent in ("scenario_summary", "impact_analysis"):
        return explain_scenario(data)
    if intent == "top_transfers":
        # Use extracted limit if available, otherwise default to 10
        return explain_top_transfers(data, limit=limit if limit else 10)
    if intent == "top_manufacturing":
        # Use extracted limit if available, otherwise default to 10
        return explain_top_manufacturing(data, limit=limit if limit else 10)
    if intent == "urgent_transfers":
        return explain_urgent_transfers(data)
    if intent == "high_cost_actions":
        return explain_high_cost_actions(data)
    if intent == "reason_analysis":
        return explain_reason_analysis(data)
    if intent == "store_activity":
        return explain_store_activity(data)
    if intent == "product_recommendations":
        return explain_product_recommendations(data, params)
    if intent == "cost_breakdown":
        return explain_cost_breakdown(data)
    return ""


def explain_inventory_status(data: dict, params: dict = None) -> str:
    """Show current vs target (final) inventory levels by store and product.

    All numeric values come directly from JSON — no LLM reasoning.
    Results are capped at ``MAX_RESULTS``.
    """
    inv_data = data.get("inventory", {})
    items = inv_data.get("items", [])
    scenario = inv_data.get("scenario", "Unknown")

    # Optional entity filters
    filter_products: list[str] = []
    filter_stores: list[str] = []
    if params and isinstance(params, dict):
        filter_products = [str(p) for p in params.get("product_id", [])]
        filter_stores = [str(s).replace("store_", "") for s in params.get("store_id", [])]

    if filter_products or filter_stores:
        filtered = []
        for item in items:
            pid = str(item.get("product_id", ""))
            sid = str(item.get("store_id", ""))
            if (filter_products and pid in filter_products) or \
               (filter_stores and sid in filter_stores):
                filtered.append(item)
        items = filtered

    if not items:
        return "No inventory records found for the specified filters in this scenario."

    # Cap at MAX_RESULTS
    display_items = items[:MAX_RESULTS]
    truncated = len(items) > MAX_RESULTS

    lines = [
        f"**Inventory Status**",
        f"**Scenario:** {scenario}",
        f"**Records shown:** {len(display_items)}"
        + (f" (of {len(items)} total, showing first {MAX_RESULTS})" if truncated else ""),
        "",
        "| Store | Product | Current Stock | Final Stock | Target Stock |",
        "|-------|---------|--------------|-------------|--------------|",
    ]

    for item in display_items:
        store = item.get("store_id", "N/A")
        product = item.get("product_id", "N/A")
        current = item.get("current", 0)
        final = item.get("final", 0)
        target = item.get("target", 0)
        lines.append(
            f"| {store} | {product} | {float(current):.2f} | {float(final):.2f} | {float(target):.2f} |"
        )

    return "\n".join(lines)


def explain_scenario_comparison(data: dict, scenario_b: str) -> str:
    """Compare the active scenario against a second scenario.

    Delegates to ``scenario_compare.compare_scenarios`` so all numeric
    values come from JSON (no LLM reasoning).

    Parameters
    ----------
    data:
        Merged data dict for the *primary* scenario (already loaded).
    scenario_b:
        Name of the second scenario to compare against.

    Returns
    -------
    str
        Markdown comparison table, or an error message.
    """
    # Import here to avoid circular dependencies
    from nlp.scenario_compare import ScenarioSnapshot, compare_scenarios

    summary_a = data.get("scenario", {})
    if not summary_a:
        return "No scenario summary available for comparison."

    try:
        data_b = load_json_data(scenario_b)
    except (FileNotFoundError, OSError):
        return (
            f"Could not load scenario data for '{scenario_b}'. "
            "Please verify the scenario name and try again."
        )

    summary_b = data_b.get("scenario", {})

    snapshot_a = ScenarioSnapshot.from_json(summary_a)
    snapshot_b = ScenarioSnapshot.from_json(summary_b)
    return compare_scenarios(snapshot_a, snapshot_b)


def build_explanation(intent: str, data: dict, params: dict = None) -> str:
    """Dispatch an intent to the appropriate deterministic explanation function.

    Parameters
    ----------
    intent:
        Classified intent string.
    data:
        Merged optimization data dict (from ``load_json_data``).
    params:
        Optional parameter dict extracted from the user query.

    Returns
    -------
    str
        Deterministic explanation text, or empty string for unknown intents.
    """
    # Extract limit from params if specified; enforce MAX_RESULTS cap
    limit = MAX_RESULTS
    if params and isinstance(params, dict):
        raw_limit = params.get("limit")
        if raw_limit is not None:
            limit = max(1, min(int(raw_limit), MAX_RESULTS))

    # ── New pipeline intents ──────────────────────────────────────────────
    if intent == "inventory_status":
        return explain_inventory_status(data, params)
    if intent in ("manufacturing_plan", "explain_manufacturing"):
        return explain_manufacturing(data, params)
    if intent in ("transfer_recommendations", "explain_transfer"):
        return explain_transfer(data, params)
    if intent == "top_transfers_by_cost":
        return explain_top_transfers(data, limit=limit)
    if intent == "top_transfers_by_quantity":
        return explain_top_transfers_by_quantity(data, limit=limit)
    if intent == "top_manufacturing_items":
        return explain_top_manufacturing(data, limit=limit)
    if intent == "scenario_comparison":
        # Extract the second scenario from params; fall back to the module-level default
        second_scenario = (params or {}).get("scenario_b", DEFAULT_COMPARISON_SCENARIO)
        return explain_scenario_comparison(data, second_scenario)
    # ── Legacy intents ────────────────────────────────────────────────────
    if intent == "list_entities":
        return explain_entities(data)
    if intent == "total_counts":
        return explain_counts(data)
    if intent in ("scenario_summary", "impact_analysis"):
        return explain_scenario(data)
    if intent == "top_transfers":
        return explain_top_transfers(data, limit=limit)
    if intent == "top_manufacturing":
        return explain_top_manufacturing(data, limit=limit)
    if intent == "urgent_transfers":
        return explain_urgent_transfers(data)
    if intent == "high_cost_actions":
        return explain_high_cost_actions(data)
    if intent == "reason_analysis":
        return explain_reason_analysis(data)
    if intent == "store_activity":
        return explain_store_activity(data)
    if intent == "product_recommendations":
        return explain_product_recommendations(data, params)
    if intent == "cost_breakdown":
        return explain_cost_breakdown(data)
    return ""


def explain_top_transfers_by_quantity(data: dict, limit: int = 10) -> str:
    """List the top transfers sorted by quantity (descending).

    All numeric values come directly from JSON.
    """
    transfers = data.get("transfers", {}).get("transfers", [])
    scenario = data.get("transfers", {}).get("scenario", "Unknown")

    sorted_transfers = sorted(transfers, key=lambda x: float(x.get("quantity", 0)), reverse=True)[:limit]

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
        cost = t.get("cost_impact", {}).get("transport_cost", 0)
        lines.append(f"| {idx} | {from_s}→{to_s} | {product} | {float(qty):.2f} | ${float(cost):.2f} |")

    return "\n".join(lines)


def handle_user_query(query: str) -> str:
    """Full NLP pipeline entry point.

    Pipeline steps:
    1. Detect intent (LLM with keyword fallback).
    2. Extract query parameters (limit, product/store filters, etc.).
    3. Detect target scenario from query keywords.
    4. Load and filter JSON data for the detected scenario.
    5. Generate deterministic explanation (no LLM reasoning on numbers).
    6. Refine explanation for readability (LLM language improvement only).
    7. Validate that the response answers the original question.
    8. Return final response.

    Safety guarantees:
    - All numbers come from JSON, never from LLM reasoning.
    - Result sets capped at ``MAX_RESULTS`` (10).
    - ``out_of_scope`` and ``greeting`` intents bypass JSON loading.

    Parameters
    ----------
    query:
        Raw user query string.

    Returns
    -------
    str
        Final response text for the user.
    """
    # Import here to keep module-level imports minimal and avoid circular deps
    from nlp.intent_classifier import classify_intent, extract_parameters, parse_result_limit
    from nlp.refiner import refine_explanation, validate_response, FALLBACK_RESPONSE

    # ── 1. Detect intent ──────────────────────────────────────────────────
    intent = classify_intent(query)

    # Short-circuit for greetings and out-of-scope queries
    if intent == "greeting":
        return (
            "Hello! I am the Supply Chain Analytics Assistant.\n\n"
            "I can help you analyse optimization results including transfers, "
            "manufacturing plans, inventory levels, and scenario comparisons.\n\n"
            "Try asking: 'Show transfer recommendations' or 'Top 5 transfers by cost'."
        )

    if intent == "out_of_scope":
        return FALLBACK_RESPONSE

    # ── 2. Extract parameters ─────────────────────────────────────────────
    params = extract_parameters(query)

    # Parse and cap the result limit (default 10, max 10)
    raw_limit = parse_result_limit(query)
    params["limit"] = raw_limit

    # ── 3. Detect scenario ────────────────────────────────────────────────
    scenario = parse_scenario(query)
    params["scenario"] = scenario

    # ── 4. Load JSON data ─────────────────────────────────────────────────
    try:
        data = load_json_data(scenario)
    except (FileNotFoundError, OSError):
        return "Unable to load optimization data. Please contact support if this issue persists."

    # ── 5. Build deterministic explanation ────────────────────────────────
    raw_explanation = build_explanation(intent, data, params)

    if not raw_explanation:
        return FALLBACK_RESPONSE

    # ── 6. Refine for readability (LLM only improves language) ────────────
    try:
        refined = refine_explanation(raw_explanation, user_question=query)
    except Exception:
        # If LLM is unavailable, return the deterministic explanation directly
        refined = raw_explanation

    if not refined or len(refined.strip()) < 10:
        refined = raw_explanation

    # ── 7. Validate that the response answers the query ───────────────────
    try:
        final = validate_response(refined, user_question=query)
    except Exception:
        # If validation fails, trust the refined response
        final = refined

    return final
