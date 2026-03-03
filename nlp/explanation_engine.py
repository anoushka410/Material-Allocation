def explain_transfer(data: dict, params: dict = None) -> str:
    transfer_data = data.get("transfers", {})
    scenario = transfer_data.get("scenario", "Unknown")
    transfers = transfer_data.get("transfers", [])
    
    if params:
        filtered = []
        p_prod = [x.lower() for x in params.get("product_id", [])]
        p_store = [x.lower() for x in params.get("store_id", [])]
        
        if p_prod or p_store:
            for t in transfers:
                product = t.get("product_id", "").lower()
                from_s = t.get("from_store", "").lower()
                to_s = t.get("to_store", "").lower()
                
                if (p_prod and product in p_prod) or \
                   (p_store and (from_s in p_store or to_s in p_store)):
                    filtered.append(t)
            transfers = filtered

    if not transfers:
        return "No transfers match the given specific product or store in this scenario."

    lines = [
        f"**Scenario:** {scenario}  ",
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

        reason_text = ", ".join(r.replace("_", " ").title() for r in reasons)

        lines.append(f"**Transfer {idx}: Product {product}**  ")
        lines.append(f"Route: Store {from_s} → Store {to_s}  ")
        lines.append(f"Quantity: **{qty:.2f} units**  ")
        lines.append(f"Reasons: {reason_text}  ")
        lines.append(f"Transport Cost: ${ci.get('transport_cost', 0):,.2f}  ")
        lines.append("")

    return "\n".join(lines).rstrip()


def explain_manufacturing(data: dict, params: dict = None) -> str:
    mfg_data = data.get("manufacturing", {})
    scenario = mfg_data.get("scenario", "Unknown")
    actions = mfg_data.get("manufacturing_actions", [])
    
    if params:
        filtered = []
        p_prod = [x.lower() for x in params.get("product_id", [])]
        
        if p_prod:
            for m in actions:
                product = m.get("product_id", "").lower()
                
                if product in p_prod:
                    filtered.append(m)
            actions = filtered
            
    if not actions:
        return "No manufacturing actions match the given specific product in this scenario."

    lines = [
        f"**Scenario:** {scenario}  ",
        f"**Manufacturing Decisions:** {len(actions)}",
        "",
    ]

    for idx, m in enumerate(actions, 1):
        product = m.get("product_id", "N/A")
        qty = m.get("manufacture_quantity", 0)
        reasons = m.get("reason_codes", [])
        ci = m.get("cost_impact", {})

        reason_text = ", ".join(r.replace("_", " ").title() for r in reasons)

        lines.append(f"**Decision {idx}: Product {product}**  ")
        lines.append(f"Quantity to Manufacture: **{qty:.2f} units**  ")
        lines.append(f"Reasons: {reason_text}  ")
        lines.append(f"Manufacturing Cost: ${ci.get('manufacturing_cost', 0):,.2f}  ")
        lines.append("")

    return "\n".join(lines).rstrip()


def explain_scenario(data: dict) -> str:
    scen_data = data.get("scenario", {})
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
    total_mfg_quantity = sum(m.get("manufacture_quantity", 0) for m in manufacturing)

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
        f"| Total Optimization Cost | ${optimized.get('total_cost', 0):,.2f} |",
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
    manufacturing = data.get("manufacturing", {}).get("manufacturing_actions", [])
    scenario = data.get("manufacturing", {}).get("scenario", "Unknown")

    # Sort by manufacturing cost
    sorted_mfg = sorted(manufacturing, key=lambda x: x['cost_impact'].get('manufacturing_cost', 0), reverse=True)[:limit]

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
        qty = m.get("manufacture_quantity", 0)
        cost = m['cost_impact'].get('manufacturing_cost', 0)
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
    top_transfers = sorted(transfers, key=lambda x: x['cost_impact'].get('transport_cost', 0), reverse=True)[:5]

    # Get top expensive manufacturing
    top_mfg = sorted(manufacturing, key=lambda x: x['cost_impact'].get('manufacturing_cost', 0), reverse=True)[:10]

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
        qty = m.get("manufacture_quantity", 0)
        cost = m['cost_impact'].get('manufacturing_cost', 0)
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
            cost = sum(t['cost_impact'].get('transport_cost', 0) for t in prod_transfers)
            lines.append(f"| {product} | Transfer | {len(prod_transfers)} | {qty:.2f} units, ${cost:.2f} |")

        if prod_mfg:
            qty = sum(m.get('manufacture_quantity', 0) for m in prod_mfg)
            cost = sum(m['cost_impact'].get('manufacturing_cost', 0) for m in prod_mfg)
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
