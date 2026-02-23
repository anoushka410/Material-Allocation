def explain_transfer(data: dict, params: dict = None) -> str:
    transfer_data = data.get("transfers", {})
    scenario = transfer_data.get("scenario", "Unknown")
    transfers = transfer_data.get("transfers", [])
    
    if params:
        filtered = []
        p_tid = [x.lower() for x in params.get("transfer_id", [])]
        p_prod = [x.lower() for x in params.get("product_id", [])]
        p_store = [x.lower() for x in params.get("store_id", [])]
        
        if p_tid or p_prod or p_store:
            for t in transfers:
                tid = t.get("transfer_id", "").lower()
                product = t.get("product_id", "").lower()
                from_s = t.get("from_store", "").lower()
                to_s = t.get("to_store", "").lower()
                
                if (p_tid and tid in p_tid) or \
                   (p_prod and product in p_prod) or \
                   (p_store and (from_s in p_store or to_s in p_store)):
                    filtered.append(t)
            transfers = filtered

    if not transfers:
        return "No transfers match the given specific ID, product, or store in this scenario."

    lines = [
        f"**Scenario:** {scenario}  ",
        f"**Matching transfers:** {len(transfers)}",
        "",
    ]

    for t in transfers:
        tid = t.get("transfer_id", "N/A")
        from_s = t.get("from_store", "N/A")
        to_s = t.get("to_store", "N/A")
        product = t.get("product_id", "N/A")
        qty = t.get("quantity", 0)
        reasons = t.get("reason_codes", [])
        ci = t.get("cost_impact", {})
        sl = t.get("service_level_impact", {})

        reason_text = ", ".join(r.replace("_", " ") for r in reasons)

        lines.append(f"---")
        lines.append(f"**{tid} — {product}**  ")
        lines.append(f"Route: {from_s} → {to_s}  ")
        lines.append(f"Quantity: **{qty} units**  ")
        lines.append(f"Reasons: {reason_text}  ")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Transport cost | ${ci.get('transport_cost', 0):,.0f} |")
        lines.append(f"| Holding cost change | ${ci.get('holding_cost_change', 0):,.0f} |")
        lines.append(f"| Stockout penalty avoided | ${ci.get('stockout_penalty_avoided', 0):,.0f} |")
        lines.append(f"| **Net cost change** | **${ci.get('net_cost_change', 0):,.0f}** |")
        lines.append(f"| Stockout before | {sl.get('baseline_stockout_units', 0)} units |")
        lines.append(f"| Stockout after | {sl.get('post_transfer_stockout_units', 0)} units |")
        lines.append(f"| Stockout reduction | {sl.get('stockout_reduction_pct', 0) * 100:.1f}% |")
        lines.append("")

    return "\n".join(lines).rstrip()


def explain_manufacturing(data: dict, params: dict = None) -> str:
    mfg_data = data.get("manufacturing", {})
    scenario = mfg_data.get("scenario", "Unknown")
    actions = mfg_data.get("manufacturing_actions", [])
    
    if params:
        filtered = []
        p_mid = [x.lower() for x in params.get("manufacturing_id", [])]
        p_prod = [x.lower() for x in params.get("product_id", [])]
        
        if p_mid or p_prod:
            for m in actions:
                mid = m.get("manufacturing_id", "").lower()
                product = m.get("product_id", "").lower()
                
                if (p_mid and mid in p_mid) or (p_prod and product in p_prod):
                    filtered.append(m)
            actions = filtered
            
    if not actions:
        return "No manufacturing actions match the given specific ID or product in this scenario."

    lines = [
        f"**Scenario:** {scenario}  ",
        f"**Matching manufacturing actions:** {len(actions)}",
        "",
    ]

    for m in actions:
        mid = m.get("manufacturing_id", "N/A")
        product = m.get("product_id", "N/A")
        qty = m.get("manufacture_quantity", 0)
        reasons = m.get("reason_codes", [])
        ci = m.get("cost_impact", {})

        reason_text = ", ".join(r.replace("_", " ") for r in reasons)

        lines.append(f"---")
        lines.append(f"**{mid} — {product}**  ")
        lines.append(f"Quantity to manufacture: **{qty} units**  ")
        lines.append(f"Reasons: {reason_text}  ")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Manufacturing cost | ${ci.get('manufacturing_cost', 0):,.0f} |")
        lines.append(f"| Distribution cost | ${ci.get('distribution_cost', 0):,.0f} |")
        lines.append(f"| **Total cost** | **${ci.get('total_manufacturing_cost', 0):,.0f}** |")
        lines.append("")

    return "\n".join(lines).rstrip()


def explain_scenario(data: dict) -> str:
    scen_data = data.get("scenario", {})
    scenario = scen_data.get("scenario", "Unknown")
    baseline = scen_data.get("baseline", {})
    optimized = scen_data.get("optimized", {})
    delta = scen_data.get("delta", {})

    lines = [
        f"**Scenario:** {scenario}",
        "",
        "| | Baseline | Optimized |",
        "|---|---|---|",
        f"| Total cost | ${baseline.get('total_cost', 0):,.0f} | ${optimized.get('total_cost', 0):,.0f} |",
        f"| Total stockouts | {baseline.get('total_stockouts', 0)} units | {optimized.get('total_stockouts', 0)} units |",
        "",
        "**Improvement after optimization:**",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Cost change | ${delta.get('cost_change', 0):,.0f} |",
        f"| Stockout units reduced | {delta.get('stockout_reduction_units', 0)} |",
        f"| Stockout reduction | {delta.get('stockout_reduction_pct', 0) * 100:.1f}% |",
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
        "**Products involved/at risk**:",
    ]
    if products:
        lines.extend(f"- {p}" for p in sorted(products))
    else:
        lines.append("- None")

    lines.append("")
    lines.append("**Stores involved**:")
    if stores:
        lines.extend(f"- {s}" for s in sorted(stores))
    else:
        lines.append("- None")
    
    return "\n".join(lines)


def build_explanation(intent: str, data: dict, params: dict = None) -> str:
    if intent == "explain_transfer":
        return explain_transfer(data, params)
    if intent == "explain_manufacturing":
        return explain_manufacturing(data, params)
    if intent == "list_entities":
        return explain_entities(data)
    if intent in ("scenario_summary", "impact_analysis"):
        return explain_scenario(data)
    return ""
