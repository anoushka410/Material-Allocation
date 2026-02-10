from reason_map import REASON_CODE_TEXT

def explain_transfer(rec):
    reasons = [REASON_CODE_TEXT[c] for c in rec["reason_codes"] if c in REASON_CODE_TEXT]
    reason_text = ", and ".join(reasons)
    return (
        f"{rec['quantity']} units of product {rec['product_id']} are recommended to be moved "
        f"from Store {rec['from_store']} to Store {rec['to_store']} because {reason_text}. "
        f"This reduces expected stockouts by "
        f"{rec['service_level_impact']['stockout_reduction_pct'] * 100:.0f}% "
        f"and results in a net cost change of ${rec['cost_impact']['net_cost_change']}."
    )
