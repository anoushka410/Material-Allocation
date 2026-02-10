from reason_map import REASON_CODE_TEXT

def explain_manufacturing(rec):
    reasons = [REASON_CODE_TEXT[c] for c in rec["reason_codes"] if c in REASON_CODE_TEXT]
    reason_text = ", and ".join(reasons)
    return (
        f"Manufacturing of {rec['manufacture_quantity']} units of product {rec['product_id']} "
        f"is triggered because {reason_text}. "
        f"The estimated manufacturing and distribution cost is "
        f"${rec['cost_impact']['total_manufacturing_cost']}."
    )
