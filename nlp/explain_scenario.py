def explain_scenario(rec):
    return (
        f"Under the {rec['scenario']} scenario, risk-aware optimization reduced total "
        f"stockouts from {rec['baseline']['total_stockouts']} to "
        f"{rec['optimized']['total_stockouts']} while lowering total cost by "
        f"${rec['delta']['cost_change']}. "
        f"This represents a stockout reduction of "
        f"{rec['delta']['stockout_reduction_pct'] * 100:.0f}%."
    )
