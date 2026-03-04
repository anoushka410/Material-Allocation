"""
Scenario Comparison and What-If Narrative Generator.

Generates natural-language contrastive narratives comparing two optimization
scenarios (e.g. baseline vs. service-first) and what-if sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class ScenarioSnapshot:
    """Lightweight representation of an optimization scenario."""
    name: str
    total_cost: float
    manufacturing_cost: float
    transfer_cost: float
    holding_cost: float
    total_transfers: int
    manufacturing_units: float
    transfer_units: float

    @classmethod
    def from_json(cls, scenario_json: dict) -> "ScenarioSnapshot":
        """Build from the project's scenario_summary.json format."""
        optimized = scenario_json.get("optimized", {})
        breakdown = scenario_json.get("cost_breakdown", {})
        return cls(
            name=scenario_json.get("scenario", "unknown"),
            total_cost=optimized.get("total_cost", 0.0),
            manufacturing_cost=breakdown.get("manufacturing_cost", 0.0),
            transfer_cost=breakdown.get("transfer_cost", 0.0),
            holding_cost=breakdown.get("holding_cost", 0.0),
            total_transfers=optimized.get("total_transfers", 0),
            manufacturing_units=optimized.get("manufacturing_units", 0.0),
            transfer_units=optimized.get("transfer_units", 0.0),
        )


# ── Narrative helpers ─────────────────────────────────────────────────────────

def _fmt_cost(v: float) -> str:
    return f"${v:,.2f}"


def _pct_change(a: float, b: float) -> str:
    if a == 0:
        return "N/A"
    delta = (b - a) / abs(a) * 100
    arrow = "▲" if delta > 0 else "▼"
    return f"{arrow} {abs(delta):.1f}%"


def _direction(a: float, b: float, good_direction: str = "lower") -> str:
    """Return 'improved', 'worsened', or 'unchanged' relative to good_direction."""
    diff = b - a
    if abs(diff) < 1e-6:
        return "unchanged"
    if good_direction == "lower":
        return "improved" if diff < 0 else "worsened"
    return "improved" if diff > 0 else "worsened"


# ── Comparison ───────────────────────────────────────────────────────────────

def compare_scenarios(
    baseline: ScenarioSnapshot,
    alternative: ScenarioSnapshot,
) -> str:
    """
    Generate a contrastive markdown narrative comparing two scenarios.

    Parameters
    ----------
    baseline    : the reference scenario (e.g., 'cost_minimization')
    alternative : the scenario to compare against (e.g., 'service_first')

    Returns
    -------
    Markdown-formatted string with tables and bullet insights.
    """
    lines = [
        f"## Scenario Comparison: {baseline.name} vs. {alternative.name}",
        "",
        "### Cost Summary",
        "",
        "| Metric | Baseline | Alternative | Change |",
        "|--------|----------|-------------|--------|",
        f"| Total Cost | {_fmt_cost(baseline.total_cost)} | {_fmt_cost(alternative.total_cost)} | {_pct_change(baseline.total_cost, alternative.total_cost)} |",
        f"| Manufacturing | {_fmt_cost(baseline.manufacturing_cost)} | {_fmt_cost(alternative.manufacturing_cost)} | {_pct_change(baseline.manufacturing_cost, alternative.manufacturing_cost)} |",
        f"| Transfer/Logistics | {_fmt_cost(baseline.transfer_cost)} | {_fmt_cost(alternative.transfer_cost)} | {_pct_change(baseline.transfer_cost, alternative.transfer_cost)} |",
        f"| Holding/Inventory | {_fmt_cost(baseline.holding_cost)} | {_fmt_cost(alternative.holding_cost)} | {_pct_change(baseline.holding_cost, alternative.holding_cost)} |",
        "",
        "### Volume Summary",
        "",
        "| Metric | Baseline | Alternative | Change |",
        "|--------|----------|-------------|--------|",
        f"| Total Transfers | {baseline.total_transfers} | {alternative.total_transfers} | {_pct_change(baseline.total_transfers, alternative.total_transfers)} |",
        f"| Manufacturing Units | {baseline.manufacturing_units:,.1f} | {alternative.manufacturing_units:,.1f} | {_pct_change(baseline.manufacturing_units, alternative.manufacturing_units)} |",
        f"| Transfer Units | {baseline.transfer_units:,.1f} | {alternative.transfer_units:,.1f} | {_pct_change(baseline.transfer_units, alternative.transfer_units)} |",
        "",
        "### Key Insights",
        "",
    ]

    cost_dir = _direction(baseline.total_cost, alternative.total_cost)
    cost_delta = alternative.total_cost - baseline.total_cost
    lines.append(
        f"- **Total cost** {cost_dir}: {_fmt_cost(abs(cost_delta))} "
        f"{'more' if cost_delta > 0 else 'less'} than baseline "
        f"({_pct_change(baseline.total_cost, alternative.total_cost)})."
    )

    mfg_delta = alternative.manufacturing_cost - baseline.manufacturing_cost
    if abs(mfg_delta) > 1.0:
        lines.append(
            f"- **Manufacturing cost** {'increased' if mfg_delta > 0 else 'decreased'} "
            f"by {_fmt_cost(abs(mfg_delta))}, suggesting "
            f"{'more local production' if mfg_delta > 0 else 'greater reliance on inter-store transfers'}."
        )

    transfer_delta = alternative.transfer_units - baseline.transfer_units
    if abs(transfer_delta) > 0.1:
        lines.append(
            f"- **Transfer volume** {'rose' if transfer_delta > 0 else 'fell'} "
            f"by {abs(transfer_delta):,.1f} units — "
            f"{'higher logistics activity implies better demand responsiveness' if transfer_delta > 0 else 'reduced logistics may indicate tighter inventory positioning'}."
        )

    holding_delta = alternative.holding_cost - baseline.holding_cost
    if abs(holding_delta) > 1.0:
        lines.append(
            f"- **Holding cost** {'increased' if holding_delta > 0 else 'decreased'} "
            f"by {_fmt_cost(abs(holding_delta))}, "
            f"{'indicating higher safety-stock buffers' if holding_delta > 0 else 'consistent with leaner inventory positioning'}."
        )

    return "\n".join(lines)


# ── What-if sensitivity ───────────────────────────────────────────────────────

def sensitivity_analysis_text(
    baseline: ScenarioSnapshot,
    parameter: str,
    values: list[float],
    simulated_costs: list[float],
) -> str:
    """
    Generate a sensitivity-analysis narrative.

    Parameters
    ----------
    baseline        : base scenario
    parameter       : name of the parameter being varied (e.g., 'demand_cv')
    values          : list of parameter values tested
    simulated_costs : total cost for each parameter value

    Returns
    -------
    Markdown string.
    """
    if len(values) != len(simulated_costs):
        raise ValueError("values and simulated_costs must have the same length.")

    lines = [
        f"## Sensitivity Analysis: {parameter}",
        "",
        f"Baseline total cost: **{_fmt_cost(baseline.total_cost)}**",
        "",
        f"| {parameter} | Total Cost | Δ vs Baseline | Δ% |",
        f"|{'--'*len(parameter)}-|------------|---------------|-----|",
    ]

    for v, c in zip(values, simulated_costs):
        delta = c - baseline.total_cost
        dpct = _pct_change(baseline.total_cost, c)
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {v} | {_fmt_cost(c)} | {sign}{_fmt_cost(delta)} | {dpct} |")

    lines.append("")
    # Find the inflection
    costs_arr = simulated_costs
    min_c = min(costs_arr)
    max_c = max(costs_arr)
    min_v = values[costs_arr.index(min_c)]
    max_v = values[costs_arr.index(max_c)]

    lines += [
        "### Narrative",
        "",
        f"As **{parameter}** varies from {values[0]} to {values[-1]}, "
        f"total cost ranges from {_fmt_cost(min_c)} (at {parameter}={min_v}) "
        f"to {_fmt_cost(max_c)} (at {parameter}={max_v}), "
        f"a spread of {_fmt_cost(max_c - min_c)} "
        f"({_pct_change(min_c, max_c)} relative to the minimum).",
        "",
        f"The baseline sits {'above' if baseline.total_cost > min_c else 'at or below'} the minimum, "
        f"{'indicating headroom for cost reduction by optimising ' + parameter if baseline.total_cost > min_c else 'suggesting the current setting is already near-optimal'}.",
    ]
    return "\n".join(lines)


# ── What-if narrative ─────────────────────────────────────────────────────────

def generate_whatif_narrative(
    baseline: ScenarioSnapshot,
    perturbed: ScenarioSnapshot,
    perturbation_label: str,
) -> str:
    """
    Generate a short what-if paragraph for a perturbed scenario.

    Parameters
    ----------
    baseline          : original scenario
    perturbed         : scenario after applying the perturbation
    perturbation_label: description of what changed (e.g., '+20% demand')

    Returns
    -------
    Plain-text paragraph.
    """
    cost_delta = perturbed.total_cost - baseline.total_cost
    cost_dir = "increase" if cost_delta > 0 else "decrease"
    mfg_delta = perturbed.manufacturing_units - baseline.manufacturing_units
    trans_delta = perturbed.transfer_units - baseline.transfer_units

    return (
        f"**What-if: {perturbation_label}**\n\n"
        f"Applying '{perturbation_label}' to the baseline scenario causes total cost to "
        f"{cost_dir} by {_fmt_cost(abs(cost_delta))} "
        f"({_pct_change(baseline.total_cost, perturbed.total_cost)}). "
        f"Manufacturing volume {'rises' if mfg_delta > 0 else 'falls'} by "
        f"{abs(mfg_delta):,.1f} units and transfer volume "
        f"{'rises' if trans_delta > 0 else 'falls'} by {abs(trans_delta):,.1f} units, "
        f"suggesting the system {'absorbs the shock through additional production' if mfg_delta > 0 else 'compensates via redistribution of existing stock'}."
    )
