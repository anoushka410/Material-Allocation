"""
Two-Stage Stochastic Linear Program with CVaR.

Stage 1: Decide manufacturing quantities x[s,p] before demand is revealed.
Stage 2: Observe demand scenario ω, then optimally transfer inventory t[i,j,p,ω].

Objective (risk-neutral):
  min Σ_ω π_ω [ Σ mfg_cost·x + Σ transport·t[ω] + Σ holding·inv[ω] + penalty·shortfall[ω] ]

CVaR (Conditional Value-at-Risk) at level α is approximated by
reformulating the auxiliary variable η:
  CVaR_α ≈ η + (1/(1-α)) Σ_ω π_ω · max(cost_ω - η, 0)

This module:
  1. generate_scenarios  — Monte Carlo demand scenarios.
  2. solve_stochastic_lp — Two-stage LP (PuLP).
  3. compute_cvar        — CVaR from a cost vector.
  4. StochasticResult    — Container for results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Scenario generation ───────────────────────────────────────────────────────

def generate_scenarios(
    mean_demands: dict[tuple, float],
    n_scenarios: int = 20,
    cv: float = 0.25,
    random_state: int = 0,
) -> tuple[list[dict[tuple, float]], list[float]]:
    """
    Generate Monte Carlo demand scenarios using a log-normal distribution.

    Parameters
    ----------
    mean_demands : dict mapping (store_id, product_id) → mean demand
    n_scenarios  : number of scenarios Ω
    cv           : coefficient of variation for demand noise
    random_state : RNG seed

    Returns
    -------
    (scenarios, probabilities)
      scenarios     : list of n_scenarios dicts (store,product)→demand
      probabilities : uniform [1/Ω, ...] list of length Ω
    """
    rng = np.random.default_rng(random_state)
    keys = list(mean_demands.keys())
    means = np.array([mean_demands[k] for k in keys], dtype=float)

    # Log-normal: E[X]=μ, CV=cv → σ_ln = sqrt(log(1+cv²))
    sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
    mu_ln = np.log(np.maximum(means, 1e-6)) - 0.5 * sigma_ln ** 2

    scenarios = []
    for _ in range(n_scenarios):
        samples = rng.lognormal(mean=mu_ln, sigma=sigma_ln)
        scenarios.append(dict(zip(keys, samples.tolist())))

    probabilities = [1.0 / n_scenarios] * n_scenarios
    return scenarios, probabilities


# ── CVaR computation ─────────────────────────────────────────────────────────

def compute_cvar(costs: np.ndarray | list, alpha: float = 0.95) -> float:
    """
    Compute CVaR (Expected Shortfall) at confidence level α.

    CVaR_α = E[cost | cost ≥ VaR_α]

    Parameters
    ----------
    costs : 1-D array of scenario costs
    alpha : confidence level (e.g., 0.95 → worst 5% scenarios)

    Returns
    -------
    float : CVaR value
    """
    c = np.asarray(costs, dtype=float)
    if len(c) == 0:
        return 0.0
    var_alpha = float(np.quantile(c, alpha))
    tail = c[c >= var_alpha]
    return float(tail.mean()) if len(tail) > 0 else var_alpha


# ── Two-stage LP ─────────────────────────────────────────────────────────────

@dataclass
class StochasticResult:
    """Results from the two-stage stochastic LP."""
    status: str
    expected_cost: float
    cvar_95: float
    var_95: float
    scenario_costs: list[float]
    manufacturing_decisions: dict[tuple, float]   # (store,product) → qty
    n_scenarios: int
    cv_used: float
    risk_premium: float = field(init=False)

    def __post_init__(self):
        self.risk_premium = self.cvar_95 - self.expected_cost

    def summary(self) -> str:
        return (
            f"Stochastic LP Result\n"
            f"  Status          : {self.status}\n"
            f"  Expected cost   : ${self.expected_cost:,.2f}\n"
            f"  VaR (95%)       : ${self.var_95:,.2f}\n"
            f"  CVaR (95%)      : ${self.cvar_95:,.2f}\n"
            f"  Risk premium    : ${self.risk_premium:,.2f}\n"
            f"  Scenarios (Ω)   : {self.n_scenarios}\n"
            f"  Demand CV used  : {self.cv_used:.2f}\n"
        )


def solve_stochastic_lp(
    mean_demands: dict[tuple, float],
    mfg_cost: dict,
    transport_cost: dict,
    stores: list,
    holding_cost: float = 1.0,
    shortage_penalty: float = 200.0,
    mfg_capacity: float = 5000.0,
    n_scenarios: int = 20,
    cv: float = 0.25,
    random_state: int = 42,
) -> StochasticResult:
    """
    Solve two-stage stochastic LP.

    Stage 1 decision : x[s,p]       — manufacture before demand realization
    Stage 2 recourse : t[i,j,p,ω]   — transfer after demand realization
                       shortfall[s,p,ω] — unmet demand (penalised)

    Returns
    -------
    StochasticResult
    """
    try:
        from pulp import (
            LpProblem, LpMinimize, LpVariable, lpSum,
            PULP_CBC_CMD, LpStatus, value,
        )
    except ImportError:
        raise ImportError("PuLP is required: pip install pulp")

    scenarios, probs = generate_scenarios(
        mean_demands, n_scenarios=n_scenarios, cv=cv, random_state=random_state
    )
    valid_pairs = list(mean_demands.keys())
    all_products = sorted({p for (_, p) in valid_pairs})
    products_per_store = {
        s: [p for (s2, p) in valid_pairs if s2 == s] for s in stores
    }

    model = LpProblem("Stochastic_Inventory", LpMinimize)

    # ── Stage-1 variables (scenario-independent) ─────────────────────────────
    x = LpVariable.dicts("mfg", valid_pairs, lowBound=0)

    # ── Stage-2 variables (scenario-dependent) ───────────────────────────────
    shortage = LpVariable.dicts(
        "shortage",
        [(s, p, w) for (s, p) in valid_pairs for w in range(n_scenarios)],
        lowBound=0,
    )
    t_var = LpVariable.dicts(
        "transfer",
        [
            (i, j, p, w)
            for i in stores for j in stores for p in all_products
            for w in range(n_scenarios)
            if i != j and (i, p) in valid_pairs and (j, p) in valid_pairs
        ],
        lowBound=0,
    )

    # ── Objective ─────────────────────────────────────────────────────────────
    mfg_term = lpSum(mfg_cost.get(s, 50) * x[(s, p)] for (s, p) in valid_pairs)

    recourse_term = lpSum(
        probs[w] * (
            lpSum(
                transport_cost.get((i, j), 5.0) * t_var[(i, j, p, w)]
                for (i, j, p, w2) in t_var if w2 == w
            ) +
            lpSum(
                holding_cost * (
                    x[(s, p)] +
                    lpSum(t_var[(i, s, p, w)] for i in stores
                          if i != s and (i, p) in valid_pairs and (i, s, p, w) in t_var) -
                    lpSum(t_var[(s, j, p, w)] for j in stores
                          if j != s and (j, p) in valid_pairs and (s, j, p, w) in t_var)
                )
                for (s, p) in valid_pairs
            ) +
            lpSum(
                shortage_penalty * shortage[(s, p, w)]
                for (s, p) in valid_pairs
            )
        )
        for w in range(n_scenarios)
    )

    model += mfg_term + recourse_term

    # ── Constraints ───────────────────────────────────────────────────────────
    # Capacity (stage-1)
    for s in stores:
        if products_per_store.get(s):
            model += lpSum(x[(s, p)] for p in products_per_store[s]) <= mfg_capacity

    # Demand feasibility per scenario
    for w, scenario in enumerate(scenarios):
        for (s, p) in valid_pairs:
            demand_w = scenario.get((s, p), 0.0)
            transfers_in = lpSum(
                t_var[(i, s, p, w)] for i in stores
                if i != s and (i, p) in valid_pairs and (i, s, p, w) in t_var
            )
            transfers_out = lpSum(
                t_var[(s, j, p, w)] for j in stores
                if j != s and (j, p) in valid_pairs and (s, j, p, w) in t_var
            )
            final_inv = x[(s, p)] + transfers_in - transfers_out
            # Meet demand or pay shortage penalty
            model += final_inv + shortage[(s, p, w)] >= demand_w

    status_code = model.solve(PULP_CBC_CMD(msg=0, timeLimit=60))
    status = LpStatus[status_code]

    # ── Extract stage-1 decisions ─────────────────────────────────────────────
    mfg_decisions = {(s, p): float(value(x[(s, p)]) or 0.0) for (s, p) in valid_pairs}

    # ── Per-scenario costs ────────────────────────────────────────────────────
    scenario_costs = []
    for w, scenario in enumerate(scenarios):
        mfg_c = sum(mfg_cost.get(s, 50) * mfg_decisions[(s, p)] for (s, p) in valid_pairs)
        transfer_c = sum(
            transport_cost.get((i, j), 5.0) * float(value(t_var[(i, j, p, w)]) or 0.0)
            for (i, j, p, w2) in t_var if w2 == w
        )
        shortage_c = sum(
            shortage_penalty * float(value(shortage[(s, p, w)]) or 0.0)
            for (s, p) in valid_pairs
        )
        scenario_costs.append(mfg_c + transfer_c + shortage_c)

    expected = float(np.average(scenario_costs, weights=probs))
    cvar = compute_cvar(np.array(scenario_costs), alpha=0.95)
    var95 = float(np.quantile(scenario_costs, 0.95))

    return StochasticResult(
        status=status,
        expected_cost=expected,
        cvar_95=cvar,
        var_95=var95,
        scenario_costs=scenario_costs,
        manufacturing_decisions=mfg_decisions,
        n_scenarios=n_scenarios,
        cv_used=cv,
    )
