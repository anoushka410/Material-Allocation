"""Tests for optimization/stochastic.py — scenario generation and CVaR."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimization"))
from stochastic import generate_scenarios, compute_cvar, StochasticResult


class TestGenerateScenarios:
    def test_returns_correct_count(self):
        demands = {(1, 100): 50.0, (1, 200): 30.0, (2, 100): 40.0}
        scenarios, probs = generate_scenarios(demands, n_scenarios=15)
        assert len(scenarios) == 15
        assert len(probs) == 15

    def test_probabilities_sum_to_one(self):
        demands = {(1, 100): 50.0}
        _, probs = generate_scenarios(demands, n_scenarios=10)
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_all_keys_present(self):
        demands = {(1, 100): 50.0, (2, 200): 25.0}
        scenarios, _ = generate_scenarios(demands, n_scenarios=5)
        for s in scenarios:
            assert (1, 100) in s
            assert (2, 200) in s

    def test_positive_demands(self):
        demands = {(1, 100): 20.0, (2, 100): 10.0}
        scenarios, _ = generate_scenarios(demands, n_scenarios=30, cv=0.3)
        for s in scenarios:
            for v in s.values():
                assert v >= 0

    def test_higher_cv_higher_variance(self):
        demands = {(1, k): 50.0 for k in range(10)}
        scen_low, _ = generate_scenarios(demands, n_scenarios=200, cv=0.1, random_state=0)
        scen_high, _ = generate_scenarios(demands, n_scenarios=200, cv=0.5, random_state=0)
        vals_low = np.array([list(s.values()) for s in scen_low]).flatten()
        vals_high = np.array([list(s.values()) for s in scen_high]).flatten()
        assert vals_high.std() > vals_low.std()

    def test_reproducible_with_seed(self):
        demands = {(1, 1): 50.0}
        s1, _ = generate_scenarios(demands, n_scenarios=10, random_state=42)
        s2, _ = generate_scenarios(demands, n_scenarios=10, random_state=42)
        assert s1[0][(1, 1)] == s2[0][(1, 1)]


class TestComputeCVaR:
    def test_cvar_equals_mean_all_exceed_var(self):
        costs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        cvar = compute_cvar(costs, alpha=0.60)
        var60 = float(np.quantile(costs, 0.60))
        tail = costs[costs >= var60]
        expected = float(tail.mean())
        assert abs(cvar - expected) < 1e-9

    def test_cvar_ge_mean(self):
        costs = np.random.default_rng(0).standard_normal(1000) * 5 + 100
        cvar = compute_cvar(costs, alpha=0.95)
        assert cvar >= costs.mean() - 1e-6

    def test_cvar_ge_var(self):
        costs = np.arange(1, 101, dtype=float)
        var95 = float(np.quantile(costs, 0.95))
        cvar95 = compute_cvar(costs, alpha=0.95)
        assert cvar95 >= var95 - 1e-9

    def test_empty_costs(self):
        assert compute_cvar([], alpha=0.95) == 0.0

    def test_constant_costs(self):
        costs = np.ones(100) * 42.0
        assert abs(compute_cvar(costs, alpha=0.95) - 42.0) < 1e-9

    def test_accepts_list(self):
        costs = [10.0, 20.0, 30.0, 40.0, 50.0]
        cvar = compute_cvar(costs, alpha=0.8)
        assert cvar >= 0


class TestStochasticResult:
    def _make_result(self):
        return StochasticResult(
            status="Optimal",
            expected_cost=100_000.0,
            cvar_95=130_000.0,
            var_95=120_000.0,
            scenario_costs=[100_000.0] * 18 + [200_000.0, 250_000.0],
            manufacturing_decisions={(1, 100): 50.0},
            n_scenarios=20,
            cv_used=0.25,
        )

    def test_risk_premium_computed(self):
        r = self._make_result()
        assert abs(r.risk_premium - (130_000.0 - 100_000.0)) < 1e-6

    def test_summary_contains_fields(self):
        r = self._make_result()
        s = r.summary()
        assert "Expected cost" in s
        assert "CVaR" in s
        assert "Risk premium" in s
