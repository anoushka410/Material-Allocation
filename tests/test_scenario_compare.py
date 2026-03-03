"""Tests for nlp/scenario_compare.py — scenario comparison and what-if."""
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nlp.scenario_compare import (
    ScenarioSnapshot,
    compare_scenarios,
    sensitivity_analysis_text,
    generate_whatif_narrative,
)


def _make_snapshot(name="baseline", cost=100_000.0, mfg=80_000.0, tr=1_000.0, hold=19_000.0, n_transfers=50, mfg_units=500.0, tr_units=50.0):
    return ScenarioSnapshot(
        name=name,
        total_cost=cost,
        manufacturing_cost=mfg,
        transfer_cost=tr,
        holding_cost=hold,
        total_transfers=n_transfers,
        manufacturing_units=mfg_units,
        transfer_units=tr_units,
    )


class TestScenarioSnapshot:
    def test_from_json(self):
        j = {
            "scenario": "test_run",
            "optimized": {
                "total_cost": 100_000.0,
                "total_transfers": 50,
                "manufacturing_units": 500.0,
                "transfer_units": 50.0,
            },
            "cost_breakdown": {
                "manufacturing_cost": 80_000.0,
                "transfer_cost": 1_000.0,
                "holding_cost": 19_000.0,
            },
        }
        snap = ScenarioSnapshot.from_json(j)
        assert snap.name == "test_run"
        assert snap.total_cost == 100_000.0
        assert snap.manufacturing_cost == 80_000.0

    def test_from_json_empty(self):
        snap = ScenarioSnapshot.from_json({})
        assert snap.total_cost == 0.0


class TestCompareScenarios:
    def test_returns_string(self):
        a = _make_snapshot("baseline")
        b = _make_snapshot("service_first", cost=120_000.0)
        out = compare_scenarios(a, b)
        assert isinstance(out, str)

    def test_contains_scenario_names(self):
        a = _make_snapshot("baseline")
        b = _make_snapshot("service_first")
        out = compare_scenarios(a, b)
        assert "baseline" in out
        assert "service_first" in out

    def test_contains_cost_table(self):
        a = _make_snapshot("baseline")
        b = _make_snapshot("alt")
        out = compare_scenarios(a, b)
        assert "Total Cost" in out
        assert "|" in out  # table rows

    def test_cost_increase_detected(self):
        a = _make_snapshot("baseline", cost=100_000.0)
        b = _make_snapshot("alt", cost=120_000.0)
        out = compare_scenarios(a, b)
        assert "worsened" in out or "▲" in out

    def test_cost_decrease_detected(self):
        a = _make_snapshot("baseline", cost=100_000.0)
        b = _make_snapshot("alt", cost=80_000.0)
        out = compare_scenarios(a, b)
        assert "improved" in out or "▼" in out

    def test_identical_scenarios(self):
        a = _make_snapshot("same")
        out = compare_scenarios(a, a)
        assert "unchanged" in out or "0.0%" in out


class TestSensitivityAnalysisText:
    def test_returns_string(self):
        baseline = _make_snapshot()
        out = sensitivity_analysis_text(baseline, "demand_cv", [0.1, 0.2, 0.3], [90_000.0, 100_000.0, 115_000.0])
        assert isinstance(out, str)

    def test_contains_parameter_name(self):
        baseline = _make_snapshot()
        out = sensitivity_analysis_text(baseline, "demand_cv", [0.1, 0.3], [90_000.0, 115_000.0])
        assert "demand_cv" in out

    def test_table_rows_present(self):
        baseline = _make_snapshot()
        out = sensitivity_analysis_text(baseline, "param", [1, 2, 3], [90.0, 100.0, 110.0])
        assert "|" in out
        # Three data rows plus header + separator
        assert out.count("|") > 6

    def test_mismatched_lengths_raises(self):
        baseline = _make_snapshot()
        with pytest.raises(ValueError):
            sensitivity_analysis_text(baseline, "p", [1, 2], [90.0])


class TestGenerateWhatifNarrative:
    def test_returns_string(self):
        a = _make_snapshot("baseline")
        b = _make_snapshot("perturbed", cost=110_000.0)
        out = generate_whatif_narrative(a, b, "+10% demand")
        assert isinstance(out, str)

    def test_contains_label(self):
        a = _make_snapshot("baseline")
        b = _make_snapshot("p")
        out = generate_whatif_narrative(a, b, "demand shock +20%")
        assert "demand shock +20%" in out

    def test_cost_direction_increase(self):
        a = _make_snapshot(cost=100_000.0)
        b = _make_snapshot(cost=120_000.0)
        out = generate_whatif_narrative(a, b, "higher demand")
        assert "increase" in out

    def test_cost_direction_decrease(self):
        a = _make_snapshot(cost=100_000.0)
        b = _make_snapshot(cost=80_000.0)
        out = generate_whatif_narrative(a, b, "lower demand")
        assert "decrease" in out
