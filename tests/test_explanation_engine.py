"""Unit tests for nlp.explanation_engine."""
import pytest
from nlp.explanation_engine import (
    explain_transfer,
    explain_manufacturing,
    explain_scenario,
    explain_entities,
    explain_counts,
    explain_top_transfers,
    explain_top_manufacturing,
    explain_urgent_transfers,
    explain_reason_analysis,
    explain_store_activity,
    explain_cost_breakdown,
    build_explanation,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SCENARIO_DATA = {
    "scenario": "test_run",
    "optimized": {
        "total_cost": 100000.0,
        "total_transfers": 5,
        "manufacturing_units": 500.0,
        "transfer_units": 50.0,
    },
    "cost_breakdown": {
        "manufacturing_cost": 80000.0,
        "transfer_cost": 1000.0,
        "holding_cost": 19000.0,
    },
}

TRANSFER_DATA = {
    "scenario": "test_run",
    "transfers": [
        {
            "from_store": "1",
            "to_store": "2",
            "product_id": "100",
            "quantity": 10.0,
            "reason_codes": ["excess_inventory_at_source", "projected_stockout_at_destination"],
            "cost_impact": {"transport_cost": 50.0},
        },
        {
            "from_store": "3",
            "to_store": "4",
            "product_id": "200",
            "quantity": 5.0,
            "reason_codes": ["safety_stock_violation_prevented"],
            "cost_impact": {"transport_cost": 20.0},
        },
    ],
}

MANUFACTURING_DATA = {
    "scenario": "test_run",
    "manufacturing_actions": [
        {
            "product_id": "100",
            "manufacture_quantity": 100.0,
            "reason_codes": ["manufacture_to_avoid_stockout"],
            "cost_impact": {"manufacturing_cost": 5000.0},
        },
        {
            "product_id": "200",
            "manufacture_quantity": 50.0,
            "reason_codes": ["high_demand_variability"],
            "cost_impact": {"manufacturing_cost": 2500.0},
        },
    ],
}

FULL_DATA = {
    "scenario": SCENARIO_DATA,
    "transfers": TRANSFER_DATA,
    "manufacturing": MANUFACTURING_DATA,
}


# ── explain_transfer ──────────────────────────────────────────────────────────

class TestExplainTransfer:
    def test_basic_output(self):
        out = explain_transfer(FULL_DATA)
        assert "test_run" in out
        assert "Transfer 1" in out
        assert "Transfer 2" in out

    def test_product_filter(self):
        params = {"product_id": ["100"], "store_id": []}
        out = explain_transfer(FULL_DATA, params)
        assert "100" in out
        assert "Transfer 1" in out

    def test_store_filter(self):
        params = {"product_id": [], "store_id": ["3"]}
        out = explain_transfer(FULL_DATA, params)
        assert "200" in out

    def test_no_match_filter(self):
        params = {"product_id": ["999"], "store_id": []}
        out = explain_transfer(FULL_DATA, params)
        assert "No transfers match" in out


# ── explain_manufacturing ─────────────────────────────────────────────────────

class TestExplainManufacturing:
    def test_basic_output(self):
        out = explain_manufacturing(FULL_DATA)
        assert "Decision 1" in out
        assert "100" in out

    def test_product_filter(self):
        params = {"product_id": ["200"]}
        out = explain_manufacturing(FULL_DATA, params)
        assert "200" in out
        assert "Decision 1" in out

    def test_no_match_filter(self):
        params = {"product_id": ["999"]}
        out = explain_manufacturing(FULL_DATA, params)
        assert "No manufacturing actions match" in out


# ── explain_scenario ──────────────────────────────────────────────────────────

class TestExplainScenario:
    def test_totals(self):
        out = explain_scenario(FULL_DATA)
        assert "100,000" in out or "100000" in out
        assert "test_run" in out

    def test_cost_breakdown_in_output(self):
        out = explain_scenario(FULL_DATA)
        assert "80,000" in out or "80000" in out


# ── explain_entities ─────────────────────────────────────────────────────────

class TestExplainEntities:
    def test_products_listed(self):
        out = explain_entities(FULL_DATA)
        assert "100" in out
        assert "200" in out

    def test_stores_listed(self):
        out = explain_entities(FULL_DATA)
        assert "1" in out
        assert "2" in out


# ── explain_counts ────────────────────────────────────────────────────────────

class TestExplainCounts:
    def test_transfer_count(self):
        out = explain_counts(FULL_DATA)
        assert "2" in out  # 2 transfers

    def test_manufacturing_count(self):
        out = explain_counts(FULL_DATA)
        assert "2" in out  # 2 manufacturing actions


# ── explain_top_transfers ─────────────────────────────────────────────────────

class TestExplainTopTransfers:
    def test_default_limit(self):
        out = explain_top_transfers(FULL_DATA)
        assert "1→2" in out or "1" in out

    def test_limit_1(self):
        out = explain_top_transfers(FULL_DATA, limit=1)
        # Should only show the most expensive (50.0)
        assert "50" in out

    def test_empty(self):
        empty = {"transfers": {"transfers": [], "scenario": "x"}, "manufacturing": {}, "scenario": {}}
        out = explain_top_transfers(empty)
        assert "No transfers" in out


# ── explain_top_manufacturing ─────────────────────────────────────────────────

class TestExplainTopManufacturing:
    def test_default_output(self):
        out = explain_top_manufacturing(FULL_DATA)
        assert "100" in out  # product_id

    def test_limit(self):
        out = explain_top_manufacturing(FULL_DATA, limit=1)
        assert "5,000" in out or "5000" in out


# ── explain_urgent_transfers ──────────────────────────────────────────────────

class TestExplainUrgentTransfers:
    def test_urgent_detected(self):
        out = explain_urgent_transfers(FULL_DATA)
        # Transfer 1 has "projected_stockout_at_destination"
        assert "Store 2" in out or "1 transfers" in out or "Stockout Prevention" in out

    def test_no_urgent(self):
        data_no_urgent = {
            "transfers": {
                "scenario": "x",
                "transfers": [
                    {"from_store": "1", "to_store": "2", "product_id": "100",
                     "quantity": 5, "reason_codes": ["excess_inventory_at_source"],
                     "cost_impact": {"transport_cost": 10}},
                ],
            },
            "manufacturing": {},
            "scenario": {},
        }
        out = explain_urgent_transfers(data_no_urgent)
        assert "No urgent" in out


# ── explain_reason_analysis ───────────────────────────────────────────────────

class TestExplainReasonAnalysis:
    def test_transfer_reasons(self):
        out = explain_reason_analysis(FULL_DATA)
        assert "Excess Inventory At Source" in out or "excess_inventory" in out.lower()

    def test_manufacturing_reasons(self):
        out = explain_reason_analysis(FULL_DATA)
        assert "Manufacture To Avoid Stockout" in out or "manufacture_to_avoid" in out.lower()


# ── explain_store_activity ────────────────────────────────────────────────────

class TestExplainStoreActivity:
    def test_stores_appear(self):
        out = explain_store_activity(FULL_DATA)
        assert "1" in out
        assert "3" in out


# ── explain_cost_breakdown ────────────────────────────────────────────────────

class TestExplainCostBreakdown:
    def test_percentages(self):
        out = explain_cost_breakdown(FULL_DATA)
        assert "80.0%" in out or "80%" in out

    def test_total(self):
        out = explain_cost_breakdown(FULL_DATA)
        assert "100,000" in out or "100000" in out


# ── build_explanation dispatcher ─────────────────────────────────────────────

class TestBuildExplanation:
    def test_explain_transfer_dispatch(self):
        out = build_explanation("explain_transfer", FULL_DATA)
        assert "Transfer" in out

    def test_explain_manufacturing_dispatch(self):
        out = build_explanation("explain_manufacturing", FULL_DATA)
        assert "Decision" in out

    def test_scenario_summary_dispatch(self):
        out = build_explanation("scenario_summary", FULL_DATA)
        assert "Scenario" in out

    def test_top_transfers_limit(self):
        out = build_explanation("top_transfers", FULL_DATA, params={"limit": 1})
        assert out  # non-empty

    def test_unknown_intent_returns_empty(self):
        out = build_explanation("nonexistent_intent", FULL_DATA)
        assert out == ""

    def test_cost_breakdown_dispatch(self):
        out = build_explanation("cost_breakdown", FULL_DATA)
        assert "%" in out
