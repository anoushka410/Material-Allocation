"""Tests for export/kpi_export.py — Parquet and DuckDB export."""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from export.kpi_export import (
    build_kpi_scenario,
    build_kpi_transfers,
    build_kpi_manufacturing,
    build_kpi_inventory,
    build_kpi_forecast_metrics,
    export_to_parquet,
    export_to_duckdb,
    export_all_kpis,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SCENARIO_JSON = {
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

TRANSFERS_JSON = {
    "scenario": "test_run",
    "transfers": [
        {
            "from_store": "1", "to_store": "2", "product_id": "100",
            "quantity": 10.0,
            "reason_codes": ["excess_inventory", "high_delay"],
            "cost_impact": {"transport_cost": 50.0},
        },
        {
            "from_store": "3", "to_store": "4", "product_id": "200",
            "quantity": 5.0, "reason_codes": [],
            "cost_impact": {"transport_cost": 20.0},
        },
    ],
}

MANUFACTURING_JSON = {
    "scenario": "test_run",
    "manufacturing_actions": [
        {
            "product_id": "100",
            "manufacture_quantity": 100.0,
            "reason_codes": ["stockout_prevention"],
            "cost_impact": {"manufacturing_cost": 5_000.0},
        },
    ],
}


# ── Schema builder tests ──────────────────────────────────────────────────────

class TestBuildKpiScenario:
    def test_shape(self):
        df = build_kpi_scenario(SCENARIO_JSON)
        assert df.shape == (1, 8)

    def test_total_cost(self):
        df = build_kpi_scenario(SCENARIO_JSON)
        assert df["total_cost"].iloc[0] == 100_000.0

    def test_scenario_name(self):
        df = build_kpi_scenario(SCENARIO_JSON)
        assert df["scenario"].iloc[0] == "test_run"

    def test_empty_json(self):
        df = build_kpi_scenario({})
        assert df.shape == (1, 8)
        assert df["total_cost"].iloc[0] == 0.0


class TestBuildKpiTransfers:
    def test_shape(self):
        df = build_kpi_transfers(TRANSFERS_JSON)
        assert df.shape == (2, 7)

    def test_reason_codes_joined(self):
        df = build_kpi_transfers(TRANSFERS_JSON)
        assert "|" in df["reason_codes"].iloc[0] or df["reason_codes"].iloc[0] != ""

    def test_empty_transfers(self):
        df = build_kpi_transfers({"transfers": []})
        assert len(df) == 0


class TestBuildKpiManufacturing:
    def test_shape(self):
        df = build_kpi_manufacturing(MANUFACTURING_JSON)
        assert df.shape == (1, 5)

    def test_cost_value(self):
        df = build_kpi_manufacturing(MANUFACTURING_JSON)
        assert df["manufacturing_cost"].iloc[0] == 5_000.0

    def test_empty(self):
        df = build_kpi_manufacturing({"manufacturing_actions": []})
        assert len(df) == 0


class TestBuildKpiInventory:
    def test_missing_file_returns_empty(self):
        df = build_kpi_inventory("/nonexistent/path.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_reads_real_file(self):
        real_path = Path(__file__).parent.parent / "optimization/output-csv/optimization_inventory.csv"
        if real_path.exists():
            df = build_kpi_inventory(real_path)
            assert len(df) > 0
            assert "current" in df.columns


class TestBuildKpiForecastMetrics:
    def test_missing_file_returns_empty(self):
        df = build_kpi_forecast_metrics("/nonexistent/path.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_reads_real_file(self):
        real_path = Path(__file__).parent.parent / "demand-forecast/output/product_forecast_metrics.csv"
        if real_path.exists():
            df = build_kpi_forecast_metrics(real_path)
            assert len(df) > 0
            assert "MAE" in df.columns


# ── Export tests ──────────────────────────────────────────────────────────────

class TestExportToParquet:
    def test_creates_parquet_files(self):
        frames = {
            "test_table": pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_to_parquet(frames, tmpdir)
            assert "test_table" in written
            assert written["test_table"].exists()

    def test_roundtrip(self):
        frames = {"my_df": pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})}
        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_to_parquet(frames, tmpdir)
            df_read = pd.read_parquet(written["my_df"])
            assert list(df_read["x"]) == [1, 2, 3]


class TestExportToDuckDB:
    def test_creates_duckdb_file(self):
        frames = {"kpi_test": pd.DataFrame({"id": [1], "val": [99.0]})}
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            result = export_to_duckdb(frames, db_path)
            assert result.exists()

    def test_queryable(self):
        import duckdb
        frames = {"kpi_test": pd.DataFrame({"id": [1, 2], "val": [10.0, 20.0]})}
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            export_to_duckdb(frames, db_path)
            con = duckdb.connect(str(db_path))
            result = con.execute("SELECT COUNT(*) FROM kpi_test").fetchone()[0]
            con.close()
            assert result == 2


class TestExportAllKpis:
    def test_returns_dataframes_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_all_kpis(
                SCENARIO_JSON, TRANSFERS_JSON, MANUFACTURING_JSON,
                "/nonexistent.csv", "/nonexistent2.csv",
                tmpdir, formats=["parquet"],
            )
            assert "dataframes" in result
            assert len(result["dataframes"]) == 5

    def test_parquet_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_all_kpis(
                SCENARIO_JSON, TRANSFERS_JSON, MANUFACTURING_JSON,
                "/nonexistent.csv", "/nonexistent2.csv",
                tmpdir, formats=["parquet"],
            )
            assert "parquet_files" in result
            for path in result["parquet_files"].values():
                assert path.exists()
