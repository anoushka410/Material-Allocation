"""
Structured KPI Export — Parquet and DuckDB.

Provides functions to build a normalised KPI schema from the project's
optimization and forecast outputs, then persist them in:
  - Apache Parquet  (for BI tools like Tableau, Power BI, Spark)
  - DuckDB          (for in-process SQL analytics / real-time dashboards)

Schema
------
kpi_scenario  — one row per optimization run
kpi_transfers — one row per transfer recommendation
kpi_manufacturing — one row per manufacturing decision
kpi_inventory — one row per (store, product) inventory state
kpi_forecast_metrics — one row per (store, product) forecast accuracy
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


# ── Schema builders ───────────────────────────────────────────────────────────

def build_kpi_scenario(scenario_json: dict) -> pd.DataFrame:
    """One-row DataFrame from scenario_summary.json."""
    optimized = scenario_json.get("optimized", {})
    breakdown = scenario_json.get("cost_breakdown", {})
    return pd.DataFrame([{
        "scenario": scenario_json.get("scenario", "unknown"),
        "total_cost": optimized.get("total_cost", 0.0),
        "total_transfers": optimized.get("total_transfers", 0),
        "manufacturing_units": optimized.get("manufacturing_units", 0.0),
        "transfer_units": optimized.get("transfer_units", 0.0),
        "manufacturing_cost": breakdown.get("manufacturing_cost", 0.0),
        "transfer_cost": breakdown.get("transfer_cost", 0.0),
        "holding_cost": breakdown.get("holding_cost", 0.0),
    }])


def build_kpi_transfers(transfers_json: dict) -> pd.DataFrame:
    """Flatten transfer_recommendations.json into a tabular DataFrame."""
    rows = transfers_json.get("transfers", [])
    if not rows:
        return pd.DataFrame(columns=[
            "scenario", "from_store", "to_store", "product_id",
            "quantity", "transport_cost", "reason_codes",
        ])
    scenario = transfers_json.get("scenario", "unknown")
    records = []
    for r in rows:
        records.append({
            "scenario": scenario,
            "from_store": r.get("from_store"),
            "to_store": r.get("to_store"),
            "product_id": r.get("product_id"),
            "quantity": r.get("quantity", 0.0),
            "transport_cost": r.get("cost_impact", {}).get("transport_cost", 0.0),
            "reason_codes": "|".join(r.get("reason_codes", [])),
        })
    return pd.DataFrame(records)


def build_kpi_manufacturing(manufacturing_json: dict) -> pd.DataFrame:
    """Flatten manufacturing_decisions.json into a tabular DataFrame."""
    rows = manufacturing_json.get("manufacturing_actions", [])
    if not rows:
        return pd.DataFrame(columns=[
            "scenario", "product_id", "manufacture_quantity",
            "manufacturing_cost", "reason_codes",
        ])
    scenario = manufacturing_json.get("scenario", "unknown")
    records = []
    for r in rows:
        records.append({
            "scenario": scenario,
            "product_id": r.get("product_id"),
            "manufacture_quantity": r.get("manufacture_quantity", 0.0),
            "manufacturing_cost": r.get("cost_impact", {}).get("manufacturing_cost", 0.0),
            "reason_codes": "|".join(r.get("reason_codes", [])),
        })
    return pd.DataFrame(records)


def build_kpi_inventory(inventory_csv_path: str | Path) -> pd.DataFrame:
    """Load optimization_inventory.csv into a typed DataFrame."""
    path = Path(inventory_csv_path)
    if not path.exists():
        return pd.DataFrame(columns=["store_id", "product_id", "current", "final", "target"])
    df = pd.read_csv(path)
    for col in ("current", "final", "target"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_kpi_forecast_metrics(forecast_metrics_path: str | Path) -> pd.DataFrame:
    """Load product_forecast_metrics.csv into a typed DataFrame."""
    path = Path(forecast_metrics_path)
    if not path.exists():
        return pd.DataFrame(columns=["store_id", "product_id", "MAE", "RMSE", "n_train", "n_test"])
    df = pd.read_csv(path)
    for col in ("MAE", "RMSE", "n_train", "n_test"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Parquet export ────────────────────────────────────────────────────────────

def export_to_parquet(
    data: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Export each DataFrame in `data` to a separate Parquet file.

    Parameters
    ----------
    data       : mapping of table_name → DataFrame
    output_dir : directory to write .parquet files

    Returns
    -------
    dict of table_name → Path written
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, df in data.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        written[name] = path
    return written


# ── DuckDB export ─────────────────────────────────────────────────────────────

def export_to_duckdb(
    data: dict[str, pd.DataFrame],
    db_path: str | Path,
) -> Path:
    """
    Export each DataFrame in `data` to a DuckDB database table.

    Parameters
    ----------
    data    : mapping of table_name → DataFrame
    db_path : path to the .duckdb file (created if absent)

    Returns
    -------
    Path to the DuckDB file
    """
    import duckdb  # optional dependency; fail loudly if missing

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for name, df in data.items():
            # Replace existing table
            con.execute(f"DROP TABLE IF EXISTS {name}")
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
    finally:
        con.close()
    return db_path


# ── All-in-one helper ─────────────────────────────────────────────────────────

def export_all_kpis(
    scenario_json: dict,
    transfers_json: dict,
    manufacturing_json: dict,
    inventory_csv: str | Path,
    forecast_metrics_csv: str | Path,
    output_dir: str | Path,
    formats: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build all KPI DataFrames and export to requested formats.

    Parameters
    ----------
    formats : list of 'parquet', 'duckdb' (default: both)

    Returns
    -------
    dict with keys 'dataframes', 'parquet_files', 'duckdb_path'
    """
    if formats is None:
        formats = ["parquet", "duckdb"]

    frames = {
        "kpi_scenario": build_kpi_scenario(scenario_json),
        "kpi_transfers": build_kpi_transfers(transfers_json),
        "kpi_manufacturing": build_kpi_manufacturing(manufacturing_json),
        "kpi_inventory": build_kpi_inventory(inventory_csv),
        "kpi_forecast_metrics": build_kpi_forecast_metrics(forecast_metrics_csv),
    }

    result: dict[str, Any] = {"dataframes": frames}

    out_dir = Path(output_dir)
    if "parquet" in formats:
        result["parquet_files"] = export_to_parquet(frames, out_dir / "parquet")

    if "duckdb" in formats:
        result["duckdb_path"] = export_to_duckdb(frames, out_dir / "supply_chain_kpis.duckdb")

    return result
