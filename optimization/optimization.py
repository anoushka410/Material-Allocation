"""
Inventory Optimization Model
Minimizes total cost (manufacturing + transfer + holding) while meeting demand + safety stock.
"""

import pandas as pd
import numpy as np
import json
import os
from pulp import *
import warnings
warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioConfig:
    """Configuration for scenario types with self-explanatory IDs and parameter overrides."""
    
    SCENARIOS = {
        "base_case_standard_conditions": {
            "description": "Normal forecast, normal transport cost, default risk penalties.",
            "demand_multiplier": 1.0,
            "transport_cost_multiplier": 1.0,
            "lead_time_multiplier": 1.0,
            "safety_stock_multiplier": 1.0,
            "delay_probability_multiplier": 1.0,
        },
        "risk_aware_high_disruption": {
            "description": "Increased delay probability and higher safety stock to simulate disruption risk.",
            "demand_multiplier": 1.0,
            "transport_cost_multiplier": 1.0,
            "lead_time_multiplier": 1.2,
            "safety_stock_multiplier": 1.4,
            "delay_probability_multiplier": 1.8,
        },
        "cost_only_no_risk_penalty": {
            "description": "Optimization minimizes cost only, without risk penalties.",
            "demand_multiplier": 1.0,
            "transport_cost_multiplier": 1.0,
            "lead_time_multiplier": 1.0,
            "safety_stock_multiplier": 0.5,
            "delay_probability_multiplier": 0.0,
        },
        "demand_spike_high_forecast": {
            "description": "Increased forecast values to simulate promotional or peak demand week.",
            "demand_multiplier": 1.25,
            "transport_cost_multiplier": 1.0,
            "lead_time_multiplier": 1.0,
            "safety_stock_multiplier": 1.2,
            "delay_probability_multiplier": 1.0,
        },
        "transport_cost_increase_fuel_shock": {
            "description": "Increased shipping costs to simulate fuel price or logistics inflation.",
            "demand_multiplier": 1.0,
            "transport_cost_multiplier": 1.5,
            "lead_time_multiplier": 1.0,
            "safety_stock_multiplier": 1.0,
            "delay_probability_multiplier": 1.0,
        },
        "extended_lead_time_supplier_delay": {
            "description": "Increased lead times to simulate supplier or customs delays.",
            "demand_multiplier": 1.0,
            "transport_cost_multiplier": 1.0,
            "lead_time_multiplier": 1.5,
            "safety_stock_multiplier": 1.3,
            "delay_probability_multiplier": 1.2,
        },
    }
    
    @classmethod
    def get_scenario(cls, scenario_id: str) -> dict:
        """Get scenario configuration by ID. Defaults to base_case_standard_conditions."""
        if scenario_id not in cls.SCENARIOS:
            print(f"Warning: Unknown scenario '{scenario_id}'. Using base_case_standard_conditions.")
            scenario_id = "base_case_standard_conditions"
        return cls.SCENARIOS[scenario_id]
    
    @classmethod
    def list_scenarios(cls) -> dict:
        """List all available scenarios with descriptions."""
        return {
            sid: config["description"] 
            for sid, config in cls.SCENARIOS.items()
        }


# Thresholds for reason codes
THRESHOLDS = {'high_cv': 0.7, 'high_delay_prob': 0.5, 'capacity_ratio': 0.9}


def assign_transfer_reasons(i, j, p, qty, data):
    """Assign reason codes for a transfer decision."""
    reasons = []
    
    current_dest = data['inv'].get((j, p), 0)
    demand_dest = data['demand'].get((j, p), 0)
    target_dest = demand_dest + data['safety'].get((j, p), 0)
    current_src = data['inv'].get((i, p), 0)
    target_src = data['demand'].get((i, p), 0) + data['safety'].get((i, p), 0)
    
    # Stockout risk at destination
    if current_dest < demand_dest:
        reasons.append("projected_stockout_at_destination")
    
    # Excess at source
    if current_src > target_src:
        reasons.append("excess_inventory_at_source")
    
    # Safety stock violation prevented
    if current_dest < data['safety'].get((j, p), 0):
        reasons.append("safety_stock_violation_prevented")
    
    # High demand variability at destination
    cv = data['cv'].get((j, p), 0)
    if cv > THRESHOLDS['high_cv']:
        reasons.append("high_demand_variability")
    
    # High delay probability at destination
    delay = data['delay'].get(j, 0)
    if delay > THRESHOLDS['high_delay_prob']:
        reasons.append("high_delay_probability")
    
    # Transfer cheaper than manufacturing
    if data['transport'].get((i, j), 999) < data['mfg'].get(j, 999):
        reasons.append("transport_cost_acceptable")
    
    return reasons if reasons else ["rebalance_inventory"]


def assign_manufacturing_reasons(s, p, qty, data, store_mfg_total):
    """Assign reason codes for a manufacturing decision."""
    reasons = []
    
    current = data['inv'].get((s, p), 0)
    demand = data['demand'].get((s, p), 0)
    safety = data['safety'].get((s, p), 0)
    
    # Demand exceeds inventory
    if current < demand:
        reasons.append("manufacture_to_avoid_stockout")
    
    # Safety stock replenishment
    if current < safety:
        reasons.append("safety_stock_violation_prevented")
    
    # High variability
    cv = data['cv'].get((s, p), 0)
    if cv > THRESHOLDS['high_cv']:
        reasons.append("high_demand_variability")
    
    # High delay probability
    delay = data['delay'].get(s, 0)
    if delay > THRESHOLDS['high_delay_prob']:
        reasons.append("high_delay_probability")
    
    # Capacity constrained
    if store_mfg_total.get(s, 0) > THRESHOLDS['capacity_ratio'] * data['capacity']:
        reasons.append("manufacturing_capacity_constrained")
    
    return reasons if reasons else ["aggregate_demand_exceeds_inventory"]


def _merge_unique(existing: list[dict], new: list[dict], key_fields: list[str]) -> list[dict]:
    """Merge two lists of dicts, de-duping by a tuple of key_fields."""
    seen = set()
    out: list[dict] = []

    def _key(r: dict) -> tuple:
        return tuple(r.get(k) for k in key_fields)

    for r in existing:
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    for r in new:
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def save_json_outputs(transfers, manufacturing, costs, output_dir, scenario_id: str = "base_case_standard_conditions"):
    """Save/append optimization results to flat JSON files.

    Output schema (exactly 3 files under output_dir):
      - transfer_recommendations.json  : {"scenario": "all", "transfers": [...]} where each transfer has scenario
      - manufacturing_decisions.json   : {"scenario": "all", "manufacturing_actions": [...]} where each action has scenario
      - scenario_summary.json          : {"scenario": "all", "scenarios": {<scenario_id>: {...}}, "scenario_ids": [...]}

    Note: This is designed for dashboarding/NLP to load a single file per entity type.
    """
    os.makedirs(output_dir, exist_ok=True)

    if scenario_id not in ScenarioConfig.SCENARIOS:
        print(f"Warning: Unknown scenario '{scenario_id}'. Using 'base_case_standard_conditions'.")
        scenario_id = "base_case_standard_conditions"

    out_dir = Path(output_dir)

    # --- Transfers (append; each record includes scenario) ---
    transfers_path = out_dir / "transfer_recommendations.json"
    existing_transfers: list[dict] = []
    if transfers_path.exists():
        try:
            existing_transfers = json.loads(transfers_path.read_text()).get("transfers", [])
        except Exception:
            existing_transfers = []

    transfers_with_scenario = []
    for t in transfers:
        tt = dict(t)
        tt["scenario"] = scenario_id
        transfers_with_scenario.append(tt)

    merged_transfers = _merge_unique(
        existing_transfers,
        transfers_with_scenario,
        key_fields=["scenario", "from_store", "to_store", "product_id"],
    )

    transfers_path.write_text(json.dumps({"scenario": "all", "transfers": merged_transfers}, indent=2))

    # --- Manufacturing (append; each record includes scenario) ---
    mfg_path = out_dir / "manufacturing_decisions.json"
    existing_mfg: list[dict] = []
    if mfg_path.exists():
        try:
            existing_mfg = json.loads(mfg_path.read_text()).get("manufacturing_actions", [])
        except Exception:
            existing_mfg = []

    # Aggregate by product within scenario as before, but attach scenario to each action
    mfg_by_product = {}
    for m in manufacturing:
        pid = m['product_id']
        if pid not in mfg_by_product:
            mfg_by_product[pid] = {'quantity': 0, 'cost': 0, 'reasons': set()}
        mfg_by_product[pid]['quantity'] += m['quantity']
        mfg_by_product[pid]['cost'] += m['cost']
        mfg_by_product[pid]['reasons'].update(m['reason_codes'])

    mfg_actions_with_scenario = [
        {
            "scenario": scenario_id,
            "product_id": str(pid),
            "manufacture_quantity": round(data['quantity'], 1),
            "reason_codes": list(data['reasons']),
            "cost_impact": {"manufacturing_cost": round(data['cost'], 2)},
        }
        for pid, data in mfg_by_product.items()
    ]

    merged_mfg = _merge_unique(
        existing_mfg,
        mfg_actions_with_scenario,
        key_fields=["scenario", "product_id"],
    )

    mfg_path.write_text(json.dumps({"scenario": "all", "manufacturing_actions": merged_mfg}, indent=2))

    # --- Scenario summary (combined; keyed by scenario_id) ---
    summary_path = out_dir / "scenario_summary.json"
    existing_summary: dict = {}
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text())
        except Exception:
            existing_summary = {}

    scenarios_dict = existing_summary.get("scenarios") if isinstance(existing_summary, dict) else None
    if not isinstance(scenarios_dict, dict):
        scenarios_dict = {}

    scenarios_dict[scenario_id] = {
        "scenario": scenario_id,
        "optimized": {
            "total_cost": round(costs['total'], 2),
            "total_transfers": len(transfers),
            "manufacturing_units": round(sum(m['quantity'] for m in manufacturing), 1),
            "transfer_units": round(sum(t['quantity'] for t in transfers), 1),
        },
        "cost_breakdown": {
            "manufacturing_cost": round(costs['manufacturing'], 2),
            "transfer_cost": round(costs['transfer'], 2),
            "holding_cost": round(costs['holding'], 2),
        },
    }

    scenario_ids = sorted(scenarios_dict.keys())
    summary_path.write_text(
        json.dumps({"scenario": "all", "scenarios": scenarios_dict, "scenario_ids": scenario_ids}, indent=2)
    )

    print(f"JSON outputs updated under {output_dir}/ (flat 3-file schema). Added scenario_id='{scenario_id}'")


def run_optimization(
    scenario_id: str = "base_case_standard_conditions",
    output_dir: str = "output-json",
    output_csv_dir: str = "output-csv",
    time_limit: int = 300,
    seed: int = 42,
) -> dict:
    """Run the optimization end-to-end and write outputs.

    This wraps the module's original script logic so it can be invoked from:
    - CLI (python optimization/optimization.py --scenario-id ...)
    - Streamlit app (subprocess call)

    Returns a small summary dict.
    """

    scen = ScenarioConfig.get_scenario(scenario_id)

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    # 1. LOAD DATA (paths resolved from repo root)
    forecast_path = repo_root / 'demand-forecast' / 'output' / 'product_forecasts_wide.csv'
    historical_path = base_dir / 'input' / 'processed_store_product_params.csv'
    store_params_path = base_dir / 'input' / 'store_supply_params.csv'
    transport_matrix_path = base_dir / 'input' / 'transport_cost_matrix.csv'

    forecast = pd.read_csv(forecast_path)
    forecast['total_demand_7d'] = forecast[[f'day+{i}' for i in range(1, 8)]].sum(axis=1)
    forecast['avg_daily_demand'] = forecast['total_demand_7d'] / 7

    historical = pd.read_csv(historical_path)
    store_params = pd.read_csv(store_params_path)
    transport_matrix = pd.read_csv(transport_matrix_path, index_col=0).values

    # 2. PREPARE DATA
    demand_df = forecast.merge(
        historical[['store_id', 'product_id', 'demand_std', 'city_id']],
        on=['store_id', 'product_id'], how='left'
    )
    demand_df['demand_std'] = demand_df['demand_std'].fillna(demand_df['avg_daily_demand'] * 0.5)
    demand_df['city_id'] = demand_df['city_id'].fillna(0).astype(int)

    demand_df = demand_df.merge(
        store_params[['store_id', 'lead_time_days_mean', 'delay_probability_mean']],
        on='store_id', how='left'
    )
    demand_df['lead_time_days_mean'] = demand_df['lead_time_days_mean'].fillna(5)
    demand_df['delay_probability_mean'] = demand_df['delay_probability_mean'].fillna(0.7)

    # Apply scenario multipliers
    demand_df['total_demand_7d'] = demand_df['total_demand_7d'] * float(scen.get('demand_multiplier', 1.0))
    demand_df['avg_daily_demand'] = demand_df['avg_daily_demand'] * float(scen.get('demand_multiplier', 1.0))
    demand_df['lead_time_days_mean'] = demand_df['lead_time_days_mean'] * float(scen.get('lead_time_multiplier', 1.0))
    demand_df['delay_probability_mean'] = demand_df['delay_probability_mean'] * float(scen.get('delay_probability_multiplier', 1.0))
    demand_df['delay_probability_mean'] = demand_df['delay_probability_mean'].clip(lower=0.0, upper=1.0)

    np.random.seed(seed)
    if 'current_inventory' in historical.columns:
        inv_lookup_ = historical.set_index(['store_id', 'product_id'])['current_inventory'].to_dict()
        demand_df['current_inventory'] = demand_df.apply(
            lambda r: inv_lookup_.get((r['store_id'], r['product_id']), r['total_demand_7d'] * np.random.uniform(0.3, 0.8)), axis=1
        )
    else:
        demand_df['current_inventory'] = demand_df['total_demand_7d'] * np.random.uniform(0.3, 0.8, len(demand_df))

    # 3. SAFETY STOCK
    Z_95 = 1.65
    demand_df['risk_factor'] = 1 + demand_df['delay_probability_mean']
    demand_df['safety_stock'] = (
        Z_95 * demand_df['demand_std'] *
        np.sqrt(demand_df['lead_time_days_mean']) *
        demand_df['risk_factor']
    )
    demand_df['safety_stock'] = demand_df['safety_stock'] * float(scen.get('safety_stock_multiplier', 1.0))
    demand_df['target_inventory'] = demand_df['total_demand_7d'] + demand_df['safety_stock']

    # 4. OPTIMIZATION SETUP
    stores = sorted(demand_df['store_id'].unique())
    valid_pairs = set(zip(demand_df['store_id'], demand_df['product_id']))
    products_per_store = {s: [p for (s2, p) in valid_pairs if s2 == s] for s in stores}

    demand_lookup = demand_df.set_index(['store_id', 'product_id'])['total_demand_7d'].to_dict()
    safety_lookup = demand_df.set_index(['store_id', 'product_id'])['safety_stock'].to_dict()
    inv_lookup = demand_df.set_index(['store_id', 'product_id'])['current_inventory'].to_dict()
    shipping_lookup = store_params.set_index('store_id')['shipping_costs_mean'].to_dict()

    MFG_BASE = 50
    HOLDING_COST = 1.0
    TRANSPORT_SCALE = 0.1
    MFG_CAPACITY = 5000

    mfg_cost = {s: MFG_BASE * (1 + shipping_lookup.get(s, 450) / 1000) for s in stores}

    transport_cost = {}
    transport_mult = float(scen.get('transport_cost_multiplier', 1.0))
    for i in stores:
        for j in stores:
            if i != j and i < transport_matrix.shape[0] and j < transport_matrix.shape[1]:
                transport_cost[(i, j)] = transport_matrix[i, j] * TRANSPORT_SCALE * transport_mult
            else:
                transport_cost[(i, j)] = 0 if i == j else 5.0 * transport_mult

    model = LpProblem("Inventory_Optimization", LpMinimize)

    x = LpVariable.dicts("mfg", list(valid_pairs), lowBound=0)
    final_inv = LpVariable.dicts("final_inv", list(valid_pairs), lowBound=0)

    all_products = set(p for (s, p) in valid_pairs)
    t = LpVariable.dicts(
        "transfer",
        [(i, j, p) for i in stores for j in stores for p in all_products
         if i != j and (i, p) in valid_pairs and (j, p) in valid_pairs],
        lowBound=0,
    )

    model += (
        lpSum(mfg_cost[s] * x[(s, p)] for (s, p) in valid_pairs) +
        lpSum(transport_cost.get((i, j), 5) * t[(i, j, p)]
              for i in stores for j in stores for p in all_products
              if i != j and (i, p) in valid_pairs and (j, p) in valid_pairs) +
        lpSum(HOLDING_COST * final_inv[(s, p)] for (s, p) in valid_pairs)
    )

    for (s, p) in valid_pairs:
        current = inv_lookup.get((s, p), 0)
        transfers_in = lpSum(t[(i, s, p)] for i in stores if i != s and (i, p) in valid_pairs and (i, s, p) in t)
        transfers_out = lpSum(t[(s, j, p)] for j in stores if j != s and (j, p) in valid_pairs and (s, j, p) in t)

        model += final_inv[(s, p)] == current + x[(s, p)] + transfers_in - transfers_out
        target = demand_lookup.get((s, p), 0) + safety_lookup.get((s, p), 0)
        model += final_inv[(s, p)] >= target
        model += transfers_out <= current

    for s in stores:
        model += lpSum(x[(s, p)] for p in products_per_store[s]) <= MFG_CAPACITY

    status = model.solve(PULP_CBC_CMD(msg=0, timeLimit=int(time_limit)))

    demand_df['demand_cv'] = (demand_df['demand_std'] / demand_df['avg_daily_demand']).fillna(0.5)
    cv_lookup = demand_df.set_index(['store_id', 'product_id'])['demand_cv'].to_dict()
    delay_lookup = demand_df.groupby('store_id')['delay_probability_mean'].first().to_dict()

    reason_data = {
        'inv': inv_lookup,
        'demand': demand_lookup,
        'safety': safety_lookup,
        'cv': cv_lookup,
        'delay': delay_lookup,
        'transport': transport_cost,
        'mfg': mfg_cost,
        'capacity': MFG_CAPACITY,
    }

    store_mfg_total = {}
    for (s, p) in valid_pairs:
        qty = value(x[(s, p)])
        if qty and qty > 0.01:
            store_mfg_total[s] = store_mfg_total.get(s, 0) + qty

    mfg_results = []
    for (s, p) in valid_pairs:
        qty = value(x[(s, p)])
        if qty and qty > 0.01:
            reasons = assign_manufacturing_reasons(s, p, qty, reason_data, store_mfg_total)
            mfg_results.append({
                'store_id': s, 'product_id': p, 'qty': round(qty, 2),
                'cost': round(qty * mfg_cost[s], 2), 'reason_codes': reasons,
            })

    transfer_results = []
    for (i, j, p) in t.keys():
        qty = value(t[(i, j, p)])
        if qty and qty > 0.01:
            reasons = assign_transfer_reasons(i, j, p, qty, reason_data)
            transfer_results.append({
                'from_store': i, 'to_store': j, 'product_id': p,
                'qty': round(qty, 2), 'cost': round(qty * transport_cost.get((i, j), 5), 2),
                'reason_codes': reasons,
            })

    inventory_results = [
        {'store_id': s, 'product_id': p,
         'current': round(inv_lookup.get((s, p), 0), 2),
         'final': round(value(final_inv[(s, p)]), 2),
         'target': round(demand_lookup.get((s, p), 0) + safety_lookup.get((s, p), 0), 2)}
        for (s, p) in valid_pairs
    ]

    mfg_df = pd.DataFrame(mfg_results)
    transfer_df = pd.DataFrame(transfer_results)
    inventory_df = pd.DataFrame(inventory_results)

    total_mfg = sum(value(x[(s, p)]) * mfg_cost[s] for (s, p) in valid_pairs)
    total_transfer = sum(value(t[k]) * transport_cost.get((k[0], k[1]), 5) for k in t.keys())
    total_holding = sum(HOLDING_COST * value(final_inv[(s, p)]) for (s, p) in valid_pairs)
    total_cost = total_mfg + total_transfer + total_holding

    Path(output_csv_dir).mkdir(parents=True, exist_ok=True)
    mfg_df.to_csv(f'{output_csv_dir}/optimization_manufacturing.csv', index=False)
    transfer_df.to_csv(f'{output_csv_dir}/optimization_transfers.csv', index=False)
    inventory_df.to_csv(f'{output_csv_dir}/optimization_inventory.csv', index=False)

    transfers_json = [
        {
            'from_store': str(r['from_store']),
            'to_store': str(r['to_store']),
            'product_id': str(r['product_id']),
            'quantity': r['qty'],
            'reason_codes': r['reason_codes'],
            'cost_impact': {'transport_cost': r['cost']},
        }
        for r in transfer_results
    ]
    mfg_json = [
        {
            'store_id': r['store_id'],
            'product_id': r['product_id'],
            'quantity': r['qty'],
            'cost': r['cost'],
            'reason_codes': r['reason_codes'],
        }
        for r in mfg_results
    ]

    costs = {'total': total_cost, 'manufacturing': total_mfg, 'transfer': total_transfer, 'holding': total_holding}
    save_json_outputs(transfers_json, mfg_json, costs, output_dir, scenario_id=scenario_id)

    return {
        'scenario': scenario_id,
        'status': LpStatus[status],
        'total_cost': float(total_cost),
        'output_dir': output_dir,
    }


def run_all_scenarios(
    output_root: str = "output-json",
    output_csv_root: str = "output-csv",
    time_limit: int = 300,
    seed: int = 42,
) -> dict:
    """Run optimization for every scenario in ScenarioConfig.

    Outputs (flat):
      - output_root/transfer_recommendations.json
      - output_root/manufacturing_decisions.json
      - output_root/scenario_summary.json

    CSVs remain scenario-specific under output_csv_root/<scenario_id>/.
    """
    results = {}

    # Ensure output_root exists and (optionally) clean legacy scenario folders
    Path(output_root).mkdir(parents=True, exist_ok=True)

    for scenario_id in ScenarioConfig.SCENARIOS.keys():
        scenario_csv = str(Path(output_csv_root) / scenario_id)
        res = run_optimization(
            scenario_id=scenario_id,
            output_dir=output_root,
            output_csv_dir=scenario_csv,
            time_limit=time_limit,
            seed=seed,
        )
        results[scenario_id] = res

    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inventory optimization and write outputs")
    p.add_argument('--scenario-id', default='base_case_standard_conditions')
    p.add_argument('--all-scenarios', action='store_true', help='Run all scenarios and write to per-scenario folders')
    p.add_argument('--output-dir', default='output-json')
    p.add_argument('--output-csv-dir', default='output-csv')
    p.add_argument('--time-limit', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args()
    if args.all_scenarios:
        run_all_scenarios(
            output_root=args.output_dir,
            output_csv_root=args.output_csv_dir,
            time_limit=args.time_limit,
            seed=args.seed,
        )
    else:
        run_optimization(
            scenario_id=args.scenario_id,
            output_dir=args.output_dir,
            output_csv_dir=args.output_csv_dir,
            time_limit=args.time_limit,
            seed=args.seed,
        )
    raise SystemExit(0)

