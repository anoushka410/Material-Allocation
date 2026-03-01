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


def save_json_outputs(transfers, manufacturing, costs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Transfer recommendations
    transfer_json = {"scenario": "optimization_run", "transfers": transfers}
    with open(f'{output_dir}/transfer_recommendations.json', 'w') as f:
        json.dump(transfer_json, f, indent=2)
    
    # 2. Manufacturing decisions (aggregate by product)
    mfg_by_product = {}
    for m in manufacturing:
        pid = m['product_id']
        if pid not in mfg_by_product:
            mfg_by_product[pid] = {'quantity': 0, 'cost': 0, 'reasons': set()}
        mfg_by_product[pid]['quantity'] += m['quantity']
        mfg_by_product[pid]['cost'] += m['cost']
        mfg_by_product[pid]['reasons'].update(m['reason_codes'])
    
    mfg_json = {
        "scenario": "optimization_run",
        "manufacturing_actions": [
            {
                "product_id": str(pid),
                "manufacture_quantity": round(data['quantity'], 1),
                "reason_codes": list(data['reasons']),
                "cost_impact": {"manufacturing_cost": round(data['cost'], 2)}
            }
            for pid, data in mfg_by_product.items()
        ]
    }
    with open(f'{output_dir}/manufacturing_decisions.json', 'w') as f:
        json.dump(mfg_json, f, indent=2)
    
    # 3. Scenario summary
    scenario_json = {
        "scenario": "optimization_run",
        "optimized": {
            "total_cost": round(costs['total'], 2),
            "total_transfers": len(transfers),
            "manufacturing_units": round(sum(m['quantity'] for m in manufacturing), 1),
            "transfer_units": round(sum(t['quantity'] for t in transfers), 1)
        },
        "cost_breakdown": {
            "manufacturing_cost": round(costs['manufacturing'], 2),
            "transfer_cost": round(costs['transfer'], 2),
            "holding_cost": round(costs['holding'], 2)
        }
    }
    with open(f'{output_dir}/scenario_summary.json', 'w') as f:
        json.dump(scenario_json, f, indent=2)
    
    print(f"JSON outputs saved to {output_dir}/")

# 1. LOAD DATA

# Demand forecasts (7-day horizon per store-product)
forecast = pd.read_csv('../demand-forecast/output/product_forecasts_wide.csv')
forecast['total_demand_7d'] = forecast[[f'day+{i}' for i in range(1, 8)]].sum(axis=1)
forecast['avg_daily_demand'] = forecast['total_demand_7d'] / 7

# Historical parameters (for demand_std, safety stock calculation)
historical = pd.read_csv('input/processed_store_product_params.csv')

# Store supply parameters (lead times, delay probability)
store_params = pd.read_csv('input/store_supply_params.csv')

# Transport cost matrix (store-to-store)
transport_matrix = pd.read_csv('input/transport_cost_matrix.csv', index_col=0).values

# 2. PREPARE DATA

# Merge demand with historical std
demand_df = forecast.merge(
    historical[['store_id', 'product_id', 'demand_std', 'city_id']],
    on=['store_id', 'product_id'], how='left'
)
demand_df['demand_std'] = demand_df['demand_std'].fillna(demand_df['avg_daily_demand'] * 0.5)
demand_df['city_id'] = demand_df['city_id'].fillna(0).astype(int)

# Add supply chain params
demand_df = demand_df.merge(
    store_params[['store_id', 'lead_time_days_mean', 'delay_probability_mean']],
    on='store_id', how='left'
)
demand_df['lead_time_days_mean'] = demand_df['lead_time_days_mean'].fillna(5)
demand_df['delay_probability_mean'] = demand_df['delay_probability_mean'].fillna(0.7)

# Simulate current inventory (in production: from inventory system)
np.random.seed(42)
if 'current_inventory' in historical.columns:
    inv_lookup = historical.set_index(['store_id', 'product_id'])['current_inventory'].to_dict()
    demand_df['current_inventory'] = demand_df.apply(
        lambda r: inv_lookup.get((r['store_id'], r['product_id']), r['total_demand_7d'] * np.random.uniform(0.3, 0.8)), axis=1
    )
else:
    demand_df['current_inventory'] = demand_df['total_demand_7d'] * np.random.uniform(0.3, 0.8, len(demand_df))

# 3. SAFETY STOCK CALCULATION
# Formula: SS = z × σ × √L × risk_factor

Z_95 = 1.65  # 95% service level
demand_df['risk_factor'] = 1 + demand_df['delay_probability_mean']
demand_df['safety_stock'] = (
    Z_95 * demand_df['demand_std'] * 
    np.sqrt(demand_df['lead_time_days_mean']) * 
    demand_df['risk_factor']
)
demand_df['target_inventory'] = demand_df['total_demand_7d'] + demand_df['safety_stock']

# 4. OPTIMIZATION SETUP

# Only consider valid (store, product) pairs from forecast
# Forecast already contains top 50 products per store (by total sale_amount)
stores = sorted(demand_df['store_id'].unique())
valid_pairs = set(zip(demand_df['store_id'], demand_df['product_id']))
products_per_store = {s: [p for (s2, p) in valid_pairs if s2 == s] for s in stores}
n_stores = len(stores)
n_pairs = len(valid_pairs)

print(f"Scope: {n_stores} stores, {n_pairs} store-product pairs (top 50 products/store)")

# Lookup dictionaries
demand_lookup = demand_df.set_index(['store_id', 'product_id'])['total_demand_7d'].to_dict()
safety_lookup = demand_df.set_index(['store_id', 'product_id'])['safety_stock'].to_dict()
inv_lookup = demand_df.set_index(['store_id', 'product_id'])['current_inventory'].to_dict()
shipping_lookup = store_params.set_index('store_id')['shipping_costs_mean'].to_dict()

# Cost parameters
MFG_BASE = 50
HOLDING_COST = 1.0
TRANSPORT_SCALE = 0.1
MFG_CAPACITY = 5000

mfg_cost = {s: MFG_BASE * (1 + shipping_lookup.get(s, 450) / 1000) for s in stores}

transport_cost = {}
for i in stores:
    for j in stores:
        if i != j and i < transport_matrix.shape[0] and j < transport_matrix.shape[1]:
            transport_cost[(i, j)] = transport_matrix[i, j] * TRANSPORT_SCALE
        else:
            transport_cost[(i, j)] = 0 if i == j else 5.0

# =============================================================================
# 5. BUILD OPTIMIZATION MODEL
#
# Decision Variables:
#   x[s,p]     = manufacturing qty for store s, product p
#   t[i,j,p]   = transfer qty from store i to store j for product p
#   final_inv  = auxiliary for final inventory
#
# Objective: Minimize Total Cost
#   min Σ(mfg_cost × x) + Σ(transport_cost × t) + Σ(holding_cost × final_inv)
#
# Constraints:
#   1. Inventory balance: final_inv = current + manufactured + in - out
#   2. Meet demand: final_inv ≥ demand + safety_stock
#   3. Transfer limit: outgoing transfers ≤ current inventory
#   4. Capacity: total manufacturing per store ≤ MFG_CAPACITY
# =============================================================================

model = LpProblem("Inventory_Optimization", LpMinimize)

# Decision variables - only for valid (store, product) pairs
x = LpVariable.dicts("mfg", list(valid_pairs), lowBound=0)
final_inv = LpVariable.dicts("final_inv", list(valid_pairs), lowBound=0)

# Transfer variables: only between stores for same product (if product exists at both)
all_products = set(p for (s, p) in valid_pairs)
t = LpVariable.dicts("transfer", 
    [(i, j, p) for i in stores for j in stores for p in all_products 
     if i != j and (i, p) in valid_pairs and (j, p) in valid_pairs], 
    lowBound=0)

# Objective: minimize total cost
model += (
    lpSum(mfg_cost[s] * x[(s, p)] for (s, p) in valid_pairs) +
    lpSum(transport_cost.get((i, j), 5) * t[(i, j, p)] 
          for i in stores for j in stores for p in all_products 
          if i != j and (i, p) in valid_pairs and (j, p) in valid_pairs) +
    lpSum(HOLDING_COST * final_inv[(s, p)] for (s, p) in valid_pairs)
)

# Constraints - only for valid pairs
for (s, p) in valid_pairs:
    current = inv_lookup.get((s, p), 0)
    transfers_in = lpSum(t[(i, s, p)] for i in stores if i != s and (i, p) in valid_pairs and (i, s, p) in t)
    transfers_out = lpSum(t[(s, j, p)] for j in stores if j != s and (j, p) in valid_pairs and (s, j, p) in t)
    
    # 1. Inventory balance
    model += final_inv[(s, p)] == current + x[(s, p)] + transfers_in - transfers_out
    
    # 2. Meet demand + safety stock
    target = demand_lookup.get((s, p), 0) + safety_lookup.get((s, p), 0)
    model += final_inv[(s, p)] >= target
    
    # 3. Transfer limit
    model += transfers_out <= current

# 4. Manufacturing capacity per store
for s in stores:
    model += lpSum(x[(s, p)] for p in products_per_store[s]) <= MFG_CAPACITY

# 6. SOLVE

print(f"Solving: {n_stores} stores, {n_pairs} store-product pairs")
status = model.solve(PULP_CBC_CMD(msg=0, timeLimit=300))
print(f"Status: {LpStatus[status]}")

# 7. EXTRACT RESULTS

# Build data dict for reason code assignment
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
    'capacity': MFG_CAPACITY
}

# Calculate total mfg per store for capacity check
store_mfg_total = {}
for (s, p) in valid_pairs:
    qty = value(x[(s, p)])
    if qty > 0.01:
        store_mfg_total[s] = store_mfg_total.get(s, 0) + qty

# Manufacturing decisions with reason codes
mfg_results = []
for (s, p) in valid_pairs:
    qty = value(x[(s, p)])
    if qty > 0.01:
        reasons = assign_manufacturing_reasons(s, p, qty, reason_data, store_mfg_total)
        mfg_results.append({
            'store_id': s, 'product_id': p, 'qty': round(qty, 2),
            'cost': round(qty * mfg_cost[s], 2), 'reason_codes': reasons
        })
mfg_df = pd.DataFrame(mfg_results)

# Transfer decisions with reason codes
transfer_results = []
for (i, j, p) in t.keys():
    qty = value(t[(i, j, p)])
    if qty > 0.01:
        reasons = assign_transfer_reasons(i, j, p, qty, reason_data)
        transfer_results.append({
            'from_store': i, 'to_store': j, 'product_id': p,
            'qty': round(qty, 2), 'cost': round(qty * transport_cost.get((i, j), 5), 2),
            'reason_codes': reasons
        })
transfer_df = pd.DataFrame(transfer_results)

# Final inventory
inventory_results = [
    {'store_id': s, 'product_id': p, 
     'current': round(inv_lookup.get((s, p), 0), 2),
     'final': round(value(final_inv[(s, p)]), 2),
     'target': round(demand_lookup.get((s, p), 0) + safety_lookup.get((s, p), 0), 2)}
    for (s, p) in valid_pairs
]
inventory_df = pd.DataFrame(inventory_results)

# 8. COST SUMMARY

total_mfg = sum(value(x[(s, p)]) * mfg_cost[s] for (s, p) in valid_pairs)
total_transfer = sum(value(t[k]) * transport_cost.get((k[0], k[1]), 5) for k in t.keys())
total_holding = sum(HOLDING_COST * value(final_inv[(s, p)]) for (s, p) in valid_pairs)
total_cost = total_mfg + total_transfer + total_holding

print(f"\n{'='*50}")
print(f"COST BREAKDOWN")
print(f"{'='*50}")
print(f"Manufacturing: ${total_mfg:>12,.2f} ({100*total_mfg/total_cost:.1f}%)")
print(f"Transfer:      ${total_transfer:>12,.2f} ({100*total_transfer/total_cost:.1f}%)")
print(f"Holding:       ${total_holding:>12,.2f} ({100*total_holding/total_cost:.1f}%)")
print(f"{'='*50}")
print(f"TOTAL:         ${total_cost:>12,.2f}")
print(f"\nManufacturing: {mfg_df['qty'].sum() if len(mfg_df) else 0:,.1f} units")
print(f"Transfers:     {transfer_df['qty'].sum() if len(transfer_df) else 0:,.1f} units")

# 9. SAVE OUTPUTS

# CSV outputs
output_dir = 'output-csv'
os.makedirs(output_dir, exist_ok=True)
mfg_df.to_csv(f'{output_dir}/optimization_manufacturing.csv', index=False)
transfer_df.to_csv(f'{output_dir}/optimization_transfers.csv', index=False)
inventory_df.to_csv(f'{output_dir}/optimization_inventory.csv', index=False)
print(f"CSV outputs saved to {output_dir}/")

# JSON outputs for NLP layer
json_output_dir = 'output-json'
transfers_json = [
    {
        'from_store': str(r['from_store']),
        'to_store': str(r['to_store']),
        'product_id': str(r['product_id']),
        'quantity': r['qty'],
        'reason_codes': r['reason_codes'],
        'cost_impact': {'transport_cost': r['cost']}
    }
    for r in transfer_results
]
mfg_json = [
    {
        'store_id': r['store_id'],
        'product_id': r['product_id'],
        'quantity': r['qty'],
        'cost': r['cost'],
        'reason_codes': r['reason_codes']
    }
    for r in mfg_results
]
costs = {'total': total_cost, 'manufacturing': total_mfg, 'transfer': total_transfer, 'holding': total_holding}
save_json_outputs(transfers_json, mfg_json, costs, json_output_dir)

