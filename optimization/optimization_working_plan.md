# Optimization Model

## Problem

Minimize total supply chain cost while meeting 7-day demand + safety stock for each store-product.

---

## Scope

| Dimension | Value |
|-----------|-------|
| Stores | 20 (top stores from forecast) |
| Products | Top 50 per store |
| Store-product pairs | ~1,000 |
| Planning horizon | 7 days |

---

## Model Formulation

### Decision Variables

| Variable | Description |
|----------|-------------|
| `x[s,p]` | Manufacturing quantity for store s, product p |
| `t[i,j,p]` | Transfer quantity from store i to store j for product p |
| `final_inv[s,p]` | Final inventory (auxiliary) |

### Objective

```
Minimize: Σ(mfg_cost × x) + Σ(transport_cost × t) + Σ(holding_cost × final_inv)
```

### Constraints

1. **Inventory balance**: `final_inv = current + manufactured + transfers_in - transfers_out`
2. **Meet demand**: `final_inv ≥ demand_7d + safety_stock`
3. **Transfer limit**: `transfers_out ≤ current_inventory`
4. **Capacity**: `Σ(x[s,*]) ≤ 5000` per store

---

## Safety Stock Formula

```
safety_stock = 1.65 × demand_std × √(lead_time_days) × (1 + delay_probability)
```

- `1.65` = z-score for 95% service level
- `delay_probability` from supply chain data adds risk buffer

---

## Cost Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Manufacturing cost | $50 × (1 + shipping_factor) | Varies by store |
| Holding cost | $1.0 per unit | Assumed |
| Transport cost | Matrix value × 0.1 | From spatial clustering |
| Capacity | 5,000 units/store | Assumed |

---

## Input Files

| File | Description |
|------|-------------|
| `../demand-forecast/output/product_forecasts_wide.csv` | 7-day demand forecasts |
| `input/processed_store_product_params.csv` | Historical demand parameters |
| `input/store_supply_params.csv` | Lead times, delay probability |
| `input/transport_cost_matrix.csv` | Store-to-store costs |

---

## Output Files

### CSV (output-csv/)
- `optimization_manufacturing.csv` — Manufacturing decisions with reason codes
- `optimization_transfers.csv` — Transfer decisions with reason codes
- `optimization_inventory.csv` — Final inventory positions

### JSON (output-json/) — For NLP Layer
- `transfer_recommendations.json` — Transfer actions with reasons
- `manufacturing_decisions.json` — Manufacturing actions by product
- `scenario_summary.json` — Cost breakdown and totals

---

## Reason Codes

### Transfer Reasons
- `projected_stockout_at_destination`
- `excess_inventory_at_source`
- `safety_stock_violation_prevented`
- `high_demand_variability` (CV > 0.7)
- `high_delay_probability` (delay > 0.5)
- `transport_cost_acceptable`

### Manufacturing Reasons
- `manufacture_to_avoid_stockout`
- `safety_stock_violation_prevented`
- `high_demand_variability`
- `high_delay_probability`
- `manufacturing_capacity_constrained` (>90% capacity)

---

## Solver

- **Library**: PuLP
- **Solver**: CBC (COIN-OR Branch and Cut)
- **Time limit**: 300 seconds
