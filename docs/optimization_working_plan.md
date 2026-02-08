# Optimization Model Working Plan
## Risk-Aware Material Allocation Optimization
---

## 1. Problem Statement

We aim to optimize product-level inventory allocation across a retail network by:
1. **Reallocating excess inventory** between stores to reduce stockouts
2. **Determining manufacturing/replenishment quantities** to meet forecasted demand
3. **Minimizing total supply chain costs** while maintaining service levels

---

## 2. Scope and Granularity

| Dimension | Level | Count |
|-----------|-------|-------|
| Geographic | Store (treated as plant) | 898 stores |
| Product | Individual SKU | 865 products |
| Temporal | Daily demand, 7-day planning horizon | 90 days historical |
| Planning | Single-period static optimization | — |

**Key Decision**: Optimization at **store × product** level to enable product-specific inventory decisions.

---

## 3. Model Formulation

### 3.1 Sets and Indices
- `S` = Set of stores (s ∈ S)
- `P` = Set of products (p ∈ P)
- `i, j` = Store indices for transfers

### 3.2 Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| `transfer[i,j,p]` | ≥ 0 | Units of product p transferred from store i to store j |
| `manufacture[p]` | ≥ 0 | Units of product p to produce/order centrally |
| `allocate[s,p]` | ≥ 0 | Units from manufacturing allocated to store s |
| `stockout[s,p]` | ≥ 0 | Unmet demand at store s for product p |
| `ending_inv[s,p]` | ≥ 0 | Ending inventory at store s for product p |

### 3.3 Parameters (Inputs)

| Parameter | Source | Description |
|-----------|--------|-------------|
| `current_inv[s,p]` | Simulated | Current inventory levels |
| `demand[s,p]` | Forecast model output | Forecasted demand for planning horizon |
| `demand_std[s,p]` | Historical data | Demand standard deviation |
| `transport_cost[i,j]` | Spatial clustering + Haversine | Cost from K-Means cluster shipping costs × distance |
| `manufacturing_cost[p]` | Assumed | Cost per unit to produce |
| `distribution_cost[s]` | Assumed | Cost to ship from central plant to store |
| `holding_cost[p]` | Assumed | Cost per unit per period to hold inventory |
| `stockout_penalty[p]` | Assumed | Penalty per unit of unmet demand |
| `safety_stock[s,p]` | Computed | z_α × σ × √(lead_time) |
| `max_capacity[s,p]` | Assumed | Maximum storage capacity |
| `max_production[p]` | Assumed | Manufacturing capacity limit |

### 3.4 Objective Function

**Minimize Total Cost:**

```
Z = Σ_{s,p} holding_cost[p] × ending_inv[s,p]
  + Σ_{s,p} stockout_penalty[p] × stockout[s,p]
  + Σ_{i,j,p} transport_cost[i,j] × transfer[i,j,p]
  + Σ_p manufacturing_cost[p] × manufacture[p]
  + Σ_{s,p} distribution_cost[s] × allocate[s,p]
```

### 3.5 Constraints

**C1. Inventory Balance** (for each store s, product p):
```
ending_inv[s,p] = current_inv[s,p]
                + Σ_i transfer[i,s,p]      # incoming
                - Σ_j transfer[s,j,p]      # outgoing
                + allocate[s,p]            # from manufacturing
                - demand[s,p]              # forecasted demand
                + stockout[s,p]            # unmet (slack)
```

**C2. Transfer Feasibility** (can only transfer excess):
```
Σ_j transfer[s,j,p] ≤ max(0, current_inv[s,p] - safety_stock[s,p])
```

**C3. Manufacturing-Allocation Balance**:
```
Σ_s allocate[s,p] = manufacture[p]
```

**C4. Manufacturing Capacity**:
```
manufacture[p] ≤ max_production[p]
```

**C5. Storage Capacity**:
```
ending_inv[s,p] ≤ max_capacity[s,p]
```

**C6. Non-Negativity**:
```
All decision variables ≥ 0
```

---

## 4. Data Pipeline

### 4.1 Data Sources

| Dataset | Records | Source | Status |
|---------|---------|--------|--------|
| Retail Demand (FreshRetailNet) | 4.5M rows | Hugging Face | ✅ Loaded |
| Supply Chain Logistics | 32K rows | Synthetic | ✅ Loaded |

### 4.2 Derived Data (from EDA notebook)

| Output File | Granularity | Purpose |
|-------------|-------------|---------|
| `processed_store_product_daily.csv` | Store × Product × Day | Input to forecasting (4.5M rows) |
| `processed_store_product_params.csv` | Store × Product | Demand parameters (50K rows) |
| `city_supply_params.csv` | City | Supply chain params from K-Means clustering |
| `store_supply_params.csv` | Store | Store-level supply chain params |
| `transport_cost_matrix.csv` | Store × Store | Distance-based transfer costs (898×898) |

### 4.3 Data Available vs. Assumed

| Data Element | Real/Derived | Assumed |
|--------------|--------------|---------|
| Demand patterns (mean, std, CV) | ✅ | |
| Store/product identifiers | ✅ | |
| **City supply params (K-Means clustering):** | | |
| — Shipping costs per city | ✅ | |
| — Lead time per city | ✅ | |
| — Delay probability per city | ✅ | |
| — Route risk, supplier reliability | ✅ | |
| Store-to-store transport costs | ✅ | |
| City GPS coordinates (cluster centroids) | ✅ | |
| Current inventory levels | | ✅ Simulated |
| Holding costs | | ✅ $0.10/unit/day |
| Stockout penalties | | ✅ $2.50/unit |
| Storage capacity | | ✅ Based on demand |
| Manufacturing costs | | ✅ $1.00/unit |

---

## 5. Implementation Plan

### Phase 1: Demand Forecasting (Notebook 02)
- **Model**: LightGBM global regressor
- **Features**: store_id, product_id, temporal features, promotions, holidays
- **Output**: `demand_forecast.csv` with predicted demand and uncertainty

### Phase 2: Optimization Model (Notebook 03)
- **Solver**: PuLP (Python LP/MIP library)
- **Approach**: Linear program with store × product granularity
- **Scaling**: May need to subset to top products if computational limits hit

### Phase 3: Scenario Analysis (Notebook 04)
- Vary stockout penalties and holding costs
- Simulate demand shocks (±20% demand)
- Compare centralized vs. decentralized allocation strategies

---

## 6. Key Assumptions and Limitations

### Assumptions
1. Single planning period (static optimization, not rolling horizon)
2. Demand forecast is known with uncertainty captured by standard deviation
3. All stores can transfer to all other stores (complete network)
4. Manufacturing has sufficient capacity for aggregate demand
5. Lead times are deterministic

### Limitations
1. No multi-period dynamics (inventory carryover effects simplified)
2. Current inventory state is simulated (no real inventory data available)
3. No supplier constraints or raw material considerations
4. City-cluster mapping based on geographic ordering (not explicit location matching)

---

## 7. Expected Outputs

1. **Optimal Transfer Plan**: Which products to move between which stores
2. **Manufacturing Schedule**: How much of each product to produce
3. **Allocation Plan**: How to distribute manufactured goods to stores
4. **Cost Breakdown**: Holding, stockout, transport, and manufacturing costs
5. **Service Level Achievement**: Percentage of demand fulfilled

---
