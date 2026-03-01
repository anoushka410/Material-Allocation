# Data Preprocessing Updates
**Risk-Aware Material Allocation Optimization | Group 7**  

---

## Overview

This document summarizes the key updates made to the data preprocessing pipeline (`01_data_preprocessing_and_eda.ipynb`) to support product-level demand forecasting and inventory optimization.

---

## 1. Store × Product Level Aggregation

### Change
Shifted from **store-level** aggregation to **store × product** level.

### Rationale
- Demand forecasting requires product-level granularity
- Inventory optimization decisions are made per product per location
- Different products have distinct demand patterns and variability

### Implementation
```python
# Before: Aggregated at store level only
plant_daily = demand_df.groupby(['store_id', 'city_id', 'dt']).agg({...})

# After: Aggregated at store × product level
store_product_daily = demand_df.groupby(['store_id', 'product_id', 'city_id', 'dt']).agg({...})
```

### Output
| Dataset | Rows | Description |
|---------|------|-------------|
| `processed_store_product_daily.csv` | 4.5M | Daily demand per store-product |
| `processed_store_product_params.csv` | 50,000 | Aggregated parameters per store-product |

---

## 2. Spatial Clustering for Supply Chain Parameter Mapping

### Change
Used **K-Means spatial clustering** to map supply chain parameters to cities using actual data values.

### Data Source
- GPS coordinates (`vehicle_gps_latitude`, `vehicle_gps_longitude`) from 32K supply chain records
- Supply chain parameters: `shipping_costs`, `lead_time_days`, `delay_probability`, `route_risk_level`, etc.

### Methodology

1. **K-Means Clustering**  
   Clustered 32K supply chain observations into 18 geographic regions:
   ```python
   kmeans = KMeans(n_clusters=18, random_state=42)
   supply_df['geo_cluster'] = kmeans.fit_predict(gps_coords)
   ```

2. **Parameter Aggregation**  
   Aggregated actual supply chain values per cluster:
   - Shipping costs (mean, std)
   - Lead times (mean, std)
   - Delay probability, route risk, supplier reliability

3. **City-Cluster Mapping**  
   Mapped demand cities to clusters based on geographic ordering (west to east).

4. **Transport Cost Derivation**  
   Combined cluster shipping costs with Haversine distances:
   ```
   Cost(i→j) = base_shipping_cost_i × distance_factor(i,j)
   ```

### Output Files
| File | Description |
|------|-------------|
| `store_supply_params.csv` | Store-level supply chain parameters (898 stores) |
| `transport_cost_matrix.csv` | 898×898 store-to-store costs |

---

## 3. Store × Product Parameters

### New Parameters Computed
| Parameter | Description |
|-----------|-------------|
| `avg_daily_demand` | Mean daily demand per store-product |
| `demand_std` | Standard deviation of daily demand |
| `demand_cv` | Coefficient of variation (std/mean) |
| `total_demand` | Sum of demand over 90-day period |
| `avg_stockout_hours` | Average stockout hours per day |
| `holiday_proportion` | Fraction of days that were holidays |
| `promotion_proportion` | Fraction of days with promotions |

These parameters directly feed into:
- **Demand forecasting** (as features and targets)
- **Safety stock calculation** (using demand_std and demand_cv)
- **Inventory optimization** (demand estimates and variability)

---

## Summary of Changes

| Aspect | Before | After |
|--------|--------|-------|
| **Aggregation level** | Store × Day | Store × Product × Day |
| **Transport costs** | Random synthetic | Spatial clustering + Haversine distance |
| **Supply chain params** | Global averages only | Per-city parameters from K-Means clusters |
| **Geographic data** | Not used | K-Means on 32K GPS coordinates |
| **Output granularity** | 80K rows (store-day) | 4.5M rows (store-product-day) |
| **Parameter source** | Assumed values | Actual aggregated supply chain data |

---

## Downstream Impact

These preprocessing changes enable:

1. **Product-level demand forecasting** — Train models on individual product demand patterns
2. **Realistic transport costs** — Optimization reflects actual geographic constraints  
3. **Granular inventory decisions** — Optimize stock levels per product per store
4. **Meaningful reallocation** — Transfer costs reflect real distances between locations

