# Capstone - Material Allocation

## Datasets

### 1. FreshRetailNet-50K
**Source:** [HuggingFace](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)  
**Size:** 4.5M rows

Fresh grocery retail demand forecasting data from Dingdong (Chinese grocery delivery). Contains daily sales per product/store with hourly breakdowns, stock availability, discounts, promotions, and weather conditions.

**Key columns:** `sale_amount`, `hours_sale`, `stock_hour6_22_cnt`, `discount`, `holiday_flag`, `avg_temperature`

---

### 2. Dynamic Supply Chain Logistics
**Size:** ~32K rows

Simulated supply chain operations data with vehicle tracking, warehouse inventory, delivery metrics, and risk scoring. Includes GPS coordinates, fuel consumption, traffic/weather conditions, and disruption likelihood.

**Key columns:** `warehouse_inventory_level`, `lead_time_days`, `delay_probability`, `risk_classification`, `delivery_time_deviation`
