# Supply Chain Optimization & NLP Analytics Platform

A comprehensive supply chain optimization system with AI-powered NLP interface for analyzing transfers, manufacturing decisions, and inventory management across multiple scenarios.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Components](#architecture--components)
3. [Datasets](#datasets)
4. [Prerequisites & Setup](#prerequisites--setup)
5. [Installation](#installation)
6. [Running the Application](#running-the-application)
7. [Key Features](#key-features)
8. [Project Structure](#project-structure)

---

## Project Overview

This capstone project combines **supply chain optimization** with **natural language processing** to provide:

- **Demand Forecasting**: Predict future product demand using causal analysis and time-series models
- **Inventory Optimization**: Determine optimal inventory levels across multiple stores
- **Transfer Recommendations**: Suggest inter-store product transfers to minimize costs and prevent stockouts
- **Manufacturing Planning**: Optimize production decisions based on demand and inventory levels
- **NLP Interface**: Query optimization results using natural language (powered by Ollama/TinyLLaMA)
- **Multi-Scenario Analysis**: Compare different optimization scenarios and their financial impacts

### Key Business Goals

✓ Minimize total supply chain costs (manufacturing + logistics + holding)
✓ Prevent stockouts while maintaining inventory safety stocks
✓ Optimize inventory distribution across stores
✓ Provide actionable insights through conversational AI

---

## Architecture & Components

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB INTERFACE                   │
│  (Chat + Optimization Dashboard + KPI Export)               │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼──────────┐  ┌──────▼─────────────┐
│   NLP Pipeline   │  │   Optimization     │
│  (Intent Classify│  │    Module          │
│   + LLM Refiner) │  │  (PuLP Solver)     │
└───────┬──────────┘  └──────┬─────────────┘
        │                     │
   ┌────▼─────────────────────▼────┐
   │    Explanation Engine          │
   │  (Markdown Tables + Badges)    │
   └────┬─────────────────────────┬─┘
        │                         │
   ┌────▼──────┐         ┌───────▼────┐
   │   Ollama   │         │  JSON Data │
   │ TinyLLaMA  │         │   Files    │
   │  (Mistral) │         │ (Transfers,│
   └────────────┘         │ Inventory) │
                          └────────────┘
```

### Core Modules

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **NLP Pipeline** | Intent classification, LLM refinement | `nlp/intent_classifier.py`, `nlp/explanation_engine.py`, `nlp/llm_client.py` |
| **Demand Forecasting** | Time-series prediction, causal analysis | `demand-forecast/forecaster.py`, `demand-forecast/causal.py` |
| **Optimization** | Cost minimization via linear programming | `optimization/optimization.py`, `optimization/stochastic.py` |
| **Monitoring** | Data drift detection | `monitoring/drift.py` |
| **Export** | KPI data export to CSV/JSON | `export/kpi_export.py` |
| **EDA** | Data preprocessing & visualization | `EDA/data_preprocessing_and_eda.ipynb` |

### Data Flow

```
Raw Data (CSV)
    ↓
EDA & Preprocessing
    ↓
Demand Forecasting
    ↓
Optimization (PuLP Solver)
    ↓
JSON Output Files
    ↓
Streamlit Interface (Chat + Dashboard)
    ↓
User Queries → NLP → Explanation Engine → LLM Refining → Response
```

---

## Datasets

### 1. FreshRetailNet-50K
**Source:** [HuggingFace](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
**File:** `data/freshretailnet_data_100.csv`
**Size:** 4.5M rows (sampled to 100 rows for demo)

Fresh grocery retail demand forecasting data from Dingdong (Chinese grocery delivery). Contains daily sales per product/store with hourly breakdowns, stock availability, discounts, promotions, and weather conditions.

**Key columns:**
- `sale_amount`: Daily sales revenue
- `hours_sale`: Hourly breakdown of sales
- `stock_hour6_22_cnt`: Inventory levels
- `discount`: Promotional discount percentage
- `holiday_flag`: Holiday indicator
- `avg_temperature`: Weather condition

### 2. Dynamic Supply Chain Logistics
**File:** `data/dynamic_supply_chain_logistics_dataset.csv`
**Size:** ~32K rows

Simulated supply chain operations data with vehicle tracking, warehouse inventory, delivery metrics, and risk scoring. Includes GPS coordinates, fuel consumption, traffic/weather conditions, and disruption likelihood.

**Key columns:**
- `warehouse_inventory_level`: Current stock levels
- `lead_time_days`: Supplier lead time
- `delay_probability`: Likelihood of delivery delay
- `risk_classification`: Risk level (low/medium/high)
- `delivery_time_deviation`: Actual vs planned delivery variance

---

## Prerequisites & Setup

### System Requirements

- **OS:** macOS (Apple Silicon recommended) / Linux / Windows
- **Python:** 3.9+
- **RAM:** 8GB minimum (16GB recommended for optimization runs)
- **Disk:** 10GB for models and data

### Required Software

#### 1. **Ollama** (for LLM inference)

Ollama provides local LLM inference without cloud dependencies.

**Install Ollama:**

```bash
# macOS (Homebrew)
brew install ollama

# Or download from official site
curl -L https://ollama.com/download -o ollama.dmg
open ollama.dmg
```

**Verify installation:**
```bash
ollama --version
```

#### 2. **TinyLLaMA or Mistral** (LLM Models)

Download at least one LLM model via Ollama:

```bash
# Option 1: TinyLLaMA (recommended for Mac, ~637MB)
ollama pull tinyllama

# Option 2: Mistral (faster, ~4.4GB)
ollama pull mistral

# List available models
ollama list
```

**Note:** Models are downloaded to `~/.ollama/models/`

#### 3. **Python Dependencies**

All dependencies are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key packages:**
- `streamlit >= 1.40.0` - Web interface
- `pandas >= 2.0.0` - Data manipulation
- `pulp >= 2.7.0` - Linear programming solver
- `plotly >= 5.18.0` - Interactive visualizations
- `pydantic >= 2.0.0` - Data validation
- `requests >= 2.31.0` - HTTP client for Ollama API

---

## Installation

### Step 1: Clone/Setup Project

```bash
cd <project-directory>
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Ollama Setup

```bash
# Check Ollama installation
ollama --version

# Check downloaded models
ollama list
# Expected output:
# NAME              ID              SIZE      MODIFIED
# tinyllama:latest  2644915ede35    637 MB    10 days ago
# mistral:latest    6577803aa9a0    4.4 GB    10 days ago
```

---

## Running the Application

### Step 1: Start Ollama Service

**Important:** Ollama must be running BEFORE starting the Streamlit app.

```bash
# Start Ollama daemon
ollama serve
```

You should see:
```
Listening on [::]:11434
```

Keep this terminal window open.

### Step 2: Open New Terminal and Start Streamlit

```bash
# Navigate to project directory
cd <project-directory>

# Activate virtual environment (if using one)
source venv/bin/activate

# Start Streamlit app
streamlit run app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 3: Access the Application

Open your browser and go to: **http://localhost:8501**

---

## Key Features

### 1. **Chat Assistant Tab**
- Query optimization results using natural language
- Examples:
  - "Give me top 10 transfers by cost"
  - "Show manufacturing decisions"
  - "Compare scenarios"
  - "What are the inventory gaps?"

### 2. **Optimization Dashboard Tab**
- View optimization results with cost breakdown
- Top transfers and manufacturing decisions
- Store activity network visualization
- KPI metrics and comparisons

### 3. **Forecasting Tab**
- Demand forecasting performance metrics
- Model accuracy analysis (MAE/RMSE)
- Store-level forecast performance
- Data preprocessing and EDA outputs

### 4. **Stochastic Scenarios Tab**
- **Demand Uncertainty Analysis**: Simulate multiple demand scenarios using log-normal distribution
- **Risk Metrics**: Expected Cost, VaR (Value at Risk), CVaR (Conditional Value at Risk)
- **Risk Premium Calculation**: Buffer needed for tail-risk management
- **Cost Distribution Analysis**: Histogram and empirical CDF of scenario outcomes
- **Real Scenario Comparison**: Compare any 2 of your 6 pre-defined optimization scenarios
  - Cost summary (total, manufacturing, transfer, holding)
  - Volume summary (transfers, manufacturing units, transfer units)
  - Auto-generated key insights and percent changes

### 5. **Monitoring Tab**
- Data drift detection for forecast models
- Sliding-window and CUSUM detection methods
- Store and product-level drift visualization
- KPI export to CSV for BI dashboards

### 6. **Multi-Scenario Optimization**
- Base case (standard conditions)
- High disruption (risk-aware)
- Demand spike (peak scenarios)
- Transport cost increase (fuel shock)
- Extended lead times (supplier delays)
- Cost-only optimization


---

## Project Structure

```
v7/
├── README.md                                    # This file
├── app.py                                       # Main Streamlit application
├── requirements.txt                             # Python dependencies
│
├── data/                                        # Raw datasets
│   ├── freshretailnet_data_100.csv
│   └── dynamic_supply_chain_logistics_dataset.csv
│
├── nlp/                                         # NLP Pipeline
│   ├── __init__.py
│   ├── intent_classifier.py                    # Classify user queries
│   ├── explanation_engine.py                   # Generate responses
│   ├── llm_client.py                           # Ollama API client
│   ├── refiner.py                              # LLM response refinement
│   ├── scenario_compare.py                     # Scenario data models (ScenarioSnapshot)
│   ├── schemas.py                              # Data models
│   └── README.md                               # NLP documentation
│
├── optimization/                                # Supply Chain Optimization
│   ├── optimization.py                         # Main solver
│   ├── scenarios.py                            # Scenario definitions
│   ├── stochastic.py                           # Stochastic optimization
│   ├── optimization_working_plan.md
│   ├── input/                                  # Input parameters
│   │   ├── processed_store_product_params.csv
│   │   ├── store_supply_params.csv
│   │   └── transport_cost_matrix.csv
│   ├── output-json/                            # Optimization results
│   │   ├── transfer_recommendations.json
│   │   ├── manufacturing_decisions.json
│   │   ├── inventory.json
│   │   └── scenario_summary.json
│   └── output-csv/
│       ├── optimization_transfers.csv
│       ├── optimization_manufacturing.csv
│       └── optimization_inventory.csv
│
├── demand-forecast/                             # Demand Prediction
│   ├── forecaster.py                           # Main forecasting engine
│   ├── causal.py                               # Causal analysis
│   ├── correlations.py                         # Correlation analysis
│   ├── reconciler.py                           # Forecast reconciliation
│   ├── Demand_prediction_V3_AsifKhan.ipynb
│   └── output/
│       ├── product_forecasts.csv
│       ├── product_forecasts_wide.csv
│       └── product_forecast_metrics.csv
│
├── monitoring/                                  # Data Quality
│   ├── __init__.py
│   └── drift.py                                # Data drift detection
│
├── export/                                      # Export Utilities
│   ├── __init__.py
│   └── kpi_export.py                           # KPI data export
│
├── EDA/                                         # Exploratory Data Analysis
│   ├── data_preprocessing_and_eda.ipynb
│   ├── fetch_data.py
│   ├── data_preprocessing_updates.md
│   └── output/
│       ├── eda_cleaned_logistics_data.csv
│       ├── city_coordinates.csv
│       └── forecast_long_topstore.csv
│
└── tests/                                       # Unit Tests
    ├── __init__.py
    ├── test_causal.py
    ├── test_correlations.py
    ├── test_drift.py
    ├── test_explanation_engine.py
    ├── test_forecaster.py
    ├── test_intent_classifier.py
    ├── test_kpi_export.py
    ├── test_reconciler.py
    ├── test_scenario_compare.py
    └── test_stochastic.py
```

---

## Troubleshooting

### Issue: "Ollama not responding" or Connection Error

**Solution:**
1. Verify Ollama is running: `ollama serve` in a separate terminal
2. Check Ollama API: `curl http://localhost:11434/api/tags`
3. Expected response: JSON list of available models

### Issue: Model not loading / Slow responses

**Solution:**
1. Check which models are available: `ollama list`
2. Ensure model is downloaded: `ollama pull tinyllama`
3. Check system RAM and available disk space
4. TinyLLaMA (~637MB) is recommended for Mac; Mistral (~4.4GB) for larger systems

### Issue: "Failed to load MLX dynamic library"

**Solution:** This is a warning, not an error. Ollama is working correctly. MLX is optional hardware acceleration for Apple Silicon. Your app will run fine with standard CPU/GPU inference.

### Issue: Placeholder text (HTMLTAG0, HTMLTAG1) in chat output

**Solution:** This has been fixed in the current version. The app uses Unicode placeholders that don't interfere with markdown processing.

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_intent_classifier.py -v

# Run with coverage
pytest tests/ --cov=nlp --cov=optimization
```

---

## Configuration

### Model Selection

Edit `nlp/llm_client.py` to change the default model:

```python
MODEL_NAME = "mistral:latest"  # Change to "tinyllama:latest" if preferred
```

### Ollama Endpoint

Default: `http://localhost:11434`

To change, edit the URL in `nlp/llm_client.py`:

```python
OLLAMA_API_URL = "http://localhost:11434"
```

---

## Example Queries

Try these natural language queries in the Chat Assistant:

```
Transfers:
- "Give me top 10 transfers by cost"
- "Show urgent transfers"
- "What are the transfer recommendations?"

Manufacturing:
- "Top 5 manufacturing decisions"
- "Detail manufacturing plans"

Inventory:
- "Show inventory gaps"
- "Inventory status summary"

Analysis:
- "Compare scenarios"
- "What's the cost breakdown?"
- "How many recommendations total?"
```

---

## Documentation

- **NLP Pipeline:** See `nlp/README.md`
- **Optimization:** See `optimization/optimization_working_plan.md`
- **EDA:** See `EDA/data_preprocessing_updates.md`
- **Formatting Fix:** See `FORMATTING_FIX_SUMMARY.md`
- **Placeholder Fix:** See `PLACEHOLDER_FIX.md`

---

## Contributors

Developed as a Capstone project for supply chain optimization with AI/NLP integration.

---

## License

Educational use - Capstone Project
