# NLP Component - Supply Chain Assistant

## Overview
Natural Language Processing module for supply chain optimization queries. Processes user questions and provides intelligent, conversational responses by analyzing optimization output data from the optimization engine.

---

## Core Components

### 1. **intent_classifier.py**
Classifies user queries into specific intents using LLM-based classification with keyword fallback.

**Supported Intents:**
- `explain_transfer` - Transfer recommendations and routing decisions
- `explain_manufacturing` - Manufacturing decisions and production actions
- `scenario_summary` - Overall scenario metrics and cost breakdown
- `impact_analysis` - Cost savings and financial impact
- `total_counts` - Summary counts and statistics
- `out_of_scope` - Unrelated queries
- `greeting` - Greeting messages

### 2. **explanation_engine.py**
Generates explanations by extracting and formatting data from optimization JSON files.

**Key Functions:**
- `explain_transfer()` - Provides transfer recommendation details
- `explain_manufacturing()` - Details manufacturing decisions
- `explain_scenario()` - Summarizes scenario metrics
- `explain_counts()` - Provides summary statistics
- `explain_entities()` - Lists affected products and stores
- `build_explanation()` - Orchestrates explanation generation

### 3. **refiner.py**
Refines raw explanations into natural, conversational responses using LLM-based text refinement.

### 4. **llm_client.py**
Handles communication with Ollama LLM service for intent classification and response refinement.

### 5. **schemas.py**
Pydantic models for JSON data validation (optional validation layer).

---

## Data Source

**Location:** `optimization/output-json/`

**Expected Files:**
1. `transfer_recommendations.json` - Transfer decisions between stores
2. `manufacturing_decisions.json` - Manufacturing action decisions
3. `scenario_summary.json` - Scenario metrics and cost breakdown

---

## JSON Data Schemas

### transfer_recommendations.json
```json
{
  "scenario": "optimization_run",
  "transfers": [
    {
      "from_store": "string",
      "to_store": "string",
      "product_id": "string",
      "quantity": "number",
      "reason_codes": ["string"],
      "cost_impact": {
        "transport_cost": "number"
      }
    }
  ]
}
```

### manufacturing_decisions.json
```json
{
  "scenario": "optimization_run",
  "manufacturing_actions": [
    {
      "product_id": "string",
      "manufacture_quantity": "number",
      "reason_codes": ["string"],
      "cost_impact": {
        "manufacturing_cost": "number"
      }
    }
  ]
}
```

### scenario_summary.json
```json
{
  "scenario": "optimization_run",
  "optimized": {
    "total_cost": "number",
    "total_transfers": "number",
    "manufacturing_units": "number",
    "transfer_units": "number"
  },
  "cost_breakdown": {
    "manufacturing_cost": "number",
    "transfer_cost": "number",
    "holding_cost": "number"
  }
}
```

---

## Query Examples

**Transfer Queries:**
- "Explain the transfer recommendations"
- "How many transfers are recommended"
- "What transfers involve store 60"
- "Show transfers for product 489"

**Manufacturing Queries:**
- "Detail the manufacturing decisions"
- "How many manufacturing actions are needed"
- "What products need to be manufactured"

**Scenario Queries:**
- "What is the total cost"
- "Provide a scenario summary"
- "Show me the cost breakdown"

**Count/Summary Queries:**
- "How many stores are affected"
- "How many unique products are involved"
- "Total recommendations count"

---

## Architecture

```
User Input (Chat)
    ↓
intent_classifier.py → Classifies intent + extracts parameters
    ↓
explanation_engine.py → Loads data & generates raw explanation
    ↓
refiner.py → Refines with LLM (if available)
    ↓
Response Output
```

---

## Key Parameters

**Transfer/Manufacturing Filters:**
- `product_id` - Filter by specific product (e.g., "product_489")
- `store_id` - Filter by store ID (e.g., "store_60")
- `is_all` - Include all results (triggered by "all", "total", etc.)

---

## Performance Characteristics

- **Data Loading:** Single load per app session (optimization/output-json files)
- **Intent Classification:** Ollama LLM with keyword fallback (~2-3 seconds)
- **Response Refinement:** Optional LLM refinement (~3-5 seconds)
- **No External APIs:** All responses derived from local JSON files
- **Read-Only:** NLP module never modifies source data

