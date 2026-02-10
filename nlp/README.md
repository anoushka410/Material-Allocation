# NLP Supply Chain Assistant

## Overview
AI-powered chatbot assistant for supply chain optimization. Analyzes optimization results and provides intelligent, conversational insights to supply chain managers. The system reads optimization recommendations from JSON files and answers open-ended questions about transfers, manufacturing actions, and scenario outcomes.

---

## Architecture

### Data Flow
1. **Optimizer Engine** produces 3 JSON files per batch run:
   - `transfer.json` - Transfer recommendations between stores
   - `manufacturing.json` - Manufacturing action recommendations
   - `scenario.json` - Scenario metrics and comparison data

2. **NLP Module** (read-only):
   - Loads the 3 JSON files
   - Processes user queries against this data
   - Returns conversational responses with insights

3. **Streamlit UI** (`app.py`):
   - Displays recommendations dashboard
   - Provides chat interface for user queries
   - Shows key metrics and global insights

### Key Principles
- **Read-Only**: NLP never computes or modifies data, only reads from JSONs
- **No External APIs**: All responses derived from JSON data
- **Batch Processing**: Each optimizer run overwrites the 3 JSON files
- **On-Demand Analysis**: Users can refresh queries anytime

---

## File Structure

```
nlp/
├── app.py                    # Streamlit UI application
├── chatbot.py               # Query handler & response generation
├── explain_transfer.py      # Transfer recommendation explanations
├── explain_manufacturing.py # Manufacturing action explanations
├── explain_scenario.py      # Scenario metrics explanations
├── schemas.py               # Data validation schemas
├── reason_map.py            # Reasoning logic and mappings
├── llm_refiner.py          # Response refinement utilities
├── main.py                  # CLI entry point (optional)
├── sample_inputs/
│   ├── transfer.json        # Sample transfer recommendations
│   ├── manufacturing.json   # Sample manufacturing actions
│   └── scenario.json        # Sample scenario metrics
└── README.md               # This file
```

---

## JSON Data Schemas

### 1. `transfer.json`
```json
{
  "transfers": [
    {
      "product_id": "string",
      "from_store": "string",
      "to_store": "string",
      "quantity": "number",
      "reason": "string",
      "urgency": "high|medium|low"
    }
  ]
}
```

### 2. `manufacturing.json`
```json
{
  "manufacturing_actions": [
    {
      "product_id": "string",
      "manufacture_quantity": "number",
      "target_location": "string",
      "reason": "string",
      "deadline": "string"
    }
  ]
}
```

### 3. `scenario.json`
```json
{
  "scenario": "string (description)",
  "baseline": {
    "total_stockouts": "number",
    "total_cost": "number",
    ...
  },
  "optimized": {
    "total_stockouts": "number",
    "total_cost": "number",
    ...
  },
  "delta": {
    "cost_change": "number",
    "stockout_reduction": "number",
    ...
  }
}
```

---

## Running the Application

### Prerequisites
```bash
pip install streamlit
```

### Start the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Features

### 1. **Recommendations Dashboard**
- View all transfers and manufacturing actions
- Select specific recommendations to analyze
- See key performance metrics (baseline vs optimized)

### 2. **Chat Interface**
- Ask open-ended questions about recommendations
- Get conversational explanations for supply chain decisions
- Chat history preserved during session

### 3. **Global Insights**
- Total transfers and manufacturing actions count
- Risk level assessment
- Cost impact analysis
- Stockout reduction metrics

### 4. **Context-Aware Responses**
- Ask general questions (e.g., "How many transfers?")
- Ask specific questions about selected items (e.g., "Why move these units?")
- Ask scenario comparisons (e.g., "What improved?")

---

## Example Queries

**General Questions:**
- "What's the overall cost impact?"
- "How many stockouts were reduced?"
- "How many manufacturing actions are needed?"

**Transfer-Specific:**
- "Why is this transfer needed?"
- "Is this urgent?"
- "What happens if we skip this transfer?"

**Manufacturing-Specific:**
- "When do we need to manufacture?"
- "How critical is this product?"
- "What's the deadline?"

**Scenario Analysis:**
- "What improved compared to baseline?"
- "Is this scenario better?"
- "How much did we save?"

---

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `app.py` | Streamlit UI, data loading, session management |
| `chatbot.py` | Main query handler, orchestrates response generation |
| `explain_transfer.py` | Extract and explain transfer recommendations |
| `explain_manufacturing.py` | Extract and explain manufacturing actions |
| `explain_scenario.py` | Analyze scenario metrics and comparisons |
| `schemas.py` | Validate JSON data structures |
| `reason_map.py` | Map reasons and decision logic |
| `llm_refiner.py` | Polish responses for better readability |

---

## Development Notes

### Adding a New Question Type
1. Identify the relevant data in the 3 JSONs
2. Add logic to `chatbot.py` to detect this question type
3. Create or modify `explain_*.py` modules to extract answers
4. Return conversational response

### Testing
Place test JSON files in `sample_inputs/` and verify queries work as expected.

### Performance
- Data loads once per app session (cached)
- No API calls or external dependencies
- Response generation is instant

---

## Troubleshooting

### App keeps loading
Check `chatbot.py` for blocking operations. Ensure `handle_query()` returns immediately.

### JSON not found
Verify 3 JSON files exist in `sample_inputs/` directory.

### Empty responses
Check that JSON files contain expected data structure matching schemas above.

---

## Future Enhancements
- Add export functionality (PDF/CSV reports)
- Implement query logging and analytics
- Add multi-scenario comparison UI
- Persist chat history to database
- Add confidence scores to recommendations

---

## Contact & Support
For questions about the supply chain optimization logic, contact the optimization team.
For NLP/UI issues, refer to the code comments and schemas in this directory.

