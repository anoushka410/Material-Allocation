import io
import json
import subprocess
import time
import requests
import streamlit as st
import csv
import ast
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import zipfile
import os

from nlp.intent_classifier import classify_intent, extract_parameters
from nlp.explanation_engine import build_explanation, handle_user_query, detect_scenario
from nlp.refiner import refine_explanation
from nlp.llm_client import MODEL_NAME as LLM_MODEL_NAME
from optimization.stochastic import generate_scenarios, compute_cvar
from optimization.scenarios import ScenarioRegistry
from monitoring.drift import ForecastDriftDetector
from export.kpi_export import export_all_kpis

SAMPLE_DATA_DIR = "optimization/output-json"
DEFAULT_NLP_SCENARIO = "base_case_standard_conditions"

# SVG Icon definitions
def svg_icon_trash():
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>'''

def svg_icon_refresh():
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36M20.49 15a9 9 0 0 1-14.85 3.36"></path></svg>'''

def svg_icon_download():
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>'''

def svg_icon_settings():
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v6m0 6v6M4.22 4.22l4.24 4.24m4.24 4.24l4.24 4.24M1 12h6m6 0h6M4.22 19.78l4.24-4.24m4.24-4.24l4.24-4.24"></path></svg>'''

def svg_icon_menu():
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>'''

def button_label(text: str, icon: str = "→") -> str:
    """Create a styled button label with icon."""
    return f"{icon} {text}"

if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = None


def _ping_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_ollama() -> tuple[bool, str]:
    """Return (is_available, status_message)."""
    if _ping_ollama():
        return True, "Ollama is running."
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, "Ollama is not installed. Using keyword fallback."
    except Exception as e:
        return False, f"Could not start Ollama: {e}. Using keyword fallback."
    for _ in range(8):
        time.sleep(1)
        if _ping_ollama():
            return True, "Ollama started automatically."
    return False, "Ollama started but did not respond in time. Using keyword fallback."


def _get_available_scenarios(output_root: str = SAMPLE_DATA_DIR) -> list[str]:
    """Return scenario IDs available in the flat scenario_summary.json."""
    try:
        summary_path = os.path.join(output_root, "scenario_summary.json")
        if not os.path.exists(summary_path):
            return []
        combined = load_json(summary_path)
        ids = combined.get("scenario_ids") or [s["scenario"] for s in combined.get("summaries", [])]
        return sorted([str(x) for x in ids])
    except Exception:
        return []


def _resolve_nlp_scenario(params: dict | None) -> str:
    """Resolve scenario for NLP queries: explicit valid scenario, else base scenario."""
    params = params or {}
    requested = str(params.get("scenario") or "").strip()
    available = _get_available_scenarios(SAMPLE_DATA_DIR)

    if requested and requested not in ("all", "unknown", "optimization_run"):
        if (not available) or (requested in available):
            return requested

    if (not available) or (DEFAULT_NLP_SCENARIO in available):
        return DEFAULT_NLP_SCENARIO
    return available[0]


def _load_all_data(data_dir: str | None = None):
    """Load all optimization and forecast data into a single dict.

    Flat JSON schema (no per-scenario folders):
      - optimization/output-json/scenario_summary.json
      - optimization/output-json/transfer_recommendations.json
      - optimization/output-json/manufacturing_decisions.json

    Transfers/manufacturing are filtered in-memory by selected scenario.
    """
    if data_dir is None:
        data_dir = SAMPLE_DATA_DIR

    d: dict = {}

    # Combined scenario summary file (all scenarios)
    combined_summary = {}
    try:
        combined_summary = load_json(f"{SAMPLE_DATA_DIR}/scenario_summary.json")
    except Exception:
        combined_summary = {}

    scenario_id = st.session_state.selected_scenario
    if not scenario_id:
        try:
            scenario_ids = combined_summary.get("scenario_ids") or [s["scenario"] for s in combined_summary.get("summaries", [])]
            # Default to base_case_standard_conditions if available
            if "base_case_standard_conditions" in scenario_ids:
                scenario_id = "base_case_standard_conditions"
            else:
                scenario_id = scenario_ids[0] if scenario_ids else None
            st.session_state.selected_scenario = scenario_id
        except Exception:
            scenario_id = None

    # Pick the selected scenario's summary for dashboard KPIs
    if scenario_id and isinstance(combined_summary, dict):
        # Find the matching scenario in the summaries array
        summaries = combined_summary.get("summaries", [])
        d["scenario"] = next((s for s in summaries if s.get("scenario") == scenario_id), {})
    else:
        d["scenario"] = {}

    # Load flat detailed JSONs
    try:
        transfers_all = load_json(f"{SAMPLE_DATA_DIR}/transfer_recommendations.json")
    except Exception:
        transfers_all = {}

    try:
        mfg_all = load_json(f"{SAMPLE_DATA_DIR}/manufacturing_decisions.json")
    except Exception:
        mfg_all = {}

    # Filter to selected scenario
    if scenario_id:
        transfers = [t for t in (transfers_all.get("transfers") or []) if str(t.get("scenario")) == str(scenario_id)]
        mfg_actions = [m for m in (mfg_all.get("manufacturing_actions") or []) if str(m.get("scenario")) == str(scenario_id)]
        d["transfers"] = {"scenario": scenario_id, "transfers": transfers}
        d["manufacturing"] = {"scenario": scenario_id, "manufacturing_actions": mfg_actions}
    else:
        d["transfers"] = {"scenario": "unknown", "transfers": []}
        d["manufacturing"] = {"scenario": "unknown", "manufacturing_actions": []}

    # Scenario-specific CSVs remain under optimization/output-csv/<scenario_id>/
    if scenario_id:
        inv_path = f"optimization/output-csv/{scenario_id}/optimization_inventory.csv"
        tr_path = f"optimization/output-csv/{scenario_id}/optimization_transfers.csv"
        mfg_path = f"optimization/output-csv/{scenario_id}/optimization_manufacturing.csv"
    else:
        inv_path = "optimization/output-csv/optimization_inventory.csv"
        tr_path = "optimization/output-csv/optimization_transfers.csv"
        mfg_path = "optimization/output-csv/optimization_manufacturing.csv"

    d["inventory"] = load_csv(inv_path) if os.path.exists(inv_path) else []
    d["opt_transfers"] = load_csv(tr_path) if os.path.exists(tr_path) else []
    d["opt_manufacturing"] = load_csv(mfg_path) if os.path.exists(mfg_path) else []
    d["forecast_metrics"] = load_csv("demand-forecast/output/product_forecast_metrics.csv")

    d["scenario_summary_all"] = combined_summary

    return d


@st.cache_data(show_spinner=False)
def _load_all_data_cached(selected_scenario_id: str):
    """Cache per selected scenario to avoid mixing scenario state."""
    # Ensure session state is set for the loader
    st.session_state.selected_scenario = selected_scenario_id
    return _load_all_data(SAMPLE_DATA_DIR)


def _run_optimization_from_app(scenario_id: str, all_scenarios: bool = False) -> tuple[bool, str]:
    """Run optimization as a subprocess and return (ok, message)."""
    try:
        cmd = [
            "python",
            "optimization/optimization.py",
            "--output-dir",
            "optimization/output-json",
            "--output-csv-dir",
            "optimization/output-csv",
        ]
        if all_scenarios:
            cmd.append("--all-scenarios")
        else:
            cmd.extend(["--scenario-id", scenario_id])

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "Optimization failed").strip()
        return True, (r.stdout or "Optimization complete").strip()
    except Exception as e:
        return False, str(e)


st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        min-height: 100vh;
    }
    
    /* Target dark mode users to ensure text is legible */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
        }
    }

    .main-header {
        text-align: left;
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        font-size: 0.9rem;
        opacity: 0.7;
    }

    .intent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.3);
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .filter-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(14, 165, 233, 0.1);
        color: #0ea5e9;
        border: 1px solid rgba(14, 165, 233, 0.3);
        margin-bottom: 0.5rem;
        margin-left: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .fallback-note {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 0.5rem;
        font-style: italic;
    }

    .stChatMessage {
        background: #ffffff !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    @media (prefers-color-scheme: dark) {
        .stChatMessage {
            background: #1e293b !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
    }

    /* Chat input stays at bottom */
    .stChatInputContainer {
        position: relative;
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        margin-top: auto;
    }
    
    /* Sidebar styling */
    .sidebar-control-panel {
        padding: 1.5rem;
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #0f172a;
    }

    /* Keep control panel permanently visible and remove collapse controls */
    section[data-testid="stSidebar"] {
        transform: none !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 21rem !important;
        max-width: 21rem !important;
    }

    button[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    @media (prefers-color-scheme: dark) {
        .sidebar-title {
            color: #f8fafc;
        }
    }

    .status-indicator-online {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }

    .status-indicator-offline {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #ef4444;
        margin-right: 0.5rem;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .status-text {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    /* Chat container with proper layout */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 300px);
        min-height: 500px;
    }

    .chat-messages {
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }

    .chat-input-wrapper {
        flex-shrink: 0;
        padding: 1rem 0;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* SVG Icon styling */
    .icon-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        padding: 0;
        border: none;
        border-radius: 6px;
        background-color: rgba(148, 163, 184, 0.1);
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .icon-button:hover {
        background-color: rgba(148, 163, 184, 0.2);
    }

    .icon-button svg {
        width: 18px;
        height: 18px;
        stroke: currentColor;
        stroke-width: 2;
    }

    /* Sidebar toggle visibility */
    .sidebar-menu-button {
        display: block;
        margin-bottom: 1rem;
    }

    /* Professional button styling without emojis */
    .download-button-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .download-icon {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid currentColor;
        border-radius: 2px;
        position: relative;
    }

    .download-icon::before {
        content: '';
        position: absolute;
        top: 2px;
        left: 50%;
        transform: translateX(-50%);
        width: 2px;
        height: 4px;
        background: currentColor;
    }

    .download-icon::after {
        content: '';
        position: absolute;
        bottom: 2px;
        left: 50%;
        transform: translateX(-50%);
        width: 6px;
        height: 2px;
        background: currentColor;
    }

    /* Visual indicator badges */
    .visual-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }

    .indicator-online {
        background-color: #22c55e;
        animation: pulse-online 2s infinite;
    }

    .indicator-offline {
        background-color: #ef4444;
    }

    @keyframes pulse-online {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Professional status label */
    .status-label {
        display: inline-block;
        font-weight: 500;
        font-size: 0.9rem;
        vertical-align: middle;
    }

    /* Download label styling */
    .download-label {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 500;
    }

    /* Enhanced dashboard styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(56, 189, 248, 0.05) 100%);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(14, 165, 233, 0.4);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
    }

    .dashboard-section {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }

    .dashboard-section h2 {
        margin-top: 0;
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(14, 165, 233, 0.3);
    }

    .chart-container {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 6px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }

    .stMetric {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(56, 189, 248, 0.08) 100%);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 8px;
        padding: 1rem !important;
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        border-color: rgba(14, 165, 233, 0.4);
        transform: translateY(-2px);
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1200px;
    }

    /* Tab styling - remove emoji display */
    .stTabs [role="tab"] {
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>Supply Chain Analytics Assistant</h1>
        <p>Query optimization scenarios, transfer recommendations, and scenario metrics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar is intentionally non-collapsible so control panel remains accessible.

INTENT_LABELS = {
    # ── Required pipeline intents ──────────────────────────────────────────────
    "transfer_recommendations": "Transfer Recommendations",
    "manufacturing_plan": "Manufacturing Plan",
    "inventory_status": "Inventory Health",
    "scenario_summary": "Scenario Summary",
    "top_transfers_by_cost": "Top Transfers (Cost)",
    "top_transfers_by_quantity": "Top Transfers (Qty)",
    "top_manufacturing_items": "Top Manufacturing Items",
    "scenario_comparison": "Scenario Comparison",
    "out_of_scope": "Out of Scope",
    # ── Additional intents ────────────────────────────────────────────────────
    "greeting": "Greeting",
    "impact_analysis": "Impact Analysis",
    "list_entities": "Entity List",
    "total_counts": "Summary Metrics",
    "urgent_transfers": "Urgent Transfers",
    "high_cost_actions": "High-Cost Actions",
    "reason_analysis": "Decision Analysis",
    "store_activity": "Store Activity",
    "product_recommendations": "Product Actions",
    "cost_breakdown": "Cost Breakdown",
    "inventory_gaps": "Inventory Gaps",
    # Backward-compatible aliases
    "explain_transfer": "Transfer Recommendations",
    "explain_manufacturing": "Manufacturing Plan",
    "top_transfers": "Top Transfers (Cost)",
    "top_manufacturing": "Top Manufacturing Items",
}


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_csv(path: str) -> list[dict]:
    """Load a CSV file and return a list of dicts.
    Attempts to parse fields that look like Python lists using ast.literal_eval.
    """
    rows = []
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Try to parse any field that looks like a Python list
                parsed = {}
                for k, v in r.items():
                    if v is None:
                        parsed[k] = v
                        continue
                    v = v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        try:
                            parsed[k] = ast.literal_eval(v)
                        except Exception:
                            parsed[k] = v
                    else:
                        # Try numeric conversion
                        try:
                            if "." in v:
                                parsed[k] = float(v)
                            else:
                                parsed[k] = int(v)
                        except Exception:
                            parsed[k] = v
                rows.append(parsed)
    except FileNotFoundError:
        return []
    return rows


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """**Welcome to Supply Chain Analytics!**

I'm here to help you explore your optimization data. You can ask me about:

• Inventory status and gaps
• Transfer recommendations
• Manufacturing decisions
• Scenario comparisons

Try asking something like "Show me top transfers by cost" or "What's the inventory status for store 5?"

What would you like to know?"""
        }
    ]

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

if "ollama_ok" not in st.session_state:
    st.session_state.ollama_ok = False
    st.session_state.ollama_msg = "Checking system status..."

# Sidebar Controls - Always Visible
with st.sidebar:
    st.markdown('<div class="sidebar-control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎛️ Control Panel</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Clear Conversation", use_container_width=True, key="clear_btn"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": """**Welcome to Supply Chain Analytics!**

I'm here to help you explore your optimization data. You can ask me about:

• Inventory status and gaps
• Transfer recommendations
• Manufacturing decisions
• Scenario comparisons

Try asking something like "Show me top transfers by cost" or "What's the inventory status for store 5?"

What would you like to know?"""
                }
            ]
            st.session_state.last_intent = None
            st.rerun()
    with col2:
        if st.button("↻", use_container_width=True, help="Refresh", key="refresh_btn"):
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-title">LLM Status</div>', unsafe_allow_html=True)
    # Display the configured model name so it's easy to see which model is active
    st.caption(f"Model: **{LLM_MODEL_NAME}**")

    if "ollama_checked" not in st.session_state:
        # Check ollama on first load quietly
        ok, msg = _ensure_ollama()
        st.session_state.ollama_ok = ok
        st.session_state.ollama_msg = msg
        st.session_state.ollama_checked = True

    # Status indicator with professional styling
    if st.session_state.ollama_ok:
        status_html = '<span class="visual-indicator indicator-online"></span><span class="status-label">Online</span>'
        st.markdown(status_html, unsafe_allow_html=True)
    else:
        status_html = '<span class="visual-indicator indicator-offline"></span><span class="status-label">Offline</span>'
        st.markdown(status_html, unsafe_allow_html=True)
        st.caption("Using Fallback")
        if st.button("Start Engine", use_container_width=True, key="start_engine"):
            with st.spinner("Starting..."):
                ok, msg = _ensure_ollama()
                st.session_state.ollama_ok = ok
                st.session_state.ollama_msg = msg
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

AVATAR_USER = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#334155" rx="20"/><text x="50" y="65" font-family="sans-serif" font-weight="bold" font-size="50" fill="#f8fafc" text-anchor="middle">U</text></svg>'''
AVATAR_AI = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#0ea5e9" rx="20"/><text x="50" y="65" font-family="sans-serif" font-weight="bold" font-size="50" fill="#f8fafc" text-anchor="middle">AI</text></svg>'''

# ── Tab layout ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Assistant",
    "Optimization",
    "Forecasting",
    "Scenarios",
    "Monitoring",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Optimization Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Run Optimization")

    # Available scenarios come from the registry (what we *can* run)
    registry_ids = list(ScenarioRegistry.SCENARIOS.keys())

    # Available scenarios already generated on disk (from flat scenario_summary.json)
    generated_ids = _get_available_scenarios(SAMPLE_DATA_DIR)

    # Use generated list if present; else fall back to registry list
    scenario_options = generated_ids if generated_ids else registry_ids

    if scenario_options and st.session_state.selected_scenario not in scenario_options:
        # Default to base_case_standard_conditions if available
        if "base_case_standard_conditions" in scenario_options:
            st.session_state.selected_scenario = "base_case_standard_conditions"
        else:
            st.session_state.selected_scenario = scenario_options[0]

    scen_col1, scen_col2, scen_col3, scen_col4 = st.columns([2, 1, 1, 1])

    with scen_col1:
        # Find the current index
        current_idx = 0
        if st.session_state.selected_scenario and st.session_state.selected_scenario in scenario_options:
            current_idx = scenario_options.index(st.session_state.selected_scenario)

        # Selectbox with on_change callback
        selected = st.selectbox(
            "View scenario",
            options=scenario_options,
            index=current_idx,
            key="scenario_selectbox",
            on_change=lambda: setattr(st.session_state, 'selected_scenario', st.session_state.scenario_selectbox)
        )

        # Ensure session state is immediately updated
        st.session_state.selected_scenario = selected

    with scen_col2:
        run_one = st.button("Run selected", use_container_width=True)
    with scen_col3:
        run_all = st.button("Run all", use_container_width=True)
    with scen_col4:
        reset_opt = st.button("Reset", use_container_width=True, help="Delete all optimization output files")

    if run_one:
        with st.spinner(f"Running optimization: {st.session_state.selected_scenario} …"):
            ok, msg = _run_optimization_from_app(st.session_state.selected_scenario, all_scenarios=False)
        if ok:
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.success("Optimization finished. Reloading outputs…")
            st.code(msg)
            st.rerun()
        else:
            st.error("Optimization failed")
            st.code(msg)

    if run_all:
        with st.spinner("Running optimization for all scenarios …"):
            ok, msg = _run_optimization_from_app("base_case_standard_conditions", all_scenarios=True)
        if ok:
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.success("All scenarios finished. Reloading outputs…")
            st.code(msg)
            st.rerun()
        else:
            st.error("Optimization failed")
            st.code(msg)

    if reset_opt:
        with st.spinner("Resetting optimization outputs…"):
            try:
                from optimization.optimization import reset_json_outputs
                import shutil

                # Reset JSON files
                reset_json_outputs("optimization/output-json")

                # Remove CSV directories for each scenario
                csv_root = "optimization/output-csv"
                if os.path.exists(csv_root):
                    for scenario_dir in os.listdir(csv_root):
                        scenario_path = os.path.join(csv_root, scenario_dir)
                        if os.path.isdir(scenario_path):
                            shutil.rmtree(scenario_path)
                            print(f"Deleted {scenario_path}")

                # Clear cache and rerun
                try:
                    st.cache_data.clear()
                except Exception:
                    pass

                st.success("All optimization outputs deleted successfully!")
                st.session_state.selected_scenario = None
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {str(e)}")

    st.caption(
        "Click **Run all** once to generate scenario_summary.json with all scenarios. Use **View scenario** to switch dashboards."
    )

    st.markdown("---")

    _all = _load_all_data_cached(st.session_state.selected_scenario or "")
    scenario_data = _all.get("scenario", {})
    optimized = scenario_data.get("optimized", {})
    cost_breakdown = scenario_data.get("cost_breakdown", {})
    transfers_list = _all.get("transfers", {}).get("transfers", [])
    mfg_list = _all.get("manufacturing", {}).get("manufacturing_actions", [])
    inv_list = _all.get("inventory", [])
    opt_transfers = _all.get("opt_transfers", [])
    opt_manufacturing = _all.get("opt_manufacturing", [])

    st.markdown("### Optimization Overview")

    # Display the scenario ID from loaded data
    current_scenario_id = scenario_data.get("scenario", "unknown")
    st.info(f"**Scenario:** {current_scenario_id}")

    # ── KPI Metric Row ─────────────────────────────────────────────────────────
    total_cost = optimized.get("total_cost", 0)
    mfg_cost = cost_breakdown.get("manufacturing_cost", 0)
    transfer_cost = cost_breakdown.get("transfer_cost", 0)
    holding_cost = cost_breakdown.get("holding_cost", 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Cost", f"${total_cost:,.0f}")
    k2.metric("Manufacturing Cost", f"${mfg_cost:,.0f}", f"{mfg_cost/total_cost*100:.1f}% of total" if total_cost else "N/A")
    k3.metric("Transfer Cost", f"${transfer_cost:,.0f}", f"{transfer_cost/total_cost*100:.1f}% of total" if total_cost else "N/A")
    k4.metric("Holding Cost", f"${holding_cost:,.0f}", f"{holding_cost/total_cost*100:.1f}% of total" if total_cost else "N/A")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Total Transfers", optimized.get("total_transfers", 0))
    k6.metric("Transfer Units", f"{optimized.get('transfer_units', 0):,.1f}")
    k7.metric("Manufacturing Units", f"{optimized.get('manufacturing_units', 0):,.1f}")
    k8.metric("Unique Products", len({t.get("product_id") for t in transfers_list} | {m.get("product_id") for m in mfg_list}))

    st.markdown("---")

    # ── Cost Breakdown chart ───────────────────────────────────────────────────
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.markdown("#### Cost Breakdown")
        pie_df = pd.DataFrame({
            "Component": ["Manufacturing", "Transfer / Logistics", "Holding / Inventory"],
            "Amount": [mfg_cost, transfer_cost, holding_cost],
        })
        fig_pie = px.pie(
            pie_df, values="Amount", names="Component",
            color_discrete_sequence=["#0ea5e9", "#38bdf8", "#7dd3fc"],
            hole=0.45,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.markdown("#### Cost Components")
        bar_df = pd.DataFrame({
            "Component": ["Manufacturing", "Transfer", "Holding"],
            "Cost ($)": [mfg_cost, transfer_cost, holding_cost],
        })
        fig_bar = px.bar(
            bar_df, x="Component", y="Cost ($)",
            color="Component",
            color_discrete_sequence=["#0ea5e9", "#38bdf8", "#7dd3fc"],
            text_auto=True,
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Top Transfers by Cost ──────────────────────────────────────────────────
    st.markdown("#### Top 15 Transfers by Transport Cost")
    if transfers_list:
        t_df = pd.DataFrame([
            {
                "Route": f"{t.get('from_store')}→{t.get('to_store')}",
                "Product": t.get("product_id", ""),
                "Quantity": t.get("quantity", 0),
                "Transport Cost ($)": t.get("cost_impact", {}).get("transport_cost", 0),
                "Reasons": ", ".join(t.get("reason_codes", [])),
            }
            for t in transfers_list
        ])
        t_df = t_df.sort_values("Transport Cost ($)", ascending=False).head(15).reset_index(drop=True)
        t_df.index += 1

        col_ttbl, col_tbar = st.columns([1, 1])
        with col_ttbl:
            st.dataframe(t_df[["Route", "Product", "Quantity", "Transport Cost ($)"]], use_container_width=True)
        with col_tbar:
            fig_t = px.bar(
                t_df.head(10), x="Transport Cost ($)", y="Route",
                orientation="h", color="Transport Cost ($)",
                color_continuous_scale="Blues", text_auto=True,
            )
            fig_t.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_t, use_container_width=True)

        csv_buf = io.StringIO()
        t_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Transfers CSV", csv_buf.getvalue(),
            file_name="top_transfers.csv", mime="text/csv",
        )
    else:
        st.info("No transfer data loaded.")

    st.markdown("---")

    # ── Top Manufacturing Decisions ────────────────────────────────────────────
    st.markdown("#### Top 15 Manufacturing Decisions by Cost")
    if mfg_list:
        m_df = pd.DataFrame([
            {
                "Product": m.get("product_id", ""),
                "Manufacture Qty": m.get("quantity", 0),
                "Manufacturing Cost ($)": m.get("cost", 0),
                "Reasons": ", ".join(m.get("reason_codes", [])),
            }
            for m in mfg_list
        ])
        m_df = m_df.sort_values("Manufacturing Cost ($)", ascending=False).head(15).reset_index(drop=True)
        m_df.index += 1

        col_mtbl, col_mbar = st.columns([1, 1])
        with col_mtbl:
            st.dataframe(m_df[["Product", "Manufacture Qty", "Manufacturing Cost ($)"]], use_container_width=True)
        with col_mbar:
            fig_m = px.bar(
                m_df.head(10), x="Manufacturing Cost ($)", y="Product",
                orientation="h", color="Manufacturing Cost ($)",
                color_continuous_scale="Blues", text_auto=True,
            )
            fig_m.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_m, use_container_width=True)

        csv_buf2 = io.StringIO()
        m_df.to_csv(csv_buf2, index=False)
        st.download_button(
            "Download Manufacturing CSV", csv_buf2.getvalue(),
            file_name="top_manufacturing.csv", mime="text/csv",
        )
    else:
        st.info("No manufacturing data loaded.")

    st.markdown("---")

    # ── Inventory Coverage ─────────────────────────────────────────────────────
    st.markdown("#### Inventory Coverage: Current vs. Target (sample)")
    if inv_list:
        inv_df = pd.DataFrame(inv_list)
        for col in ("current", "final", "target"):
            if col in inv_df.columns:
                inv_df[col] = pd.to_numeric(inv_df[col], errors="coerce")
        if "store_id" in inv_df.columns and "current" in inv_df.columns and "target" in inv_df.columns:
            store_inv = (
                inv_df.groupby("store_id", as_index=False)
                .agg({"current": "sum", "final": "sum", "target": "sum"})
                .sort_values("store_id")
                .head(20)
            )
            store_inv["store_id"] = store_inv["store_id"].astype(str)
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Bar(name="Current", x=store_inv["store_id"], y=store_inv["current"], marker_color="#94a3b8"))
            fig_inv.add_trace(go.Bar(name="Final (Post-Opt)", x=store_inv["store_id"], y=store_inv["final"], marker_color="#0ea5e9"))
            fig_inv.add_trace(go.Scatter(name="Target", x=store_inv["store_id"], y=store_inv["target"], mode="lines+markers", line=dict(color="#f97316", width=2, dash="dot")))
            fig_inv.update_layout(barmode="group", xaxis_title="Store ID", yaxis_title="Units", margin=dict(t=10, b=10))
            st.plotly_chart(fig_inv, use_container_width=True)

    st.markdown("---")

    # ── Inventory Health Metrics ───────────────────────────────────────────────
    st.markdown("#### Inventory Health Summary (BEFORE Optimization)")
    if inv_list and len(inv_list) > 0:
        inv_df_health = pd.DataFrame(inv_list)
        for col in ("current", "final", "target"):
            if col in inv_df_health.columns:
                inv_df_health[col] = pd.to_numeric(inv_df_health[col], errors="coerce")

        if all(c in inv_df_health.columns for c in ("current", "target")):
            # Check CURRENT inventory against TARGET (before optimization)
            inv_df_health["coverage_ratio"] = inv_df_health["current"] / inv_df_health["target"].replace(0, 1)
            inv_df_health["status"] = inv_df_health["coverage_ratio"].apply(
                lambda x: "Critical" if x < 0.8 else ("Warning" if x < 1.0 else "Healthy")
            )

            # Health counts
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            health_counts = inv_df_health["status"].value_counts()
            col_h1.metric("At Target", health_counts.get("Healthy", 0))
            col_h2.metric("Below Target", health_counts.get("Warning", 0))
            col_h3.metric("Critical Gap", health_counts.get("Critical", 0))
            col_h4.metric("Avg Current Coverage", f"{inv_df_health['coverage_ratio'].mean():.1%}")

            st.markdown("---")

            # Top products with gaps (CURRENT vs TARGET, not final)
            st.markdown("#### Top Store-Products with Inventory Gaps (Current < Target)")
            gaps_df = inv_df_health[inv_df_health["current"] < inv_df_health["target"]].copy()
            gaps_df["gap"] = gaps_df["target"] - gaps_df["current"]

            if len(gaps_df) > 0:
                # Top 10 by gap
                top_gaps = gaps_df.nlargest(10, "gap")[["product_id", "store_id", "current", "final", "target", "gap"]].copy()
                top_gaps["product_id"] = top_gaps["product_id"].astype(str)
                top_gaps["store_id"] = top_gaps["store_id"].astype(str)
                top_gaps = top_gaps.round(2)

                col_gap_tbl, col_gap_chart = st.columns([1, 1])
                with col_gap_tbl:
                    st.dataframe(top_gaps.rename(columns={
                        "product_id": "Product",
                        "store_id": "Store",
                        "current": "Current",
                        "final": "Final",
                        "target": "Target",
                        "gap": "Gap"
                    }), use_container_width=True, hide_index=True)

                with col_gap_chart:
                    gap_chart_data = top_gaps.sort_values("gap")
                    gap_chart_data["label"] = gap_chart_data["product_id"] + " (Store " + gap_chart_data["store_id"] + ")"
                    fig_gaps = px.bar(
                        gap_chart_data,
                        x="gap",
                        y="label",
                        orientation="h",
                        color="gap",
                        color_continuous_scale="Reds",
                        labels={"gap": "Gap (units)", "label": "Product - Store"}
                    )
                    fig_gaps.update_layout(showlegend=False, margin=dict(t=10, b=10), yaxis_title="")
                    st.plotly_chart(fig_gaps, use_container_width=True)
            else:
                st.success("All inventory levels meet or exceed targets!")

    # ── Reason Code Analysis ───────────────────────────────────────────────────
    st.markdown("#### Transfer Decision Reasons")
    if transfers_list:
        reason_counts: dict[str, int] = {}
        for t in transfers_list:
            for r in t.get("reason_codes", []):
                reason_counts[r] = reason_counts.get(r, 0) + 1
        r_df = pd.DataFrame(list(reason_counts.items()), columns=["Reason", "Count"])
        r_df["Reason"] = r_df["Reason"].str.replace("_", " ").str.title()
        r_df = r_df.sort_values("Count", ascending=True)
        fig_r = px.bar(r_df, x="Count", y="Reason", orientation="h", color="Count", color_continuous_scale="Blues", text_auto=True)
        fig_r.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Full JSON export ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Export Raw Optimization Data")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            "Download scenario_summary.json",
            json.dumps(scenario_data, indent=2),
            file_name="scenario_summary.json",
            mime="application/json",
        )
    with col_dl2:
        st.download_button(
            "Download transfer_recommendations.json",
            json.dumps(_all.get("transfers", {}), indent=2),
            file_name="transfer_recommendations.json",
            mime="application/json",
        )
    with col_dl3:
        st.download_button(
            "Download manufacturing_decisions.json",
            json.dumps(_all.get("manufacturing", {}), indent=2),
            file_name="manufacturing_decisions.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Demand Forecast Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    _all3 = _load_all_data()
    fm_list = _all3.get("forecast_metrics", [])

    st.markdown("### Demand Forecast Model Performance")

    if fm_list:
        fm_df = pd.DataFrame(fm_list)
        for col in ("MAE", "RMSE", "n_train", "n_test"):
            if col in fm_df.columns:
                fm_df[col] = pd.to_numeric(fm_df[col], errors="coerce")

        # ── KPIs ─────────────────────────────────────────────────────────────
        fk1, fk2, fk3, fk4 = st.columns(4)
        fk1.metric("Store-Product Pairs", len(fm_df))
        fk2.metric("Mean MAE", f"{fm_df['MAE'].mean():.3f}" if "MAE" in fm_df else "N/A")
        fk3.metric("Mean RMSE", f"{fm_df['RMSE'].mean():.3f}" if "RMSE" in fm_df else "N/A")
        fk4.metric("Unique Stores", fm_df["store_id"].nunique() if "store_id" in fm_df else "N/A")

        st.markdown("---")

        # ── MAE Distribution ─────────────────────────────────────────────────
        col_hist, col_store = st.columns(2)
        with col_hist:
            st.markdown("#### MAE Distribution")
            if "MAE" in fm_df.columns:
                fig_h = px.histogram(fm_df, x="MAE", nbins=30, color_discrete_sequence=["#0ea5e9"])
                fig_h.update_layout(margin=dict(t=10, b=10))
                st.plotly_chart(fig_h, use_container_width=True)

        with col_store:
            st.markdown("#### Mean MAE per Store (top 20)")
            if "store_id" in fm_df.columns and "MAE" in fm_df.columns:
                store_mae = fm_df.groupby("store_id", as_index=False)["MAE"].mean().sort_values("MAE", ascending=False).head(20)
                store_mae["store_id"] = store_mae["store_id"].astype(str)
                fig_s = px.bar(store_mae, x="store_id", y="MAE", color="MAE", color_continuous_scale="Oranges", text_auto=True)
                fig_s.update_layout(xaxis_title="Store ID", showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("---")

        # ── MAE vs RMSE scatter ───────────────────────────────────────────────
        st.markdown("#### MAE vs RMSE by Store-Product")
        if "MAE" in fm_df.columns and "RMSE" in fm_df.columns:
            scatter_df = fm_df.copy()
            scatter_df["store_id"] = scatter_df["store_id"].astype(str)
            fig_sc = px.scatter(
                scatter_df, x="MAE", y="RMSE",
                color="store_id", hover_data=["product_id"] if "product_id" in scatter_df.columns else None,
                opacity=0.7,
            )
            fig_sc.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("---")

        # ── Full metrics table ────────────────────────────────────────────────
        st.markdown("#### Full Forecast Metrics Table")
        display_cols = [c for c in ("store_id", "product_id", "MAE", "RMSE", "n_train", "n_test") if c in fm_df.columns]
        st.dataframe(fm_df[display_cols].sort_values("MAE", ascending=False).reset_index(drop=True), use_container_width=True)

        csv_fm = io.StringIO()
        fm_df[display_cols].to_csv(csv_fm, index=False)
        st.download_button(
            "Download Forecast Metrics CSV", csv_fm.getvalue(),
            file_name="forecast_metrics.csv", mime="text/csv",
        )
    else:
        st.info("Forecast metrics not found at demand-forecast/output/product_forecast_metrics.csv.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stochastic Scenarios
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    _all4 = _load_all_data()
    _scenario4 = _all4.get("scenario", {})
    _transfers4 = _all4.get("transfers", {}).get("transfers", []) if isinstance(_all4.get("transfers"), dict) else (_all4.get("transfers") or [])

    st.markdown("### Stochastic Demand Scenario Analysis")
    st.markdown(
        "Explore how cost and risk metrics change when demand is uncertain. "
        "Scenarios are generated with a **log-normal** demand distribution around point forecasts."
    )
    with st.expander("How to read this section", expanded=False):
        st.markdown(
            "- **Expected Cost**: average across all simulated demand outcomes.\n"
            "- **VaR (alpha)**: cost threshold that only the worst `(1-alpha)` fraction exceed.\n"
            "- **CVaR (alpha)**: average cost inside that worst tail, so it captures severity of bad outcomes.\n"
            "- **Risk Premium**: `CVaR - Expected Cost`; think of this as a risk buffer over the average case."
        )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        n_scen = st.slider(
            "Number of scenarios (Omega)",
            10,
            100,
            30,
            step=10,
            help="How many demand futures to simulate. More scenarios give a smoother risk estimate.",
        )
    with col_cfg2:
        demand_cv = st.slider(
            "Demand CV (coefficient of variation)",
            0.05,
            0.60,
            0.20,
            step=0.05,
            help="Relative demand volatility. Example: CV=0.20 means roughly 20% spread around mean demand.",
        )
    with col_cfg3:
        alpha_cvar = st.slider(
            "CVaR confidence level (alpha)",
            0.80,
            0.99,
            0.95,
            step=0.01,
            help="Tail-risk level. Alpha=0.95 focuses on the worst 5% scenarios; alpha=0.97 focuses on worst 3%.",
        )

    @st.cache_data(show_spinner=False)
    def _build_mean_demands_from_forecast(path_wide: str, max_pairs: int = 30):
        """Build mean daily demand per (store_id, product_id) from forecast output files."""
        if not os.path.exists(path_wide):
            return {}
        df = pd.read_csv(path_wide)
        if "store_id" not in df.columns or "product_id" not in df.columns:
            return {}
        day_cols = [c for c in df.columns if c.startswith("day+")]
        if not day_cols:
            return {}
        df["mean_demand"] = df[day_cols].mean(axis=1)
        df = df.sort_values("mean_demand", ascending=False).head(max_pairs)
        out = {}
        for _, r in df.iterrows():
            try:
                out[(int(r["store_id"]), str(r["product_id"]))] = float(r["mean_demand"])  # type: ignore[arg-type]
            except Exception:
                continue
        return out

    @st.cache_data(show_spinner=False)
    def _build_mean_demands_from_transfers(transfers_json_str: str, max_pairs: int = 30):
        """Approximate mean demands from transfer destination quantities."""
        import json as _json
        transfers = _json.loads(transfers_json_str)
        demands: dict[str, float] = {}
        for t in transfers:
            try:
                key = f"{t.get('to_store', 0)}_{t.get('product_id', '')}"
                demands[key] = demands.get(key, 0.0) + float(t.get("quantity", 0.0))
            except Exception:
                pass
        result = {}
        for k, v in demands.items():
            parts = k.split("_", 1)
            try:
                result[(int(parts[0]), parts[1])] = v
            except Exception:
                continue
        return dict(list(result.items())[:max_pairs])

    forecast_path = "demand-forecast/output/product_forecasts_wide.csv"
    mean_demands = _build_mean_demands_from_forecast(forecast_path, max_pairs=30)
    if not mean_demands and _transfers4:
        mean_demands = _build_mean_demands_from_transfers(json.dumps(_transfers4), max_pairs=30)

    if not mean_demands:
        st.warning(
            "No demand data available for scenario generation. "
            "Generate demand forecasts (demand-forecast/output/product_forecasts_wide.csv) or run optimization first."
        )
    else:
        with st.spinner(f"Generating {n_scen} demand scenarios…"):
            scenarios, probs = generate_scenarios(
                mean_demands,
                n_scenarios=n_scen,
                cv=demand_cv,
                random_state=42,
            )

        unit_cost = 50.0
        scenario_costs = [sum(v * unit_cost for v in s.values()) for s in scenarios]

        expected_cost = float(sum(c * p for c, p in zip(scenario_costs, probs)))
        var_val = float(pd.Series(scenario_costs).quantile(alpha_cvar))
        cvar_val = compute_cvar(scenario_costs, alpha=alpha_cvar)

        sk1, sk2, sk3, sk4 = st.columns(4)
        sk1.metric("Expected Cost", f"${expected_cost:,.0f}")
        sk2.metric(f"VaR ({alpha_cvar:.0%})", f"${var_val:,.0f}")
        sk3.metric(f"CVaR ({alpha_cvar:.0%})", f"${cvar_val:,.0f}")
        risk_premium = cvar_val - expected_cost
        sk4.metric("Risk Premium", f"${risk_premium:,.0f}", delta=f"{risk_premium/expected_cost*100:.1f}% of EV" if expected_cost else None)
        st.caption(
            "Interpretation: VaR is a threshold; CVaR is the average beyond that threshold. "
            "A larger Risk Premium means heavier downside tail risk relative to the average case."
        )

        st.markdown("---")

        col_dist, col_cdf = st.columns(2)
        with col_dist:
            st.markdown("#### Scenario Cost Distribution")
            scen_df = pd.DataFrame({"scenario": range(1, n_scen + 1), "cost": scenario_costs})
            fig_hist = px.histogram(scen_df, x="cost", nbins=20, color_discrete_sequence=["#0ea5e9"])
            fig_hist.add_vline(x=expected_cost, line_dash="solid", line_color="#f97316", annotation_text="E[cost]")
            fig_hist.add_vline(x=var_val, line_dash="dash", line_color="#dc2626", annotation_text=f"VaR({alpha_cvar:.0%})")
            fig_hist.add_vline(x=cvar_val, line_dash="dot", line_color="#7c3aed", annotation_text=f"CVaR({alpha_cvar:.0%})")
            fig_hist.update_layout(margin=dict(t=30, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("Each bar shows how many scenarios land in a cost range. Wider spread means higher uncertainty.")

        with col_cdf:
            st.markdown("#### Empirical CDF")
            sorted_costs = sorted(scenario_costs)
            cdf_y = [(i + 1) / n_scen for i in range(n_scen)]
            fig_cdf = px.line(x=sorted_costs, y=cdf_y, labels={"x": "Cost ($)", "y": "Cumulative Probability"})
            fig_cdf.add_vline(x=var_val, line_dash="dash", line_color="#dc2626", annotation_text=f"VaR({alpha_cvar:.0%})")
            fig_cdf.update_layout(margin=dict(t=30, b=10))
            st.plotly_chart(fig_cdf, use_container_width=True)
            st.caption("At any cost value on the x-axis, the y-axis tells the fraction of scenarios at or below that cost.")

        st.markdown("---")

        st.markdown("### Compare Two Optimization Scenarios")

        # Get list of available generated scenarios
        available_scen_ids = _get_available_scenarios(SAMPLE_DATA_DIR)
        if not available_scen_ids:
            st.warning("No optimization scenarios available. Run 'Optimization' tab first.")
        else:
            col_scen1, col_scen2 = st.columns(2)

            with col_scen1:
                scen1_id = st.selectbox(
                    "Scenario 1",
                    options=available_scen_ids,
                    index=0,
                    key="scen1_compare",
                )

            with col_scen2:
                # Default to second scenario if available, else first
                default_idx2 = 1 if len(available_scen_ids) > 1 else 0
                scen2_id = st.selectbox(
                    "Scenario 2",
                    options=available_scen_ids,
                    index=default_idx2,
                    key="scen2_compare",
                )

            if scen1_id != scen2_id:
                # Load both scenario summaries from the flat JSON
                combined_summary = _all4.get("scenario_summary_all", {})
                summaries = combined_summary.get("summaries", [])

                scen1_data = next((s for s in summaries if s.get("scenario") == scen1_id), None)
                scen2_data = next((s for s in summaries if s.get("scenario") == scen2_id), None)

                if scen1_data and scen2_data:
                    # Extract cost and volume metrics
                    scen1_opt = scen1_data.get("optimized", {})
                    scen2_opt = scen2_data.get("optimized", {})
                    scen1_cost = scen1_data.get("cost_breakdown", {})
                    scen2_cost = scen2_data.get("cost_breakdown", {})

                    # Cost Summary Table
                    st.markdown("#### Cost Summary")
                    cost_comparison = pd.DataFrame({
                        "Metric": ["Total Cost", "Manufacturing", "Transfer/Logistics", "Holding/Inventory"],
                        scen1_id: [
                            scen1_opt.get("total_cost", 0),
                            scen1_cost.get("manufacturing_cost", 0),
                            scen1_cost.get("transfer_cost", 0),
                            scen1_cost.get("holding_cost", 0),
                        ],
                        scen2_id: [
                            scen2_opt.get("total_cost", 0),
                            scen2_cost.get("manufacturing_cost", 0),
                            scen2_cost.get("transfer_cost", 0),
                            scen2_cost.get("holding_cost", 0),
                        ],
                    })

                    # Calculate change
                    cost_comparison["Change ($)"] = cost_comparison[scen2_id] - cost_comparison[scen1_id]
                    cost_comparison["Change (%)"] = (cost_comparison["Change ($)"] / cost_comparison[scen1_id] * 100).round(1)
                    st.dataframe(cost_comparison, use_container_width=True, hide_index=True)

                    # Volume Summary Table
                    st.markdown("#### Volume Summary")
                    vol_comparison = pd.DataFrame({
                        "Metric": ["Total Transfers", "Manufacturing Units", "Transfer Units"],
                        scen1_id: [
                            scen1_opt.get("total_transfers", 0),
                            scen1_opt.get("manufacturing_units", 0),
                            scen1_opt.get("transfer_units", 0),
                        ],
                        scen2_id: [
                            scen2_opt.get("total_transfers", 0),
                            scen2_opt.get("manufacturing_units", 0),
                            scen2_opt.get("transfer_units", 0),
                        ],
                    })

                    vol_comparison["Change"] = vol_comparison[scen2_id] - vol_comparison[scen1_id]
                    vol_comparison["Change (%)"] = (vol_comparison["Change"] / vol_comparison[scen1_id] * 100).round(1)
                    st.dataframe(vol_comparison, use_container_width=True, hide_index=True)

                    # Key Insights
                    st.markdown("#### Key Insights")
                    total_cost_diff = scen2_opt.get("total_cost", 0) - scen1_opt.get("total_cost", 0)
                    total_cost_pct = (total_cost_diff / scen1_opt.get("total_cost", 1)) * 100

                    insights = []
                    if total_cost_diff > 0:
                        insights.append(f"**{scen2_id}** costs **${abs(total_cost_diff):,.0f}** ({total_cost_pct:.1f}%) **more** than {scen1_id}.")
                    elif total_cost_diff < 0:
                        insights.append(f"**{scen2_id}** costs **${abs(total_cost_diff):,.0f}** ({abs(total_cost_pct):.1f}%) **less** than {scen1_id}.")
                    else:
                        insights.append(f"Both scenarios have equal total cost.")

                    mfg_diff = scen2_cost.get("manufacturing_cost", 0) - scen1_cost.get("manufacturing_cost", 0)
                    if mfg_diff != 0:
                        mfg_pct = (mfg_diff / max(scen1_cost.get("manufacturing_cost", 1), 1)) * 100
                        direction = "increases" if mfg_diff > 0 else "decreases"
                        insights.append(f"Manufacturing cost {direction} by **${abs(mfg_diff):,.0f}** ({abs(mfg_pct):.1f}%).")

                    transfer_diff = scen2_cost.get("transfer_cost", 0) - scen1_cost.get("transfer_cost", 0)
                    if transfer_diff != 0:
                        transfer_pct = (transfer_diff / max(scen1_cost.get("transfer_cost", 1), 1)) * 100
                        direction = "increases" if transfer_diff > 0 else "decreases"
                        insights.append(f"Transfer/Logistics cost {direction} by **${abs(transfer_diff):,.0f}** ({abs(transfer_pct):.1f}%).")

                    transfer_units_diff = scen2_opt.get("transfer_units", 0) - scen1_opt.get("transfer_units", 0)
                    if transfer_units_diff != 0:
                        units_pct = (transfer_units_diff / max(scen1_opt.get("transfer_units", 1), 1)) * 100
                        direction = "increases" if transfer_units_diff > 0 else "decreases"
                        insights.append(f"Transfer units {direction} by **{abs(transfer_units_diff):,.1f}** ({abs(units_pct):.1f}%).")

                    for insight in insights:
                        st.markdown(f"• {insight}")
                else:
                    st.warning("Could not load one or both scenario data. Ensure both scenarios have been optimized.")
            else:
                st.info("Please select two different scenarios to compare.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Monitoring & Export
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    _all5 = _load_all_data()
    fm5 = _all5.get("forecast_metrics", [])

    # Data for export
    _scen5 = _all5.get("scenario", {})
    _trans5 = _all5.get("transfers", {})
    _mfg5 = _all5.get("manufacturing", {})

    st.markdown("### Forecast Drift Monitoring")
    st.markdown(
        "Simulates a rolling prediction monitor using the stored forecast metrics. "
        "Drift is flagged when recent MAE exceeds the baseline by the configured multiplier."
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        baseline_win = st.slider("Baseline window (obs)", 10, 50, 30)
    with col_m2:
        alert_win = st.slider("Alert window (obs)", 3, 20, 7)
    with col_m3:
        mae_thresh = st.slider("MAE alert threshold (×)", 1.1, 3.0, 1.5, step=0.1)

    if fm5:
        fm5_df = pd.DataFrame(fm5)
        for col in ("MAE", "RMSE"):
            if col in fm5_df.columns:
                fm5_df[col] = pd.to_numeric(fm5_df[col], errors="coerce")

        # Build drift detector with current slider parameters (no caching for full interactivity)
        detector = ForecastDriftDetector(
            baseline_window=baseline_win, alert_window=alert_win, mae_threshold=mae_thresh
        )
        rng = np.random.default_rng(42)
        for _, row in fm5_df.iterrows():
            sid = row.get("store_id")
            pid = row.get("product_id")
            mae = row.get("MAE", 1.0)
            if not mae or np.isnan(mae):
                continue
            n_obs = max(baseline_win + alert_win + 5, 50)
            for i in range(n_obs):
                # More realistic noise: baseline 0.8x, recent 1.8x (not 2.5x)
                noise_scale = mae * (1.8 if i >= n_obs - alert_win else 0.8)
                err = float(rng.normal(0, noise_scale))
                detector.add_observation(sid, pid, f"day_{i}", 10.0 + err, 10.0)

        alerts_df = detector.get_alerts_df()
        summary_df = detector.summary()

        # Executive KPIs
        monitored_pairs = len(summary_df)
        total_alerts = len(alerts_df)
        # Alert rate: % of pairs with at least one drift alert (0-100%)
        alert_rate = (total_alerts / monitored_pairs * 100) if monitored_pairs else 0.0

        mk1, mk2, mk3, mk4, mk5 = st.columns(5)
        mk1.metric("Monitored Pairs", monitored_pairs)
        mk2.metric("Total Alerts", total_alerts)
        mk3.metric("Sliding-Window", int((alerts_df["method"] == "sliding_window").sum()) if total_alerts else 0)
        mk4.metric("CUSUM", int((alerts_df["method"] == "cusum").sum()) if total_alerts else 0)
        mk5.metric("Alert Rate (%)", f"{alert_rate:.1f}%")

        st.markdown("---")

        # Make large tables usable: summarize, then let users drill down
        if total_alerts > 0:
            alerts_df = alerts_df.copy()
            for c in ("baseline_mae", "recent_mae", "threshold_ratio"):
                if c in alerts_df.columns:
                    alerts_df[c] = pd.to_numeric(alerts_df[c], errors="coerce")

            # Severity buckets on ratio (simple, explainable)
            def _severity_bucket(r: float) -> str:
                try:
                    if r >= 3.0:
                        return "Critical (≥3×)"
                    if r >= 2.0:
                        return "High (2–3×)"
                    if r >= 1.5:
                        return "Medium (1.5–2×)"
                    return "Low (<1.5×)"
                except Exception:
                    return "Unknown"

            alerts_df["severity"] = alerts_df.get("threshold_ratio", pd.Series([None] * len(alerts_df))).apply(_severity_bucket)
            alerts_df["store_id"] = alerts_df["store_id"].astype(str)
            alerts_df["product_id"] = alerts_df["product_id"].astype(str)

            # Primary summary visuals
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("#### Where is drift happening? (Top stores)")
                store_top = (
                    alerts_df.groupby("store_id", as_index=False)
                    .agg(alerts=("store_id", "size"), avg_ratio=("threshold_ratio", "mean"))
                    .sort_values(["alerts", "avg_ratio"], ascending=False)
                    .head(15)
                )
                fig_store = px.bar(
                    store_top,
                    x="alerts",
                    y="store_id",
                    orientation="h",
                    color="avg_ratio",
                    color_continuous_scale="Reds",
                    labels={"alerts": "# Alerts", "store_id": "Store", "avg_ratio": "Avg ratio"},
                    text_auto=True,
                )
                fig_store.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10))
                st.plotly_chart(fig_store, use_container_width=True)

            with col_a2:
                st.markdown("#### What is drifting? (Top products)")
                prod_top = (
                    alerts_df.groupby("product_id", as_index=False)
                    .agg(alerts=("product_id", "size"), avg_ratio=("threshold_ratio", "mean"))
                    .sort_values(["alerts", "avg_ratio"], ascending=False)
                    .head(15)
                )
                fig_prod = px.bar(
                    prod_top,
                    x="alerts",
                    y="product_id",
                    orientation="h",
                    color="avg_ratio",
                    color_continuous_scale="Reds",
                    labels={"alerts": "# Alerts", "product_id": "Product", "avg_ratio": "Avg ratio"},
                    text_auto=True,
                )
                fig_prod.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10))
                st.plotly_chart(fig_prod, use_container_width=True)

            st.markdown("---")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("#### Severity distribution")
                sev_df = (
                    alerts_df.groupby(["severity"], as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                )
                # ordering
                order = ["Critical (≥3×)", "High (2–3×)", "Medium (1.5–2×)", "Low (<1.5×)", "Unknown"]
                sev_df["severity"] = pd.Categorical(sev_df["severity"], categories=order, ordered=True)
                sev_df = sev_df.sort_values("severity")
                fig_sev = px.bar(sev_df, x="severity", y="count", color="severity", text_auto=True)
                fig_sev.update_layout(showlegend=False, xaxis_title=None, yaxis_title="# Alerts", margin=dict(t=10, b=10))
                st.plotly_chart(fig_sev, use_container_width=True)

            with col_b2:
                st.markdown("#### Ratio distribution (recent MAE ÷ baseline MAE)")
                fig_hist = px.histogram(
                    alerts_df,
                    x="threshold_ratio",
                    nbins=30,
                    color_discrete_sequence=["#ef4444"],
                    labels={"threshold_ratio": "MAE ratio"},
                )
                fig_hist.add_vline(x=mae_thresh, line_dash="dash", line_color="#f97316", annotation_text="threshold")
                fig_hist.update_layout(margin=dict(t=30, b=10))
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")

            st.markdown("#### Drill-down (show only what matters)")
            f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
            with f1:
                method_pick = st.multiselect(
                    "Method",
                    options=sorted(alerts_df["method"].unique().tolist()),
                    default=sorted(alerts_df["method"].unique().tolist()),
                )
            with f2:
                severity_pick = st.multiselect(
                    "Severity",
                    options=[x for x in ["Critical (≥3×)", "High (2–3×)", "Medium (1.5–2×)", "Low (<1.5×)"] if x in alerts_df["severity"].unique()],
                    default=[x for x in ["Critical (≥3×)", "High (2–3×)", "Medium (1.5–2×)"] if x in alerts_df["severity"].unique()],
                )
            with f3:
                top_n = st.number_input("Show top N rows", min_value=10, max_value=200, value=50, step=10)
            with f4:
                min_ratio = st.number_input("Min ratio", min_value=1.0, max_value=10.0, value=float(mae_thresh), step=0.1)

            filtered = alerts_df[
                alerts_df["method"].isin(method_pick)
                & alerts_df["severity"].isin(severity_pick if severity_pick else alerts_df["severity"].unique())
                & (alerts_df["threshold_ratio"] >= float(min_ratio))
            ].copy()

            filtered = filtered.sort_values(["threshold_ratio", "recent_mae"], ascending=False)

            st.dataframe(
                filtered[[
                    "store_id",
                    "product_id",
                    "method",
                    "detected_at",
                    "severity",
                    "baseline_mae",
                    "recent_mae",
                    "threshold_ratio",
                ]].head(int(top_n)),
                use_container_width=True,
            )

            # Always allow full download (no scrolling)
            csv_alerts = io.StringIO()
            alerts_df.to_csv(csv_alerts, index=False)
            st.download_button(
                "Download Drift Alerts CSV",
                csv_alerts.getvalue(),
                file_name="drift_alerts.csv",
                mime="text/csv",
            )

        else:
            st.success("No drift detected with current thresholds.")

        st.markdown("---")
        st.markdown("### Structured KPI Export")
        st.markdown("Export all optimization and forecast KPIs to **CSV** for BI dashboards.")

        if st.button("Generate CSV Export", use_container_width=True):
            with st.spinner("Building KPI schema and exporting..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        result = export_all_kpis(
                            scenario_json=_scen5,
                            transfers_json=_trans5,
                            manufacturing_json=_mfg5,
                            inventory_csv="optimization/output-csv/optimization_inventory.csv",
                            forecast_metrics_csv="demand-forecast/output/product_forecast_metrics.csv",
                            output_dir=tmpdir,
                            formats=["csv"],
                        )
                        # Zip all output files for download
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                            csv_dir = os.path.join(tmpdir, "csv")
                            if os.path.exists(csv_dir):
                                for fname in os.listdir(csv_dir):
                                    zf.write(os.path.join(csv_dir, fname), arcname=f"csv/{fname}")
                        zip_buf.seek(0)
                        st.success("Export ready!")
                        st.download_button(
                            "Download KPI Export (ZIP)",
                            zip_buf.getvalue(),
                            file_name="supply_chain_kpis.zip",
                            mime="application/zip",
                        )
                        # Preview DataFrames
                        st.markdown("**Schema Preview:**")
                        for name, df in result["dataframes"].items():
                            with st.expander(f"{name} ({len(df)} rows)"):
                                st.dataframe(df.head(10), use_container_width=True)
                    except Exception as e:
                        st.error(f"Export failed: {e}")

with tab1:
    st.markdown("### Chat Assistant")
    st.markdown("Query optimization scenarios, transfers, manufacturing decisions, and metrics.")

    # Enhanced chat UI with messages INSIDE the container
    st.markdown("""
    <style>
    /* Main chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 600px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Messages scroll area - INSIDE container */
    .messages-area {
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        scroll-behavior: smooth;  /* Smooth scrolling */
    }
    
    /* Message bubble */
    .message-bubble {
        display: flex;
        animation: slideIn 0.3s ease-out;
        margin-bottom: 0.5rem;
    }
    
    .message-bubble.user-msg {
        justify-content: flex-end;
    }
    
    /* Message content */
    .msg-content {
        max-width: 75%;
        padding: 1rem;
        border-radius: 12px;
        word-wrap: break-word;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .msg-user {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.85) 0%, rgba(56, 189, 248, 0.65) 100%);
        color: #f8fafc;
        border: 1px solid rgba(14, 165, 233, 0.4);
    }
    
    .msg-assistant {
        background: rgba(30, 41, 59, 0.8);
        color: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    
    /* Markdown content styling inside assistant messages */
    .msg-assistant table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        background: rgba(15, 23, 42, 0.6);
    }
    
    .msg-assistant th {
        background: rgba(14, 165, 233, 0.2);
        color: #0ea5e9;
        padding: 0.5rem;
        text-align: left;
        border: 1px solid rgba(148, 163, 184, 0.2);
        font-weight: 600;
    }
    
    .msg-assistant td {
        padding: 0.5rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    
    .msg-assistant tr:hover {
        background: rgba(14, 165, 233, 0.05);
    }
    
    .msg-assistant strong {
        color: #38bdf8;
        font-weight: 600;
    }
    
    .msg-assistant code {
        background: rgba(15, 23, 42, 0.8);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #7dd3fc;
    }
    
    .msg-assistant pre {
        background: rgba(15, 23, 42, 0.8);
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .msg-assistant pre code {
        background: none;
        padding: 0;
    }
    
    /* Badge styling */
    .intent-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.3), rgba(56, 189, 248, 0.2));
        color: #38bdf8;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(14, 165, 233, 0.4);
        margin-right: 0.5rem;
    }
    
    .filter-badge {
        display: inline-block;
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        margin-right: 0.5rem;
    }
    
    .msg-assistant h1,
    .msg-assistant h2,
    .msg-assistant h3,
    .msg-assistant h4 {
        color: #0ea5e9;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .msg-assistant ul,
    .msg-assistant ol {
        padding-left: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .msg-assistant li {
        margin: 0.25rem 0;
    }
    
    .msg-assistant p {
        margin: 0.5rem 0;
    }
    
    /* Empty state */
    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: rgba(148, 163, 184, 0.5);
        text-align: center;
        font-style: italic;
        font-size: 0.95rem;
    }
    
    /* Input section */
    .input-section {
        flex-shrink: 0;
        padding: 1rem;
        border-top: 1px solid rgba(14, 165, 233, 0.2);
        background: rgba(15, 23, 42, 0.95);
    }
    
    /* Scrollbar styling */
    .messages-area::-webkit-scrollbar {
        width: 6px;
    }
    
    .messages-area::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .messages-area::-webkit-scrollbar-thumb {
        background: rgba(14, 165, 233, 0.3);
        border-radius: 3px;
    }
    
    .messages-area::-webkit-scrollbar-thumb:hover {
        background: rgba(14, 165, 233, 0.5);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Hide Streamlit's default block spacing inside chat */
    .chat-container [class*="block-container"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .chat-container .stMarkdown {
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Chat container starts
    import html
    import re

    def simple_markdown_to_html(text):
        """Convert basic markdown to HTML without external dependencies."""
        # Preserve HTML tags (like badges) by temporarily replacing them
        html_tags = []
        def save_html(match):
            html_tags.append(match.group(0))
            # Use a unique placeholder that won't be affected by markdown processing
            # Format: ≪HTMLTAG·0≫ (using special unicode characters)
            return f"≪HTMLTAG·{len(html_tags)-1}≫"

        # Save HTML tags
        text = re.sub(r'<[^>]+>', save_html, text)

        # Convert markdown tables to HTML
        lines = text.split('\n')
        result = []
        in_table = False

        for i, line in enumerate(lines):
            # Detect table rows
            if '|' in line and line.strip().startswith('|'):
                cells = [cell.strip() for cell in line.strip().split('|')[1:-1]]

                # Check if this is a separator line (|------|)
                if all(re.match(r'^-+$', cell) for cell in cells if cell):
                    # Skip separator line, table header was previous line
                    continue

                # Check if next line is separator (this is a header)
                is_header = False
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if '|' in next_line and all(re.match(r'^-+$', cell.strip()) for cell in next_line.split('|')[1:-1] if cell.strip()):
                        is_header = True
                        if not in_table:
                            result.append('<table>')
                            in_table = True
                        result.append('<thead><tr>')
                        for cell in cells:
                            result.append(f'<th>{cell}</th>')
                        result.append('</tr></thead><tbody>')
                        continue

                # Regular table row
                if not in_table:
                    result.append('<table><tbody>')
                    in_table = True

                result.append('<tr>')
                for cell in cells:
                    result.append(f'<td>{cell}</td>')
                result.append('</tr>')
            else:
                # Close table if we were in one
                if in_table:
                    result.append('</tbody></table>')
                    in_table = False

                # Skip markdown conversion for lines with placeholders
                has_placeholder = '≪HTMLTAG·' in line

                if not has_placeholder:
                    # Convert markdown bold (**text** - avoid converting __text__ to prevent placeholder corruption)
                    line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)

                    # Convert markdown italic (*text* - avoid _text_ to prevent placeholder corruption)
                    line = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', line)

                # Add line breaks for non-empty lines
                if line.strip():
                    result.append(line + '<br>')
                else:
                    result.append('<br>')

        # Close table if still open
        if in_table:
            result.append('</tbody></table>')

        html_output = '\n'.join(result)

        # Restore HTML tags
        for idx, tag in enumerate(html_tags):
            html_output = html_output.replace(f'≪HTMLTAG·{idx}≫', tag)

        return html_output

    messages_html = '<div class="chat-container"><div class="messages-area" id="messagesArea">'

    # Build all messages as HTML string with proper escaping
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                # Escape user input to prevent HTML injection
                content = html.escape(msg["content"])
                content = content.replace('\n', '<br>')
                messages_html += f'''<div class="message-bubble user-msg">
                    <div class="msg-content msg-user">{content}</div>
                </div>'''
            else:
                # For assistant messages, convert markdown to HTML
                content = msg["content"]
                content_html = simple_markdown_to_html(content)

                messages_html += f'''<div class="message-bubble">
                    <div class="msg-content msg-assistant">{content_html}</div>
                </div>'''
    else:
        messages_html += '<div class="empty-state">Start a conversation by typing your question below...</div>'

    # Add anchor element at the bottom for scroll targeting
    messages_html += '<div id="messagesBottom" style="scroll-margin-top: 10px;"></div>'

    # Close container divs
    messages_html += '</div></div>'

    # Render the entire container as one block
    st.markdown(messages_html, unsafe_allow_html=True)

    # Input section OUTSIDE the container
    col1, col2 = st.columns([20, 1])

    with col1:
        if prompt := st.chat_input("Ask about transfers, manufacturing, or scenario metrics…"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # ── Classify intent for badge display (lightweight, also used for
            #    contextual follow-up logic) ─────────────────────────────────
            with st.spinner("Classifying intent…"):
                intent = classify_intent(prompt)

            params = extract_parameters(prompt)

            # Contextual fallback for follow-up questions
            if intent == "out_of_scope" and st.session_state.last_intent:
                has_specifics = any(params.get(k) for k in ["product_id", "store_id"])
                if params.get("is_all") or has_specifics:
                    intent = st.session_state.last_intent

            if intent not in ("out_of_scope", "greeting"):
                st.session_state.last_intent = intent

            label = INTENT_LABELS.get(intent, intent)
            nlp_scenario = detect_scenario(prompt)

            # ── Run the full NLP pipeline via the orchestrator ─────────────
            with st.spinner("Processing query…"):
                final_response = handle_user_query(prompt)

            # ── Build display with intent and scenario badges ──────────────
            has_specifics = any(params.get(k) for k in ["product_id", "store_id"])
            badge_html = f'<span class="intent-badge">{label}</span>'
            if has_specifics:
                badge_html += ' <span class="filter-badge">Specific Filter Applied</span>'
            scenario_badge = f'<span class="filter-badge">Scenario: {nlp_scenario}</span>'

            if intent in ("greeting", "out_of_scope"):
                full_display = final_response
            else:
                full_display = f"{badge_html} {scenario_badge}\n\n{final_response}"

            st.session_state.messages.append({"role": "assistant", "content": full_display})
            st.rerun()

    # Simple auto-scroll script - ONLY scrolls the container
    st.markdown("""
    <script>
        const messagesArea = document.getElementById('messagesArea');
        if (messagesArea) {
            messagesArea.scrollTop = messagesArea.scrollHeight;
        }
    </script>
    """, unsafe_allow_html=True)
