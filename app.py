import io
import json
import subprocess
import time
import requests
import streamlit as st
import csv
import ast
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from nlp.intent_classifier import classify_intent, extract_parameters
from nlp.explanation_engine import build_explanation
from nlp.refiner import refine_explanation
from nlp.scenario_compare import ScenarioSnapshot, compare_scenarios, sensitivity_analysis_text
from optimization.stochastic import generate_scenarios, compute_cvar
from monitoring.drift import ForecastDriftDetector
from export.kpi_export import export_all_kpis


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

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon=None,
    layout="wide",
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
        padding: 2rem 0 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 2rem;
    }

    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        font-size: 0.95rem;
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

    .stChatInputContainer {
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1000px;
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

SAMPLE_DATA_DIR = "optimization/output-json"

INTENT_LABELS = {
    "explain_transfer": "Transfer Details",
    "explain_manufacturing": "Manufacturing Details",
    "scenario_summary": "Scenario Summary",
    "impact_analysis": "Impact Analysis",
    "list_entities": "Entity List",
    "total_counts": "Summary Metrics",
    "top_transfers": "Top Transfers",
    "top_manufacturing": "Top Manufacturing",
    "urgent_transfers": "Urgent Transfers",
    "high_cost_actions": "High-Cost Actions",
    "reason_analysis": "Decision Analysis",
    "store_activity": "Store Activity",
    "product_recommendations": "Product Actions",
    "cost_breakdown": "Cost Breakdown",
    "out_of_scope": "Out of Scope",
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
    st.session_state.messages = []

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

if "ollama_ok" not in st.session_state:
    st.session_state.ollama_ok = False
    st.session_state.ollama_msg = "Checking system status..."

# Sidebar Controls
with st.sidebar:
    st.markdown("### Control Panel")
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_intent = None
        st.rerun()

    st.markdown("---")
    st.markdown("### System Status")
    
    if "ollama_checked" not in st.session_state:
        # Check ollama on first load quietly
        ok, msg = _ensure_ollama()
        st.session_state.ollama_ok = ok
        st.session_state.ollama_msg = msg
        st.session_state.ollama_checked = True

    if st.session_state.ollama_ok:
        st.success("Refinement Engine: Online")
    else:
        st.warning("Refinement Engine: Offline\n\n(Using Deterministic Fallback)")
        st.caption(st.session_state.ollama_msg)
        if st.button("Start Engine", use_container_width=True):
            with st.spinner("Starting engine..."):
                ok, msg = _ensure_ollama()
                st.session_state.ollama_ok = ok
                st.session_state.ollama_msg = msg
                st.rerun()

AVATAR_USER = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#334155" rx="20"/><text x="50" y="65" font-family="sans-serif" font-weight="bold" font-size="50" fill="#f8fafc" text-anchor="middle">U</text></svg>'''
AVATAR_AI = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#0ea5e9" rx="20"/><text x="50" y="65" font-family="sans-serif" font-weight="bold" font-size="50" fill="#f8fafc" text-anchor="middle">AI</text></svg>'''

# ── Helper: load data once per session ────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_all_data():
    """Load all optimization and forecast data into a single dict."""
    d = {}
    try:
        d["scenario"] = load_json(f"{SAMPLE_DATA_DIR}/scenario_summary.json")
    except Exception:
        d["scenario"] = {}
    try:
        d["transfers"] = load_json(f"{SAMPLE_DATA_DIR}/transfer_recommendations.json")
    except Exception:
        d["transfers"] = {}
    try:
        d["manufacturing"] = load_json(f"{SAMPLE_DATA_DIR}/manufacturing_decisions.json")
    except Exception:
        d["manufacturing"] = {}
    d["inventory"] = load_csv("optimization/output-csv/optimization_inventory.csv")
    d["opt_transfers"] = load_csv("optimization/output-csv/optimization_transfers.csv")
    d["opt_manufacturing"] = load_csv("optimization/output-csv/optimization_manufacturing.csv")
    d["forecast_metrics"] = load_csv("demand-forecast/output/product_forecast_metrics.csv")
    return d


# ── Tab layout ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 AI Assistant",
    "📊 Optimization Dashboard",
    "📈 Demand Forecast",
    "🎲 Stochastic Scenarios",
    "📡 Monitoring & Export",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Optimization Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    _all = _load_all_data()
    scenario_data = _all.get("scenario", {})
    optimized = scenario_data.get("optimized", {})
    cost_breakdown = scenario_data.get("cost_breakdown", {})
    transfers_list = _all.get("transfers", {}).get("transfers", [])
    mfg_list = _all.get("manufacturing", {}).get("manufacturing_actions", [])
    inv_list = _all.get("inventory", [])
    opt_transfers = _all.get("opt_transfers", [])
    opt_manufacturing = _all.get("opt_manufacturing", [])

    st.markdown("### Optimization Overview")

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
            "⬇ Download Transfers CSV", csv_buf.getvalue(),
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
                "Manufacture Qty": m.get("manufacture_quantity", 0),
                "Manufacturing Cost ($)": m.get("cost_impact", {}).get("manufacturing_cost", 0),
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
            "⬇ Download Manufacturing CSV", csv_buf2.getvalue(),
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
            "⬇ scenario_summary.json",
            json.dumps(scenario_data, indent=2),
            file_name="scenario_summary.json",
            mime="application/json",
        )
    with col_dl2:
        st.download_button(
            "⬇ transfer_recommendations.json",
            json.dumps(_all.get("transfers", {}), indent=2),
            file_name="transfer_recommendations.json",
            mime="application/json",
        )
    with col_dl3:
        st.download_button(
            "⬇ manufacturing_decisions.json",
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
            "⬇ Download Forecast Metrics CSV", csv_fm.getvalue(),
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
    _transfers4 = _all4.get("transfers", {}).get("transfers", [])

    st.markdown("### Stochastic Demand Scenario Analysis")
    st.markdown(
        "Explore how cost and risk metrics change when demand is uncertain. "
        "Scenarios are generated with a **log-normal** demand distribution around point forecasts."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        n_scen = st.slider("Number of scenarios (Ω)", 10, 100, 30, step=10)
    with col_cfg2:
        demand_cv = st.slider("Demand CV (coefficient of variation)", 0.05, 0.60, 0.20, step=0.05)
    with col_cfg3:
        alpha_cvar = st.slider("CVaR confidence level (α)", 0.80, 0.99, 0.95, step=0.01)

    # Build mean demands from the transfers/manufacturing data
    @st.cache_data(show_spinner=False)
    def _build_mean_demands(transfers_json_str: str):
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
        # Convert back to tuple keys for generate_scenarios
        result = {}
        for k, v in demands.items():
            parts = k.split("_", 1)
            result[(int(parts[0]), parts[1])] = v
        return result

    mean_demands = _build_mean_demands(json.dumps(_transfers4))

    # Use a small subset for display (≤ 30 pairs for performance)
    sample_demands = dict(list(mean_demands.items())[:30]) if len(mean_demands) > 30 else mean_demands

    if not sample_demands:
        st.warning("No demand data available for scenario generation.")
    else:
        with st.spinner(f"Generating {n_scen} demand scenarios…"):
            scenarios, probs = generate_scenarios(
                sample_demands,
                n_scenarios=n_scen,
                cv=demand_cv,
                random_state=42,
            )

        # Scenario cost proxy: sum of demands × unit cost for each scenario
        unit_cost = 50.0
        scenario_costs = [
            sum(v * unit_cost for v in s.values()) for s in scenarios
        ]

        expected_cost = float(sum(c * p for c, p in zip(scenario_costs, probs)))
        var_val = float(pd.Series(scenario_costs).quantile(alpha_cvar))
        cvar_val = compute_cvar(scenario_costs, alpha=alpha_cvar)

        # ── KPI row ─────────────────────────────────────────────────────────
        sk1, sk2, sk3, sk4 = st.columns(4)
        sk1.metric("Expected Cost", f"${expected_cost:,.0f}")
        sk2.metric(f"VaR ({alpha_cvar:.0%})", f"${var_val:,.0f}")
        sk3.metric(f"CVaR ({alpha_cvar:.0%})", f"${cvar_val:,.0f}")
        risk_premium = cvar_val - expected_cost
        sk4.metric("Risk Premium", f"${risk_premium:,.0f}",
                   delta=f"{risk_premium/expected_cost*100:.1f}% of EV" if expected_cost else None)

        st.markdown("---")

        # ── Distribution chart ───────────────────────────────────────────────
        col_dist, col_cdf = st.columns(2)
        with col_dist:
            st.markdown("#### Scenario Cost Distribution")
            scen_df = pd.DataFrame({"scenario": range(1, n_scen + 1), "cost": scenario_costs})
            fig_hist = px.histogram(scen_df, x="cost", nbins=20, color_discrete_sequence=["#0ea5e9"])
            fig_hist.add_vline(x=expected_cost, line_dash="solid", line_color="#f97316",
                               annotation_text="E[cost]")
            fig_hist.add_vline(x=var_val, line_dash="dash", line_color="#dc2626",
                               annotation_text=f"VaR({alpha_cvar:.0%})")
            fig_hist.add_vline(x=cvar_val, line_dash="dot", line_color="#7c3aed",
                               annotation_text=f"CVaR({alpha_cvar:.0%})")
            fig_hist.update_layout(margin=dict(t=30, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_cdf:
            st.markdown("#### Empirical CDF")
            sorted_costs = sorted(scenario_costs)
            cdf_y = [(i + 1) / n_scen for i in range(n_scen)]
            fig_cdf = px.line(x=sorted_costs, y=cdf_y, labels={"x": "Cost ($)", "y": "Cumulative Probability"})
            fig_cdf.add_vline(x=var_val, line_dash="dash", line_color="#dc2626",
                              annotation_text=f"VaR({alpha_cvar:.0%})")
            fig_cdf.update_layout(margin=dict(t=30, b=10))
            st.plotly_chart(fig_cdf, use_container_width=True)

        st.markdown("---")

        # ── Scenario comparison ───────────────────────────────────────────────
        st.markdown("### Scenario Comparison (Baseline vs. Service-First)")
        st.markdown("Simulate a *service-first* scenario with 20% more demand (proxy for prioritising service level over cost).")

        baseline_snap = ScenarioSnapshot.from_json(_scenario4) if _scenario4 else ScenarioSnapshot(
            name="baseline", total_cost=expected_cost,
            manufacturing_cost=expected_cost * 0.8, transfer_cost=expected_cost * 0.01,
            holding_cost=expected_cost * 0.19, total_transfers=len(_transfers4),
            manufacturing_units=500.0, transfer_units=50.0,
        )

        service_first_snap = ScenarioSnapshot(
            name="service_first",
            total_cost=baseline_snap.total_cost * 1.18,
            manufacturing_cost=baseline_snap.manufacturing_cost * 1.22,
            transfer_cost=baseline_snap.transfer_cost * 1.35,
            holding_cost=baseline_snap.holding_cost * 1.05,
            total_transfers=int(baseline_snap.total_transfers * 1.30),
            manufacturing_units=baseline_snap.manufacturing_units * 1.22,
            transfer_units=baseline_snap.transfer_units * 1.35,
        )

        narrative = compare_scenarios(baseline_snap, service_first_snap)
        st.markdown(narrative)

        st.markdown("---")

        # ── Sensitivity analysis ─────────────────────────────────────────────
        st.markdown("### Sensitivity: Cost vs. Demand CV")
        cv_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        sens_costs = []
        for cv_val in cv_values:
            scen_cv, probs_cv = generate_scenarios(sample_demands, n_scenarios=30, cv=cv_val, random_state=42)
            cost_cv = sum(c * p for c, p in zip([sum(v * unit_cost for v in s.values()) for s in scen_cv], probs_cv))
            sens_costs.append(cost_cv)

        sens_text = sensitivity_analysis_text(baseline_snap, "demand_cv", cv_values, sens_costs)
        with st.expander("📄 Sensitivity Analysis Narrative"):
            st.markdown(sens_text)

        sens_df = pd.DataFrame({"Demand CV": cv_values, "Expected Cost ($)": sens_costs})
        fig_sens = px.line(sens_df, x="Demand CV", y="Expected Cost ($)",
                           markers=True, color_discrete_sequence=["#0ea5e9"])
        fig_sens.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_sens, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Monitoring & Export
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    _all5 = _load_all_data()
    fm5 = _all5.get("forecast_metrics", [])

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

        @st.cache_data(show_spinner=False)
        def _build_drift_data(_fm_df_json, _bw, _aw, _thresh):
            """Build drift detector from simulated rolling observations, return serializable results."""
            import json as _json
            import numpy as np
            fm_df = pd.read_json(_json.loads(_fm_df_json))
            detector = ForecastDriftDetector(
                baseline_window=_bw, alert_window=_aw, mae_threshold=_thresh
            )
            rng = np.random.default_rng(42)
            for _, row in fm_df.iterrows():
                sid = row.get("store_id")
                pid = row.get("product_id")
                mae = row.get("MAE", 1.0)
                if not mae or np.isnan(mae):
                    continue
                n_obs = max(_bw + _aw + 5, 50)
                for i in range(n_obs):
                    noise_scale = mae * (2.5 if i >= n_obs - _aw else 0.8)
                    err = float(rng.normal(0, noise_scale))
                    detector.add_observation(sid, pid, f"day_{i}", 10.0 + err, 10.0)
            return detector.get_alerts_df(), detector.summary()

        alerts_df, summary_df = _build_drift_data(
            fm5_df.to_json(),
            baseline_win, alert_win, mae_thresh,
        )

        # ── Metric cards ─────────────────────────────────────────────────────
        mk1, mk2, mk3, mk4 = st.columns(4)
        mk1.metric("Monitored Pairs", len(summary_df))
        mk2.metric("Total Alerts", len(alerts_df))
        sw_alerts = len(alerts_df[alerts_df["method"] == "sliding_window"]) if len(alerts_df) > 0 else 0
        cusum_alerts = len(alerts_df[alerts_df["method"] == "cusum"]) if len(alerts_df) > 0 else 0
        mk3.metric("Sliding-Window Alerts", sw_alerts)
        mk4.metric("CUSUM Alerts", cusum_alerts)

        st.markdown("---")

        if len(alerts_df) > 0:
            st.markdown("#### 🚨 Drift Alerts")
            st.dataframe(
                alerts_df[["store_id", "product_id", "method", "detected_at",
                            "baseline_mae", "recent_mae", "threshold_ratio", "message"]],
                use_container_width=True,
            )
            csv_alerts = io.StringIO()
            alerts_df.to_csv(csv_alerts, index=False)
            st.download_button("⬇ Download Alerts CSV", csv_alerts.getvalue(),
                               file_name="drift_alerts.csv", mime="text/csv")
        else:
            st.success("✅ No drift detected with current thresholds.")

        st.markdown("---")
        st.markdown("#### Monitoring Summary")
        st.dataframe(
            summary_df.sort_values("mae", ascending=False).head(20).reset_index(drop=True),
            use_container_width=True,
        )

    else:
        st.info("Forecast metrics not available for drift monitoring.")

    st.markdown("---")
    st.markdown("### Structured KPI Export")
    st.markdown("Export all optimization and forecast KPIs to **Parquet** and **DuckDB** for BI dashboards.")

    _scen5 = _all5.get("scenario", {})
    _trans5 = _all5.get("transfers", {})
    _mfg5 = _all5.get("manufacturing", {})

    import tempfile, zipfile, os

    if st.button("🚀 Generate Parquet + DuckDB Export", use_container_width=True):
        with st.spinner("Building KPI schema and exporting…"):
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    result = export_all_kpis(
                        scenario_json=_scen5,
                        transfers_json=_trans5,
                        manufacturing_json=_mfg5,
                        inventory_csv="optimization/output-csv/optimization_inventory.csv",
                        forecast_metrics_csv="demand-forecast/output/product_forecast_metrics.csv",
                        output_dir=tmpdir,
                        formats=["parquet", "duckdb"],
                    )
                    # Zip all output files for download
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        parquet_dir = os.path.join(tmpdir, "parquet")
                        if os.path.exists(parquet_dir):
                            for fname in os.listdir(parquet_dir):
                                zf.write(os.path.join(parquet_dir, fname),
                                         arcname=f"parquet/{fname}")
                        db_path = os.path.join(tmpdir, "supply_chain_kpis.duckdb")
                        if os.path.exists(db_path):
                            zf.write(db_path, arcname="supply_chain_kpis.duckdb")
                    zip_buf.seek(0)
                    st.success("✅ Export ready!")
                    st.download_button(
                        "⬇ Download KPI Export (Parquet + DuckDB)",
                        zip_buf.getvalue(),
                        file_name="supply_chain_kpis.zip",
                        mime="application/zip",
                    )
                    # Preview DataFrames
                    st.markdown("**Schema Preview:**")
                    for name, df in result["dataframes"].items():
                        with st.expander(f"📋 {name} ({len(df)} rows)"):
                            st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Export failed: {e}")

with tab1:
    for msg in st.session_state.messages:
        avatar_val = AVATAR_USER if msg["role"] == "user" else AVATAR_AI
        with st.chat_message(msg["role"], avatar=avatar_val):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about transfers, manufacturing, or scenario metrics…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AVATAR_AI):
            with st.spinner("Classifying intent…"):
                intent = classify_intent(prompt)

            params = extract_parameters(prompt)

            # Contextual fallback for follow-up questions
            if intent == "out_of_scope" and st.session_state.last_intent:
                has_specifics = any(params.get(k) for k in ["product_id", "store_id"])
                if params["is_all"] or has_specifics:
                    intent = st.session_state.last_intent

            if intent not in ("out_of_scope", "greeting"):
                st.session_state.last_intent = intent

            label = INTENT_LABELS.get(intent, intent)

            if intent == "greeting":
                response = (
                    "Hello. I am the Supply Chain Analytics Assistant.\n\n"
                    "I can help you analyze optimization recommendations across multiple dimensions:\n\n"
                    "**Overview & Summaries:**\n"
                    "- *\"Provide a scenario summary\"* - Overall optimization results\n"
                    "- *\"Show cost breakdown\"* - Detailed cost composition\n"
                    "- *\"How many recommendations total\"* - Summary counts\n\n"
                    "**Transfer Analysis:**\n"
                    "- *\"Explain transfer recommendations\"* - Detailed transfer details\n"
                    "- *\"Top transfers by cost\"* - Prioritized high-cost transfers\n"
                    "- *\"Urgent transfers\"* - Transfers for stockout prevention\n"
                    "- *\"Store activity\"* - Store involvement in transfers\n\n"
                    "**Manufacturing Analysis:**\n"
                    "- *\"Detail manufacturing decisions\"* - Production action specifics\n"
                    "- *\"Top manufacturing by cost\"* - Highest priority manufacturing\n\n"
                    "**Decision Insights:**\n"
                    "- *\"Why these decisions\"* - Analysis of decision reasons\n"
                    "- *\"High-cost actions\"* - Most expensive recommendations\n"
                    "- *\"Product specific details\"* - Product-level recommendations\n\n"
                    "Please enter your query below to get started."
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif intent == "out_of_scope":
                response = (
                    "This query appears to be outside my designated scope. I am calibrated strictly for supply chain "
                    "optimization analysis, including inventory transfers, production runs, and cost diagnostics.\n\n"
                    "Please rephrase your request. For example:\n"
                    "- *\"Why was inventory transferred between facilities?\"*\n"
                    "- *\"What manufacturing actions were recommended?\"*\n"
                    "- *\"Compare the optimized scenario against the baseline.\"*"
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                has_specifics = any(params.get(k) for k in ["product_id", "store_id"])

                badge_html = f'<span class="intent-badge">{label}</span>'
                if has_specifics:
                    badge_html += ' <span class="filter-badge">Specific Filter Applied</span>'
                st.markdown(badge_html, unsafe_allow_html=True)
                data = {
                    "scenario": load_json(f"{SAMPLE_DATA_DIR}/scenario_summary.json"),
                    "transfers": load_json(f"{SAMPLE_DATA_DIR}/transfer_recommendations.json"),
                    "manufacturing": load_json(f"{SAMPLE_DATA_DIR}/manufacturing_decisions.json"),
                    # Additional contextual sources (optimization CSVs and demand-forecast outputs)
                    "optimization_summary": None,
                    "inventory": [],
                    "optimization_transfers_csv": [],
                    "optimization_manufacturing_csv": [],
                    "forecast_metrics": [],
                    "product_forecasts": [],
                }

                # Try to load extra files (non-fatal)
                try:
                    data["optimization_summary"] = load_json("optimization/output-csv/optimization_summary.json")
                except Exception:
                    data["optimization_summary"] = None

                data["inventory"] = load_csv("optimization/output-csv/optimization_inventory.csv")
                data["optimization_transfers_csv"] = load_csv("optimization/output-csv/optimization_transfers.csv")
                data["optimization_manufacturing_csv"] = load_csv("optimization/output-csv/optimization_manufacturing.csv")

                # Demand forecast context
                data["forecast_metrics"] = load_csv("demand-forecast/output/product_forecast_metrics.csv")
                # product_forecasts.csv can be large; load if present
                data["product_forecasts"] = load_csv("demand-forecast/output/product_forecasts.csv")

                # Build a small RAG context (only include rows relevant to the user's filters)
                def _normalize_param_ids(params):
                    # Accept forms like 'product_489' or '489'
                    pids = []
                    sids = []
                    for p in params.get("product_id", []):
                        if isinstance(p, str) and p.startswith("product_"):
                            pids.append(p.split("product_")[-1])
                        else:
                            pids.append(str(p))
                    for s in params.get("store_id", []):
                        if isinstance(s, str) and s.startswith("store_"):
                            sids.append(s.split("store_")[-1])
                        else:
                            sids.append(str(s))
                    return pids, sids

                def _build_rag_text(data, params, intent, limit=10):
                    pids, sids = _normalize_param_ids(params or {})
                    lines = []
                    # Collect transfers (CSV prioritized for tabular rows)
                    transfers_rows = data.get("optimization_transfers_csv") or data.get("transfers", {}).get("transfers", [])
                    mfg_rows = data.get("optimization_manufacturing_csv") or data.get("manufacturing", {}).get("manufacturing_actions", [])
                    inv_rows = data.get("inventory", [])
                    forecast_rows = data.get("forecast_metrics", [])

                    def _match_row(row):
                        # row may use keys like product_id, from_store, to_store, store_id
                        try:
                            prod = str(row.get("product_id", ""))
                            if pids and prod not in pids:
                                return False
                            if sids:
                                # check any store field
                                if (str(row.get("from_store", "")) not in sids) and (str(row.get("to_store", "")) not in sids) and (str(row.get("store_id", "")) not in sids):
                                    return False
                            return True
                        except Exception:
                            return False

                    # add matching transfers
                    added = 0
                    for r in transfers_rows:
                        if _match_row(r):
                            lines.append(f"TRANSFER | from={r.get('from_store')} to={r.get('to_store')} product={r.get('product_id')} qty={r.get('qty') or r.get('quantity')} cost={r.get('cost') or (r.get('cost_impact') and r.get('cost_impact').get('transport_cost'))} reasons={r.get('reason_codes')}")
                            added += 1
                            if added >= limit:
                                break
                    # manufacturing
                    added = 0
                    for r in mfg_rows:
                        if _match_row(r):
                            lines.append(f"MANUFACTURE | store={r.get('store_id') or ''} product={r.get('product_id')} qty={r.get('qty') or r.get('manufacture_quantity')} cost={r.get('cost') or (r.get('cost_impact') and r.get('cost_impact').get('manufacturing_cost'))} reasons={r.get('reason_codes')}")
                            added += 1
                            if added >= limit:
                                break
                    # inventory short snapshot
                    added = 0
                    for r in inv_rows:
                        if pids and str(r.get('product_id')) not in pids:
                            continue
                        if sids and str(r.get('store_id')) not in sids:
                            continue
                        lines.append(f"INVENTORY | store={r.get('store_id')} product={r.get('product_id')} current={r.get('current')} final={r.get('final')} target={r.get('target')}")
                        added += 1
                        if added >= limit:
                            break
                    # forecast metrics
                    added = 0
                    for r in forecast_rows:
                        if pids and str(r.get('product_id')) not in pids:
                            continue
                        if sids and str(r.get('store_id')) not in sids:
                            continue
                        lines.append(f"FORECAST_METRIC | store={r.get('store_id')} product={r.get('product_id')} MAE={r.get('MAE')} RMSE={r.get('RMSE')}")
                        added += 1
                        if added >= limit:
                            break
                    if not lines:
                        return ""  # nothing relevant
                    # just return the structured rows; raw_explanation will be appended after it's computed
                    rag = "\n".join(lines)
                    return rag

                with st.spinner("Building explanation…"):
                    raw_explanation = build_explanation(intent, data, params)

                # Build RAG context AFTER raw_explanation is computed
                rag_rows_only = _build_rag_text(data, params, intent)
                # Append raw_explanation to the RAG context
                rag_text = (rag_rows_only + "\n\nRAW_EXPLANATION:\n" + raw_explanation) if rag_rows_only else raw_explanation

                refined = None
                fallback = False
                refinement_rejected = False

                # Check for empty state responses directly from explanation_engine
                is_empty_state = raw_explanation.startswith("No transfers match") or raw_explanation.startswith("No manufacturing actions match") or raw_explanation.startswith("No urgent")

                # Bypass LLM refinement for tabular/list data to prevent hallucination
                table_intents = ("list_entities", "total_counts", "top_transfers", "top_manufacturing",
                               "urgent_transfers", "high_cost_actions", "reason_analysis",
                               "store_activity", "product_recommendations", "cost_breakdown")
                skip_refiner = is_empty_state or intent in table_intents

                if skip_refiner:
                    refined = raw_explanation
                else:
                    try:
                        with st.spinner("Refining with TinyLlama…"):
                            # If we have a small RAG context, pass that as the factual input to the refiner
                            refined_input = rag_text if rag_text else raw_explanation
                            refined = refine_explanation(refined_input, user_question=prompt)
                            # Safety checks on refined output to detect possible hallucination
                            if not refined or len(refined) < 10:
                                refinement_rejected = True
                            else:
                                lower_refined = refined.lower()
                                # Reject outputs that contain uncertainty cues or unrelated long narratives
                                uncertain_phrases = ["i think", "maybe", "could be", "possibly", "as an ai", "as a model"]
                                unrelated_indicators = ["transportation system", "buses", "passengers", "cargo", "trams"]
                                if any(p in lower_refined for p in uncertain_phrases):
                                    refinement_rejected = True
                                if any(p in lower_refined for p in unrelated_indicators):
                                    refinement_rejected = True
                                # Bound size to prevent huge unrelated text
                                if len(refined) > 5000:
                                    refinement_rejected = True
                    except Exception:
                        fallback = True

                if refinement_rejected or fallback:
                    # Prefer deterministic root explanation in case of doubt
                    final_response = raw_explanation or "Insufficient data to answer."

                    # Add a provenance note mapping by intent to the primary data source
                    provenance_map = {
                        "explain_transfer": "optimization/output-json/transfer_recommendations.json",
                        "top_transfers": "optimization/output-json/transfer_recommendations.json",
                        "urgent_transfers": "optimization/output-json/transfer_recommendations.json",
                        "store_activity": "optimization/output-json/transfer_recommendations.json",
                        "explain_manufacturing": "optimization/output-json/manufacturing_decisions.json",
                        "top_manufacturing": "optimization/output-json/manufacturing_decisions.json",
                        "high_cost_actions": "optimization/output-json/manufacturing_decisions.json",
                        "product_recommendations": "optimization/output-json/manufacturing_decisions.json",
                        "total_counts": "optimization/output-json/scenario_summary.json",
                        "scenario_summary": "optimization/output-json/scenario_summary.json",
                        "cost_breakdown": "optimization/output-json/scenario_summary.json",
                        "reason_analysis": "optimization/output-json/transfer_recommendations.json",
                    }
                    prov = provenance_map.get(intent)
                    if prov:
                        final_response += f"\n\n[Source: {prov} — deterministic summary]"

                    if refinement_rejected and not fallback:
                        final_response += "\n\n[Note: A refined summary was suppressed because it appeared to contain unrelated or uncertain content. Displaying the original deterministic output.]"
                    if fallback:
                        final_response += "\n\n[Note: LLM refinement unavailable. Displaying root deterministic evaluation.]"
                else:
                    final_response = refined if refined else raw_explanation

                st.markdown(final_response)

                if fallback or refinement_rejected:
                    st.markdown(
                        '<p class="fallback-note">System indicator: LLM refinement suppressed. Displaying root deterministic evaluation.</p>',
                        unsafe_allow_html=True,
                    )

                full_display = f"{badge_html}\n\n{final_response}"
                if fallback or refinement_rejected:
                    full_display += '\n\n<p class="fallback-note">System indicator: LLM refinement suppressed. Displaying root deterministic evaluation.</p>'

                st.session_state.messages.append({"role": "assistant", "content": full_display})
