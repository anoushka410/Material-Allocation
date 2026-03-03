import json
import subprocess
import time
import requests
import streamlit as st
import csv
import ast

from nlp.intent_classifier import classify_intent, extract_parameters
from nlp.explanation_engine import build_explanation
from nlp.refiner import refine_explanation


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
