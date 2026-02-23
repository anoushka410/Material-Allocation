import json
import subprocess
import time
import requests
import streamlit as st

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

SAMPLE_DATA_DIR = "nlp/sample_inputs"

INTENT_LABELS = {
    "explain_transfer": "Transfer Explanation",
    "explain_manufacturing": "Manufacturing Explanation",
    "scenario_summary": "Scenario Summary",
    "impact_analysis": "Impact Analysis",
    "list_entities": "Entity List",
    "total_counts": "Summary Metrics",
    "out_of_scope": "Out of Scope",
}


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


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
            has_specifics = any(params.get(k) for k in ["transfer_id", "manufacturing_id", "product_id", "store_id"])
            if params["is_all"] or has_specifics:
                intent = st.session_state.last_intent

        if intent not in ("out_of_scope", "greeting"):
            st.session_state.last_intent = intent

        label = INTENT_LABELS.get(intent, intent)

        if intent == "greeting":
            response = (
                "Hello. I am the Supply Chain Analytics Assistant.\n\n"
                "I am equipped to analyze optimization recommendations based on the current scenario data. "
                "You can query me on the following topics:\n\n"
                "- *\"Explain the transfer recommendations\"*\n"
                "- *\"Detail the manufacturing decisions\"*\n"
                "- *\"Provide a high-level scenario summary\"*\n"
                "- *\"Review the cost impact\"*\n\n"
                "Please enter your query below."
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
            has_specifics = any(params.get(k) for k in ["transfer_id", "manufacturing_id", "product_id", "store_id"])

            badge_html = f'<span class="intent-badge">{label}</span>'
            if has_specifics:
                badge_html += ' <span class="filter-badge">Specific Filter Applied</span>'
            st.markdown(badge_html, unsafe_allow_html=True)
            data = {
                "scenario": load_json(f"{SAMPLE_DATA_DIR}/scenario.json"),
                "transfers": load_json(f"{SAMPLE_DATA_DIR}/transfer.json"),
                "manufacturing": load_json(f"{SAMPLE_DATA_DIR}/manufacturing.json"),
            }

            with st.spinner("Building explanation…"):
                raw_explanation = build_explanation(intent, data, params)

            refined = None
            fallback = False
            
            # Check for empty state responses directly from explanation_engine
            is_empty_state = raw_explanation.startswith("No transfers match") or raw_explanation.startswith("No manufacturing actions match")
            
            # Bypass LLM refinement for tabular/list data (entities and counts) to prevent hallucination
            skip_refiner = is_empty_state or intent in ("list_entities", "total_counts")

            if skip_refiner:
                refined = raw_explanation
            else:
                try:
                    with st.spinner("Refining with TinyLlama…"):
                        refined = refine_explanation(raw_explanation, user_question=prompt)
                except Exception:
                    fallback = True

            final_response = refined if refined else raw_explanation
            st.markdown(final_response)

            if fallback:
                st.markdown(
                    '<p class="fallback-note">System indicator: LLM refinement unavailable. Displaying root deterministic evaluation.</p>',
                    unsafe_allow_html=True,
                )

            full_display = f"{badge_html}\n\n{final_response}"
            if fallback:
                full_display += '\n\n<p class="fallback-note">System indicator: LLM refinement unavailable. Displaying root deterministic evaluation.</p>'

            st.session_state.messages.append({"role": "assistant", "content": full_display})
