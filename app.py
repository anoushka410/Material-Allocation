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
    page_title="Supply Chain NLP Assistant",
    page_icon="🔗",
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
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    .intent-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-bottom: 0.5rem;
    }

    .fallback-note {
        font-size: 0.8rem;
        color: #f59e0b;
        margin-top: 0.4rem;
        font-style: italic;
    }

    .stChatMessage {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }

    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>🔗 Supply Chain NLP Assistant</h1>
        <p>Ask about transfers, manufacturing actions, or scenario summaries</p>
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
    with st.spinner("Checking Ollama… starting if needed."):
        ok, msg = _ensure_ollama()
    st.session_state.ollama_ok = ok
    st.session_state.ollama_msg = msg

if st.session_state.ollama_ok:
    st.success(f"🤖 {st.session_state.ollama_msg}", icon="✅")
else:
    st.warning(
        f"⚡ **{st.session_state.ollama_msg}** "
        "Explanations will be shown without LLM refinement.",
        icon="🤖",
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask about transfers, manufacturing, or scenario metrics…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
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
                "👋 Hey there! I'm your **Supply Chain NLP Assistant**.\n\n"
                "I can help you understand the optimization decisions made for the current scenario. "
                "Here are a few things you can ask me:\n\n"
                "- 🔄 *\"Explain the transfer recommendations\"*\n"
                "- 🏭 *\"Why were these manufacturing decisions made?\"*\n"
                "- 📊 *\"Give me a scenario summary\"*\n"
                "- 💰 *\"What was the cost impact?\"*\n\n"
                "What would you like to know?"
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        elif intent == "out_of_scope":
            response = (
                "🤔 Hmm, that one's a bit outside my area! I'm specialised in supply chain "
                "optimization — things like inventory transfers, manufacturing decisions, and scenario analysis.\n\n"
                "Try asking something like:\n"
                "- *\"Why was inventory transferred between stores?\"*\n"
                "- *\"What manufacturing actions were taken?\"*\n"
                "- *\"How did the optimized scenario compare to baseline?\"*"
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            has_specifics = any(params.get(k) for k in ["transfer_id", "manufacturing_id", "product_id", "store_id"])

            if intent in ("explain_transfer", "explain_manufacturing") and not has_specifics and not params["is_all"]:
                item_name = "transfer recommendations" if intent == "explain_transfer" else "manufacturing decisions"
                example_id = "T001" if intent == "explain_transfer" else "M001"
                response = (
                    f"I can explain the {item_name}. Do you want an overview of **all** of them, "
                    f"or do you want to know about a specific ID (e.g., `{example_id}`), product (e.g., `product_892`), or store?"
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                badge_html = f'<span class="intent-badge">🏷 {label}</span>'
                if has_specifics:
                    badge_html += ' <span class="intent-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border-color: rgba(16, 185, 129, 0.4);">🎯 Specific Filter</span>'
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

                if is_empty_state:
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
                        '<p class="fallback-note">⚡ Ollama unavailable — showing deterministic explanation.</p>',
                        unsafe_allow_html=True,
                    )

                full_display = f"{badge_html}\n\n{final_response}"
                if fallback:
                    full_display += '\n\n<p class="fallback-note">⚡ Ollama unavailable — showing deterministic explanation.</p>'

                st.session_state.messages.append({"role": "assistant", "content": full_display})
