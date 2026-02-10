import streamlit as st
import json
import os
from chatbot import handle_query

# Page Configuration
st.set_page_config(
    page_title="Supply Chain AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metrics and Cards */
    div[data-testid="metric-container"], .stCard {
        background-color: #21262d;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff; 
        font-weight: 600;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Custom Button */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #2ea043;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_inputs_dir = os.path.join(base_dir, "sample_inputs")
    
    with open(os.path.join(sample_inputs_dir, "transfer.json")) as f:
        transfer = json.load(f)
    with open(os.path.join(sample_inputs_dir, "manufacturing.json")) as f:
        manufacturing = json.load(f)
    with open(os.path.join(sample_inputs_dir, "scenario.json")) as f:
        scenario = json.load(f)
    return transfer, manufacturing, scenario

transfer_data, manufacturing_data, scenario = load_data()
transfers = transfer_data.get("transfers", [])
manufacturing_actions = manufacturing_data.get("manufacturing_actions", [])

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Content
with st.sidebar:
    st.title("Logistics Command Center")
    st.markdown("---")
    
    st.subheader("Current Scenario")
    st.caption(scenario["scenario"])
    
    st.markdown("### Key Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Baseline Stockouts", scenario["baseline"]["total_stockouts"], delta_color="inverse")
    with col2:
        st.metric("Optimized Stockouts", scenario["optimized"]["total_stockouts"], 
                 delta=scenario["baseline"]["total_stockouts"] - scenario["optimized"]["total_stockouts"],
                 delta_color="normal")
    
    st.metric("Cost Impact", f"${scenario['delta']['cost_change']:,.2f}", 
             delta="Savings" if scenario['delta']['cost_change'] < 0 else "Cost Increase",
             delta_color="inverse")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Content
st.title("Supply Chain Optimization Assistant")
st.markdown("Welcome to your intelligent decision support system.")

# Global Insights / Summary Dashboard
with st.container():
    st.subheader("Global Insights")
    g_col1, g_col2, g_col3 = st.columns(3)
    
    with g_col1:
        st.metric("Total Transfers", len(transfers))
    with g_col2:
        st.metric("Manufacturing Actions", len(manufacturing_actions))
    with g_col3:
        risk_level = "High" if "High" in scenario.get("scenario", "") else "Moderate" # Simple inference
        st.metric("Risk Level", risk_level, delta="Analysis Complete")

st.markdown("---")

# Selection Logic
st.subheader("Recommendations Dashboard")

# Create a list of options for the selectbox
transfer_options = [f"Transfer: {t['quantity']} units of {t['product_id']} ({t['from_store']} -> {t['to_store']})" for t in transfers]
manufacturing_options = [f"Manufacture: {m['manufacture_quantity']} units of {m['product_id']}" for m in manufacturing_actions]
all_options = ["None (General Analysis)"] + transfer_options + manufacturing_options

selected_option = st.selectbox("Select a recommendation to analyze or ask about:", all_options)

# Find the selected item data
selected_item = None
selected_type = None

if selected_option and selected_option != "None (General Analysis)":
    if selected_option.startswith("Transfer"):
        index = transfer_options.index(selected_option)
        selected_item = transfers[index]
        selected_type = "transfer"
    elif selected_option.startswith("Manufacture"):
        index = manufacturing_options.index(selected_option)
        selected_item = manufacturing_actions[index]
        selected_type = "manufacturing"

# Prepare data for chatbot
# We pass global data AND the selected item (if any)
current_context_data = {
    "scenario": scenario,
    "all_transfers": transfers,
    "all_manufacturing": manufacturing_actions,
    "transfer": selected_item if selected_type == "transfer" else None,
    "manufacturing": selected_item if selected_type == "manufacturing" else None
}

# Display Active Recommendation Card
with st.container():
    if selected_item:
        st.markdown("#### Selected Action Details")
        rec_col1, rec_col2 = st.columns([1, 2])
        
        with rec_col1:
            st.info(f"**Type:** {selected_type.capitalize()}")
        
        with rec_col2:
            if selected_type == "transfer":
                st.success(
                    f"Move **{selected_item['quantity']} units** of **{selected_item['product_id']}** "
                    f"from **{selected_item['from_store']}** to **{selected_item['to_store']}**"
                )
            else:
                st.success(
                    f"Manufacture **{selected_item['manufacture_quantity']} units** of **{selected_item['product_id']}**"
                )
    else:
        st.info("Select a recommendation above to view details, or ask general questions below.")

st.markdown("---")

# Chat Interface
st.subheader("AI Analyst")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
input_placeholder = f"Ask about this {selected_type}..." if selected_item else "Ask about global insights (e.g. 'How many transfers?', 'Summary')"
if prompt := st.chat_input(input_placeholder):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Pass the context-specific data to the chatbot
        full_response = handle_query(prompt, current_context_data)
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
