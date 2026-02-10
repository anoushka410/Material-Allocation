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
    
    /* Metrics */
    div[data-testid="metric-container"] {
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

# Load data with robust path handling
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

transfer, manufacturing, scenario = load_data()

data_store = {
    "transfer": transfer,
    "manufacturing": manufacturing,
    "scenario": scenario
}

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Content
with st.sidebar:
    st.title("📦 Logistics Command Center")
    st.markdown("---")
    
    st.subheader("Current Scenario")
    st.caption(scenario["scenario"])
    
    st.markdown("### 📊 Key Metrics")
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

# Recommendation Section
with st.container():
    st.subheader("🚀 Active Recommendation")
    rec_col1, rec_col2 = st.columns([1, 2])
    
    with rec_col1:
        st.info(f"**Action:** Transfer Stock")
    
    with rec_col2:
        st.success(
            f"Move **{transfer['quantity']} units** of **{transfer['product_id']}** "
            f"from **{transfer['from_store']}** to **{transfer['to_store']}**"
        )

st.markdown("---")

# Chat Interface
st.subheader("💬 AI Analyst")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about this recommendation..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = handle_query(prompt, data_store)
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
