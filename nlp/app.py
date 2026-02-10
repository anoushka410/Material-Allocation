import streamlit as st
import json
from chatbot import handle_query

# Load data
with open("sample_inputs/transfer.json") as f:
    transfer = json.load(f)

with open("sample_inputs/manufacturing.json") as f:
    manufacturing = json.load(f)

with open("sample_inputs/scenario.json") as f:
    scenario = json.load(f)

data_store = {
    "transfer": transfer,
    "manufacturing": manufacturing,
    "scenario": scenario
}

st.set_page_config(page_title="Supply Chain Decision Assistant", layout="wide")

st.title("📦 Supply Chain Optimization Assistant")

# Sidebar
st.sidebar.header("Scenario")
st.sidebar.write(scenario["scenario"])

st.sidebar.header("Key Metrics")
st.sidebar.metric("Baseline Stockouts", scenario["baseline"]["total_stockouts"])
st.sidebar.metric("Optimized Stockouts", scenario["optimized"]["total_stockouts"])
st.sidebar.metric(
    "Cost Change",
    f"${scenario['delta']['cost_change']}"
)

# Main area
st.subheader("Recommended Transfer")
st.write(
    f"Move **{transfer['quantity']} units** of **{transfer['product_id']}** "
    f"from **{transfer['from_store']}** to **{transfer['to_store']}**"
)

st.subheader("Ask the Assistant")
user_query = st.text_input(
    "Ask a question (e.g. Why is this recommended? What is the impact?)"
)

if user_query:
    response = handle_query(user_query, data_store)
    st.markdown("### Explanation")
    st.write(response)
