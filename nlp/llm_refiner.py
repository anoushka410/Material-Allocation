def refine_with_llm(draft_text, user_type="planner"):
    prompt = f"""You are a supply chain decision assistant.
Rewrite the following explanation for a {user_type}.
Do not add or remove facts.
Do not change numbers.
Improve clarity and tone.
Explanation:
{draft_text}
"""
    return draft_text
