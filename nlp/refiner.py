from nlp.llm_client import call_llm


def refine_explanation(raw_explanation: str, user_question: str = "") -> str:
    tone_instruction = (
        f'The user asked: "{user_question}". Match the tone of their question — '
        "casual questions should get a conversational answer, formal questions a professional one. "
        if user_question
        else ""
    )
    prompt = (
        f"You are a helpful supply chain assistant. Explain the following data in a readable, natural way. "
        f"{tone_instruction}"
        "Write conversational sentences explaining the actions, the reasoning, and the financial impact. "
        "Do not change any numbers, costs, or percentages, and do not hallucinate new facts.\n\n"
        f"DATA TO EXPLAIN:\n{raw_explanation}"
    )
    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages)

