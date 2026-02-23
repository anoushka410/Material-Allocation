from nlp.llm_client import call_llm


def refine_explanation(raw_explanation: str, user_question: str = "") -> str:
    tone_instruction = (
        f'The user asked: "{user_question}". Match the tone of their question — '
        "casual questions should get a conversational answer, formal questions a professional one. "
        if user_question
        else ""
    )
    prompt = (
        "Focus only on summarizing the following data into a single, clear, conversational paragraph. "
        "Do not write any dialogue, greetings, or conversational filler. "
        "State the facts directly based only on the provided data.\n\n"
        f"DATA TO EXPLAIN:\n{raw_explanation}\n\n"
        "SUMMARY PARAGRAPH:\n"
    )
    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages)

