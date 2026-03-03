from nlp.llm_client import call_llm


def refine_explanation(raw_explanation: str, user_question: str = "") -> str:
    """Refine a deterministic raw explanation using the LLM.

    Safety measures:
    - If raw_explanation is empty or only whitespace, return it immediately (do not call LLM).
    - Instruct the LLM explicitly not to invent facts and only to rephrase the provided data.
    """
    if not raw_explanation or not raw_explanation.strip():
        # Nothing to refine — avoid calling the LLM which could hallucinate
        return raw_explanation

    tone_instruction = (
        f'The user asked: "{user_question}". Match the tone of their question — '
        "casual questions should get a conversational answer, formal questions a professional one. "
        if user_question
        else ""
    )
    prompt = (
        "You will be given a piece of factual text labeled 'DATA TO EXPLAIN'. "
        "Your ONLY task is to produce a concise, factual summary paragraph that contains NO new facts, "
        "NO assumptions, and NO invented data not present in the input. If the given data is insufficient, "
        "do not attempt to fill gaps — instead return the original data or a short statement: 'Insufficient data to answer.'\n\n"
        f"DATA TO EXPLAIN:\n{raw_explanation}\n\n"
        "SUMMARY PARAGRAPH:\n"
    )
    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages)
