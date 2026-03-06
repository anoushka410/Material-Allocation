from nlp.llm_client import call_llm

# Phrases that indicate the LLM introduced hallucinated or uncertain content
_HALLUCINATION_INDICATORS = [
    "i think", "maybe", "could be", "possibly", "as an ai", "as a model",
    "i believe", "i'm not sure", "not certain", "cannot confirm",
]

# Phrases that indicate the LLM drifted to unrelated topics
_UNRELATED_INDICATORS = [
    "transportation system", "buses", "passengers", "cargo ship", "trams",
    "railway", "airline", "taxi", "subway",
]


def refine_explanation(raw_explanation: str, user_question: str = "") -> str:
    """Refine a deterministic raw explanation using the LLM.

    Safety measures:
    - If raw_explanation is empty or only whitespace, return it immediately (do not call LLM).
    - Instruct the LLM explicitly not to invent facts and only to rephrase the provided data.
    - Numbers in the raw explanation must not be changed by the LLM.
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
        "NO assumptions, and NO invented data not present in the input. "
        "CRITICAL: Do NOT change any numbers, quantities, or costs — reproduce them exactly as given. "
        "If the given data is insufficient, "
        "do not attempt to fill gaps — instead return the original data or a short statement: 'Insufficient data to answer.'\n\n"
        f"{tone_instruction}"
        f"DATA TO EXPLAIN:\n{raw_explanation}\n\n"
        "SUMMARY PARAGRAPH:\n"
    )
    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages)


def validate_response(raw_explanation: str, refined: str) -> bool:
    """Check whether the refined LLM response is acceptable to return to the user.

    Validation rules:
    - refined must not be empty or too short (< 10 characters).
    - refined must not contain hallucination indicators (uncertainty phrases).
    - refined must not have drifted to unrelated topics.
    - refined must not be excessively long (> 5 000 characters).

    Parameters
    ----------
    raw_explanation : str
        The original deterministic explanation.
    refined : str
        The LLM-refined response.

    Returns
    -------
    bool
        True  → the refined response is safe to display.
        False → fall back to raw_explanation.
    """
    if not refined or len(refined.strip()) < 10:
        return False

    if len(refined) > 5000:
        return False

    lower = refined.lower()

    if any(phrase in lower for phrase in _HALLUCINATION_INDICATORS):
        return False

    if any(phrase in lower for phrase in _UNRELATED_INDICATORS):
        return False

    return True
