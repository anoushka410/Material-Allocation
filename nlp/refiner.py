"""Response refinement and validation module.

Responsibilities:
  - ``refine_explanation``: Send deterministic explanation to LLM and improve
    readability without changing numbers or adding facts.
  - ``validate_response``: Ask the LLM whether the refined answer actually
    answers the user's original question.  Returns a fallback message if not.
"""

from nlp.llm_client import call_llm

# Shown to the user when the pipeline cannot produce a valid answer.
FALLBACK_RESPONSE = (
    "I cannot answer this question using the available optimization data."
)


def refine_explanation(raw_explanation: str, user_question: str = "") -> str:
    """Refine a deterministic raw explanation using the LLM.

    Safety guarantees enforced via the prompt:
    - Do NOT change any numbers.
    - Do NOT add new facts not present in the input.
    - Only improve clarity and readability.

    If ``raw_explanation`` is empty the function returns it immediately
    to avoid an LLM call that could hallucinate content.

    Parameters
    ----------
    raw_explanation:
        Deterministic text produced by the explanation engine.
    user_question:
        Original user query — used to match the tone of the response.

    Returns
    -------
    str
        Refined text, or the original text if the LLM call fails or produces
        an empty / low-quality result.
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
        "You will be given supply chain optimization data labeled 'DATA TO EXPLAIN'.\n"
        "Your ONLY task is to produce a concise, factual summary paragraph.\n\n"
        "STRICT RULES — you MUST follow ALL of these:\n"
        "1. Do NOT change ANY numbers, quantities, costs, or percentages.\n"
        "2. Do NOT add new facts, assumptions, or information not present in the input.\n"
        "3. Do NOT perform calculations or derive new numeric values.\n"
        "4. Only improve clarity, grammar, and readability.\n"
        "5. If the data is insufficient, return exactly: 'Insufficient data to answer.'\n\n"
        f"{tone_instruction}"
        f"DATA TO EXPLAIN:\n{raw_explanation}\n\n"
        "IMPROVED SUMMARY:\n"
    )
    messages = [{"role": "user", "content": prompt}]
    refined = call_llm(messages)

    # Fall back to raw explanation if refinement produces unusable output
    if not refined or len(refined.strip()) < 10:
        return raw_explanation
    return refined


def validate_response(refined_response: str, user_question: str) -> str:
    """Validate that the refined response answers the user's question.

    Sends a short yes/no check to the LLM.  If the LLM indicates that the
    response does NOT answer the question, the canonical fallback message is
    returned instead.

    Parameters
    ----------
    refined_response:
        The response text produced after refinement.
    user_question:
        The original user query.

    Returns
    -------
    str
        Either ``refined_response`` (if valid) or ``FALLBACK_RESPONSE``.
    """
    if not refined_response or not refined_response.strip():
        return FALLBACK_RESPONSE

    prompt = (
        "You are a response validator for a supply chain analytics assistant.\n\n"
        f"USER QUESTION: {user_question}\n\n"
        f"ASSISTANT RESPONSE:\n{refined_response}\n\n"
        "Does this response directly answer the user's question using the information provided?\n"
        "Reply with exactly one word: YES or NO."
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        verdict = call_llm(messages).strip().upper()
    except Exception:
        # If validation call fails, trust the response rather than blocking it
        return refined_response

    if verdict.startswith("NO"):
        return FALLBACK_RESPONSE
    return refined_response
