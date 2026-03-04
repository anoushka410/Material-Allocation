from nlp.chatbot import answer as _chatbot_answer


def refine_explanation(raw_explanation: str, user_question: str = "") -> str:
    """Refine a deterministic raw explanation using OpenAI.

    Safety measures:
    - If raw_explanation is empty or only whitespace, return it immediately
      (do not call the LLM, which could hallucinate on empty input).
    - The chatbot's system prompt forbids inventing facts, so the model
      only rephrases what is already present in raw_explanation combined
      with the full output-file context.
    """
    if not raw_explanation or not raw_explanation.strip():
        return raw_explanation

    question = user_question if user_question else "Summarize the following supply chain data."
    return _chatbot_answer(question, additional_context=raw_explanation)
