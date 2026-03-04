"""chatbot.py — Simplified OpenAI-powered supply-chain chatbot.

Design goals
------------
* Loads the **entire** content of ``optimization/output-json/`` and
  ``demand-forecast/output/`` as context so every answer is grounded
  in real output data.
* Uses a strict anti-hallucination system prompt: the model is forbidden
  from inventing facts that are not present in the supplied data.
* Stateless helper function ``answer()`` for easy integration; a stateful
  ``SupplyChainChatbot`` class for multi-turn conversation support.
"""
from __future__ import annotations

from nlp.context_loader import load_context
from nlp.llm_client import call_llm

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = """You are a supply chain analytics assistant.
Your ONLY knowledge source is the structured data provided below under DATA.
You MUST NOT invent, guess, or extrapolate any facts beyond what is explicitly
stated in that data.  If the data does not contain enough information to answer
a question, say exactly: "I don't have enough data to answer that question."

Rules:
- Answer concisely and factually.
- Quote specific numbers from the data when relevant.
- Do not mention that you are an AI or reference OpenAI.
- Do not use uncertain language such as "I think", "maybe", or "possibly".
- If asked about something unrelated to supply-chain optimization or demand
  forecasting, respond: "That is outside my scope."

DATA:
{context}
"""


def _build_system_prompt(context: str) -> str:
    """Inject *context* into the system prompt template."""
    if not context:
        context = "(No output data available yet. Run the optimization and demand-forecast pipelines first.)"
    return _SYSTEM_PROMPT_TEMPLATE.format(context=context)


# ---------------------------------------------------------------------------
# Stateless helper
# ---------------------------------------------------------------------------
def answer(
    question: str,
    additional_context: str = "",
    conversation_history: list[dict] | None = None,
    base_path: str = ".",
) -> str:
    """Answer *question* using all available output files as context.

    Parameters
    ----------
    question:
        The user's natural-language question.
    additional_context:
        Extra structured text (e.g. a deterministic explanation already
        built by ``explanation_engine``) prepended to the file-based context.
    conversation_history:
        Optional list of previous ``{"role": ..., "content": ...}`` turns
        for multi-turn support.
    base_path:
        Project root directory (default: current working directory).

    Returns
    -------
    str
        Factual answer grounded in the loaded output data, or a safe
        fallback message if the LLM call fails.
    """
    file_context = load_context(base_path)
    full_context = (
        f"{additional_context}\n\n{file_context}" if additional_context else file_context
    )
    system_msg = {"role": "system", "content": _build_system_prompt(full_context)}

    messages: list[dict] = [system_msg]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    result = call_llm(messages)
    return result if result else "I don't have enough data to answer that question."


# ---------------------------------------------------------------------------
# Stateful chatbot class (multi-turn)
# ---------------------------------------------------------------------------
class SupplyChainChatbot:
    """Stateful wrapper around :func:`answer` that keeps conversation history.

    Usage::

        bot = SupplyChainChatbot()
        print(bot.chat("What is the total transfer cost?"))
        print(bot.chat("Which product has the highest manufacturing cost?"))
        bot.reset()  # clear conversation history
    """

    def __init__(self, base_path: str = ".") -> None:
        self.base_path = base_path
        self._history: list[dict] = []

    def chat(self, question: str, additional_context: str = "") -> str:
        """Send *question* and return the assistant's reply.

        The full conversation history is automatically included so the model
        can handle follow-up questions.
        """
        reply = answer(
            question,
            additional_context=additional_context,
            conversation_history=self._history,
            base_path=self.base_path,
        )
        # Append the exchange to history for subsequent turns
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear the conversation history."""
        self._history = []
