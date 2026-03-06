"""LLM client for the NLP pipeline.

Wraps OpenAI API calls used for:
  - Intent classification
  - Language refinement
  - Answer validation

The OpenAI API key is read from the ``OPENAI_API_KEY`` environment variable.
If the key is not set the client raises a clear ``RuntimeError`` rather than
silently producing an empty result.
"""

import os
from openai import OpenAI

# Model to use for all LLM calls.  Keeping temperature at 0 ensures
# deterministic classification results and minimises hallucination risk.
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0


def _get_client() -> OpenAI:
    """Return an authenticated OpenAI client.

    Raises
    ------
    RuntimeError
        If ``OPENAI_API_KEY`` is not set in the environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please export it before running the NLP pipeline."
        )
    return OpenAI(api_key=api_key)


def call_llm(messages: list[dict]) -> str:
    """Call the OpenAI chat completion API.

    Parameters
    ----------
    messages:
        List of message dicts in OpenAI format, e.g.
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``.

    Returns
    -------
    str
        The assistant reply, stripped of leading/trailing whitespace.
        Returns an empty string if the response is malformed or empty.

    Raises
    ------
    RuntimeError
        If ``OPENAI_API_KEY`` is not set.
    openai.OpenAIError
        On API communication failures (let callers decide how to handle).
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=TEMPERATURE,
    )
    # Defensive check — return empty string rather than raising if content missing
    if not response.choices:
        return ""
    content = response.choices[0].message.content
    return content.strip() if content else ""
