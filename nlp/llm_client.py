from openai import OpenAI, APIError
from nlp.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return a cached OpenAI client, initialised lazily."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def call_llm(messages: list[dict]) -> str:
    """Send *messages* to the OpenAI chat API and return the reply text.

    Returns an empty string on any error so callers can fall back gracefully.
    Temperature is forced to 0 to keep answers deterministic and factual.
    """
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except APIError:
        return ""
    except Exception:
        return ""
