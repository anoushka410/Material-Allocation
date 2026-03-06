import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# ── Model selection ───────────────────────────────────────────────────────────
# Switch between local models by changing MODEL_NAME below.
# Supported options: "tinyllama", "mistral"
MODEL_NAME = "mistral"

TIMEOUT = 30


def call_llm(messages: list[dict]) -> str:
    """Send a chat message list to the local Ollama LLM and return the response text.

    Uses MODEL_NAME defined above. Temperature is fixed at 0.0 to minimize
    hallucination risk and ensure deterministic outputs.

    Returns an empty string if the LLM is unavailable or returns a malformed
    response — callers are responsible for handling the empty-string fallback.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        # Force deterministic output to reduce hallucination risk
        "options": {"temperature": 0.0},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    # Defensive checks — return an explicit placeholder if the response is malformed or empty
    if not data or "message" not in data or "content" not in data["message"]:
        return ""
    return data["message"]["content"].strip()


def get_llm_status() -> dict:
    """Check whether the local Ollama service is reachable.

    Returns a status dict with keys:
      - online (bool): True if Ollama is responding.
      - model  (str):  The configured model name.
      - url    (str):  The Ollama API base URL.
    """
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if r.status_code == 200:
            return {"online": True, "model": MODEL_NAME, "url": OLLAMA_URL}
    except Exception:
        pass
    return {"online": False, "model": MODEL_NAME, "url": OLLAMA_URL}
