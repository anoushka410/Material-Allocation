import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "tinyllama"
TIMEOUT = 30


def call_llm(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
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
