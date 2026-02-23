import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "tinyllama"
TIMEOUT = 30


def call_llm(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
