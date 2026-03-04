import os

# ── OpenAI Configuration ─────────────────────────────────────────────────────
# Place your OpenAI API key below OR set the OPENAI_API_KEY environment variable.
# The environment variable takes precedence when set.
# Obtain a key at https://platform.openai.com/api-keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Model to use for chat completions.
# "gpt-4o-mini" is cost-effective and accurate; change to "gpt-4o" for higher quality.
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Maximum tokens allowed in a single completion response.
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))

# Temperature: 0.0 = deterministic, avoids hallucination.
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

