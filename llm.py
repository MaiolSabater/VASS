"""Thin wrapper around the Ollama OpenAI-compatible chat API."""

from openai import OpenAI, APIConnectionError, APIStatusError

import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    return _client


def query_llm(text: str, history: list) -> str:
    history.append({"role": "user", "content": text})
    try:
        response = _get_client().chat.completions.create(
            model=config.OLLAMA_MODEL,
            messages=history,
        )
        reply = response.choices[0].message.content or ""
    except APIConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Is the server running?"
        )
    except APIStatusError as e:
        raise RuntimeError(f"Ollama returned an error: {e.status_code} — {e.message}")
    history.append({"role": "assistant", "content": reply})
    return reply
