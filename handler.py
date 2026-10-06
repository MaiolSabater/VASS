"""
Central dispatch layer — the handle() seam.

All input from the REPL (and, later, from the speech recogniser) passes through
here.  Right now every message goes straight to the LLM, but this is the future
home of the intent router: smart-home commands, timers, local queries, etc.
Add branches here without touching the REPL or TTS layers.
"""

from llm import query_llm

def handle(text: str, history: list) -> str:
    # ── future router goes here: command vs. question ──
    return query_llm(text, history)
