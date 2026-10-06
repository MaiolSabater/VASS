"""
Central dispatch layer — the handle() seam.

All input from the REPL (and, later, from the speech recogniser) passes through
here.  Local commands (currently: the LED strip) are matched by keyword and
handled directly; everything else goes to the LLM.
"""

import re

import lights
from llm import query_llm

# Light commands must mention "lights"/"LEDs" (plural), so phrases like
# "shed some light on this" or "light blue" in normal chat don't trigger them.
_LIGHTS_WORD = re.compile(r"\b(lights|leds)\b")
# Longest names first, so "light blue" wins over "blue".
_COLOR_NAMES = sorted(lights.COLORS, key=len, reverse=True)


def _handle_lights(text: str) -> str | None:
    """Return Vass's reply if text is a lights command, else None."""
    words = re.sub(r"[^a-z ]", " ", text.lower())
    words = " ".join(words.split())
    if not _LIGHTS_WORD.search(words):
        return None

    if re.search(r"\bdisconnect\b", words):
        return lights.disconnect()
    if re.search(r"\bconnect\b", words):
        return lights.connect()
    if re.search(r"\boff\b", words):
        return lights.power_off()
    for name in _COLOR_NAMES:
        if re.search(rf"\b{name}\b", words):
            return lights.set_color(name)
    if re.search(r"\bon\b", words):
        return lights.power_on()
    return None


def handle(text: str, history: list) -> str:
    reply = _handle_lights(text)
    if reply is not None:
        return reply
    return query_llm(text, history)
