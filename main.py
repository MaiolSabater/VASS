"""Entry point — wake word OR keypress opens a conversation session: one
"yeah?" cue, then repeated turns (no re-trigger needed) until a close phrase
or a silence timeout ends the session and control returns to wake-word
listening."""
import sys

# Force UTF-8 stdout: on some Windows consoles (cp1252 codepage), printing the
# emoji used for status messages (🔔, 🎤, ⚠️) raises UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import queue
import threading

import wake
from handler import handle
from tts import speak
from stt import record_with_vad, transcribe
import config

_EXIT_COMMANDS = {"quit", "exit", "q", "bye"}
_EOF = "__eof__"

_MIN_AUDIO_FRAMES = 16_000 * 0.3   # 300 ms at 16 kHz — anything shorter is noise


def _keypress_thread(trigger_queue: "queue.Queue[str]") -> None:
    """Background thread: blocks on input(), forwards each line to the main loop.

    Runs concurrently with the wake-word listening loop on the main thread —
    input() blocks this thread only, so it never stalls mic capture. Lines are
    handed off through a thread-safe Queue, which the main loop drains with a
    non-blocking get_nowait() between audio chunks.
    """
    while True:
        try:
            line = input()
        except EOFError:
            trigger_queue.put(_EOF)
            return
        trigger_queue.put(line.strip().lower())


def _wait_for_trigger(detector: wake.WakeWordDetector, trigger_queue: "queue.Queue[str]") -> str | None:
    """Block until the wake word fires or Enter is pressed.

    Returns "wake", "key", or None (caller should quit).
    """
    with wake.open_stream() as stream:
        wake.flush_stream(stream)   # drop any stale frames left from before this stream opened
        while True:
            try:
                line = trigger_queue.get_nowait()
                if line == _EOF or line in _EXIT_COMMANDS:
                    return None
                return "key"
            except queue.Empty:
                pass

            chunk, overflowed = stream.read(wake.CHUNK_SAMPLES)
            if overflowed:
                print("[wake] ⚠️  input overflow — a chunk was dropped")

            detected, score = detector.process_chunk(chunk.flatten())
            if detected:
                print(f"🔔 Wake word detected (score: {score:.2f})")
                return "wake"


def _drain(trigger_queue: "queue.Queue[str]") -> None:
    """Discard any stray keypresses queued up while a conversation was running."""
    while True:
        try:
            trigger_queue.get_nowait()
        except queue.Empty:
            return


def _is_close_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in config.CONVERSATION_END_PHRASES)


def _run_turn(text: str, history: list) -> None:
    """Send transcribed text to the LLM and speak the reply."""
    print(f"You: {text}\n")

    try:
        reply = handle(text, history)
    except RuntimeError as e:
        print(f"[LLM error] {e}\n")
        return

    print(f"Vass: {reply}\n")
    try:
        speak(reply)
    except RuntimeError as e:
        print(f"[TTS error] {e}\n")


def _converse() -> None:
    """Run one whole conversation session.

    History is fresh per session (only the system prompt persists across
    sessions) — each new wake word/Enter starts a clean conversation. The
    first turn waits for speech the same way a single-shot trigger always
    has; every turn after that applies CONVERSATION_TIMEOUT, so silence ends
    the session instead of waiting forever. speak() blocks until playback
    finishes, so a follow-up recording never starts until Vass has stopped
    talking — it won't hear/transcribe itself.
    """
    history = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    try:
        speak(config.WAKE_ACK)
    except RuntimeError as e:
        print(f"[TTS error] {e}\n")

    first_turn = True
    while True:
        timeout = None if first_turn else config.CONVERSATION_TIMEOUT
        try:
            audio = record_with_vad(timeout=timeout)
        except Exception as e:
            print(f"[Mic error] {e}\n")
            return

        if audio.size == 0:
            if first_turn:
                print("[Conversation] Didn't hear anything — going back to sleep.\n")
            else:
                print("[Conversation] No follow-up heard — ending conversation.\n")
            return

        if audio.size < _MIN_AUDIO_FRAMES:
            print("[Too short — try again]\n")
            first_turn = False
            continue

        print("Transcribing…")
        try:
            text = transcribe(audio)
        except Exception as e:
            print(f"[STT error] {e}\n")
            first_turn = False
            continue

        if not text:
            print("[Nothing heard — try again]\n")
            first_turn = False
            continue

        if _is_close_phrase(text):
            print(f"You: {text}")
            print("[Conversation] Close phrase detected — ending conversation.\n")
            try:
                speak(config.CONVERSATION_END_ACK)
            except RuntimeError as e:
                print(f"[TTS error] {e}\n")
            return

        _run_turn(text, history)
        first_turn = False


def main() -> None:
    print(
        "Vass is listening for \"what's up Vass\" — or press Enter to talk. "
        "Type 'quit' to exit.\n"
    )

    trigger_queue: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_keypress_thread, args=(trigger_queue,), daemon=True).start()

    detector = wake.WakeWordDetector()

    while True:
        # ── wait for trigger ──────────────────────────────────────────────────
        try:
            reason = _wait_for_trigger(detector, trigger_queue)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if reason is None:
            print("Goodbye.")
            break

        if reason == "key":
            print("⌨️  Manual trigger (Enter)")

        # ── conversation session (ack cue, turns, follow-ups) ─────────────────
        _converse()
        _drain(trigger_queue)
        # reset the wake-word detector before listening resumes: clears
        # openWakeWord's internal buffers + cooldown, and arms a short grace
        # period, so stale audio/context from the just-finished conversation
        # can't cause an immediate false re-trigger
        detector.reset()


if __name__ == "__main__":
    main()
