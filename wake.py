"""Wake-word detection via openWakeWord, using a custom ONNX model.

Public API
----------
WakeWordDetector
    Loads the custom model once. Call .process_chunk(audio) per 80 ms chunk to
    get back (detected, score), with config.WAKE_THRESHOLD/WAKE_COOLDOWN applied.
    Call .reset() when resuming wake-word listening after a pause (e.g. after
    a conversation session) to clear openWakeWord's internal buffers and this
    detector's cooldown/grace-period state, so stale audio/context can't cause
    an immediate false trigger.
open_stream() -> sd.InputStream
    Opens a mic InputStream at the sample rate/chunk size openWakeWord expects.
flush_stream(stream) -> None
    Discards audio frames already sitting in a stream's buffer, so the next
    read only sees genuinely new audio.
listen() -> None
    Standalone loop: continuously print scores/detections until Ctrl+C.

Standalone test mode: `python wake.py` runs listen() directly.
This module does NOT do VAD — detection only. VAD lives in stt.record_with_vad().
"""

import os
import sys
import time

# Force UTF-8 stdout: on some Windows consoles (cp1252 codepage), printing the
# emoji used for status messages (🔔, ⚠️) raises UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import sounddevice as sd

import config

SAMPLE_RATE = 16_000   # Hz — openWakeWord expects 16 kHz mono
CHUNK_SAMPLES = 1280   # 80 ms per chunk @ 16 kHz — openWakeWord's native frame size
WAKE_SCORE_FLOOR = 0.1  # print scores above this so you can watch/tune against config.WAKE_THRESHOLD

# openWakeWord keys predictions by the model filename without its extension
_WAKE_KEY = os.path.splitext(os.path.basename(config.WAKE_MODEL_PATH))[0]

_model = None


def _ensure_feature_models() -> None:
    """Make sure openWakeWord's shared melspectrogram/embedding ONNX models
    are present locally; download them once (cached for future runs) if not."""
    import openwakeword
    from openwakeword.utils import download_models

    models_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    required = ["melspectrogram.onnx", "embedding_model.onnx"]
    if not all(os.path.exists(os.path.join(models_dir, f)) for f in required):
        print("[wake] Downloading openWakeWord shared feature models (one-time)…")
        download_models()


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(config.WAKE_MODEL_PATH):
            raise FileNotFoundError(
                f"Custom wake-word model not found at '{config.WAKE_MODEL_PATH}'. "
                "Update WAKE_MODEL_PATH in config.py."
            )
        _ensure_feature_models()
        from openwakeword.model import Model
        print(f"[wake] Loading custom wake-word model '{_WAKE_KEY}'…")
        _model = Model(
            wakeword_models=[config.WAKE_MODEL_PATH],
            inference_framework="onnx",
        )
        print("[wake] Model loaded.")
    return _model


def open_stream() -> sd.InputStream:
    """Open a mic InputStream at the sample rate/chunk size openWakeWord expects."""
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
    )


def flush_stream(stream: sd.InputStream) -> None:
    """Discard whatever audio is already buffered in `stream`, so the next
    read only returns genuinely new audio. Call right after opening a stream
    and before the detection loop starts, to drop any stale frames left over
    from just before the stream was (re)opened."""
    available = stream.read_available
    if available > 0:
        stream.read(available)


# ── Public API ────────────────────────────────────────────────────────────────

class WakeWordDetector:
    """Loads the custom model once; feed it 80 ms chunks to get detections.

    Usage (e.g. from main.py):
        detector = WakeWordDetector()
        with open_stream() as stream:
            while True:
                chunk, _ = stream.read(CHUNK_SAMPLES)
                detected, score = detector.process_chunk(chunk.flatten())
                if detected:
                    ...
    """

    def __init__(self):
        self.model = _get_model()
        self._last_detection = 0.0
        self._suppress_until = 0.0   # grace-period deadline; 0 = no active suppression

    def reset(self) -> None:
        """Clear openWakeWord's internal buffers and this detector's cooldown
        state. Call before resuming wake-word listening after a pause (e.g.
        once a conversation session ends), so leftover audio context from
        before the pause can't score as an immediate false detection.

        Uses openWakeWord's own Model.reset(), which clears both the
        per-model prediction_buffer (recent score history used for
        patience/debounce) and the shared AudioFeatures preprocessor's
        buffers (raw_data_buffer, melspectrogram_buffer, and the ~4 s
        embedding feature_buffer, which it re-warms with random noise
        instead of leftover real audio) — this is the officially exposed
        way to reset the model's streaming state without re-instantiating it.
        """
        self.model.reset()
        self._last_detection = 0.0
        self._suppress_until = time.time() + config.WAKE_GRACE_PERIOD

    def process_chunk(self, audio: np.ndarray) -> tuple[bool, float]:
        """audio: 1-D int16 array of CHUNK_SAMPLES samples.

        Returns (detected, score). `detected` applies config.WAKE_THRESHOLD,
        config.WAKE_COOLDOWN, and — for a short window right after reset() —
        config.WAKE_GRACE_PERIOD, so one utterance only fires once and a
        just-resumed listener can't trigger instantly on stale audio. `score`
        is always the model's real score, even during the grace period, so
        standalone tuning/debugging still sees genuine numbers.
        """
        score = self.model.predict(audio).get(_WAKE_KEY, 0.0)
        now = time.time()
        detected = (
            now >= self._suppress_until
            and score >= config.WAKE_THRESHOLD
            and (now - self._last_detection) >= config.WAKE_COOLDOWN
        )
        if detected:
            self._last_detection = now
        return detected, score


def listen() -> None:
    """Continuously capture mic audio and run wake-word detection until Ctrl+C."""
    detector = WakeWordDetector()

    print(
        f"=== Wake-word test mode — say \"What's up Vass\" (Ctrl+C to quit) ===\n"
        f"Threshold: {config.WAKE_THRESHOLD}  |  printing scores above {WAKE_SCORE_FLOOR}\n"
    )

    with open_stream() as stream:
        flush_stream(stream)
        while True:
            chunk, overflowed = stream.read(CHUNK_SAMPLES)
            if overflowed:
                print("[wake] ⚠️  input overflow — a chunk was dropped")

            audio = chunk.flatten()  # (1280, 1) int16 -> (1280,) int16
            detected, score = detector.process_chunk(audio)

            if detected:
                print(f'🔔 "What\'s up Vass" DETECTED! (score: {score:.2f})')
            elif score >= WAKE_SCORE_FLOOR:
                print(f"   score: {score:.2f}")


# ── Standalone test mode ──────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        listen()
    except KeyboardInterrupt:
        print("\nBye.")
