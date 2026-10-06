"""Speech-to-text via faster-whisper + sounddevice mic capture.

Public API
----------
record_until_keypress() -> np.ndarray
    Record from the default microphone until the user presses Enter.
    Returns a float32 numpy array at 16 kHz (Whisper's native format).

record_with_vad(timeout: float | None = None) -> np.ndarray
    Record from the default microphone, automatically stopping once Silero
    VAD detects end-of-speech (or after VAD_MAX_RECORDING_SECONDS as a safety
    cap). Returns a float32 numpy array at 16 kHz, or an empty array if no
    speech was detected within `timeout` seconds (used for conversation
    follow-ups).

transcribe(audio: np.ndarray) -> str
    Transcribe a float32 audio array.  The Whisper model is loaded once
    (lazy singleton) and reused for every call.
"""

import os
import threading

import numpy as np
import sounddevice as sd

import config

import os

import os

# Make cuBLAS/cuDNN DLLs in the local libs/ folder loadable by CTranslate2.
_libs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
if os.path.isdir(_libs_dir):
    os.add_dll_directory(_libs_dir)                       # newer mechanism
    os.environ["PATH"] = _libs_dir + os.pathsep + os.environ["PATH"]  # fallback: prepend to PATH

# ── Whisper model singleton ───────────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()

SAMPLE_RATE = 16_000  # Hz — Whisper always expects 16 kHz


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:   # double-checked locking
                from faster_whisper import WhisperModel
                print(
                    f"[STT] Loading Whisper '{config.WHISPER_MODEL}' "
                    f"on {config.WHISPER_DEVICE} ({config.WHISPER_COMPUTE})…"
                )
                _model = WhisperModel(
                    config.WHISPER_MODEL,
                    device=config.WHISPER_DEVICE,
                    compute_type=config.WHISPER_COMPUTE,
                )
    return _model


# ── Silero VAD singleton ──────────────────────────────────────────────────────
# Reuses the silero_vad.onnx file that openWakeWord already downloads into its
# own resources directory (see wake.py / openwakeword.utils.download_models),
# loaded through openWakeWord's own onnxruntime-based VAD wrapper.

_vad = None
_vad_lock = threading.Lock()

_VAD_FRAME_SAMPLES = 480  # 30 ms @ 16 kHz — Silero's recommended frame size


def _get_vad():
    global _vad
    if _vad is None:
        with _vad_lock:
            if _vad is None:   # double-checked locking
                import openwakeword
                from openwakeword.utils import download_models
                from openwakeword.vad import VAD

                model_path = openwakeword.VAD_MODELS["silero_vad"]["model_path"]
                if not os.path.exists(model_path):
                    print("[STT] Downloading Silero VAD model (one-time)…")
                    download_models()
                print("[STT] Loading Silero VAD…")
                _vad = VAD(model_path=model_path)
    return _vad


# ── Public API ────────────────────────────────────────────────────────────────

def record_until_keypress() -> np.ndarray:
    """Record from the default microphone until the user presses Enter.

    Captures audio in a sounddevice callback thread while the main thread
    blocks on input().  On Enter, the stream closes and the accumulated
    chunks are concatenated into a 1-D float32 array at SAMPLE_RATE Hz.
    """
    chunks: list[np.ndarray] = []

    def _callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"[audio] {status}")
        chunks.append(indata.copy())

    print("🎤  Recording… press Enter to stop")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_callback,
    ):
        input()   # main thread blocks here; audio flows in the callback

    if not chunks:
        return np.array([], dtype=np.float32)

    # shape after concat: (total_frames, 1) → flatten to (total_frames,)
    return np.concatenate(chunks, axis=0).flatten()


def record_with_vad(timeout: float | None = None) -> np.ndarray:
    """Record from the default microphone, stopping automatically once Silero
    VAD detects end-of-speech.

    Captures int16 PCM in 30 ms frames (Silero's native frame size). Recording
    stops after VAD_SILENCE_DURATION seconds of continuous silence *following*
    detected speech, or after VAD_MAX_RECORDING_SECONDS of active speech,
    whichever comes first.

    timeout: if given, stop waiting and return an empty array if no speech
        starts within this many seconds — lets a caller (e.g. a conversation
        follow-up) distinguish "nothing was said" from a recorded utterance.
        If None, falls back to VAD_MAX_RECORDING_SECONDS as the bound on how
        long to wait for speech to start (the original behaviour).

    Returns a float32 1-D array at SAMPLE_RATE Hz, or an empty array if no
    speech was ever detected.
    """
    vad = _get_vad()
    vad.reset_states()   # clear LSTM hidden state left over from any prior call

    max_samples = int(SAMPLE_RATE * config.VAD_MAX_RECORDING_SECONDS)
    timeout_samples = int(SAMPLE_RATE * timeout) if timeout is not None else None
    frame_duration = _VAD_FRAME_SAMPLES / SAMPLE_RATE

    chunks: list[np.ndarray] = []
    speech_started = False
    silence_run = 0.0
    total_samples = 0

    print("🎤  Listening… (auto-stops on silence)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=_VAD_FRAME_SAMPLES,
    ) as stream:
        while True:
            frame, overflowed = stream.read(_VAD_FRAME_SAMPLES)
            if overflowed:
                print("[audio] input overflow")
            frame = frame.flatten()
            chunks.append(frame)
            total_samples += frame.shape[0]

            score = vad.predict(frame, frame_size=_VAD_FRAME_SAMPLES)
            if score >= config.VAD_THRESHOLD:
                speech_started = True
                silence_run = 0.0
            elif speech_started:
                silence_run += frame_duration
                if silence_run >= config.VAD_SILENCE_DURATION:
                    break   # natural end-of-speech

            if not speech_started:
                # Waiting for speech to start: bounded by `timeout` if given,
                # else by the same VAD_MAX_RECORDING_SECONDS safety cap as before.
                wait_limit = timeout_samples if timeout_samples is not None else max_samples
                if total_samples >= wait_limit:
                    print("[STT] No speech detected.")
                    break
            elif total_samples >= max_samples:
                print("[STT] Max recording length reached.")
                break

    if not speech_started:
        return np.array([], dtype=np.float32)

    audio_int16 = np.concatenate(chunks, axis=0)
    return (audio_int16.astype(np.float32) / 32768.0)


def transcribe(audio: np.ndarray) -> str:
    """Transcribe a 1-D float32 audio array sampled at 16 kHz."""
    if audio.size == 0:
        return ""

    segments, _ = _get_model().transcribe(
        audio,
        beam_size=5,
        language="en",     # remove if you want auto language detection
    )
    return " ".join(seg.text for seg in segments).strip()


# ── Standalone test mode ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== STT test mode — Ctrl+C to quit ===\n")

    # Warm up the model before the first recording so the first round feels fast
    print("Warming up Whisper (first run downloads the model if needed)…")
    transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))   # 1 s of silence
    print("Ready.\n")

    while True:
        try:
            audio = record_until_keypress()
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if audio.size < SAMPLE_RATE * 0.3:   # less than 300 ms → probably noise
            print("[Too short — nothing transcribed]\n")
            continue

        print("Transcribing…")
        try:
            text = transcribe(audio)
        except Exception as e:
            print(f"[Transcription error] {e}\n")
            continue

        if text:
            print(f"Heard: {text}\n")
        else:
            print("[Empty transcription — no speech detected]\n")
