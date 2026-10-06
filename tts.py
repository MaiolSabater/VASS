"""Text-to-speech via Piper + sounddevice playback."""

import os
import subprocess
import tempfile

import sounddevice as sd
import soundfile as sf

import config
import re

def speak(text: str) -> None:
    # remove stage directions / actions in asterisks: *pause*, *laughs*
    text = re.sub(r"\*[^*]*\*", "", text)
    # remove emojis and unspeakable chars (the surrogate fix from before)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    text = text.strip()
    if not text:
        return
    tmp_path = _synthesise(text)
    try:
        _play_wav(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── internals ─────────────────────────────────────────────────────────────────

def _synthesise(text: str) -> str:
    """Run Piper and return the path to the generated WAV file."""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        subprocess.run(
            [
                config.PIPER_EXECUTABLE,
                "--model", config.PIPER_VOICE_MODEL,
                "--output_file", tmp_path,
            ],
            input=text.encode("utf-8", errors="ignore"),
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        os.unlink(tmp_path)
        raise RuntimeError(
            f"Piper binary not found at '{config.PIPER_EXECUTABLE}'. "
            "Install Piper and/or update PIPER_EXECUTABLE in config.py."
        )
    except subprocess.CalledProcessError as e:
        os.unlink(tmp_path)
        stderr = e.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Piper failed: {stderr or e}")

    return tmp_path


def _play_wav(path: str) -> None:
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()
