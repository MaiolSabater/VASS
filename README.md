# VASS — Local Voice Assistant

VASS is a fully local, always-listening voice assistant. Say **"What's up Vass"** (or press Enter), talk, and it answers out loud. Nothing leaves your machine.

```
 mic ──► wake word ──► VAD recording ──► Whisper STT ──► Ollama LLM ──► Piper TTS ──► speakers
        (openWakeWord)   (Silero VAD)    (faster-whisper)  (qwen2.5)
```

## Features

- **Custom wake word.** "What's up Vass" is detected by a custom-trained openWakeWord model (`whats_up_vass.onnx`).
- **Hands-free recording.** Silero VAD stops recording automatically when you stop talking.
- **GPU speech-to-text** with faster-whisper (CUDA, falls back to CPU).
- **Local LLM** through Ollama's OpenAI-compatible API.
- **Natural speech output** with Piper TTS.
- **Conversation mode.** After the wake word you can keep talking without re-triggering it. A session ends after a silence timeout or when you say a closing phrase like "thanks Vass" or "that's all".
- **Keyboard fallback.** Press Enter instead of saying the wake word.

## Requirements

- Windows 10/11 (Linux/macOS should work with small changes; see [Notes](#notes))
- Python 3.10+
- A microphone and speakers
- An NVIDIA GPU with CUDA 12 for fast STT (optional; CPU works too)

## Setup

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/MaiolSabater/VASS.git
cd VASS
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ollama (LLM)

Install [Ollama](https://ollama.com), then create the `qwen2.5-fast` model from the included `Modelfile`. It is `qwen2.5` with a smaller 4096-token context, which makes replies faster.

```bash
ollama pull qwen2.5
ollama create qwen2.5-fast -f Modelfile
```

Ollama must be running (`ollama serve`, or the tray app) while VASS runs.

### 3. Piper (TTS)

1. Download the Piper binary from the [Piper releases](https://github.com/rhasspy/piper/releases) and either put it on your `PATH` or set `PIPER_EXECUTABLE` in `config.py` to its full path.
2. Download the **en_US-joe-medium** voice (`.onnx` file) from [rhasspy/piper-voices on Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/joe/medium) into `voices/EN/`. The matching `.onnx.json` is already in the repo.

Any other Piper voice works too; point `PIPER_VOICE_MODEL` in `config.py` at it.

### 4. CUDA libraries (GPU STT only)

faster-whisper (CTranslate2) needs the cuBLAS 12 and cuDNN 9 DLLs. They are too large for git, so they aren't in the repo. Either:

- install the CUDA 12 toolkit and cuDNN 9 system-wide, **or**
- copy these DLLs into a `libs/` folder in the project root (`stt.py` adds it to the DLL search path automatically):
  `cublas64_12.dll`, `cublasLt64_12.dll`, `cudnn*64_9.dll`

No NVIDIA GPU? Set `WHISPER_DEVICE = "cpu"` and `WHISPER_COMPUTE = "int8"` in `config.py`.

The Whisper model and openWakeWord's shared feature/VAD models download automatically on first run.

## Usage

```bash
python main.py
```

1. Say **"What's up Vass"** or press **Enter**.
2. VASS replies "yeah?". Then ask your question.
3. Keep talking for follow-ups. End the session with "thanks Vass", "that's all", "goodbye", or just stay quiet for 25 s.
4. Type `quit` (or `exit`, `q`, `bye`) and press Enter to close the program.

### Test modes

Each component can run on its own:

| Command | What it does |
|---|---|
| `python wake.py` | Prints live wake-word scores. Use it to tune `WAKE_THRESHOLD`. |
| `python stt.py` | Push-to-talk transcription test (press Enter to stop recording). |

## Configuration

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `qwen2.5-fast` | Model name |
| `PIPER_EXECUTABLE` | `piper` | Path to the Piper binary |
| `PIPER_VOICE_MODEL` | `voices\EN\en_US-joe-medium.onnx` | Piper voice |
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE` | `cuda` / `int8_float16` | Use `cpu` / `int8` without a GPU |
| `WAKE_THRESHOLD` | `0.4` | Wake-word score needed to trigger |
| `WAKE_COOLDOWN` / `WAKE_GRACE_PERIOD` | `2.0` / `1.0` s | Prevent double and false triggers |
| `VAD_THRESHOLD` | `0.5` | Speech probability per frame |
| `VAD_SILENCE_DURATION` | `0.8` s | Silence that ends an utterance |
| `VAD_MAX_RECORDING_SECONDS` | `15` | Hard cap per utterance |
| `CONVERSATION_TIMEOUT` | `25.0` s | Follow-up wait before the session ends |
| `CONVERSATION_END_PHRASES` | "thanks vass", ... | Phrases that close a session |
| `SYSTEM_PROMPT` | | Vass's personality |

## Project layout

```
VASS/
├── main.py              # Entry point: trigger loop + conversation sessions
├── wake.py              # Wake-word detection (openWakeWord)
├── stt.py               # Mic capture, Silero VAD, faster-whisper transcription
├── handler.py           # handle(): dispatch seam / future intent router
├── llm.py               # Ollama chat client
├── tts.py               # Piper synthesis + playback
├── config.py            # All tunable settings
├── Modelfile            # Ollama model definition (qwen2.5-fast)
├── whats_up_vass.onnx   # Custom wake-word model
├── voices/EN/           # Piper voice (download the .onnx, see Setup)
└── libs/                # Optional CUDA DLLs (not tracked, see Setup)
```

## Extending

| To add | Where |
|---|---|
| Smart-home / local commands | Add an intent branch in `handle()` in `handler.py` |
| A different LLM or host | `config.py`, or rewrite `llm.py` |
| A different TTS engine | Rewrite `tts.py`; keep the `speak(text)` signature |
| A different wake word | Train an openWakeWord model and set `WAKE_MODEL_PATH` |

## Notes

- `stt.py` calls the Windows-only `os.add_dll_directory` when a `libs/` folder exists. On Linux/macOS, don't create `libs/`; install CUDA system-wide instead.
- The joe voice is licensed CC0 (see `voices/EN/MODEL_CARD`).
