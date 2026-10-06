# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL    = "qwen2.5-fast"

# ── Piper TTS ─────────────────────────────────────────────────────────────────
# Full path to the piper binary (or just "piper" if it's on PATH)
PIPER_EXECUTABLE  = "piper"
# Full path to the .onnx voice model file — MUST be set before running
PIPER_VOICE_MODEL = r"voices\EN\en_US-joe-medium.onnx"   # ← update this

# ── Whisper STT ───────────────────────────────────────────────────────────────
WHISPER_MODEL       = "small" # tiny / base / small / medium / large-v3
WHISPER_DEVICE      = "cuda"   # "cuda" or "cpu"
WHISPER_COMPUTE     = "int8_float16"  # float16 (GPU) | int8 (CPU fallback)

# ── openWakeWord wake-word detection ────────────────────────────────────────────
# Path to the custom-trained openWakeWord ONNX classifier
WAKE_MODEL_PATH = r"whats_up_vass.onnx"
# Score (0-1) the model must cross to count as a detection
WAKE_THRESHOLD  = 0.4
WAKE_COOLDOWN = 2.0   # seconds to ignore detections after a trigger
# Spoken acknowledgment played when the wake word fires or Enter is pressed
WAKE_ACK = "yeah?"
# Seconds to suppress wake-word detections right after (re)starting wake-word
# listening — covers any stale audio/context that slips past the buffer flush
WAKE_GRACE_PERIOD = 1.0

# ── Voice activity detection (Silero VAD, reused from openWakeWord's models) ───
VAD_THRESHOLD = 0.5             # speech probability (0-1) a frame must cross to count as speech
VAD_SILENCE_DURATION = 0.8      # seconds of continuous silence after speech before recording stops
VAD_MAX_RECORDING_SECONDS = 15  # hard safety cap on a single utterance once speech has started

# ── Conversation mode ────────────────────────────────────────────────────────────
# How long (seconds) to wait for a follow-up before a conversation session times out
CONVERSATION_TIMEOUT = 25.0
# Case-insensitive substrings that end a conversation session when heard in the transcript
CONVERSATION_END_PHRASES = [
    "thanks vass",
    "thank you vass",
    "that's all",
    "that is all",
    "goodbye",
    "bye vass",
]
# Spoken sign-off played when a close phrase ends the conversation
CONVERSATION_END_ACK = "later!"

# ── BLE LED strip (lights.py) ────────────────────────────────────────────────────
# Found with `python light.py scan`; the HappyLighting app must be closed while VASS uses it
LIGHTS_ADDRESS          = "69:AB:00:CE:5C:8A"   # QHM-5C8A
LIGHTS_WRITE_CHAR       = "0000ffd9-0000-1000-8000-00805f9b34fb"
LIGHTS_CONNECT_TIMEOUT  = 15.0   # seconds

# ── Assistant behaviour ───────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Vass, Maiol's voice assistant and friend who hangs out in his room. "
    "Talk like a close bro — casual, relaxed, friendly. "
    "IMPORTANT: Your replies are spoken aloud by a text-to-speech engine, not read. "
    "Write exactly as you would speak out loud. "
    "Never use emojis, asterisks, stage directions, action descriptions (like *pause* or *laughs*), "
    "markdown, bullet points, or any formatting. "
    "Only output words that should actually be said aloud. "
    "Keep replies short and punchy — a sentence or two."
)