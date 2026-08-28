# ============================================================
#  config.py
#  JD, skills, experience are NO LONGER hardcoded here.
#  They are collected from the user at runtime in interview.py.
# ============================================================

import os
import torch

os.environ["HF_HOME"] = r"D:\hf_cache"

# ── Device ────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Remote GPU services ───────────────────────────────────────
GPU_SERVER_IP      = "10.1.12.31"
WHISPER_SERVER_URL = f"http://127.0.0.1:8000"        # Whisper runs locally now
OLLAMA_BASE_URL    = f"http://{GPU_SERVER_IP}:11434"  # LLaMA still on GPU

# ── Audio ─────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
RECORD_SECONDS = 30

# ── Edge TTS ──────────────────────────────────────────────────
TTS_VOICE = "en-US-GuyNeural"
TTS_RATE  = "+0%"
TTS_PITCH = "+0Hz"

# ── Camera ────────────────────────────────────────────────────
CAMERA_INDEX   = 0
FRAME_INTERVAL = 5

# ── Interview ─────────────────────────────────────────────────
NUM_QUESTIONS = 7   # number of main questions (follow-ups are extra)

# ── Ollama ────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3"

# ── Output ────────────────────────────────────────────────────
REPORT_PATH = "interview_report.json"
