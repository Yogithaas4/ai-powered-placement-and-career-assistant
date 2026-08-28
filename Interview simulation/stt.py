from __future__ import annotations

# ============================================================
#  stt.py  –  Records audio on laptop, sends to GPU for
#             transcription via faster-whisper server
# ============================================================

import tempfile
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests

from config import SAMPLE_RATE, RECORD_SECONDS, WHISPER_SERVER_URL


# ── Recording ────────────────────────────────────────────────

def record_audio(max_seconds: int = RECORD_SECONDS,
                 silence_threshold: float = 0.01,
                 silence_duration: float = 30.0) -> np.ndarray:
    """
    Record from microphone.
    Stops after silence_duration seconds of silence OR max_seconds total.
    Returns float32 numpy array at SAMPLE_RATE.
    """
    print("[STT] Recording ... (speak now, silence stops recording)")

    chunk_size   = int(SAMPLE_RATE * 0.5)
    max_chunks   = int(max_seconds / 0.5)
    silent_limit = int(silence_duration / 0.5)

    frames       = []
    silent_count = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=chunk_size) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk.copy())

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < silence_threshold:
                silent_count += 1
                if silent_count >= silent_limit:
                    print("[STT] Silence detected - stopping.")
                    break
            else:
                silent_count = 0

    audio = np.concatenate(frames, axis=0).flatten()
    print(f"[STT] Recorded {len(audio)/SAMPLE_RATE:.1f}s of audio.")
    return audio


# ── Transcribe via GPU server ─────────────────────────────────

def transcribe_audio(audio: np.ndarray) -> str:
    """
    Save audio to temp .wav, POST to GPU Whisper server, return transcript.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        sf.write(tmp_path, audio, SAMPLE_RATE)

        with open(tmp_path, "rb") as f:
            files    = {"file": ("audio.wav", f, "audio/wav")}
            response = requests.post(
                f"{WHISPER_SERVER_URL}/transcribe",
                files=files,
                timeout=120,
            )

        response.raise_for_status()
        text = response.json().get("text", "").strip()
        print(f"[STT] Transcribed: {text[:120]}{'...' if len(text) > 120 else ''}")
        return text

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"[STT] Cannot reach Whisper server at {WHISPER_SERVER_URL}. "
            "Make sure the GPU server is running and VPN is connected."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "[STT] Whisper server timed out. "
            "Audio may be too long or GPU is busy."
        )
    except Exception as e:
        raise RuntimeError(f"[STT] Transcription request failed: {e}")
    finally:
        os.unlink(tmp_path)


# ── Convenience function ──────────────────────────────────────

def record_and_transcribe(max_seconds: int = RECORD_SECONDS) -> str:
    """Record from mic → send to GPU → return transcript text."""
    audio = record_audio(max_seconds=max_seconds)
    return transcribe_audio(audio)


# ── Connection check (call once at startup) ───────────────────

def check_whisper_server():
    """
    Ping the GPU server health endpoint.
    Call this at the start of interview.py to catch connection
    issues before the interview begins.
    """
    try:
        resp = requests.get(f"{WHISPER_SERVER_URL}/health", timeout=5)
        resp.raise_for_status()
        info = resp.json()
        print(f"[STT] Whisper server OK — model: {info['model']} "
              f"device: {info['device']}")
    except Exception:
        raise RuntimeError(
            f"[STT] Whisper server not reachable at {WHISPER_SERVER_URL}.\n"
            "  1. Check VPN is connected\n"
            "  2. Check GPU server is running:  "
            "uvicorn server:app --host 0.0.0.0 --port 8000\n"
            f"  3. Check WHISPER_SERVER_URL in config.py"
        )
