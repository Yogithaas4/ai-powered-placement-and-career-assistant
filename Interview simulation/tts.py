# ============================================================
#  tts.py  –  Text-to-Speech via edge-tts  (Microsoft Edge TTS)
# ============================================================

import asyncio
import tempfile
import os
import edge_tts
import pygame

from config import TTS_VOICE, TTS_RATE, TTS_PITCH


# ── Init pygame mixer once ───────────────────────────────────

_mixer_ready = False

def _ensure_mixer():
    global _mixer_ready
    if not _mixer_ready:
        pygame.mixer.init()
        _mixer_ready = True


# ── Core async synthesis ─────────────────────────────────────

async def _synthesise_async(text: str, out_path: str):
    communicate = edge_tts.Communicate(
        text,
        voice=TTS_VOICE,
        rate=TTS_RATE,
        pitch=TTS_PITCH,
    )
    await communicate.save(out_path)


# ── Public API ────────────────────────────────────────────────

def speak(text: str, block: bool = True):
    """
    Convert text to speech and play it through the system speakers.

    block=True  → wait until audio finishes before returning
    block=False → fire-and-forget (non-blocking, useful for overlays)
    """
    if not text.strip():
        return

    _ensure_mixer()

    # Write to temp mp3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    try:
        # Run async synthesis in a sync context
        asyncio.run(_synthesise_async(text, tmp_path))

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()

        if block:
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
    finally:
        # Small delay to let pygame release file handle before deleting
        pygame.time.wait(200)
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass   # Windows sometimes holds the file; safe to skip cleanup


def speak_async(text: str):
    """Non-blocking speak — returns immediately."""
    speak(text, block=False)


def stop_speaking():
    """Interrupt any currently playing speech."""
    if _mixer_ready:
        pygame.mixer.music.stop()
