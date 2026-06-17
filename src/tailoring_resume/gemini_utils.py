from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict

# Reduce noisy low-level gRPC/absl logging from the Google client on local runs.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import google.generativeai as genai


class GeminiRateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def gemini_is_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _normalize_model_name(model: str) -> str:
    value = (model or "").strip() or "gemini-2.5-flash"
    return value if value.startswith("models/") else f"models/{value}"


def _extract_retry_after_seconds(message: str) -> int | None:
    if not message:
        return None

    patterns = [
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retry_delay\s*\{\s*seconds:\s*([0-9]+)",
        r"please retry in\s+([0-9]+(?:\.[0-9]+)?)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return max(1, int(float(match.group(1))))
            except (TypeError, ValueError):
                continue
    return None


def _is_rate_limit_message(message: str) -> bool:
    lowered = (message or "").lower()
    return any(
        token in lowered
        for token in (
            "429",
            "quota exceeded",
            "rate limit",
            "rate-limits",
            "retry_delay",
            "too many requests",
            "resource_exhausted",
        )
    )


def call_gemini_json(prompt: str, model: str, retries: int = 5) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(_normalize_model_name(model))

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = model_obj.generate_content(prompt)
            text = (getattr(response, "text", "") or "").strip()
            if text.startswith("```json"):
                text = text.replace("```json", "", 1).replace("```", "").strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            data = json.loads(text)
            if isinstance(data, dict):
                data["_llm_backend"] = "gemini"
                data["_llm_model"] = model
                return data
            raise RuntimeError("Gemini did not return a JSON object.")
        except Exception as exc:
            last_error = exc
            raw_message = str(exc)
            message = raw_message.lower()
            retry_after = _extract_retry_after_seconds(raw_message)
            retryable = _is_rate_limit_message(raw_message) or "503" in message or "temporarily unavailable" in message
            if retryable and attempt < retries - 1:
                time.sleep(retry_after if retry_after is not None else 2**attempt)
                continue
            if retryable:
                raise GeminiRateLimitError(raw_message, retry_after_seconds=retry_after)
            raise

    raise last_error or RuntimeError("Gemini request failed.")
