from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from config import DATA_DIR
from tailoring_resume.gemini_utils import (
    GeminiRateLimitError,
    call_gemini_json,
    gemini_is_configured,
)

CACHE_FILE = DATA_DIR / "processed" / "llm_explanations_cache.json"
COOLDOWN_FILE = DATA_DIR / "processed" / "llm_explanations_cooldowns.json"
DEFAULT_MODEL = os.environ.get("CAREER_EXPLAINER_MODEL", "gemini-2.5-flash")
PROMPT_VERSION = "career-explainer-v2"

DEVELOPER_PROMPT = """
You are an expert AI career mentor and hiring-market analyst.

Your job is to analyze:
- the user's current skills
- the missing skills from recommended jobs
- the top matching roles
- the relationship between existing skills and next best skills to learn

Write:
1. A short, smart explanation for each recommended job.
2. A global summary with strategic advice.
3. A revised learning path ordered intelligently.
4. Mark some steps Optional when they are useful but not essential.

Rules:
- Ground your advice in the resume profile and recommendation signals provided in the input.
- You may infer logical next skills even if they are not explicitly in `missing_skills`.
- You may revise the grounded learning-path draft when there is a better sequence.
- Prioritize skills that:
  1. build naturally on the user's existing strengths
  2. complete a known stack or workflow
  3. fit the user's strongest domain and the recommended roles
  4. unlock more roles quickly
- You may deprioritize or mark optional skills when they are less relevant.
- Do not say the user already knows a skill unless it appears in `current_skills` or `matching_skills`.
- Do not invent projects, experience, companies, certifications, degrees, or resume facts.
- Keep the writing practical, encouraging, and human.
- Use priority values only from: Core, Next, Optional.
""".strip()


def llm_is_configured() -> bool:
    return gemini_is_configured()


def _load_cache() -> Dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _load_cooldowns() -> Dict:
    if COOLDOWN_FILE.exists():
        try:
            with open(COOLDOWN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def _save_cooldowns(cooldowns: Dict) -> None:
    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COOLDOWN_FILE, "w", encoding="utf-8") as f:
        json.dump(cooldowns, f, indent=2, ensure_ascii=False)


def _active_cooldown(cooldowns: Dict, cache_key: str) -> Dict | None:
    item = cooldowns.get(cache_key)
    if not isinstance(item, dict):
        return None
    retry_at = item.get("retry_at")
    if not retry_at:
        return None
    try:
        retry_dt = datetime.fromisoformat(retry_at)
    except ValueError:
        return None
    now = datetime.now(timezone.utc)
    if retry_dt <= now:
        cooldowns.pop(cache_key, None)
        _save_cooldowns(cooldowns)
        return None
    remaining = max(1, int((retry_dt - now).total_seconds()))
    return {
        "message": item.get("message", "Gemini is temporarily rate limited."),
        "retry_after_seconds": remaining,
        "retry_at": retry_at,
    }


def _hash_payload(payload: Dict, model: str) -> str:
    raw = json.dumps(
        {"version": PROMPT_VERSION, "model": model, "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _learning_path_item_schema() -> Dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "step": {"type": "string"},
            "reason": {"type": "string"},
            "priority": {"type": "string", "enum": ["Core", "Next", "Optional"]},
        },
        "required": ["step", "reason", "priority"],
    }


def _responses_schema() -> Dict:
    learning_path_item = _learning_path_item_schema()
    return {
        "type": "json_schema",
        "name": "career_explanations",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "jobs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rank": {"type": "integer"},
                            "explanation": {"type": "string"},
                            "learning_path": {
                                "type": "array",
                                "items": learning_path_item,
                            },
                        },
                        "required": ["rank", "explanation", "learning_path"],
                    },
                },
                "global": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "headline": {"type": "string"},
                        "summary": {"type": "string"},
                        "priority_path": {"type": "string"},
                        "learning_path": {
                            "type": "array",
                            "items": learning_path_item,
                        },
                    },
                    "required": ["headline", "summary", "priority_path", "learning_path"],
                },
            },
            "required": ["jobs", "global"],
        },
    }


def _call_gemini(payload: Dict, model: str) -> Dict:
    prompt = f"""
{DEVELOPER_PROMPT}

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in this format:
{json.dumps(_responses_schema()["schema"], indent=2)}
"""
    return call_gemini_json(prompt, model, retries=1)


def _fallback_learning_path(steps: List, max_items: int = 3) -> List[Dict]:
    result = []
    for idx, step in enumerate(steps[:max_items]):
        if isinstance(step, str):
            parts = [part.strip() for part in step.split("->", 1)]
            target = parts[-1] if parts else ""
            reason = f"{target} keeps showing up in the grounded learning path." if target else ""
        else:
            target = (step or {}).get("to") or (step or {}).get("step") or ""
            reason = (step or {}).get("reason") or f"{target} closes a visible skill gap."
        if not target:
            continue
        priority = "Core" if idx == 0 else "Next"
        result.append({"step": target, "reason": reason, "priority": priority})
    return result


def _job_fallback(job: Dict) -> Dict:
    matching = job.get("matching_skills") or []
    missing = job.get("missing_skills") or []
    learning_path = _fallback_learning_path(job.get("grounded_learning_path") or [])

    lead = "This role already lines up with your "
    if matching:
        lead += ", ".join(matching[:2]) + " background."
    else:
        lead = "This role has some profile overlap, but the strongest alignment signals are limited."

    if missing:
        missing_text = ", ".join(missing[:3])
        next_steps = " ".join(
            f"{step.get('step')} because {step.get('reason')}"
            for step in learning_path[:2]
        )
        explanation = f"{lead} To improve your chances, focus on {missing_text}. {next_steps}".strip()
    else:
        explanation = (
            f"{lead} Your current profile already covers most of the surfaced skills for this recommendation."
        )

    return {
        "rank": job.get("rank"),
        "explanation": explanation,
        "learning_path": learning_path,
    }


def _global_fallback(payload: Dict) -> Dict:
    overview = payload.get("global_overview", {})
    closest_roles = overview.get("closest_roles") or []
    top_missing = overview.get("top_missing_skills") or []
    path = overview.get("grounded_recommended_path") or []
    learning_path = _fallback_learning_path(path, max_items=4)

    roles_text = " / ".join(closest_roles) if closest_roles else "your recommended roles"
    missing_text = ", ".join(item["skill"] for item in top_missing[:3]) if top_missing else "no major repeated gaps"
    path_text = " -> ".join(step["step"] for step in learning_path) if learning_path else "Keep building on your current strengths"

    return {
        "headline": f"Closest to {roles_text}",
        "summary": (
            f"Your strongest matches are closest to {roles_text}. Across the top recommendations, "
            f"the most repeated missing skills are {missing_text}. "
            f"That makes {overview.get('focus_track', 'Core Engineering')} the best improvement track right now."
        ),
        "priority_path": path_text,
        "learning_path": learning_path,
    }


def _fallback_result(payload: Dict) -> Dict:
    jobs = [_job_fallback(job) for job in payload.get("jobs", [])]
    return {
        "jobs": jobs,
        "global": _global_fallback(payload),
        "used_llm": False,
        "model": None,
    }


def build_llm_payload(analysis: Dict, top_n_jobs: int = 10) -> Dict:
    jobs_payload = []
    for job in (analysis.get("jobs") or [])[:top_n_jobs]:
        jobs_payload.append(
            {
                "rank": int(job.get("rank") or 0),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "domain": job.get("domain", ""),
                "score": job.get("score"),
                "matching_skills": job.get("matching_skills") or [],
                "missing_skills": job.get("missing_skills") or [],
                "grounded_learning_path": job.get("learning_path") or [],
            }
        )

    return {
        "current_skills": analysis.get("resume_skills") or [],
        "jobs": jobs_payload,
        "global_overview": {
            "closest_roles": analysis.get("global", {}).get("closest_roles") or [],
            "focus_track": analysis.get("global", {}).get("focus_track", ""),
            "top_missing_skills": analysis.get("global", {}).get("top_missing_skills") or [],
            "grounded_recommended_path": analysis.get("global", {}).get("recommended_path") or [],
            "dominant_domains": analysis.get("global", {}).get("dominant_domains") or [],
            "summary": analysis.get("global", {}).get("summary", ""),
        },
    }


def explain_analysis(analysis: Dict, top_n_jobs: int = 3, model: str = DEFAULT_MODEL) -> Dict:
    payload = build_llm_payload(analysis, top_n_jobs=top_n_jobs)
    cache_key = _hash_payload(payload, model)
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    if not llm_is_configured():
        return _fallback_result(payload)

    cooldowns = _load_cooldowns()
    active = _active_cooldown(cooldowns, cache_key)
    if active:
        result = _fallback_result(payload)
        result["error"] = active["message"]
        result["rate_limited"] = True
        result["retry_after_seconds"] = active["retry_after_seconds"]
        result["retry_at"] = active["retry_at"]
        return result

    try:
        result = _call_gemini(payload, model)
        result["used_llm"] = True
        result["model"] = result.get("_llm_model") or model
        result["backend"] = result.get("_llm_backend") or "gemini"
        cooldowns.pop(cache_key, None)
        _save_cooldowns(cooldowns)
    except GeminiRateLimitError as exc:
        retry_after = max(15, int(exc.retry_after_seconds or 60))
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=retry_after)
        cooldowns[cache_key] = {
            "message": str(exc),
            "retry_at": retry_at.isoformat(),
        }
        _save_cooldowns(cooldowns)
        result = _fallback_result(payload)
        result["error"] = str(exc)
        result["rate_limited"] = True
        result["retry_after_seconds"] = retry_after
        result["retry_at"] = retry_at.isoformat()
        return result
    except Exception as exc:
        result = _fallback_result(payload)
        result["error"] = str(exc)
        return result

    cache[cache_key] = result
    _save_cache(cache)
    return result
