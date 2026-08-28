from __future__ import annotations

# ============================================================
#  llm.py  –  LLM via Ollama
#  Context (resume + JD) passed in at runtime from user input.
#  No hardcoded JD/skills/experience.
# ============================================================

import json
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, NUM_QUESTIONS


# ── Raw completion ────────────────────────────────────────────

def ollama_chat(system_prompt: str,
                user_message: str,
                temperature: float = 0.7,
                stream: bool = False) -> str:
    url     = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model":    OLLAMA_MODEL,
        "stream":   stream,
        "options":  {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "[LLM] Cannot connect to Ollama. Make sure ollama serve is running."
        )

    if stream:
        full = []
        for line in resp.iter_lines():
            if line:
                data  = json.loads(line)
                token = data.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                full.append(token)
                if data.get("done"):
                    break
        print()
        return "".join(full)

    return resp.json()["message"]["content"].strip()


# ── Build system prompt from runtime context ──────────────────

def _build_system_prompt(context: dict) -> str:
    domain          = context.get("domain", "Software Engineering")
    resume_text     = context.get("resume", "Not provided")
    job_description = context.get("job_description", "Not provided")

    return f"""You are a real interviewer at a company hiring for {domain} roles.
You are having a live spoken conversation with a candidate sitting in front of you.

Your personality:
- Warm but professional. You genuinely want the candidate to do well.
- You speak like a human, not like a document. Short, direct sentences.
- You react naturally to what they say. A brief "got it" or "interesting" is fine.
- You never use bullet points, numbered lists, asterisks, or markdown of any kind.
- You never say things like "Certainly!" or "Great question!" or "That's a wonderful answer."
- You never lecture. You never explain what the candidate should have said.
- If the candidate missed something, probe it as a follow-up — don't tell them the answer.

Here is the candidate's resume:
{resume_text[:3000]}

Here is the job description they are applying for:
{job_description[:2000]}

Use both documents to ask targeted, relevant questions and give feedback
specific to this candidate's background and the role requirements.
Keep every response under 3 sentences. Never summarise what the candidate just said back to them."""


# ── Question style per slot ───────────────────────────────────

def _question_style(q_num: int, total: int) -> str:
    if q_num == 1:
        return (
            "This is the very first question. Start warm and casual — "
            "ask them to introduce themselves or walk through their background. "
            "Completely non-technical. One short sentence."
        )
    if q_num == 2:
        return (
            "Pick up on something specific from their introduction — "
            "a project, role, or tool they mentioned. Still light, just showing "
            "you were listening. One sentence."
        )
    if q_num == total:
        return (
            "Final question. Ask something open-ended and forward-looking — "
            "what excites them about this role, or where they want to grow. "
            "One sentence."
        )

    depth = "light-to-moderate" if q_num == 3 else "deep" if q_num >= total - 1 else "moderate"
    return (
        f"Ask a {depth} technical question relevant to the job description and resume. "
        "Either go deeper on something they've mentioned, or probe a gap between "
        "their background and the JD. One focused question only. No preamble."
    )


# ── Public API ────────────────────────────────────────────────

def generate_question(question_number: int,
                      previous_qa: list[dict] | None = None,
                      context: dict | None = None) -> str:
    context = context or {}
    system  = _build_system_prompt(context)

    history = ""
    if previous_qa:
        history = "\nConversation so far:\n"
        for i, qa in enumerate(previous_qa, 1):
            history += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"

    style   = _question_style(question_number, NUM_QUESTIONS)
    user_msg = (
        f"{style}\n\n{history}"
        "Output only the question itself. Nothing before it, nothing after it."
    )
    return ollama_chat(system, user_msg, temperature=0.75)


def generate_feedback(question: str,
                      answer: str,
                      previous_qa: list[dict] | None = None,
                      context: dict | None = None) -> str:
    """
    Short, human, reactive feedback — 2 sentences max.
    If something is missing, ask them to elaborate (don't explain it).
    If the feedback is a follow-up question, end with '?'.
    """
    context = context or {}
    system  = _build_system_prompt(context)

    covered = ""
    if previous_qa:
        covered = "Topics already covered: " + " | ".join(qa["question"] for qa in previous_qa)

    user_msg = f"""
The candidate just answered: "{question}"
Their answer: "{answer}"
{covered}

React in 1-2 sentences as a human interviewer:
- If solid: brief genuine reaction + optionally probe one specific thing deeper (end with "?").
- If weak or incomplete: ask them to elaborate on the thin part (end with "?").
- If interesting: pick up on it naturally.

Never say "you should have said" or "you could have mentioned".
Output only your spoken response. No labels, no preamble.
"""
    return ollama_chat(system, user_msg, temperature=0.65)


def generate_final_report(all_qa: list[dict],
                          context: dict | None = None) -> str:
    """Overall performance report spoken aloud at the end."""
    context = context or {}
    system  = _build_system_prompt(context)

    summary_parts = []
    for i, item in enumerate(all_qa, 1):
        summary_parts.append(
            f"Q{i}: {item['question']}\n"
            f"Answer: {item['answer'][:300]}\n"
            f"Feedback: {item['feedback']}\n"
        )

    user_msg = f"""
Here is the complete interview transcript:

{"".join(summary_parts)}

Write a spoken performance summary the candidate will hear out loud.
Cover naturally in flowing paragraphs — no headers, no bullets, no numbers:
- Overall readiness for this role out of 10, with a one-line honest justification
- What they did well, with a specific example from their answers
- One or two genuine gaps relative to the job description
- Three concrete things to do before a real interview, specific to their gaps

Write directly to them. Warm, honest, specific. Under 200 words.
"""
    return ollama_chat(system, user_msg, temperature=0.5, stream=True)


# ── Connection check ──────────────────────────────────────────

def check_ollama_server():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            raise RuntimeError(
                f"[LLM] Model '{OLLAMA_MODEL}' not found.\n"
                f"  Available: {models}\n"
                f"  Run: ollama pull {OLLAMA_MODEL}"
            )
        print(f"[LLM] Ollama OK — model '{OLLAMA_MODEL}' ready.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"[LLM] Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            "  1. Check VPN / network\n"
            "  2. On GPU: set OLLAMA_HOST=0.0.0.0:11434 && ollama serve\n"
            "  3. Check GPU_SERVER_IP in config.py"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"[LLM] Ollama at {OLLAMA_BASE_URL} timed out.")
