# ============================================================
#  interview.py  –  Main orchestrator  (run this file)
#
#  Flow:
#    1. User provides resume (PDF/TXT path) + job description (typed)
#    2. LLM reads both and prepares interview context
#    3. Questions auto-generate and spoken — no ENTER needed
#    4. User presses ENTER to start answer, ENTER to stop
#    5. Short natural feedback spoken after each answer
#    6. Follow-up question asked if needed (also auto, no ENTER)
#    7. Next question auto-starts after feedback gap
#    8. Final report at the end
# ============================================================

import json
import time
import sys
import threading

from config import NUM_QUESTIONS, REPORT_PATH, DEVICE
from camera import CameraSession
from stt    import record_and_transcribe, check_whisper_server
from tts    import speak, stop_speaking
from llm    import (
    generate_question,
    generate_feedback,
    generate_final_report,
    check_ollama_server,
)

# ── Timing constants ──────────────────────────────────────────
GAP_AFTER_INTRO     = 2.0   # seconds after intro before Q1
GAP_AFTER_FEEDBACK  = 2.5   # seconds after feedback before next question
GAP_AFTER_QUESTION  = 0.8   # seconds after question spoken before prompting ENTER


# ── Resume / JD input ─────────────────────────────────────────

def read_resume(path: str) -> str:
    """Read resume from .pdf, .docx, or .txt file."""
    path = path.strip().strip('"')

    if path.endswith('.pdf'):
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text.strip()

    elif path.endswith('.docx'):
        from docx import Document
        doc = Document(path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return text.strip()

    elif path.endswith('.txt'):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()

    else:
        raise ValueError(f"Unsupported file format. Please use .pdf, .docx, or .txt")


def get_user_inputs() -> tuple:
    """Prompt user for resume path, job description, and role."""
    print("\n" + "═" * 60)
    print("  MOCK INTERVIEW SETUP")
    print("═" * 60)

    # ── Resume ───────────────────────────────────────────────
    print("\n  Step 1: Resume")
    print("  Supported formats: .pdf, .txt")
    while True:
        resume_path = input("  Enter path to your resume file: ").strip()
        if not resume_path:
            print("  Please enter a valid path.")
            continue
        try:
            resume_text = read_resume(resume_path)
            if not resume_text:
                print("  File appears empty. Try another.")
                continue
            print(f"  Resume loaded ({len(resume_text)} characters)")
            break
        except FileNotFoundError:
            print(f"  File not found: {resume_path}")
        except Exception as e:
            print(f"  Error reading file: {e}")

    # ── Job Description ──────────────────────────────────────
    print("\n  Step 2: Job Description")
    print("  Paste the job description below.")
    print("  Press ENTER twice (blank line) when done.\n")

    jd_lines = []
    while True:
        line = input()
        if line == "" and jd_lines and jd_lines[-1] == "":
            break
        jd_lines.append(line)
    job_description = "\n".join(jd_lines).strip()

    if not job_description:
        job_description = "Software Engineering role requiring strong problem-solving and communication skills."
        print("  No JD entered — using generic role description.")
    else:
        print(f"  Job description received ({len(job_description)} characters)")

    # ── Role name ────────────────────────────────────────────
    print("\n  Step 3: Role Title")
    domain = input("  Enter the role title (e.g. Software Engineer, Data Scientist): ").strip()
    if not domain:
        domain = "Software Engineering"
    print(f"  Role: {domain}")

    return resume_text, job_description, domain


# ── Print helpers ─────────────────────────────────────────────

def banner(text: str):
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


def print_emotion_summary(summary: dict, label: str = "Emotion Summary"):
    print(f"\n  |-- {label}")
    print(f"  |   Dominant : {summary.get('dominant', 'N/A')}")
    scores = summary.get("scores", {})
    if scores:
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        for emo, pct in top3:
            bar = "█" * int(pct / 5)
            print(f"  |   {emo:<12} {pct:5.1f}%  {bar}")


def build_emotion_speech(session_summary: dict) -> str:
    overall = session_summary.get("overall_dominant", "Neutral")
    per_q   = session_summary.get("per_question", [])
    parts   = [f"Your overall dominant emotion throughout the interview was {overall}."]
    for item in per_q:
        parts.append(f"During question {item['question_num']} you appeared most {item['dominant']}.")
    return " ".join(parts)


# ── Recording helper ──────────────────────────────────────────

def record_answer(cam: CameraSession, q_num: int) -> tuple:
    """ENTER to start, speak, ENTER to stop. Returns (answer_text, emotion_dict)."""
    time.sleep(GAP_AFTER_QUESTION)
    input("\n  Press ENTER to START your answer  ")

    cam.start_recording(q_num=q_num)

    answer_holder = [""]
    stt_done      = threading.Event()

    def _stt_thread():
        answer_holder[0] = record_and_transcribe()
        stt_done.set()

    t = threading.Thread(target=_stt_thread, daemon=True)
    t.start()

    print("  Recording ... press ENTER to STOP\n")
    input()
    stt_done.set()
    t.join(timeout=2)

    q_emotion = cam.stop_recording()
    answer    = answer_holder[0].strip() or "[No answer detected]"

    print(f"\n  Your answer : {answer}")
    print_emotion_summary(q_emotion, label=f"Q{q_num} Emotion")

    return answer, q_emotion


# ── Main interview flow ───────────────────────────────────────

def run_interview():

    # ── 1. Get inputs ─────────────────────────────────────────
    resume_text, job_description, domain = get_user_inputs()

    banner(f"MOCK INTERVIEW  |  {domain}  |  Device: {DEVICE}")

    # ── 2. Server checks ──────────────────────────────────────
    print("\n[Setup] Checking Whisper server ...")
    check_whisper_server()

    print("[Setup] Checking Ollama ...")
    check_ollama_server()

    # ── 3. Open camera ────────────────────────────────────────
    print("\n[Setup] Opening camera ...")
    cam = CameraSession()
    cam.open()
    time.sleep(1)

    # ── 4. Intro ──────────────────────────────────────────────
    intro = (
        f"Welcome to your mock interview for {domain}. "
        f"I will ask you {NUM_QUESTIONS} questions one by one. "
        "After each question is read out, press ENTER to start your answer. "
        "Speak clearly into the microphone. "
        "Press ENTER again when you have finished answering. "
        "Let's begin."
    )
    banner("INTRO")
    print(f"\n  {intro}\n")
    speak(intro)
    time.sleep(GAP_AFTER_INTRO)   # auto-start, no ENTER needed

    # ── 5. Build context for LLM ──────────────────────────────
    context = {
        "resume":          resume_text,
        "job_description": job_description,
        "domain":          domain,
    }

    # ── 6. Question loop ──────────────────────────────────────
    all_qa:       list[dict] = []
    all_emotions: list[dict] = []

    q_num = 1
    while q_num <= NUM_QUESTIONS:
        banner(f"QUESTION  {q_num} / {NUM_QUESTIONS}")

        # Generate + speak question — auto, no ENTER
        print("[LLM] Generating question ...")
        question = generate_question(q_num, all_qa, context=context)
        print(f"\n  Q{q_num}: {question}\n")
        speak(question)

        # Record answer — ENTER to start, ENTER to stop
        answer, q_emotion = record_answer(cam, q_num)

        # Generate feedback — auto after recording stops
        speak("Okay.")
        print("[LLM] Generating feedback ...")
        feedback = generate_feedback(question, answer, previous_qa=all_qa, context=context)
        print(f"\n  Feedback: {feedback}\n")
        speak(feedback)

        # Emotion note
        dominant = q_emotion.get("dominant", "Neutral")
        speak(f"During this answer you appeared most {dominant}.")

        # Store
        all_qa.append({"question": question, "answer": answer, "feedback": feedback})
        all_emotions.append({"question_num": q_num, **q_emotion})

        # Follow-up: if feedback ends with "?" treat it as a follow-up question
        if feedback.strip().endswith("?") and q_num <= NUM_QUESTIONS:
            print("\n  [Follow-up — please respond]\n")
            time.sleep(1.0)
            followup_answer, _ = record_answer(cam, q_num)
            speak("Got it, thank you.")
            # Append follow-up answer to context so next Q is aware
            all_qa[-1]["answer"] += f" | Follow-up: {followup_answer}"

        # Auto-gap before next question — no ENTER
        if q_num < NUM_QUESTIONS:
            time.sleep(GAP_AFTER_FEEDBACK)

        q_num += 1

    # ── 7. Final report ───────────────────────────────────────
    banner("FINAL REPORT")
    speak("The interview is complete. Here is your overall performance report.")
    time.sleep(0.5)

    print("[LLM] Generating final report ...\n")
    final_report = generate_final_report(all_qa, context=context)
    speak(final_report)

    # ── 8. Emotion session summary ────────────────────────────
    session_emotion = cam.get_session_summary()

    banner("EMOTION SUMMARY")
    print(f"\n  Overall dominant emotion: {session_emotion.get('overall_dominant', 'N/A')}\n")
    for item in session_emotion.get("per_question", []):
        print_emotion_summary(
            {"dominant": item["dominant"], "scores": item.get("scores", {})},
            label=f"Q{item['question_num']}",
        )

    speak(build_emotion_speech(session_emotion))

    # ── 9. Save report ────────────────────────────────────────
    report_data = {
        "domain":          domain,
        "device":          DEVICE,
        "resume_snippet":  resume_text[:500],
        "jd_snippet":      job_description[:500],
        "questions":       all_qa,
        "emotion_per_q":   all_emotions,
        "emotion_session": session_emotion,
        "final_report":    final_report,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    banner(f"Report saved to {REPORT_PATH}")
    speak("Your interview report has been saved. Good luck with your real interview!")

    cam.close()


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_interview()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        stop_speaking()
        print("\n\n[Interrupted] Exiting.")
        sys.exit(0)
