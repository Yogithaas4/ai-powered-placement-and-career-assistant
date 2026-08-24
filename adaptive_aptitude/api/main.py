"""
api/main.py
-----------
FastAPI backend exposing the adaptive platform endpoints.

Endpoints:
  POST /session/start            — Start a practice session
  GET  /question/next            — Get next question for student
  POST /question/answer          — Submit an answer, get updated mastery
  GET  /student/{id}/summary     — Full mastery dashboard
  GET  /student/{id}/history     — Recent interaction log
  GET  /student/{id}/skills      — All skill scores for a subject
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uuid
import time
import pandas as pd
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.concept_dag import build_default_dag
from core.knowledge_model import StudentKnowledgeModel
from core.question_selector import QuestionSelector
from data.db_loader import load_question_dataset


# ── App init ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Adaptive Test Prep Platform",
    description="BKT + Concept DAG powered adaptive question engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load resources at startup ──────────────────────────────────────────────

DAG = build_default_dag()
KM  = StudentKnowledgeModel(db_path="adaptive_platform.db")

QUESTIONS_DF = None

try:
    QUESTIONS_DF = load_question_dataset()
except Exception as e:
    print(f"Could not load questions from Postgres ({e}); starting with an empty question bank. "
          f"Run `docker compose up -d` and `python scripts/ingest_questions.py` first.")
    QUESTIONS_DF = pd.DataFrame(columns=[
        "subject","topic","subtopic","question",
        "option_a","option_b","option_c","option_d",
        "correct_answer","difficulty","time_expected_minutes",
        "question_index","has_image","image_url"
    ])


def get_selector() -> QuestionSelector:
    return QuestionSelector(QUESTIONS_DF, DAG, KM, epsilon=0.20)


# ── Request / Response models ──────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    student_id: str
    subject: str
    num_questions: int = 10

class AnswerRequest(BaseModel):
    student_id: str
    question_index: str
    concept_id: str
    subject: str
    topic: str
    subtopic: str
    difficulty: str
    selected_answer: str      # "A", "B", "C", or "D"
    correct_answer: str       # ground truth from question
    time_taken_sec: float = 0.0


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "service": "Adaptive Test Prep Platform"}


@app.get("/subjects")
def list_subjects():
    """List all available subjects."""
    subjects = QUESTIONS_DF["subject"].unique().tolist() if QUESTIONS_DF is not None else []
    return {"subjects": subjects}


@app.post("/session/start")
def start_session(req: SessionStartRequest):
    """
    Start a practice session. Returns session_id and first batch of question indices
    selected by the adaptive engine.
    """
    selector = get_selector()
    questions = selector.select_session_questions(req.student_id, req.subject, req.num_questions)

    if not questions:
        raise HTTPException(404, f"No questions found for subject: {req.subject}")

    # Start session log
    session_id = str(uuid.uuid4())
    with __import__("sqlite3").connect("adaptive_platform.db") as conn:
        conn.execute(
            "INSERT INTO student_session (student_id, subject, start_time) VALUES (?,?,?)",
            (req.student_id, req.subject, datetime.utcnow().isoformat())
        )

    return {
        "session_id": session_id,
        "student_id": req.student_id,
        "subject":    req.subject,
        "questions":  questions,
        "count":      len(questions)
    }


@app.get("/question/next")
def get_next_question(student_id: str, subject: str):
    """
    Get the single next best question for a student.
    Called after each answered question during a live session.
    """
    selector = get_selector()
    question = selector.select_question(student_id, subject)

    if question is None:
        raise HTTPException(404, "No suitable question found")

    # Clean NaN values for JSON serialization
    return {k: (v if v == v else None) for k, v in question.items()}


@app.post("/question/answer")
def submit_answer(req: AnswerRequest):
    """
    Submit a student's answer. Updates BKT/EMA mastery and returns:
    - Whether the answer was correct
    - Updated skill scores for this concept
    - Next recommended question
    """
    is_correct = req.selected_answer.upper() == req.correct_answer.upper()

    # Update knowledge model
    updated_skill = KM.update_skill(
        student_id    = req.student_id,
        question_id   = req.question_index,
        concept_id    = req.concept_id,
        subject       = req.subject,
        topic         = req.topic,
        subtopic      = req.subtopic,
        difficulty    = req.difficulty,
        correct       = is_correct,
        time_taken_sec= req.time_taken_sec
    )

    # Get next question
    selector = get_selector()
    next_q   = selector.select_question(req.student_id, req.subject)

    # Mastery label
    score = updated_skill["skill_score"]
    if score >= 0.80:
        mastery_label = "Mastered ✅"
    elif score >= 0.60:
        mastery_label = "Proficient 🟡"
    elif score >= 0.40:
        mastery_label = "Developing 🟠"
    else:
        mastery_label = "Needs Practice 🔴"

    return {
        "correct":         is_correct,
        "correct_answer":  req.correct_answer,
        "updated_skill": {
            "concept_id":    req.concept_id,
            "topic":         req.topic,
            "subtopic":      req.subtopic,
            "bkt_score":     round(updated_skill["bkt_score"], 3),
            "ema_score":     round(updated_skill["ema_score"], 3),
            "skill_score":   round(updated_skill["skill_score"], 3),
            "mastery_label": mastery_label,
            "attempts":      updated_skill["attempts"],
        },
        "next_question": {k: (v if v == v else None) for k, v in next_q.items()} if next_q else None
    }


@app.get("/student/{student_id}/summary")
def student_summary(student_id: str, subject: Optional[str] = None):
    """Full mastery summary for a student, optionally filtered by subject."""
    if subject:
        summaries = [KM.get_subject_summary(student_id, subject)]
    else:
        subjects = QUESTIONS_DF["subject"].unique().tolist()
        summaries = [KM.get_subject_summary(student_id, s) for s in subjects]

    return {"student_id": student_id, "subjects": summaries}


@app.get("/student/{student_id}/skills")
def student_skills(student_id: str, subject: str):
    """Detailed per-concept skill scores for a student in a subject."""
    skills = KM.get_all_skills(student_id, subject)
    coverage = get_selector().get_coverage_stats(student_id, subject)

    return {
        "student_id": student_id,
        "subject":    subject,
        "coverage":   coverage,
        "skills":     sorted(skills, key=lambda x: x["skill_score"])
    }


@app.get("/student/{student_id}/history")
def student_history(student_id: str, n: int = 20):
    """Recent interaction log."""
    history = KM.get_recent_history(student_id, n)
    return {"student_id": student_id, "history": history}


@app.get("/dag/{subject}")
def get_dag(subject: str):
    """Return the concept graph for a subject (for visualization)."""
    nodes = []
    edges = []
    for cid, node in DAG.nodes.items():
        if node.subject == subject:
            nodes.append({
                "id": cid, "topic": node.topic, "subtopic": node.subtopic
            })
            for dep in node.dependents:
                if dep in DAG.nodes and DAG.nodes[dep].subject == subject:
                    edges.append({"from": cid, "to": dep})
    return {"subject": subject, "nodes": nodes, "edges": edges}
