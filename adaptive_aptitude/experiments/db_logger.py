"""
experiments/db_logger.py
--------------------------
Writes experiment results into the tables already designed for this in
db/schema_extended.sql (experiment_runs, experiment_metrics) and, so
individual simulated sessions can be inspected later, into
student_sessions / student_responses using synthetic student_ids prefixed
"sim_" so they can never collide with real students and are trivially
filterable out of any real-usage analytics later.

SCOPING: `student_sessions.practice_category` is the session-scoping
column (a session pulls from a whole category, e.g. "Core CS"). This
requires the additive schema fix at the bottom of db/schema_extended.sql
(adds practice_category, drops NOT NULL on the now-legacy canonical_subject
column) -- run that migration before calling log_session, or it will fail
with a clear Postgres error naming the missing column, not a silent
mismatch.

Uses data/db_loader.get_pg_connection() -- same connection config as the
rest of the codebase, nothing new to configure.
"""

import json
import sys
import os
from typing import List, Dict, Optional
from uuid import UUID

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.db_loader import get_pg_connection  # noqa: E402
from experiments.interfaces import ResponseRecord  # noqa: E402


def create_experiment_run(algorithm: str, description: str = "",
                           hyperparameters: Optional[dict] = None) -> UUID:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_runs (algorithm, description, hyperparameters)
                VALUES (%s, %s, %s)
                RETURNING run_id
                """,
                (algorithm, description, json.dumps(hyperparameters or {})),
            )
            run_id = cur.fetchone()[0]
        conn.commit()
    return run_id


def finish_experiment_run(run_id: UUID, notes: str = "") -> None:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE experiment_runs
                SET finished_at = now(), notes = %s
                WHERE run_id = %s
                """,
                (notes, run_id),
            )
        conn.commit()


def log_metrics(run_id: UUID, metrics: Dict[str, float], context: Optional[dict] = None) -> None:
    """One row per metric, skipping NaN (Postgres double precision has no
    portable NaN literal via parameterized insert -- and a NaN row isn't
    useful for the paper's tables anyway; its absence is itself informative
    e.g. questions_to_mastery being NaN because no concept reached
    threshold)."""
    import math

    rows = [
        (run_id, name, value, json.dumps(context or {}))
        for name, value in metrics.items()
        if not (isinstance(value, float) and math.isnan(value))
    ]
    if not rows:
        return
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO experiment_metrics (run_id, metric_name, metric_value, context)
                VALUES (%s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()


def ensure_sim_student(student_id: str) -> None:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO students (student_id, display_name)
                VALUES (%s, %s)
                ON CONFLICT (student_id) DO NOTHING
                """,
                (student_id, f"[simulated] {student_id}"),
            )
        conn.commit()


def log_session(run_id: UUID, student_id: str, practice_category: str, algorithm: str,
                 records: List[ResponseRecord]) -> None:
    """Persist one simulated student's session + its responses, mirroring
    the shape a real live session would take (student_sessions +
    student_responses), so Phase 1 results can be inspected question-by-
    question later, not just as aggregate metrics.

    canonical_subject is deliberately left NULL on student_sessions (a
    category-scoped session spans multiple subjects) -- each response's own
    canonical_subject is resolvable later via student_responses.concept_id
    -> concepts.canonical_subject, never duplicated here."""
    ensure_sim_student(student_id)
    n_correct = sum(1 for r in records if r.is_correct)

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO student_sessions
                    (student_id, practice_category, algorithm, experiment_run_id,
                     ended_at, questions_asked, correct_count)
                VALUES (%s, %s, %s, %s, now(), %s, %s)
                RETURNING session_id
                """,
                (student_id, practice_category, algorithm, str(run_id), len(records), n_correct),
            )
            session_id = cur.fetchone()[0]

            response_rows = [
                (
                    session_id,
                    student_id,
                    r.question_id,
                    r.concept_id,
                    r.difficulty,
                    r.is_correct,
                    json.dumps({"true_mastery": r.true_mastery_at_ask}),
                    json.dumps({"predicted_p_correct": r.predicted_p_correct}),
                )
                for r in records
            ]
            cur.executemany(
                """
                INSERT INTO student_responses
                    (session_id, student_id, question_id, concept_id, difficulty_at_ask,
                     is_correct, mastery_before, mastery_after)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                response_rows,
            )
        conn.commit()


def log_subject_breakdown(run_id: UUID, practice_category: str, breakdown: Dict[str, Dict[str, float]]) -> None:
    """
    Log the per-canonical_subject rollup (experiments.metrics.
    compute_subject_breakdown) as extra experiment_metrics rows -- one row
    per (subject, metric), tagged via context so
    `SELECT * FROM experiment_metrics WHERE metric_name LIKE 'subject_%%'
    AND context->>'canonical_subject' = 'Databases'` reconstructs the
    per-subject view later without a separate table.
    """
    rows = []
    for subject, subject_metrics in breakdown.items():
        for metric_name, value in subject_metrics.items():
            import math
            if isinstance(value, float) and math.isnan(value):
                continue
            rows.append((
                run_id,
                f"subject_{metric_name}",
                float(value),
                json.dumps({"practice_category": practice_category, "canonical_subject": subject}),
            ))
    if not rows:
        return
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO experiment_metrics (run_id, metric_name, metric_value, context)
                VALUES (%s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()
