"""
experiments/runner.py
------------------------
Orchestrates one experiment: N simulated students x M questions each,
driven by one Selector + one Simulator, over the real Postgres question
pool and concept DAG. Logs to experiment_runs/experiment_metrics (and
optionally per-session detail) so every algorithm's results are
comparable and reproducible.

SCOPING: a run is scoped by `practice_category` (the 5 broad, student-
facing buckets a real session picks from) -- NOT canonical_subject. The
question pool for one run spans every canonical_subject inside that
category (e.g. "Core CS" pulls from Computer Networks, OS, Databases,
Digital Logic, ... together), exactly like a real session would. Per-
question canonical_subject is still recorded on every ResponseRecord so
per-subject mastery/accuracy can be rolled up afterward -- see
experiments/metrics.compute_subject_breakdown.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from experiments.interfaces import Selector, Simulator, StudentState, ResponseRecord
from experiments import db_logger


@dataclass
class ExperimentConfig:
    practice_category: str
    n_students: int = 200
    n_questions_per_student: int = 30
    log_to_db: bool = True
    log_sessions: bool = False   # per-response DB rows; off by default (N*M rows), turn on for spot-checks
    student_id_prefix: str = "sim"


def run_experiment(
    selector: Selector,
    simulator: Simulator,
    question_pool: pd.DataFrame,
    dag,
    config: ExperimentConfig,
) -> dict:
    """
    Runs the full experiment and returns
    {"run_id": ..., "metrics": {...}, "subject_breakdown": {...}, "n_records": ...}.
    """
    category_pool = question_pool[question_pool["practice_category"] == config.practice_category]
    if category_pool.empty:
        raise ValueError(
            f"No questions found for practice_category={config.practice_category!r} -- "
            f"check spelling against `SELECT DISTINCT practice_category FROM questions_resolved`."
        )

    n_concepts_in_scope = len(dag.get_concepts_by_practice_category(config.practice_category))

    run_id = None
    if config.log_to_db:
        run_id = db_logger.create_experiment_run(
            algorithm=selector.name,
            description=f"{selector.name} vs {simulator.name} simulator",
            hyperparameters={
                "practice_category": config.practice_category,
                "n_students": config.n_students,
                "n_questions_per_student": config.n_questions_per_student,
                "simulator": simulator.name,
            },
        )

    all_records: List[ResponseRecord] = []

    for i in range(config.n_students):
        student_id = f"{config.student_id_prefix}_{selector.name}_{i:05d}"
        state = StudentState(student_id=student_id)
        truth = simulator.init_student(student_id)

        session_records: List[ResponseRecord] = []
        for step in range(config.n_questions_per_student):
            question = selector.select(state, config.practice_category, category_pool, dag)
            if question is None:
                break  # exhausted eligible questions for this student

            concept_id = question.get("concept_id")
            predicted_p = selector.predict_p_correct(state, question)
            true_mastery_before = (
                simulator.true_mastery(truth, concept_id) if concept_id else None
            )

            correct = simulator.respond(truth, question)

            record = ResponseRecord(
                student_id=student_id,
                question_id=str(question.get("question_id")),
                concept_id=concept_id,
                canonical_subject=question.get("canonical_subject"),  # the QUESTION's own subject
                practice_category=config.practice_category,            # the SESSION's scope
                difficulty=question.get("difficulty"),
                is_correct=bool(correct),
                predicted_p_correct=predicted_p,
                true_mastery_at_ask=true_mastery_before,
                step_index=step,
            )
            session_records.append(record)

            selector.update(state, question, correct)

        all_records.extend(session_records)

        if config.log_to_db and config.log_sessions and session_records:
            db_logger.log_session(
                run_id=run_id,
                student_id=student_id,
                practice_category=config.practice_category,
                algorithm=selector.name,
                records=session_records,
            )

    if not all_records:
        raise RuntimeError(
            "No responses were generated at all -- every student's first select() "
            "call returned None. Check the question pool / practice_category filter."
        )

    from experiments.metrics import compute_metrics, compute_subject_breakdown
    metrics = compute_metrics(all_records, all_concepts_in_scope=n_concepts_in_scope)
    subject_breakdown = compute_subject_breakdown(all_records)

    if config.log_to_db:
        db_logger.log_metrics(
            run_id=run_id,
            metrics=metrics,
            context={
                "practice_category": config.practice_category,
                "n_students": config.n_students,
                "n_records": len(all_records),
                "simulator": simulator.name,
            },
        )
        db_logger.log_subject_breakdown(
            run_id=run_id,
            practice_category=config.practice_category,
            breakdown=subject_breakdown,
        )
        db_logger.finish_experiment_run(
            run_id, notes=f"{len(all_records)} responses across {config.n_students} simulated students"
        )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "subject_breakdown": subject_breakdown,
        "n_records": len(all_records),
    }
