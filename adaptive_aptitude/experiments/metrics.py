"""
experiments/metrics.py
------------------------
Computes every metric named in the Phase 1 evaluation protocol from a list
of ResponseRecord, for one experiment run (one algorithm, one simulator,
one cohort of simulated students).

Two families:

  Prediction quality (needs a P(correct) estimate per response):
    roc_auc, pr_auc, brier_score, log_loss, calibration_error (ECE)

  Learning efficiency (needs question/session structure, not predictions):
    questions_to_mastery, coverage, diversity, repeat_rate,
    difficulty_match_rate

IMPORTANT CAVEAT (mention this in the paper): algorithms with no internal
P(correct) belief (Random, rule-based weakest-topic) get `predicted_p_correct`
filled in by the harness as the running empirical accuracy rate at that
point in the session -- NOT a real prediction. Their prediction-quality
metrics are included for completeness of the comparison table but are
expected to be mediocre by construction, not a fair test of "can Random
predict responses" (it isn't trying to). The metrics that actually
differentiate baselines from the adaptive algorithms are the learning-
efficiency ones plus how much *better than the naive predictor* the
adaptive algorithms' calibration is.
"""

from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

from experiments.interfaces import ResponseRecord


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: bin predictions into n_bins equal-width buckets over
    [0,1], weight each bucket's |accuracy - mean predicted prob| by bucket
    size."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        bucket_acc = y_true[mask].mean()
        bucket_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bucket_acc - bucket_conf)
    return float(ece)


def _prediction_quality_metrics(records: List[ResponseRecord]) -> Dict[str, float]:
    y_true = np.array([1.0 if r.is_correct else 0.0 for r in records])

    # Fill missing predictions with running empirical accuracy up to that
    # point (naive predictor) -- see module docstring.
    y_prob = []
    running_correct, running_n = 0, 0
    for r in records:
        if r.predicted_p_correct is not None:
            p = r.predicted_p_correct
        else:
            p = (running_correct / running_n) if running_n > 0 else 0.5
        y_prob.append(min(max(p, 1e-6), 1 - 1e-6))
        running_correct += int(r.is_correct)
        running_n += 1
    y_prob = np.array(y_prob)

    metrics = {}
    # ROC-AUC / PR-AUC undefined if all responses are the same class.
    if len(set(y_true.tolist())) < 2:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    metrics["calibration_error"] = _expected_calibration_error(y_true, y_prob)
    return metrics


def _questions_to_mastery(records: List[ResponseRecord], mastery_threshold: float = 0.8) -> Optional[float]:
    """
    Mean, across concepts that reached mastery at all, of the number of
    questions asked on that concept before true_mastery_at_ask first
    crossed the threshold. Uses SIMULATOR ground truth (true_mastery_at_ask),
    not the algorithm's own belief -- see interfaces.py docstring for why:
    this keeps the metric comparable even for algorithms (Random) that
    don't track a mastery estimate at all.
    """
    by_concept_student = defaultdict(list)
    for r in records:
        if r.concept_id is not None and r.true_mastery_at_ask is not None:
            by_concept_student[(r.student_id, r.concept_id)].append(r)

    counts = []
    for _, recs in by_concept_student.items():
        recs.sort(key=lambda r: r.step_index)
        for i, r in enumerate(recs, start=1):
            if r.true_mastery_at_ask >= mastery_threshold:
                counts.append(i)
                break
    if not counts:
        return None
    return float(np.mean(counts))


def _learning_efficiency_metrics(records: List[ResponseRecord], all_concepts_in_scope: int) -> Dict[str, float]:
    metrics = {}

    qtm = _questions_to_mastery(records)
    metrics["questions_to_mastery"] = qtm if qtm is not None else float("nan")

    concepts_touched = {r.concept_id for r in records if r.concept_id is not None}
    metrics["coverage"] = (
        len(concepts_touched) / all_concepts_in_scope if all_concepts_in_scope else float("nan")
    )

    n = len(records)
    if n > 0:
        metrics["diversity"] = len(concepts_touched) / n
    else:
        metrics["diversity"] = float("nan")

    # Repeat rate: fraction of (student, question) pairs that recur within
    # that student's session -- should be ~0 given seen_question_ids
    # exclusion, included as a plumbing sanity check as much as a metric.
    seen_pairs = defaultdict(int)
    for r in records:
        seen_pairs[(r.student_id, r.question_id)] += 1
    repeats = sum(c - 1 for c in seen_pairs.values() if c > 1)
    metrics["repeat_rate"] = repeats / n if n else float("nan")

    # Difficulty-match rate: fraction of responses where question difficulty
    # matches the band implied by true_mastery_at_ask (same bands
    # core/question_selector.py already uses), i.e. was the student neither
    # bored nor overwhelmed.
    def band(m):
        if m is None:
            return None
        if m < 0.40:
            return "Easy"
        if m < 0.70:
            return "Medium"
        return "Hard"

    matched = 0
    scored = 0
    for r in records:
        target = band(r.true_mastery_at_ask)
        if target is not None and r.difficulty is not None:
            scored += 1
            if r.difficulty == target:
                matched += 1
    metrics["difficulty_match_rate"] = (matched / scored) if scored else float("nan")

    return metrics


def compute_metrics(records: List[ResponseRecord], all_concepts_in_scope: int) -> Dict[str, float]:
    """Main entry point: one dict of metric_name -> value, for the WHOLE
    practice_category-scoped run, ready to be written into
    `experiment_metrics` (one row per key). For a per-canonical_subject
    breakdown (dashboard/appendix use), see compute_subject_breakdown."""
    if not records:
        raise ValueError("compute_metrics called with zero records")

    metrics = {}
    metrics.update(_prediction_quality_metrics(records))
    metrics.update(_learning_efficiency_metrics(records, all_concepts_in_scope))
    return metrics


def compute_subject_breakdown(records: List[ResponseRecord]) -> Dict[str, Dict[str, float]]:
    """
    Per-canonical_subject rollup within one practice_category-scoped run --
    this is the "individual subject mastery maintained in the backend"
    view: a "Core CS" session's 30 questions might be 12 OS, 10 Databases,
    8 Digital Logic; this answers "how did the student do on each, and how
    much of each subject's concepts got touched" separately, even though
    the student only ever picked one category.

    Returns {canonical_subject: {n_asked, accuracy, mean_true_mastery,
    concepts_touched}}. Only over records whose question actually carries a
    canonical_subject (should be all of them via questions_resolved, but
    the source data occasionally leaves a subject unmapped -- see the
    practice_category-IS-NULL warning in experiments/data.py).
    """
    by_subject = defaultdict(list)
    for r in records:
        if r.canonical_subject is not None:
            by_subject[r.canonical_subject].append(r)

    breakdown = {}
    for subject, recs in by_subject.items():
        n = len(recs)
        n_correct = sum(1 for r in recs if r.is_correct)
        mastery_vals = [r.true_mastery_at_ask for r in recs if r.true_mastery_at_ask is not None]
        concepts_touched = {r.concept_id for r in recs if r.concept_id is not None}
        breakdown[subject] = {
            "n_asked": n,
            "accuracy": n_correct / n if n else float("nan"),
            "mean_true_mastery": float(np.mean(mastery_vals)) if mastery_vals else float("nan"),
            "concepts_touched": len(concepts_touched),
        }
    return breakdown
