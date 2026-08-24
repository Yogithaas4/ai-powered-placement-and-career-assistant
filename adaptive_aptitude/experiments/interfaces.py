"""
experiments/interfaces.py
--------------------------
The contract every Phase 1 algorithm and simulator must satisfy. This is
the piece that makes the model comparison fair for the paper: every
Selector sees the same StudentState shape, the same question_pool
DataFrame (from questions_resolved), and the same ConceptDAG. Every
Simulator produces responses the same way regardless of which Selector is
driving the session.

Only two roles:

  Selector  -- decides which question to ask next.
  Simulator -- decides whether the (simulated) student gets it right, and
               tracks the GROUND-TRUTH mastery used for metrics like
               questions-to-mastery. Ground truth lives on the simulator,
               not on the selector -- deliberately, so "how good is BKT's
               mastery estimate" can be scored against a mastery process
               BKT did not generate. See simulators.py for why this matters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import pandas as pd


@dataclass
class StudentState:
    """
    Everything a Selector is allowed to see about one simulated student at
    decision time. Selector.update() is the only thing allowed to mutate
    `mastery_estimate` / `attempts` -- this is the ALGORITHM's belief about
    the student, which may be wrong. It is intentionally separate from the
    simulator's ground truth (see simulators.StudentTruth).
    """
    student_id: str

    # Algorithm's own belief state (opaque per-selector; e.g. BKT keeps
    # P(known) here, EMA keeps a smoothed score, Random ignores it).
    mastery_estimate: Dict[str, float] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)
    correct_counts: Dict[str, int] = field(default_factory=dict)

    seen_question_ids: Set[str] = field(default_factory=set)
    n_asked: int = 0
    n_correct: int = 0

    # Free-form per-student scratch space for selectors that need to track
    # more than one raw signal (e.g. BKT+EMA blended tracker keeps separate
    # bkt_score / ema_score dicts here, since mastery_estimate only holds
    # the blended value used for selection/difficulty banding). Single-
    # tracker selectors (BKT-only, EMA-only) don't need this at all --
    # they store their one raw score directly in mastery_estimate.
    extra: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ResponseRecord:
    """One logged (question asked, response) event, used both for
    per-response DB rows (student_responses-shaped) and for metrics
    computation."""
    student_id: str
    question_id: str
    concept_id: Optional[str]
    canonical_subject: Optional[str]          # the QUESTION's own subject (e.g. "Databases") --
                                               # not the session's practice_category
    practice_category: Optional[str]           # the session's scope (e.g. "Core CS (Systems & Theory)")
    difficulty: Optional[str]
    is_correct: bool
    predicted_p_correct: Optional[float]   # algorithm's P(correct) BEFORE asking, if it has one
    true_mastery_at_ask: Optional[float]    # simulator ground truth, for questions-to-mastery
    step_index: int                          # position within this student's session (0-indexed)


class Selector(ABC):
    """Base class every selection algorithm implements."""

    name: str = "base"

    @abstractmethod
    def select(
        self,
        state: StudentState,
        practice_category: str,
        question_pool: pd.DataFrame,
        dag,
    ) -> Optional[pd.Series]:
        """
        Return one row (as a pandas Series) from question_pool to ask
        next, or None if no eligible question remains.

        `practice_category` is the SESSION's scope (one of the 5 broad,
        student-facing buckets) -- question_pool has already been filtered
        to it by the caller, but selectors that consult the DAG need it to
        call dag.get_concepts_by_practice_category(...) /
        dag.get_unlocked_concepts_in(...) rather than the (narrower)
        canonical_subject variants. Individual questions still carry their
        own canonical_subject column for mastery bookkeeping -- see
        ResponseRecord.
        """
        raise NotImplementedError

    def predict_p_correct(self, state: StudentState, question: pd.Series) -> Optional[float]:
        """
        Optional: the algorithm's own P(correct) estimate for this student
        on this question, computed BEFORE the response is observed. Used
        for ROC-AUC / Brier / log-loss / calibration.

        Algorithms with no probabilistic belief (Random, rule-based) return
        None -- the metrics module then falls back to the running empirical
        accuracy rate as a naive predictor, which is what makes those
        baselines score poorly on calibration by design. That's expected
        and is itself a paper finding, not a bug.
        """
        return None

    def update(self, state: StudentState, question: pd.Series, correct: bool) -> None:
        """Update the algorithm's belief state after observing a response.
        Default: just bookkeeping (attempts/seen/counts). Selectors with a
        mastery model (BKT, EMA, bandits) override this to also update
        `state.mastery_estimate`."""
        concept_id = question.get("concept_id")
        qid = str(question.get("question_id"))
        state.seen_question_ids.add(qid)
        state.n_asked += 1
        if concept_id is not None:
            state.attempts[concept_id] = state.attempts.get(concept_id, 0) + 1
            if correct:
                state.correct_counts[concept_id] = state.correct_counts.get(concept_id, 0) + 1
        if correct:
            state.n_correct += 1


class Simulator(ABC):
    """Base class for a simulated-student response model. Owns ground
    truth; a Selector never sees it directly."""

    name: str = "base"

    @abstractmethod
    def init_student(self, student_id: str) -> Any:
        """Create and return a fresh ground-truth state object for one
        simulated student (implementation-specific -- e.g. an IRT theta,
        or a dict of true per-concept mastery probabilities)."""
        raise NotImplementedError

    @abstractmethod
    def respond(self, truth: Any, question: pd.Series) -> bool:
        """Return True/False: whether this simulated student answers the
        given question correctly, given their ground-truth state. May
        mutate `truth` in place to reflect learning-from-practice."""
        raise NotImplementedError

    @abstractmethod
    def true_mastery(self, truth: Any, concept_id: str) -> float:
        """Ground-truth P(mastered) for one concept, used ONLY for
        computing the questions-to-mastery metric -- never exposed to the
        Selector."""
        raise NotImplementedError
