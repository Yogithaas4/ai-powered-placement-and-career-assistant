"""
experiments/selectors/mastery_based.py
------------------------------------------
Shared base class for the three "mastery + epsilon-greedy DAG-aware
selection" algorithms: EMA-only, BKT-only, and BKT+EMA blended (your
original core/question_selector.py, ported here in-memory).

WHY A SHARED BASE CLASS: it isolates the SELECTION STRATEGY (DAG-aware
epsilon-greedy explore/exploit + difficulty banding -- ported faithfully
from core/question_selector.py) from the MASTERY TRACKING METHOD (how the
P(known)/P(correct) belief for one concept evolves after an answer). Only
`_update_mastery` and `predict_p_correct` differ between the three
subclasses below; everything about HOW a concept gets chosen is identical.
That's deliberate: it lets the Phase 1 comparison separate two different
questions that would otherwise get confounded --
  "does epsilon-greedy exploration help?"        (compare vs rule_weakest_topic)
  "does the tracking method matter, given the
   same selection strategy?"                     (compare ema_only vs bkt vs
                                                    bkt_ema_epsilon_greedy)

The pure update math (bkt_update, ema_update, subject-tuned BKTParams,
score_concept_for_student, mastery_to_difficulty) is imported directly
from core/knowledge_model.py and core/question_selector.py rather than
re-derived -- those functions are pure (no SQLite/DB coupling), so this
keeps the live prototype and the experiment harness from ever drifting
apart on the actual math.
"""

import math
import random
from abc import abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from experiments.interfaces import Selector, StudentState
from experiments.selectors._common import pick_question_with_fallback
from core.knowledge_model import bkt_update, ema_update, BKTParams, SUBJECT_BKT_PARAMS
from core.question_selector import score_concept_for_student, mastery_to_difficulty


class MasteryBasedSelector(Selector):
    """Abstract. Subclasses implement `_update_mastery` (write the new
    per-concept mastery value into state.mastery_estimate after observing
    a response) and `predict_p_correct` (their own P(correct) belief).
    Everything about concept/question CHOICE lives here, unchanged across
    subclasses."""

    epsilon: float = 0.20                 # exploration probability, matches original default
    mastery_threshold_unlock: float = 0.60
    default_mastery_prior: float = 0.30   # used ONLY for pre-attempt selection heuristics
                                           # (DAG unlock check, explore/exploit scoring) --
                                           # NOT the same as a tracker's own p_init, which is
                                           # subject-specific and only kicks in on first update.
                                           # This mirrors core/question_selector.py exactly:
                                           # it always used a flat 0.3 fallback in selection
                                           # scoring regardless of subject, even though
                                           # knowledge_model.get_skill() initializes a fresh
                                           # BKT row with a subject-specific p_init.

    def select(self, state: StudentState, practice_category: str,
               question_pool: pd.DataFrame, dag) -> Optional[pd.Series]:
        all_concepts = dag.get_concepts_by_practice_category(practice_category)
        if not all_concepts:
            return None

        mastery = {cid: state.mastery_estimate.get(cid, self.default_mastery_prior) for cid in all_concepts}
        unlocked = set(dag.get_unlocked_concepts_in(all_concepts, mastery, self.mastery_threshold_unlock))

        if random.random() < self.epsilon:
            target_concept = self._explore(all_concepts, state, unlocked)
        else:
            target_concept = self._exploit(all_concepts, state, unlocked)

        if target_concept is None:
            target_concept = (
                random.choice(list(unlocked)) if unlocked
                else (random.choice(all_concepts) if all_concepts else None)
            )
        if target_concept is None:
            return None

        mastery_val = state.mastery_estimate.get(target_concept, self.default_mastery_prior)
        target_diff = mastery_to_difficulty(mastery_val)

        return pick_question_with_fallback(
            question_pool, target_concept, practice_category, target_diff, state.seen_question_ids
        )

    def _explore(self, concepts: List[str], state: StudentState, unlocked: set) -> Optional[str]:
        """Prefer never-attempted unlocked concepts (breadth); fall back to
        the weakest third of unlocked concepts (depth on the shakiest
        ground) if everything unlocked has already been touched."""
        unseen = [c for c in concepts if c not in state.mastery_estimate and c in unlocked]
        if unseen:
            return random.choice(unseen)

        unlocked_list = [(c, state.mastery_estimate.get(c, self.default_mastery_prior)) for c in concepts if c in unlocked]
        if not unlocked_list:
            return None
        unlocked_list.sort(key=lambda x: x[1])
        bottom_n = max(1, len(unlocked_list) // 3)
        return random.choice(unlocked_list[:bottom_n])[0]

    def _exploit(self, concepts: List[str], state: StudentState, unlocked: set) -> Optional[str]:
        """Softmax over urgency score (temperature=3) -- not pure argmin,
        so exploitation still occasionally samples a moderately-weak
        concept instead of hammering the single lowest one every time."""
        scored = []
        for cid in concepts:
            m = state.mastery_estimate.get(cid, self.default_mastery_prior)
            attempts = state.attempts.get(cid, 0)
            score = score_concept_for_student(cid, m, attempts, cid in unlocked)
            if score >= 0:
                scored.append((cid, score))
        if not scored:
            return None
        scores = [s for _, s in scored]
        total = sum(math.exp(s * 3) for s in scores)
        probs = [math.exp(s * 3) / total for s in scores]
        concept_ids = [c for c, _ in scored]
        return random.choices(concept_ids, weights=probs, k=1)[0]

    def update(self, state: StudentState, question: pd.Series, correct: bool) -> None:
        super().update(state, question, correct)  # bookkeeping: seen/attempts/correct_counts
        concept_id = question.get("concept_id")
        subject = question.get("canonical_subject")
        if concept_id is not None:
            self._update_mastery(state, concept_id, subject, correct)

    @abstractmethod
    def _update_mastery(self, state: StudentState, concept_id: str, canonical_subject: str, correct: bool) -> None:
        """Compute the new mastery value and write it into
        state.mastery_estimate[concept_id]. May also stash raw sub-scores
        in state.extra if the tracker needs more than one number (see
        BKTEMAEpsilonGreedySelector)."""
        raise NotImplementedError


class EMAOnlySelector(MasteryBasedSelector):
    """Standalone EMA tracking, no BKT slip/guess modeling. Prior for a
    never-attempted concept is 0.5 ("no information" / coin-flip), matching
    knowledge_model.get_skill()'s ema_score default -- deliberately
    different from the 0.3 selection-heuristic fallback above (EMA has no
    p_init concept of its own to justify a lower prior)."""

    name = "ema_only"
    EMA_ALPHA = 0.30  # matches StudentKnowledgeModel.EMA_ALPHA

    def _update_mastery(self, state: StudentState, concept_id: str, canonical_subject: str, correct: bool) -> None:
        prior = state.mastery_estimate.get(concept_id, 0.5)
        state.mastery_estimate[concept_id] = ema_update(prior, correct, alpha=self.EMA_ALPHA)

    def predict_p_correct(self, state: StudentState, question: pd.Series) -> Optional[float]:
        concept_id = question.get("concept_id")
        if concept_id is None:
            return None
        # EMA's smoothed accuracy score IS its P(correct) estimate -- no
        # separate slip/guess model to layer on top.
        return state.mastery_estimate.get(concept_id, 0.5)


class BKTOnlySelector(MasteryBasedSelector):
    """Standard Bayesian Knowledge Tracing, no EMA blending. Subject-tuned
    parameters (SUBJECT_BKT_PARAMS) are used both as the BKT prior on first
    real update AND to compute the slip/guess-aware P(correct) prediction.

    THIS IS WHERE TO CHANGE/SWEEP BKT PARAMETERS: pass override_params to
    force EVERY subject to use one fixed BKTParams instead of looking up
    SUBJECT_BKT_PARAMS -- this is what scripts/sweep_bkt_params.py uses to
    test different p_slip/p_guess/p_transit/p_init values against the
    Step 3 simulators, instead of hand-editing SUBJECT_BKT_PARAMS and
    re-running blind. Leave it None (default) to use the real per-subject
    tuned values from core/knowledge_model.py, i.e. normal behavior."""

    name = "bkt"

    def __init__(self, override_params: Optional[BKTParams] = None):
        self.override_params = override_params

    def _params_for(self, canonical_subject: str) -> BKTParams:
        if self.override_params is not None:
            return self.override_params
        return SUBJECT_BKT_PARAMS.get(canonical_subject, BKTParams())

    def _update_mastery(self, state: StudentState, concept_id: str, canonical_subject: str, correct: bool) -> None:
        params = self._params_for(canonical_subject)
        prior = state.mastery_estimate.get(concept_id, params.p_init)
        state.mastery_estimate[concept_id] = bkt_update(prior, correct, params)

    def predict_p_correct(self, state: StudentState, question: pd.Series) -> Optional[float]:
        concept_id = question.get("concept_id")
        canonical_subject = question.get("canonical_subject")
        if concept_id is None:
            return None
        params = self._params_for(canonical_subject)
        p_known = state.mastery_estimate.get(concept_id, params.p_init)
        return p_known * (1 - params.p_slip) + (1 - p_known) * params.p_guess


class BKTEMAEpsilonGreedySelector(MasteryBasedSelector):
    """The current production algorithm (core/question_selector.py +
    core/knowledge_model.py), ported to run in-memory over the real
    Postgres question pool/DAG instead of SQLite + the CN-only DAG.
    skill_score = 0.6*bkt + 0.4*ema, exactly matching
    knowledge_model.update_skill()'s blend weights.

    Needs BOTH raw scores tracked (bkt_score, ema_score), not just the
    blended value, because the blend isn't itself a valid input to either
    update rule next time -- stored in state.extra['bkt'] / state.extra['ema'].
    """

    name = "bkt_ema_epsilon_greedy"
    EMA_ALPHA = 0.30
    BKT_WEIGHT = 0.6
    EMA_WEIGHT = 0.4

    def __init__(self, override_params: Optional[BKTParams] = None):
        """See BKTOnlySelector.__init__ -- same override mechanism, applied
        to this selector's BKT half only (EMA half is unaffected)."""
        self.override_params = override_params

    def _params_for(self, canonical_subject: str) -> BKTParams:
        if self.override_params is not None:
            return self.override_params
        return SUBJECT_BKT_PARAMS.get(canonical_subject, BKTParams())

    def _update_mastery(self, state: StudentState, concept_id: str, canonical_subject: str, correct: bool) -> None:
        params = self._params_for(canonical_subject)
        bkt_scores = state.extra.setdefault("bkt", {})
        ema_scores = state.extra.setdefault("ema", {})

        bkt_prior = bkt_scores.get(concept_id, params.p_init)
        ema_prior = ema_scores.get(concept_id, 0.5)

        bkt_after = bkt_update(bkt_prior, correct, params)
        ema_after = ema_update(ema_prior, correct, alpha=self.EMA_ALPHA)

        bkt_scores[concept_id] = bkt_after
        ema_scores[concept_id] = ema_after
        state.mastery_estimate[concept_id] = self.BKT_WEIGHT * bkt_after + self.EMA_WEIGHT * ema_after

    def predict_p_correct(self, state: StudentState, question: pd.Series) -> Optional[float]:
        concept_id = question.get("concept_id")
        canonical_subject = question.get("canonical_subject")
        if concept_id is None:
            return None
        params = self._params_for(canonical_subject)
        # Use the BKT half's own p_known for the slip/guess-aware
        # prediction -- EMA has no slip/guess model to contribute here;
        # this is the same quantity BKT literature evaluates AUC/Brier
        # against, applied to this blended tracker's BKT component.
        bkt_p = state.extra.get("bkt", {}).get(concept_id, params.p_init)
        return bkt_p * (1 - params.p_slip) + (1 - bkt_p) * params.p_guess
