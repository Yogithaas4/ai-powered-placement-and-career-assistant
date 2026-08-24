"""
experiments/simulators.py
---------------------------
Simulated-student response models.

`SimpleAbilitySimulator` is the Step 1/2 placeholder -- kept for backward
compatibility with earlier runs, but Step 3 replaces it as the thing your
paper's numbers should actually rest on. It's a single, somewhat arbitrary
generative process; any algorithm's apparent win or loss against it could
just be a coincidence of that one process, not a real property of the
algorithm.

Step 3 adds TWO structurally independent simulators instead:

  BKTGenerativeSimulator -- ground truth literally follows BKT dynamics
      (a student's true P(known) evolves via the standard BKT transition
      update). Deliberately uses its own TRUE parameters, sampled
      independently per concept -- NOT the exact SUBJECT_BKT_PARAMS values
      the BKTOnlySelector assumes -- so this isn't circular ("BKT wins
      because the world literally is BKT with the algorithm's own
      parameters"). It tests something real: how well does BKT-based
      selection hold up when the true parameters are close to, but not
      exactly, what the algorithm assumes? That's the realistic case --
      you will never know a real student's true slip/guess/transit rates
      exactly.

  IRTGenerativeSimulator -- ground truth follows a 2PL item-response
      process instead: each student has ONE fixed latent ability (no
      learning-from-practice at all), each question has a difficulty
      parameter derived from its difficulty label. This is intentionally
      the opposite kind of process from BKT (a fixed trait vs. an
      evolving belief) -- an algorithm that only does well because it
      happens to share assumptions with its ground truth should fail
      here, and one that's genuinely picking useful questions should
      still do reasonably.

An algorithm (e.g. bkt_ema_epsilon_greedy) that performs well against BOTH
is a real finding for the paper. One that only wins against
BKTGenerativeSimulator and does no better than Random against
IRTGenerativeSimulator is telling you its apparent advantage was an
artifact of matching assumptions, not real adaptivity -- exactly the
failure mode a single-simulator evaluation would hide.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from experiments.interfaces import Simulator

DIFFICULTY_LOGIT_OFFSET = {
    "Easy": +1.5,
    "Medium": 0.0,
    "Hard": -1.5,
}

# 2PL item difficulty (b-parameter), derived from the same difficulty
# labels the selectors already reason about -- keeps the IRT simulator
# comparable to the BKT one instead of inventing an unrelated scale.
IRT_DIFFICULTY_B = {
    "Easy": -1.0,
    "Medium": 0.0,
    "Hard": +1.0,
}


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class _SimpleTruth:
    student_id: str
    true_ability: float                              # fixed latent ability, ~N(0, 1)
    true_concept_mastery: Dict[str, float] = field(default_factory=dict)  # grows with practice


class SimpleAbilitySimulator(Simulator):
    """
    LEGACY Step 1/2 placeholder -- kept only so old scripts/results still
    run. For anything paper-bound, use BKTGenerativeSimulator and
    IRTGenerativeSimulator below instead (see module docstring for why a
    single simulator isn't trustworthy on its own).
    """

    name = "simple_ability_placeholder"

    def __init__(self, ability_mean: float = 0.0, ability_sd: float = 1.0,
                 learning_rate: float = 0.08, mastery_threshold: float = 0.8,
                 seed: int = None):
        self.ability_mean = ability_mean
        self.ability_sd = ability_sd
        self.learning_rate = learning_rate
        self.mastery_threshold = mastery_threshold
        self._rng = random.Random(seed)

    def init_student(self, student_id: str) -> _SimpleTruth:
        ability = self._rng.gauss(self.ability_mean, self.ability_sd)
        return _SimpleTruth(student_id=student_id, true_ability=ability)

    def respond(self, truth: _SimpleTruth, question: pd.Series) -> bool:
        concept_id = question.get("concept_id")
        difficulty = question.get("difficulty") or "Medium"

        prior_mastery = truth.true_concept_mastery.get(concept_id, 0.3) if concept_id else 0.3

        logit = (
            truth.true_ability
            + 2.0 * (prior_mastery - 0.5)
            + DIFFICULTY_LOGIT_OFFSET.get(difficulty, 0.0)
        )
        p_correct = _logistic(logit)
        correct = self._rng.random() < p_correct

        if concept_id:
            delta = self.learning_rate * (1.2 if correct else 0.3)
            new_mastery = prior_mastery + delta * (1.0 - prior_mastery)
            truth.true_concept_mastery[concept_id] = min(new_mastery, 0.99)

        return correct

    def true_mastery(self, truth: _SimpleTruth, concept_id: str) -> float:
        return truth.true_concept_mastery.get(concept_id, 0.3)


@dataclass
class _BKTTruth:
    student_id: str
    true_p_known: Dict[str, float] = field(default_factory=dict)
    # per-concept TRUE params, sampled once per student per concept on
    # first contact -- deliberately independent of SUBJECT_BKT_PARAMS
    true_params: Dict[str, "tuple"] = field(default_factory=dict)  # concept_id -> (p_init, p_transit, p_slip, p_guess)


class BKTGenerativeSimulator(Simulator):
    """
    Ground truth literally IS a BKT process -- but with TRUE parameters
    sampled independently per (student, concept), not copied from
    SUBJECT_BKT_PARAMS. Sampling is centered near jitter_center with
    +/- jitter_width uniform noise, clipped to valid probability ranges.

    Why the jitter matters: if the true process used the exact same
    numbers BKTOnlySelector assumes, "BKT-based selection wins" would be
    trivially true by construction. Jittering the true parameters away
    from the algorithm's assumed ones tests something real: robustness to
    the algorithm's parameters being APPROXIMATELY right rather than
    exactly right, which is the realistic situation with any real student
    population.
    """

    name = "bkt_generative"

    def __init__(self, jitter_center=(0.30, 0.10, 0.10, 0.20), jitter_width: float = 0.06,
                 seed: int = None):
        """jitter_center = (p_init, p_transit, p_slip, p_guess) midpoint;
        each true param is drawn uniformly from [center - width, center + width],
        clipped to [0.02, 0.5] (p_init/p_transit/p_slip) or [0.02, 0.4] (p_guess)."""
        self.jitter_center = jitter_center
        self.jitter_width = jitter_width
        self._rng = random.Random(seed)

    def _sample_true_params(self):
        p_init_c, p_trans_c, p_slip_c, p_guess_c = self.jitter_center
        w = self.jitter_width
        p_init = min(max(self._rng.uniform(p_init_c - w, p_init_c + w), 0.02), 0.60)
        p_transit = min(max(self._rng.uniform(p_trans_c - w, p_trans_c + w), 0.02), 0.50)
        p_slip = min(max(self._rng.uniform(p_slip_c - w, p_slip_c + w), 0.02), 0.40)
        p_guess = min(max(self._rng.uniform(p_guess_c - w, p_guess_c + w), 0.02), 0.40)
        return (p_init, p_transit, p_slip, p_guess)

    def init_student(self, student_id: str) -> _BKTTruth:
        return _BKTTruth(student_id=student_id)

    def respond(self, truth: _BKTTruth, question: pd.Series) -> bool:
        concept_id = question.get("concept_id")
        if concept_id is None:
            return self._rng.random() < 0.5

        if concept_id not in truth.true_params:
            truth.true_params[concept_id] = self._sample_true_params()
            p_init, _, _, _ = truth.true_params[concept_id]
            truth.true_p_known[concept_id] = p_init

        p_init, p_transit, p_slip, p_guess = truth.true_params[concept_id]
        p_known = truth.true_p_known[concept_id]

        p_correct = p_known * (1 - p_slip) + (1 - p_known) * p_guess
        correct = self._rng.random() < p_correct

        # standard BKT posterior update using the TRUE (jittered) params
        if correct:
            numerator = p_known * (1 - p_slip)
            denominator = numerator + (1 - p_known) * p_guess
        else:
            numerator = p_known * p_slip
            denominator = numerator + (1 - p_known) * (1 - p_guess)
        p_known_given_obs = numerator / denominator if denominator > 0 else p_known
        p_known_after = p_known_given_obs + (1 - p_known_given_obs) * p_transit

        truth.true_p_known[concept_id] = min(max(p_known_after, 0.0), 1.0)
        return correct

    def true_mastery(self, truth: _BKTTruth, concept_id: str) -> float:
        return truth.true_p_known.get(concept_id, self.jitter_center[0])


@dataclass
class _IRTTruth:
    student_id: str
    theta: float = 0.0  # fixed latent ability -- no learning, deliberately


class IRTGenerativeSimulator(Simulator):
    """
    Ground truth follows a fixed-ability 2PL item-response process: each
    student has ONE latent ability theta, sampled once and NEVER updated
    (no practice effect, no mastery growth) -- deliberately the opposite
    kind of process from BKT's evolving belief. Each question's difficulty
    label maps to a b-parameter (IRT_DIFFICULTY_B); discrimination `a` is
    shared (or lightly jittered per question) across items.

    Because there's no concept-level mastery growth here, true_mastery()
    returns a STATIC pseudo-mastery derived from theta vs. a reference
    item (b=0) -- it does not change with practice. Consequently
    `questions_to_mastery` will be NaN for nearly every student under this
    simulator UNLESS their fixed ability already implies high accuracy from
    the start. That's not a bug -- it's the point: this simulator isolates
    PREDICTION quality (ROC-AUC/Brier/calibration) from LEARNING-efficiency
    metrics, since under this ground truth there is no "learning" to be
    efficient about.
    """

    name = "irt_generative"

    def __init__(self, ability_mean: float = 0.0, ability_sd: float = 1.0,
                 discrimination: float = 1.2, discrimination_jitter: float = 0.2,
                 seed: int = None):
        self.ability_mean = ability_mean
        self.ability_sd = ability_sd
        self.discrimination = discrimination
        self.discrimination_jitter = discrimination_jitter
        self._rng = random.Random(seed)

    def init_student(self, student_id: str) -> _IRTTruth:
        theta = self._rng.gauss(self.ability_mean, self.ability_sd)
        return _IRTTruth(student_id=student_id, theta=theta)

    def respond(self, truth: _IRTTruth, question: pd.Series) -> bool:
        difficulty = question.get("difficulty") or "Medium"
        b = IRT_DIFFICULTY_B.get(difficulty, 0.0)
        a = max(0.3, self._rng.gauss(self.discrimination, self.discrimination_jitter))
        p_correct = _logistic(a * (truth.theta - b))
        return self._rng.random() < p_correct

    def true_mastery(self, truth: _IRTTruth, concept_id: str) -> float:
        # Static pseudo-mastery vs. a Medium (b=0) reference item -- does
        # NOT evolve with practice, unlike the other two simulators. See
        # class docstring for why this deliberately makes
        # questions_to_mastery mostly NaN under this simulator.
        return _logistic(self.discrimination * truth.theta)

