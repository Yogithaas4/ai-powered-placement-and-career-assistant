"""
experiments/selectors/rule_based.py
---------------------------------------
The "rule baseline" from the Phase 1 plan: deterministic weakest-topic
targeting + difficulty band, with NO exploration and NO BKT/EMA modeling.
This is the honest middle ground between Random (no targeting at all) and
the epsilon-greedy trackers (targeting + a probabilistic belief model) --
it isolates "does simply always drilling the observed-weakest concept
help" from "does a proper mastery-tracking model help on top of that".

Mastery here is the simplest possible signal: raw observed accuracy
(correct / attempts) per concept, no smoothing, no priors beyond a flat
default for never-attempted concepts. Always picks the single weakest
UNLOCKED concept (ties broken randomly) -- no softmax, no randomness in
the exploit direction, by design.
"""

import random
from typing import Optional

import pandas as pd

from experiments.interfaces import Selector, StudentState
from experiments.selectors._common import pick_question_with_fallback
from core.question_selector import mastery_to_difficulty


class RuleWeakestTopicSelector(Selector):
    name = "rule_weakest_topic"

    default_mastery_prior: float = 0.30
    mastery_threshold_unlock: float = 0.60

    def _observed_accuracy(self, state: StudentState, concept_id: str) -> float:
        attempts = state.attempts.get(concept_id, 0)
        if attempts == 0:
            return self.default_mastery_prior
        return state.correct_counts.get(concept_id, 0) / attempts

    def select(self, state: StudentState, practice_category: str,
               question_pool: pd.DataFrame, dag) -> Optional[pd.Series]:
        all_concepts = dag.get_concepts_by_practice_category(practice_category)
        if not all_concepts:
            return None

        mastery = {cid: self._observed_accuracy(state, cid) for cid in all_concepts}
        unlocked = dag.get_unlocked_concepts_in(all_concepts, mastery, self.mastery_threshold_unlock)
        scope = unlocked if unlocked else all_concepts

        min_mastery = min(mastery[c] for c in scope)
        weakest = [c for c in scope if mastery[c] == min_mastery]
        target_concept = random.choice(weakest)  # tie-break only, not exploration

        target_diff = mastery_to_difficulty(mastery[target_concept])
        return pick_question_with_fallback(
            question_pool, target_concept, practice_category, target_diff, state.seen_question_ids
        )

    # predict_p_correct: not overridden -> None. This selector makes no
    # probabilistic claim about P(correct); metrics.py's naive running-
    # accuracy fallback is used for its prediction-quality numbers, same
    # as RandomSelector -- expected, not a bug (see metrics.py docstring).

    # update(): base Selector.update() already maintains attempts/
    # correct_counts per concept, which is exactly what _observed_accuracy
    # needs -- no override required.
