"""
experiments/selectors/random_selector.py
------------------------------------------
The Phase 1 floor baseline: true uniform-random question selection within
the session's practice_category, ignoring mastery, difficulty, and the
concept DAG entirely.

NOTE: this is deliberately NOT the same thing as the existing
core/question_selector.py's epsilon=1.0 setting. That "always explore"
mode still restricts to unlocked-by-DAG concepts and still scores concepts
before sampling -- it is not uniform over questions. This class is the
true floor: every not-yet-seen question anywhere in the category has equal
probability. Every other algorithm in Phase 1 needs to beat THIS, not the
"always explore" mode.

Scope: practice_category (the 5 broad, student-facing buckets), not
canonical_subject -- see experiments/data.py module docstring for why.
"""

import random
from typing import Optional

import pandas as pd

from experiments.interfaces import Selector, StudentState


class RandomSelector(Selector):
    name = "random"

    def select(
        self,
        state: StudentState,
        practice_category: str,
        question_pool: pd.DataFrame,
        dag,
    ) -> Optional[pd.Series]:
        candidates = question_pool[
            (question_pool["practice_category"] == practice_category)
            & (~question_pool["question_id"].astype(str).isin(state.seen_question_ids))
        ]
        if candidates.empty:
            return None
        return candidates.sample(1, random_state=random.randint(0, 2**31 - 1)).iloc[0]

    # predict_p_correct: not overridden -> returns None. The metrics module
    # falls back to running empirical accuracy as the naive predictor, so
    # Random still produces a (poor) calibration/Brier/ROC-AUC number
    # rather than a blank cell in the comparison table.
