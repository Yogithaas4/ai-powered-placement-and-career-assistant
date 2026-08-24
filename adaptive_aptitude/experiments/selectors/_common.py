"""
experiments/selectors/_common.py
-----------------------------------
Small helpers shared by every mastery-aware selector, factored out so the
"pick a question for this concept, relax constraints if nothing matches"
cascade is defined once instead of copy-pasted per algorithm. Ported
faithfully from core/question_selector.py's `_pick_question` fallback
logic (difficulty match -> any difficulty -> any question in scope).
"""

from typing import Optional, Set
import pandas as pd


def pick_question_with_fallback(
    question_pool: pd.DataFrame,
    concept_id: str,
    practice_category: str,
    difficulty: Optional[str],
    exclude_ids: Set[str],
) -> Optional[pd.Series]:
    """
    3-stage cascade, same as core/question_selector.py._pick_question:
      1. exact concept + difficulty match
      2. exact concept, any difficulty (relax difficulty constraint)
      3. any not-yet-seen question in the practice_category at all
         (last resort so a session never dead-ends just because one
         concept happens to be exhausted)
    """
    mask = (question_pool["concept_id"] == concept_id) & (question_pool["practice_category"] == practice_category)
    if difficulty:
        mask = mask & (question_pool["difficulty"] == difficulty)
    candidates = question_pool[mask]
    if exclude_ids:
        candidates = candidates[~candidates["question_id"].astype(str).isin(exclude_ids)]
    if not candidates.empty:
        return candidates.sample(1).iloc[0]

    if difficulty is not None:
        return pick_question_with_fallback(question_pool, concept_id, practice_category, None, exclude_ids)

    fallback = question_pool[
        (question_pool["practice_category"] == practice_category)
        & (~question_pool["question_id"].astype(str).isin(exclude_ids))
    ]
    if fallback.empty:
        return None
    return fallback.sample(1).iloc[0]
