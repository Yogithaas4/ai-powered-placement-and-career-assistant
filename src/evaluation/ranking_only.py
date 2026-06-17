"""
Pairwise ranking-agreement diagnostics for recommendation lists.

These metrics do not require human labels. They are useful for understanding
whether two matchers surface similar jobs near the top of the ranking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_job_index(rec: dict) -> Optional[int]:
    job_index = rec.get("job_index", rec.get("job_idx"))
    try:
        value = int(job_index)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def ranked_job_ids(recs: List[dict]) -> List[int]:
    return [job_id for job_id in (_safe_job_index(rec) for rec in (recs or [])) if job_id is not None]


def jaccard_at_k(a: List[int], b: List[int], k: int = 10) -> float:
    left = set(a[:k])
    right = set(b[:k])
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlap_count_at_k(a: List[int], b: List[int], k: int = 10) -> int:
    return len(set(a[:k]) & set(b[:k]))


def rbo(a: List[int], b: List[int], p: float = 0.9, k: int = 25) -> float:
    """
    Rank-Biased Overlap: top-weighted agreement for two ranked lists.
    """
    left = a[:k]
    right = b[:k]

    seen_left = set()
    seen_right = set()
    cumulative = 0.0

    for depth in range(1, k + 1):
        if depth <= len(left):
            seen_left.add(left[depth - 1])
        if depth <= len(right):
            seen_right.add(right[depth - 1])
        overlap = len(seen_left & seen_right)
        cumulative += (overlap / depth) * (p ** (depth - 1))

    return (1 - p) * cumulative


def compare_model_rankings(
    recs_by_model: Dict[str, List[dict]],
    *,
    top_k: int = 10,
    rbo_depth: int = 25,
) -> Dict[str, Any]:
    model_names = list(recs_by_model.keys())
    ranked_lists = {name: ranked_job_ids(recs_by_model.get(name, [])) for name in model_names}

    pairwise = []
    for index, model_a in enumerate(model_names):
        for model_b in model_names[index + 1 :]:
            left = ranked_lists[model_a]
            right = ranked_lists[model_b]
            pairwise.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "rbo": round(rbo(left, right, k=rbo_depth), 4),
                    f"jaccard@{top_k}": round(jaccard_at_k(left, right, top_k), 4),
                    f"overlap@{top_k}": overlap_count_at_k(left, right, top_k),
                }
            )

    return {"pairwise": pairwise}
