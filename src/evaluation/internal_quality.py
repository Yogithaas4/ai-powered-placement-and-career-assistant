"""
Lightweight per-model diagnostics that do not require labels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _safe_score(rec: dict) -> Optional[float]:
    value = rec.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def score_distribution_metrics(recs: List[dict]) -> Dict[str, float]:
    scores = [_safe_score(rec) for rec in (recs or [])]
    scores = [score for score in scores if score is not None]
    if not scores:
        return {}

    arr = np.array(scores, dtype=float)
    top5 = arr[:5]
    next5 = arr[5:10]

    return {
        "avg_score": round(float(np.mean(arr)), 4),
        "top5_avg": round(float(np.mean(top5)), 4),
        "top5_vs_next5_gap": round(float(np.mean(top5) - np.mean(next5)), 4) if len(next5) else 0.0,
    }


def evaluate_internal_quality(recs_by_model: Dict[str, List[dict]]) -> Dict[str, Any]:
    rows = []
    for model, recs in recs_by_model.items():
        rows.append(
            {
                "model": model,
                "n_results": len(recs or []),
                **score_distribution_metrics(recs or []),
            }
        )
    return {"internal_quality": rows}
