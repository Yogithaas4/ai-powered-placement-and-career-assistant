"""
Evaluation helpers for the Streamlit UI.

This module intentionally keeps only the metrics that are defensible for a
resume-to-job recommendation project:
- semantic alignment (label-free diagnostic)
- label-based ranking metrics when explicit relevant job_index values are given
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set


def _safe_job_index(rec: dict) -> Optional[int]:
    job_index = rec.get("job_index", rec.get("job_idx"))
    if job_index is None:
        return None
    try:
        value = int(job_index)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def ranked_job_ids(recs: List[dict]) -> List[int]:
    return [job_id for job_id in (_safe_job_index(rec) for rec in (recs or [])) if job_id is not None]


def recall_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> Optional[float]:
    if not relevant_ids or not ranked_ids:
        return None
    top = set(ranked_ids[: min(k, len(ranked_ids))])
    return len(top & relevant_ids) / len(relevant_ids)


def mrr(ranked_ids: List[int], relevant_ids: Set[int]) -> Optional[float]:
    if not relevant_ids or not ranked_ids:
        return None
    for rank, job_id in enumerate(ranked_ids, start=1):
        if job_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    total = 0.0
    for rank, job_id in enumerate(ranked_ids[:k], start=1):
        gain = 1.0 if job_id in relevant_ids else 0.0
        total += gain / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> Optional[float]:
    if not relevant_ids:
        return None
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits <= 0:
        return None
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal <= 0:
        return None
    return _dcg_at_k(ranked_ids, relevant_ids, k) / ideal


def success_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> Optional[float]:
    if not relevant_ids or not ranked_ids:
        return None
    return 1.0 if any(job_id in relevant_ids for job_id in ranked_ids[: min(k, len(ranked_ids))]) else 0.0


def build_reference_from_resume(preprocessed: dict, max_chars: int = 2500) -> str:
    embeddings = preprocessed.get("embeddings") or {}
    sections = preprocessed.get("sections") or {}
    chunks: List[str] = []

    query_string = (embeddings.get("query_string") or "").strip()
    if query_string:
        chunks.append(query_string)

    for section_name in ("summary", "skills", "experience", "education", "projects"):
        text = (sections.get(section_name) or "").strip()
        if text:
            chunks.append(text[:800])

    return "\n".join(chunks).strip()[:max_chars]


def build_hypothesis_from_recommendations(recs: List[dict], max_jobs: int = 10) -> str:
    chunks: List[str] = []
    for rec in (recs or [])[:max_jobs]:
        title = str(rec.get("title") or "").strip()
        skills = str(rec.get("skills") or "").strip()
        description = str(rec.get("description") or "").strip()[:350]
        parts = [part for part in (title, skills, description) if part]
        if parts:
            chunks.append(". ".join(parts))
    return " ".join(chunks)


def cosine_embedding_similarity(reference: str, hypothesis: str) -> Optional[float]:
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()
    if not ref or not hyp:
        return None

    try:
        import numpy as np

        from resume_processing.step4_embeddings import embed

        left = np.array(embed(ref[:8000]), dtype=np.float64)
        right = np.array(embed(hyp[:8000]), dtype=np.float64)
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm < 1e-12 or right_norm < 1e-12:
            return None
        return float(np.dot(left, right) / (left_norm * right_norm))
    except Exception:
        return None


def aggregate_metrics_for_models(
    preprocessed: dict,
    recs_by_model: Dict[str, List[dict]],
    *,
    user_relevant_indices: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    reference = build_reference_from_resume(preprocessed)
    diagnostic_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []

    for model_name, recs in recs_by_model.items():
        hypothesis = build_hypothesis_from_recommendations(recs)
        semantic_alignment = cosine_embedding_similarity(reference, hypothesis)
        diagnostic_rows.append(
            {
                "model": model_name,
                "semantic_alignment": round(semantic_alignment, 4) if semantic_alignment is not None else None,
                "n_recommendations": len(recs or []),
            }
        )

        if user_relevant_indices:
            ranked_ids = ranked_job_ids(recs)
            label_rows.append(
                {
                    "model": model_name,
                    "ndcg@10": round(ndcg_at_k(ranked_ids, user_relevant_indices, 10) or 0, 4),
                    "recall@10": round(recall_at_k(ranked_ids, user_relevant_indices, 10) or 0, 4),
                    "mrr": round(mrr(ranked_ids, user_relevant_indices) or 0, 4),
                    "success@10": int(success_at_k(ranked_ids, user_relevant_indices, 10) or 0),
                }
            )

    if user_relevant_indices:
        note = (
            "Ranking metrics below use only the explicit `job_index` labels you entered, "
            "which makes them the most trustworthy evaluation view in this app."
        )
        relevance_mode = "user_job_index_labels"
    else:
        note = (
            "No explicit relevance labels were provided, so this view shows only label-free diagnostics: "
            "semantic alignment, score separation, and cross-model agreement. "
            "Add `job_index` labels to unlock proper ranking metrics."
        )
        relevance_mode = "diagnostics_only"

    return {
        "diagnostic_rows": diagnostic_rows,
        "label_rows": label_rows,
        "reference_preview": reference[:500],
        "relevance_mode": relevance_mode,
        "relevance_size": len(user_relevant_indices or set()),
        "evaluation_note": note,
    }
