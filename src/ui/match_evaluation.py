"""
Evaluation: semantic similarity, ranking vs Chroma-defined relevance (no model
leakage), optional user ground-truth job_index list, pairwise agreement, and
optional lexical BLEU/ROUGE (secondary).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from evaluation.relevance import job_indices_from_chroma_topn, jobs_collection_count


def _safe_job_index(rec: dict) -> Optional[int]:
    ji = rec.get("job_index", rec.get("job_idx"))
    if ji is None:
        return None
    try:
        j = int(ji)
        return j if j >= 0 else None
    except (TypeError, ValueError):
        return None


def ranked_job_ids(recs: List[dict]) -> List[int]:
    out: List[int] = []
    for r in recs or []:
        j = _safe_job_index(r)
        if j is not None:
            out.append(j)
    return out


def precision_at_k(ranked_ids: List[int], rel: Set[int], k: int) -> Optional[float]:
    if k <= 0 or not ranked_ids or not rel:
        return None
    top = ranked_ids[: min(k, len(ranked_ids))]
    hits = sum(1 for x in top if x in rel)
    return hits / len(top)


def recall_at_k(ranked_ids: List[int], rel: Set[int], k: int) -> Optional[float]:
    if not rel or not ranked_ids:
        return None
    top = set(ranked_ids[: min(k, len(ranked_ids))])
    return len(top & rel) / len(rel)


def mrr(ranked_ids: List[int], rel: Set[int]) -> Optional[float]:
    if not rel or not ranked_ids:
        return None
    for i, jid in enumerate(ranked_ids, start=1):
        if jid in rel:
            return 1.0 / i
    return 0.0


def _dcg(ranked_ids: List[int], rel: Set[int], k: int) -> float:
    dcg = 0.0
    for i, jid in enumerate(ranked_ids[:k], start=1):
        gain = 1.0 if jid in rel else 0.0
        dcg += gain / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: List[int], rel: Set[int], k: int) -> Optional[float]:
    if not rel:
        return None
    dcg = _dcg(ranked_ids, rel, k)
    num_rel = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_rel + 1))
    if idcg <= 0:
        return None
    return dcg / idcg


def success_at_k(ranked_ids: List[int], rel: Set[int], k: int) -> Optional[float]:
    """1.0 if at least one relevant job appears in top-k, else 0.0."""
    if not rel or not ranked_ids:
        return None
    top = ranked_ids[: min(k, len(ranked_ids))]
    return 1.0 if any(j in rel for j in top) else 0.0


def corpus_recall_at_k(ranked_ids: List[int], rel: Set[int], k: int, corpus_size: int) -> Optional[float]:
    """Hits in top-k ∩ relevance, divided by total jobs in DB (usually small)."""
    if corpus_size <= 0 or not rel or not ranked_ids:
        return None
    top = set(ranked_ids[: min(k, len(ranked_ids))])
    return len(top & rel) / float(corpus_size)


def build_reference_from_resume(preprocessed: dict, max_chars: int = 2500) -> str:
    emb = preprocessed.get("embeddings") or {}
    parts: List[str] = []
    qs = emb.get("query_string") or ""
    if qs:
        parts.append(qs)
    sections = preprocessed.get("sections") or {}
    for key in ("summary", "skills", "experience", "education", "projects"):
        t = (sections.get(key) or "").strip()
        if t:
            parts.append(t[:800])
    ref = "\n".join(parts).strip()
    return ref[:max_chars]


def build_hypothesis_from_recommendations(recs: List[dict], max_jobs: int = 12) -> str:
    chunks: List[str] = []
    for r in (recs or [])[:max_jobs]:
        title = str(r.get("title") or "")
        skills = str(r.get("skills") or "")
        desc = str(r.get("description") or "")[:400]
        chunks.append(f"{title}. {skills}. {desc}".strip())
    return " ".join(chunks)


def cosine_embedding_similarity(reference: str, hypothesis: str) -> Optional[float]:
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()
    if not ref or not hyp:
        return None
    try:
        import numpy as np

        from resume_processing.step4_embeddings import embed

        a = np.array(embed(ref[:8000]), dtype=np.float64)
        b = np.array(embed(hyp[:8000]), dtype=np.float64)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return None
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return None


def bertscore_f1(reference: str, hypothesis: str) -> Optional[float]:
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()
    if not ref or not hyp:
        return None
    try:
        from bert_score import score as bert_score_fn

        ref_t = ref[:3500]
        hyp_t = hyp[:3500]
        _, _, f1 = bert_score_fn(
            [hyp_t], [ref_t], lang="en", verbose=False, rescale_with_baseline=True
        )
        return float(f1.mean().item())
    except Exception:
        return None


def compute_bleu_rouge(
    reference: str,
    hypothesis: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()
    if not ref or not hyp:
        return None, None, None, None

    bleu_val: Optional[float] = None
    r1: Optional[float] = None
    r2: Optional[float] = None
    rl: Optional[float] = None

    try:
        from sacrebleu import corpus_bleu

        bleu = corpus_bleu([hyp], [[ref]])
        bleu_val = float(getattr(bleu, "score", bleu)) / 100.0
    except Exception:
        bleu_val = None

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(ref, hyp)
        r1 = float(scores["rouge1"].fmeasure)
        r2 = float(scores["rouge2"].fmeasure)
        rl = float(scores["rougeL"].fmeasure)
    except Exception:
        r1 = r2 = rl = None

    return bleu_val, r1, r2, rl


def jaccard_top_titles(a: List[dict], b: List[dict], k: int = 10) -> float:
    ta = {str(x.get("title") or "").strip().lower() for x in (a or [])[:k] if x.get("title")}
    tb = {str(x.get("title") or "").strip().lower() for x in (b or [])[:k] if x.get("title")}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def jaccard_top_job_ids(a: List[dict], b: List[dict], k: int = 10) -> float:
    sa = {_safe_job_index(x) for x in (a or [])[:k]}
    sb = {_safe_job_index(x) for x in (b or [])[:k]}
    sa = {x for x in sa if x is not None}
    sb = {x for x in sb if x is not None}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def aggregate_metrics_for_models(
    preprocessed: dict,
    recs_by_model: Dict[str, List[dict]],
    *,
    chroma_rel_n: int = 80,
    user_relevant_indices: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """
    Ranking metrics use **relevance = user ground truth** if provided and non-empty,
    else **Chroma top-N job_index** for this resume (retrieval-only, no reranker).
    """
    ref = build_reference_from_resume(preprocessed)
    chroma_rel = job_indices_from_chroma_topn(preprocessed, n=chroma_rel_n)
    corpus_n = jobs_collection_count()

    if user_relevant_indices:
        rel: Set[int] = set(user_relevant_indices)
        rel_mode = "user_job_index_labels"
    else:
        rel = chroma_rel
        rel_mode = f"chroma_top_{chroma_rel_n}_job_indices"

    rel_empty = len(rel) == 0

    primary_rows: List[Dict[str, Any]] = []
    lexical_rows: List[Dict[str, Any]] = []

    for name, recs in recs_by_model.items():
        hyp = build_hypothesis_from_recommendations(recs)
        cos_sim = cosine_embedding_similarity(ref, hyp)
        bsf1 = bertscore_f1(ref, hyp)
        bleu, r1, r2, rl = compute_bleu_rouge(ref, hyp)
        rids = ranked_job_ids(recs)

        row: Dict[str, Any] = {
            "model": name,
            "cosine_embed_sim": round(cos_sim, 4) if cos_sim is not None else None,
            "bertscore_f1": round(bsf1, 4) if bsf1 is not None else None,
            "n_recommendations": len(recs or []),
        }
        if not rel_empty:
            row.update(
                {
                    "precision@5": round(precision_at_k(rids, rel, 5) or 0, 4),
                    "precision@10": round(precision_at_k(rids, rel, 10) or 0, 4),
                    "recall@5": round(recall_at_k(rids, rel, 5) or 0, 4),
                    "recall@10": round(recall_at_k(rids, rel, 10) or 0, 4),
                    "mrr": round(mrr(rids, rel) or 0, 4),
                    "ndcg@5": round(ndcg_at_k(rids, rel, 5) or 0, 4),
                    "ndcg@10": round(ndcg_at_k(rids, rel, 10) or 0, 4),
                    "success@5": int(success_at_k(rids, rel, 5) or 0),
                    "success@10": int(success_at_k(rids, rel, 10) or 0),
                    "corpus_recall@10": round(corpus_recall_at_k(rids, rel, 10, corpus_n) or 0, 6)
                    if corpus_n > 0
                    else None,
                }
            )
        else:
            row.update(
                {
                    "precision@5": None,
                    "precision@10": None,
                    "recall@5": None,
                    "recall@10": None,
                    "mrr": None,
                    "ndcg@5": None,
                    "ndcg@10": None,
                    "success@5": None,
                    "success@10": None,
                    "corpus_recall@10": None,
                }
            )

        primary_rows.append(row)

        scores = [r.get("score") for r in (recs or []) if isinstance(r.get("score"), (int, float))]
        avg_top = sum(scores[:5]) / min(5, len(scores)) if scores else None
        lexical_rows.append(
            {
                "model": name,
                "bleu": bleu,
                "rouge1_f": r1,
                "rouge2_f": r2,
                "rougeL_f": rl,
                "avg_top5_match_score": round(avg_top, 4) if avg_top is not None else None,
            }
        )

    model_names = list(recs_by_model.keys())
    pairwise: List[Dict[str, Any]] = []
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1 :]:
            a = recs_by_model.get(m1, [])
            b = recs_by_model.get(m2, [])
            pairwise.append(
                {
                    "model_a": m1,
                    "model_b": m2,
                    "jaccard_job_id_top10": round(jaccard_top_job_ids(a, b, 10), 4),
                    "jaccard_title_top10": round(jaccard_top_titles(a, b, 10), 4),
                }
            )

    note = (
        f"**Relevance mode:** `{rel_mode}`. "
        f"Size |rel|={len(rel)}. "
        "When using Chroma top-N, Precision/Recall/MRR/nDCG measure agreement with **vector retrieval**, "
        "not human labels — values are discriminative across models but do not prove hiring quality. "
        "Enter **job_index** ground truth in the UI for meaningful ranking metrics."
    )
    if rel_empty:
        note += " **Warning:** relevance set is empty (check jobs Chroma DB and job_index metadata)."

    return {
        "primary_rows": primary_rows,
        "lexical_rows": lexical_rows,
        "pairwise": pairwise,
        "reference_preview": ref[:500],
        "relevance_mode": rel_mode,
        "relevance_size": len(rel),
        "chroma_pool_size": chroma_rel_n,
        "corpus_job_count": corpus_n,
        "pseudo_relevance_note": note,
    }
