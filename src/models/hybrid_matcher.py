"""
hybrid_matcher.py
------------------
Reciprocal Rank Fusion (RRF) hybrid across ConFit v2, ColBERT, and CrossEncoder.

Rationale
---------
Evaluate-tab diagnostics on this resume showed the three matchers agree on only
1-3 jobs out of their top 10 (Jaccard@10 ~0.05-0.18). That disagreement is
useful signal: a job that multiple independently-architected matchers rank
highly is a more robust recommendation than any single matcher's pick.

RRF formula (standard, k=60 per common practice):
    score(job) = sum over models of  1 / (60 + rank_in_that_model)

A job that doesn't appear in a given model's returned list contributes 0 for
that model (not penalized further) — this keeps the fusion tolerant of models
with fewer/more results.

Does NOT modify confit_v2_fixed.py, colbert_matcher_fixed.py, or
cross_encoder_matcher_fixed.py. Pure downstream consumer of their outputs,
using the same List[Dict] shape defined in BaseMatcher.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from config import RECOMMENDATIONS_DIR

RRF_K = 60  # standard RRF constant


def _job_key(rec: dict) -> str:
    """
    Stable identity for a job across matchers.
    Prefers job_index (present in all three matchers' output); falls back to
    title+company for any edge case where job_index is missing/-1.
    """
    job_index = rec.get("job_index", -1)
    try:
        idx = int(job_index)
    except (TypeError, ValueError):
        idx = -1
    if idx >= 0:
        return f"idx::{idx}"
    return f"tc::{rec.get('title','')}::{rec.get('company','')}"


def reciprocal_rank_fusion(
    recs_by_model: Dict[str, List[dict]],
    top_k: int = 20,
    weights: Optional[Dict[str, float]] = None,
    k: int = RRF_K,
) -> List[Dict]:
    """
    Fuse multiple ranked lists into one via weighted Reciprocal Rank Fusion.

    Args:
        recs_by_model: e.g. {"ConFit v2": [...], "ColBERT": [...], "CrossEncoder": [...]}
                       each value already ranked (rank 1 = best) and same
                       List[Dict] shape produced by BaseMatcher.recommend().
        top_k: how many fused results to return
        weights: optional per-model weight multiplier (default 1.0 each).
                 e.g. {"CrossEncoder": 1.3} to favor its more decisive ranking.
        k: RRF constant (60 is the standard default from the original RRF paper)

    Returns:
        List of fused recommendation dicts, each carrying:
            - rrf_score: the fused score (for transparency/debugging)
            - contributing_models: which models included this job, and at what rank
            - all display fields (title, company, domain, etc.) taken from
              whichever model ranked it highest, so downstream UI code needs
              no changes.
    """
    weights = weights or {}
    fused: Dict[str, dict] = {}

    for model_name, recs in recs_by_model.items():
        w = weights.get(model_name, 1.0)
        for rec in recs or []:
            rank = rec.get("rank")
            if rank is None:
                continue
            key = _job_key(rec)

            contribution = w * (1.0 / (k + int(rank)))

            if key not in fused:
                fused[key] = {
                    "rrf_score": 0.0,
                    "contributing_models": {},
                    "_best_rec": rec,       # display fields from best-ranking model
                    "_best_rank": int(rank),
                }

            fused[key]["rrf_score"] += contribution
            fused[key]["contributing_models"][model_name] = int(rank)

            # Keep display fields from whichever model ranked this job highest
            if int(rank) < fused[key]["_best_rank"]:
                fused[key]["_best_rec"] = rec
                fused[key]["_best_rank"] = int(rank)

    # Sort by fused score, descending
    ordered = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    if ordered:
        max_rrf = ordered[0]["rrf_score"] or 1.0
        for entry in ordered:
            entry["display_score"] = entry["rrf_score"] / max_rrf

    final: List[Dict] = []
    for new_rank, entry in enumerate(ordered[:top_k], 1):
        base = dict(entry["_best_rec"])  # copy display fields (title, company, domain, etc.)
        base["rank"] = new_rank
        base["rrf_score"] = round(entry["rrf_score"], 6)
        base["score"] = round(entry.get("display_score", entry["rrf_score"]), 4) # so existing UI code (which reads "score") still works
        base["contributing_models"] = entry["contributing_models"]
        base["n_models_agreeing"] = len(entry["contributing_models"])
        final.append(base)

    return final


class HybridMatcher:
    """
    Thin wrapper so this fits the same call pattern as other matchers in the UI.

    Usage:
        hybrid = HybridMatcher()
        fused = hybrid.recommend(recs_by_model, top_k=20)
        hybrid.export(fused, resume_name="arjun_sharma")
    """

    MODEL_NAME = "Hybrid (RRF)"

    def recommend(
        self,
        recs_by_model: Dict[str, List[dict]],
        top_k: int = 20,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        return reciprocal_rank_fusion(recs_by_model, top_k=top_k, weights=weights)

    def export(self, recommendations: List[Dict], output_path: Optional[str] = None,
               resume_name: str = "resume") -> str:
        if not recommendations:
            return ""

        # contributing_models is a dict per row; flatten for CSV readability
        rows = []
        for rec in recommendations:
            row = dict(rec)
            cm = row.pop("contributing_models", {})
            row["contributing_models"] = "; ".join(f"{m}:#{r}" for m, r in cm.items())
            rows.append(row)

        df = pd.DataFrame(rows)

        if output_path is None:
            RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RECOMMENDATIONS_DIR / f"hybrid_rrf_{resume_name}_{ts}.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[Hybrid] Saved → {output_path}")
        return str(output_path)