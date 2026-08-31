"""
hybrid_matcher.py
------------------
Reciprocal Rank Fusion (RRF) hybrid — final architecture: ColBERT + CrossEncoder.

Why ConFit v2 is excluded (single source of truth for this decision)
----------------------------------------------------------------------
Based on a 304-resume evaluation (data/evaluation_results/batch_20260828_104025),
ConFit v2 had:
  - the weakest score separation of the three matchers (top5_vs_next5_gap ~0.012,
    vs ColBERT ~0.085 and CrossEncoder ~0.059)
  - the lowest pairwise agreement with both ColBERT and CrossEncoder
    (jaccard@10 ~0.21-0.28, vs ColBERT<->CrossEncoder's ~0.33)

An ablation confirmed removing ConFit v2 from the fusion matched or improved
every Hybrid metric (semantic_alignment, top5_avg, top5_vs_next5_gap) while
simplifying the architecture. ConFit v2 remains available and useful as a
STANDALONE matcher (it represents a genuinely different matching philosophy —
comparing against a synthetic ideal resume rather than direct text comparison)
— it simply does not feed into the Hybrid fusion.

This exclusion lives in exactly ONE place (EXCLUDED_FROM_HYBRID below). Any
caller — the batch evaluation script, the Streamlit UI, future features — can
pass the full recs_by_model dict (including ConFit v2) and the fusion itself
filters it out. Nothing downstream needs to remember to build a filtered dict.

RRF formula (standard, k=60 per common practice):
    score(job) = sum over included models of  1 / (60 + rank_in_that_model)
"""

from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
import pandas as pd

from config import RECOMMENDATIONS_DIR

RRF_K = 60  # standard RRF constant

# ── SINGLE SOURCE OF TRUTH: which matchers feed the Hybrid fusion ──────────
# To change the final Hybrid architecture, edit ONLY this set.
EXCLUDED_FROM_HYBRID: Set[str] = {"ConFit v2"}


def _job_key(rec: dict) -> str:
    """
    Stable identity for a job across matchers.
    Prefers job_index (present in all matchers' output); falls back to
    title+company for any edge case where job_index is missing/-1.
    """
    job_index = rec.get("job_index", -1)
    try:
        idx = int(job_index)
    except (TypeError, ValueError):
        idx = -1
    if idx >= 0:
        return f"idx::{idx}"
    return f"tc::{rec.get('title', '')}::{rec.get('company', '')}"


def reciprocal_rank_fusion(
    recs_by_model: Dict[str, List[dict]],
    top_k: int = 20,
    weights: Optional[Dict[str, float]] = None,
    k: int = RRF_K,
    exclude: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Fuse multiple ranked lists into one via weighted Reciprocal Rank Fusion.

    Args:
        recs_by_model: e.g. {"ConFit v2": [...], "ColBERT": [...], "CrossEncoder": [...]}
                       Pass the FULL dict of whatever matchers you have results
                       for — models named in `exclude` (default:
                       EXCLUDED_FROM_HYBRID) are automatically dropped before
                       fusion, so callers never need to filter manually.
        top_k: how many fused results to return
        weights: optional per-model weight multiplier. If None, every
                 INCLUDED model (after exclusion) gets equal weight 1.0 —
                 this auto-derives from whichever models are actually present,
                 so it never goes stale if the included-model set changes.
        k: RRF constant (60 is the standard default from the original RRF paper)
        exclude: model names to drop before fusion. Defaults to
                 EXCLUDED_FROM_HYBRID (currently just "ConFit v2").

    Returns:
        List of fused recommendation dicts, each carrying:
            - rrf_score: the raw fused RRF score (for transparency/debugging)
            - score: rrf_score normalized to 0-1 (so it reads on the same
              scale as other matchers' scores in the UI)
            - contributing_models: which models included this job, and at
              what rank
            - n_models_agreeing: how many included models surfaced this job
            - all display fields (title, company, domain, etc.) taken from
              whichever included model ranked it highest
    """
    exclude = EXCLUDED_FROM_HYBRID if exclude is None else exclude

    included = {name: recs for name, recs in recs_by_model.items() if name not in exclude}
    weights = weights or {name: 1.0 for name in included}

    fused: Dict[str, dict] = {}

    for model_name, recs in included.items():
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
                    "_best_rec": rec,
                    "_best_rank": int(rank),
                }

            fused[key]["rrf_score"] += contribution
            fused[key]["contributing_models"][model_name] = int(rank)

            if int(rank) < fused[key]["_best_rank"]:
                fused[key]["_best_rec"] = rec
                fused[key]["_best_rank"] = int(rank)

    ordered = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Normalize displayed score to 0-1 range (raw RRF scores are tiny, ~1/60,
    # which reads confusingly low next to other matchers' 0-1 similarity scores)
    if ordered:
        max_rrf = ordered[0]["rrf_score"] or 1.0
        for entry in ordered:
            entry["display_score"] = entry["rrf_score"] / max_rrf

    final: List[Dict] = []
    for new_rank, entry in enumerate(ordered[:top_k], 1):
        base = dict(entry["_best_rec"])
        base["rank"] = new_rank
        base["rrf_score"] = round(entry["rrf_score"], 6)
        base["score"] = round(entry.get("display_score", entry["rrf_score"]), 4)
        base["contributing_models"] = entry["contributing_models"]
        base["n_models_agreeing"] = len(entry["contributing_models"])
        final.append(base)

    return final


class HybridMatcher:
    """
    Thin wrapper so this fits the same call pattern as other matchers in the UI.

    Usage:
        hybrid = HybridMatcher()
        # pass the FULL recs_by_model dict (including ConFit v2, if present) --
        # exclusion is handled internally, per EXCLUDED_FROM_HYBRID above.
        fused = hybrid.recommend(recs_by_model, top_k=20)
        hybrid.export(fused, resume_name="arjun_sharma")
    """

    MODEL_NAME = "Hybrid (RRF)"

    def recommend(
        self,
        recs_by_model: Dict[str, List[dict]],
        top_k: int = 20,
        weights: Optional[Dict[str, float]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> List[Dict]:
        return reciprocal_rank_fusion(recs_by_model, top_k=top_k, weights=weights, exclude=exclude)

    def export(self, recommendations: List[Dict], output_path: Optional[str] = None,
               resume_name: str = "resume") -> str:
        if not recommendations:
            return ""

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
        print(f"[Hybrid] Saved -> {output_path}")
        return str(output_path)