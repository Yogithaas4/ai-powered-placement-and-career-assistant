
"""
batch_evaluation.py
--------------------
Large-scale, automated evaluation of ColBERT / CrossEncoder / Hybrid (RRF)
across an entire resume dataset (no ground-truth labels; diagnostic/ranking
metrics only).

ConFit v2 has been fully removed from this project. The final architecture is
two base matchers (ColBERT, CrossEncoder) fused via Reciprocal Rank Fusion --
see models/hybrid_matcher.py for why.

Reuses the EXACT SAME metric functions as the single-resume Evaluate tab:
    ui.match_evaluation.build_reference_from_resume
    ui.match_evaluation.build_hypothesis_from_recommendations
    ui.match_evaluation.cosine_embedding_similarity        -> semantic_alignment
    evaluation.internal_quality.score_distribution_metrics -> avg_score, top5_avg,
                                                               top5_vs_next5_gap
    evaluation.ranking_only.compare_model_rankings         -> rbo, jaccard@10, overlap@10

Deliberately does NOT introduce ground-truth labels, precision/recall/ndcg/mrr.

Two selection modes:
    --limit N                          random sample of N resumes from the
                                        WHOLE dataset (reproducible, seed=42)
    --full-categories "A,B"             include EVERY resume from categories A
                                        and B, plus a small random sample
                                        (--other-per-category, default 3) from
                                        every other category folder

Output layout (all under a single timestamped run directory):

    data/evaluation_results/<run_id>/
        per_resume_model_metrics.csv    one row per (resume, model)
        per_resume_pair_metrics.csv     one row per (resume, model_a, model_b)
        failures.csv                    resume_id, filename, category, error
        aggregate_model_metrics.csv     per model x metric: n, mean, median, std,
                                         ci95_low, ci95_high
        aggregate_pair_metrics.csv      per model-pair x metric: same stats
        run_summary.json                config, counts, timing
        processed_ids.txt               resume_ids already completed

Usage (CLI)
-----------
    python -m evaluation.batch_evaluation \
        --resume-root "C:/Users/Student1/Downloads/archive/data/data" \
        --full-categories "ENGINEERING,INFORMATION-TECHNOLOGY" \
        --other-per-category 3

    python -m evaluation.batch_evaluation \
        --resume-root "C:/Users/Student1/Downloads/archive/data/data" \
        --limit 50

    python -m evaluation.batch_evaluation \
        --resume-root "C:/Users/Student1/Downloads/archive/data/data" \
        --resume-from data/evaluation_results/batch_20260824_120000
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR
from resume_processing.main_pipeline import process_one
from resume_processing.step1_parser import parse_file, SUPPORTED_EXTENSIONS

from models.colbert_matcher_fixed import ColBERTEngine
from models.cross_encoder_matcher_fixed import CrossEncoderEngine
from models.hybrid_matcher import HybridMatcher

from evaluation.internal_quality import score_distribution_metrics
from evaluation.ranking_only import compare_model_rankings
from ui.match_evaluation import (
    build_reference_from_resume,
    build_hypothesis_from_recommendations,
    cosine_embedding_similarity,
)

HYBRID_NAME = "Hybrid (RRF)"
MODEL_ORDER = ("ColBERT", "CrossEncoder", HYBRID_NAME)
EVAL_RESULTS_DIR = DATA_DIR / "evaluation_results"

MODEL_METRIC_COLUMNS = [
    "semantic_alignment",
    "n_recommendations",
    "avg_score",
    "top5_avg",
    "top5_vs_next5_gap",
]
PAIR_METRIC_COLUMNS = ["rbo", "jaccard_at_10", "overlap_at_10"]


# ==============================================================================
#  Resume discovery
# ==============================================================================

def find_resume_files(root: str) -> List[Path]:
    """Recursively find every supported resume file (.docx/.pdf/.txt) under root."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Resume root not found: {root_path}")

    files = [
        p for p in root_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort()
    return files


def _resume_id_and_category(file_path: Path, root: Path) -> Tuple[str, str]:
    """
    resume_id: path relative to root (stable, human-readable, unique)
    category:  first path component under root, or "" if none.
    """
    rel = file_path.relative_to(root)
    resume_id = rel.as_posix()
    parts = rel.parts
    category = parts[0] if len(parts) > 1 else ""
    return resume_id, category


def select_resume_subset(
    root: str,
    full_categories: List[str],
    other_per_category: int = 3,
    seed: int = 42,
) -> List[Path]:
    """
    Build a curated subset: ALL resumes from `full_categories` (case-insensitive
    match against the category folder name), plus a small reproducible random
    sample from every other category folder.
    """
    random.seed(seed)

    root_path = Path(root)
    all_files = find_resume_files(root)

    full_categories_lower = {c.strip().lower() for c in full_categories}

    by_category: Dict[str, List[Path]] = {}
    for f in all_files:
        _, category = _resume_id_and_category(f, root_path)
        by_category.setdefault(category, []).append(f)

    selected: List[Path] = []
    for category, files in sorted(by_category.items()):
        if category.lower() in full_categories_lower:
            selected.extend(files)
            print(f"[Batch] '{category}': including all {len(files)} resume(s)")
        else:
            sample_size = min(other_per_category, len(files))
            sampled = random.sample(files, sample_size)
            selected.extend(sampled)
            print(f"[Batch] '{category}': sampling {sample_size} of {len(files)} resume(s)")

    selected.sort()
    return selected


# ==============================================================================
#  Statistics helpers (no scipy dependency -- normal approximation for 95% CI)
# ==============================================================================

def _ci95(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = statistics.mean(values)
    if n == 1:
        return (mean, mean)
    std = statistics.stdev(values)
    margin = 1.96 * (std / math.sqrt(n))
    return (mean - margin, mean + margin)


def _describe(values: List[float]) -> Dict[str, float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = statistics.mean(clean)
    median = statistics.median(clean)
    std = statistics.stdev(clean) if n > 1 else 0.0
    ci_low, ci_high = _ci95(clean)
    return {
        "n": n,
        "mean": round(mean, 6),
        "median": round(median, 6),
        "std": round(std, 6),
        "ci95_low": round(ci_low, 6),
        "ci95_high": round(ci_high, 6),
    }


# ==============================================================================
#  Engine bundle -- load ColBERT + CrossEncoder once, reuse across all resumes
# ==============================================================================

class EngineBundle:
    """
    Loads ColBERT and CrossEncoder once, and computes Hybrid (RRF) on top of
    their combined output for every resume. No ConFit v2 anywhere in this
    project anymore -- final architecture is exactly these two base matchers
    plus their fusion.
    """

    def __init__(self, chroma_pool: int = 60, top_k: int = 15,
                 hybrid_weights: Optional[Dict[str, float]] = None):
        self.chroma_pool = chroma_pool
        self.top_k = top_k
        self.hybrid_weights = hybrid_weights  # None -> HybridMatcher auto-assigns equal weight

        print("[Batch] Loading ColBERT engine...")
        self.colbert = ColBERTEngine()

        print("[Batch] Loading CrossEncoder engine...")
        self.cross_encoder = CrossEncoderEngine()

        print("[Batch] Initializing Hybrid (RRF) combiner...")
        self.hybrid = HybridMatcher()

        print("[Batch] All engines loaded.")

    def run_all(self, preprocessed: dict) -> Dict[str, List[dict]]:
        recs = {
            "ColBERT": self.colbert.recommend(
                preprocessed, top_k=self.top_k, stage1_n_results=self.chroma_pool
            ),
            "CrossEncoder": self.cross_encoder.recommend(
                preprocessed, top_k=self.top_k, stage1_n_results=self.chroma_pool
            ),
        }
        recs[HYBRID_NAME] = self.hybrid.recommend(
            recs, top_k=self.top_k, weights=self.hybrid_weights
        )
        return recs


# ==============================================================================
#  Per-resume evaluation (raises on failure -- caller decides how to handle it)
# ==============================================================================

def evaluate_one_resume(
    file_path: Path,
    resume_id: str,
    category: str,
    engines: EngineBundle,
) -> Tuple[List[dict], List[dict]]:
    """
    Returns (model_rows, pair_rows) for this single resume.
    Raises on any failure -- the batch loop below catches and records it.
    """
    t0 = time.time()

    raw_text = parse_file(str(file_path))
    t1 = time.time()

    parsed = {
        "filename": file_path.name,
        "raw_text": raw_text,
        "file_type": file_path.suffix.lstrip("."),
    }
    preprocessed = process_one(parsed)
    t2 = time.time()
    print(f"    [timing] parse={t1 - t0:.2f}s process_one={t2 - t1:.2f}s")

    recs_by_model = engines.run_all(preprocessed)
    t3 = time.time()
    print(f"    [timing] run_all_matchers={t3 - t2:.2f}s")

    reference = build_reference_from_resume(preprocessed)

    model_rows: List[dict] = []
    for model_name, recs in recs_by_model.items():
        hypothesis = build_hypothesis_from_recommendations(recs)
        semantic_alignment = cosine_embedding_similarity(reference, hypothesis)
        dist_metrics = score_distribution_metrics(recs or [])

        model_rows.append({
            "resume_id": resume_id,
            "filename": file_path.name,
            "category": category,
            "model": model_name,
            "semantic_alignment": round(semantic_alignment, 6) if semantic_alignment is not None else None,
            "n_recommendations": len(recs or []),
            "avg_score": dist_metrics.get("avg_score"),
            "top5_avg": dist_metrics.get("top5_avg"),
            "top5_vs_next5_gap": dist_metrics.get("top5_vs_next5_gap"),
        })

    rank_cmp = compare_model_rankings(recs_by_model, top_k=10, rbo_depth=25)
    pair_rows: List[dict] = []
    for pair in rank_cmp.get("pairwise", []):
        pair_rows.append({
            "resume_id": resume_id,
            "filename": file_path.name,
            "category": category,
            "model_a": pair.get("model_a"),
            "model_b": pair.get("model_b"),
            "rbo": pair.get("rbo"),
            "jaccard_at_10": pair.get("jaccard@10"),
            "overlap_at_10": pair.get("overlap@10"),
        })

    t4 = time.time()
    print(f"    [timing] metrics_computation={t4 - t3:.2f}s TOTAL={t4 - t0:.2f}s")

    return model_rows, pair_rows


# ==============================================================================
#  Incremental CSV writers
# ==============================================================================

def _append_rows(path: Path, rows: List[dict], columns: List[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def _load_processed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _mark_processed(path: Path, resume_id: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(resume_id + "\n")


# ==============================================================================
#  Aggregation
# ==============================================================================

def aggregate_model_metrics(per_resume_model_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_resume_model_csv, encoding="utf-8-sig")
    rows = []
    for model_name, group in df.groupby("model"):
        for metric in MODEL_METRIC_COLUMNS:
            stats = _describe(group[metric].dropna().tolist())
            rows.append({"model": model_name, "metric": metric, **stats})
    return pd.DataFrame(rows)


def aggregate_pair_metrics(per_resume_pair_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_resume_pair_csv, encoding="utf-8-sig")
    rows = []
    for (model_a, model_b), group in df.groupby(["model_a", "model_b"]):
        for metric in PAIR_METRIC_COLUMNS:
            stats = _describe(group[metric].dropna().tolist())
            rows.append({"model_a": model_a, "model_b": model_b, "metric": metric, **stats})
    return pd.DataFrame(rows)


# ==============================================================================
#  Main batch runner -- generator, so both CLI and Streamlit can consume progress
# ==============================================================================

def run_batch_evaluation(
    resume_root: str,
    output_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    limit: Optional[int] = None,
    top_k: int = 15,
    chroma_pool: int = 60,
    full_categories: Optional[List[str]] = None,
    other_per_category: int = 3,
    hybrid_weights: Optional[Dict[str, float]] = None,
) -> Iterator[Dict]:
    """
    Generator. Yields a progress dict after every resume.
    Final yield is {"done": True, "summary": {...}, "output_dir": str}
    """
    if resume_from:
        run_dir = Path(resume_from)
        if not run_dir.exists():
            raise FileNotFoundError(f"--resume-from directory not found: {run_dir}")
    else:
        run_id = output_dir or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(output_dir) if output_dir else (EVAL_RESULTS_DIR / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_csv = run_dir / "per_resume_model_metrics.csv"
    pair_csv = run_dir / "per_resume_pair_metrics.csv"
    failures_csv = run_dir / "failures.csv"
    processed_ids_file = run_dir / "processed_ids.txt"

    already_done = _load_processed_ids(processed_ids_file)
    if already_done:
        print(f"[Batch] Resuming run -- {len(already_done)} resumes already completed, will be skipped.")

    resume_root_path = Path(resume_root)

    if full_categories:
        all_files = select_resume_subset(
            resume_root, full_categories=full_categories,
            other_per_category=other_per_category, seed=42,
        )
    else:
        all_files = find_resume_files(resume_root)
        if limit:
            random.seed(42)
            all_files = random.sample(all_files, min(limit, len(all_files)))
            all_files.sort()

    total = len(all_files)
    if total == 0:
        raise RuntimeError(f"No supported resume files (.docx/.pdf/.txt) found under {resume_root}")

    print(f"[Batch] Selected {total} resume file(s) under {resume_root}")

    engines = EngineBundle(chroma_pool=chroma_pool, top_k=top_k, hybrid_weights=hybrid_weights)

    n_ok, n_failed, n_skipped = 0, 0, 0
    start_time = datetime.now()

    for i, file_path in enumerate(all_files, 1):
        resume_id, category = _resume_id_and_category(file_path, resume_root_path)

        if resume_id in already_done:
            n_skipped += 1
            yield {"index": i, "total": total, "resume_id": resume_id, "status": "skipped", "error": None}
            continue

        try:
            model_rows, pair_rows = evaluate_one_resume(file_path, resume_id, category, engines)
            _append_rows(model_csv, model_rows, ["resume_id", "filename", "category", "model"] + MODEL_METRIC_COLUMNS)
            _append_rows(pair_csv, pair_rows, ["resume_id", "filename", "category", "model_a", "model_b"] + PAIR_METRIC_COLUMNS)
            _mark_processed(processed_ids_file, resume_id)
            n_ok += 1
            yield {"index": i, "total": total, "resume_id": resume_id, "status": "ok", "error": None}

        except Exception as exc:
            n_failed += 1
            error_text = f"{type(exc).__name__}: {exc}"
            _append_rows(
                failures_csv,
                [{"resume_id": resume_id, "filename": file_path.name, "category": category, "error": error_text}],
                ["resume_id", "filename", "category", "error"],
            )
            print(f"[Batch] FAILED [{i}/{total}] {resume_id}: {error_text}")
            traceback.print_exc()
            yield {"index": i, "total": total, "resume_id": resume_id, "status": "failed", "error": error_text}

        if i % 25 == 0 or i == total:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"[Batch] Progress: {i}/{total} (ok={n_ok}, failed={n_failed}, skipped={n_skipped}) "
                  f"-- {elapsed:.0f}s elapsed")

    print("[Batch] Aggregating results...")
    agg_model_df = aggregate_model_metrics(model_csv) if model_csv.exists() else pd.DataFrame()
    agg_pair_df = aggregate_pair_metrics(pair_csv) if pair_csv.exists() else pd.DataFrame()

    agg_model_df.to_csv(run_dir / "aggregate_model_metrics.csv", index=False, encoding="utf-8-sig")
    agg_pair_df.to_csv(run_dir / "aggregate_pair_metrics.csv", index=False, encoding="utf-8-sig")

    elapsed_total = (datetime.now() - start_time).total_seconds()
    summary = {
        "resume_root": str(resume_root_path),
        "output_dir": str(run_dir),
        "config": {
            "top_k": top_k,
            "chroma_pool": chroma_pool,
            "full_categories": full_categories,
            "other_per_category": other_per_category,
        },
        "total_selected": total,
        "succeeded": n_ok,
        "failed": n_failed,
        "skipped_already_done": n_skipped,
        "elapsed_seconds": round(elapsed_total, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Batch] Done. {n_ok} succeeded, {n_failed} failed, {n_skipped} skipped. "
          f"Results in {run_dir}")

    yield {
        "done": True,
        "summary": summary,
        "output_dir": str(run_dir),
        "aggregate_model_metrics": agg_model_df,
        "aggregate_pair_metrics": agg_pair_df,
    }


# ==============================================================================
#  CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch-evaluate ColBERT / CrossEncoder / Hybrid (RRF) across a resume dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--resume-root", required=True, help="Root folder containing resumes (recursively scanned)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: auto-timestamped under data/evaluation_results)")
    parser.add_argument("--resume-from", default=None, help="Path to an existing (partial) run directory to resume")
    parser.add_argument("--limit", type=int, default=None, help="Random sample of N resumes from the whole dataset (ignored if --full-categories is set)")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--chroma-pool", type=int, default=60, help="Stage-1 Chroma retrieval pool size, shared across both matchers")
    parser.add_argument("--full-categories", default=None,
                        help="Comma-separated category folder names to include fully, e.g. 'ENGINEERING,INFORMATION-TECHNOLOGY'")
    parser.add_argument("--other-per-category", type=int, default=3,
                        help="How many resumes to sample from each non-full category when --full-categories is set (default: 3)")
    parser.add_argument("--hybrid-weights", default=None,
                        help="Comma-separated model:weight pairs for Hybrid RRF, e.g. 'ColBERT:1.0,CrossEncoder:1.0'. Default: equal weights.")
    args = parser.parse_args()

    parsed_hybrid_weights = None
    if args.hybrid_weights:
        parsed_hybrid_weights = {}
        for pair in args.hybrid_weights.split(","):
            name, weight = pair.split(":")
            parsed_hybrid_weights[name.strip()] = float(weight.strip())

    final_summary = None
    for progress in run_batch_evaluation(
        resume_root=args.resume_root,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        limit=args.limit,
        top_k=args.top_k,
        chroma_pool=args.chroma_pool,
        full_categories=args.full_categories.split(",") if args.full_categories else None,
        other_per_category=args.other_per_category,
        hybrid_weights=parsed_hybrid_weights,
    ):
        if progress.get("done"):
            final_summary = progress["summary"]
        else:
            status = progress["status"]
            marker = {"ok": "+", "failed": "x", "skipped": "-"}.get(status, "?")
            print(f"  [{marker}] [{progress['index']}/{progress['total']}] {progress['resume_id']} ({status})")

    print("\n" + "=" * 70)
    print("  BATCH EVALUATION SUMMARY")
    print("=" * 70)
    print(json.dumps(final_summary, indent=2))
