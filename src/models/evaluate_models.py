"""
evaluate_models.py
------------------
Unified runner to compare all 3 models (+ 3 ConFit v2 variants) on one resume.

Pipeline integration
--------------------
Runs the resume through steps 1-4 ONCE, then passes the preprocessed dict
to all models. No re-parsing, no re-embedding per model.

Usage:
    # Run pipeline first (if not done):
    python src/main_pipeline.py --folder data/resumes

    # Compare all models:
    python src/models/evaluate_models.py --resume arjun_sharma_resume.docx

    # Compare only some models:
    python src/models/evaluate_models.py --resume arjun_sharma_resume.docx \
        --models confit_rule confit_llm cross

    # With ground-truth titles for metric computation:
    python src/models/evaluate_models.py --resume arjun_sharma_resume.docx \
        --relevant-titles "Software Engineer" "Backend Developer" "ML Engineer"
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RECOMMENDATIONS_DIR
from resume_processing.main_pipeline import process_one
from resume_processing.step1_parser import parse_file

from models.confit_v2 import (
    ConFitV2Engine,
    evaluate as confit_evaluate,
)
from models.colbert_matcher import (
    ColBERTEngine,
    evaluate as colbert_evaluate,
)
from models.cross_encoder_matcher import (
    CrossEncoderEngine,
    evaluate as ce_evaluate,
)

ALL_MODELS = ["confit_rule", "confit_llm", "confit_local", "colbert", "cross"]


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline loader — runs ONCE, shared across all models
# ══════════════════════════════════════════════════════════════════════════════

def load_preprocessed(resume_filename: str) -> dict:
    """
    Run pipeline steps 1-4 for a resume and return the preprocessed dict.
    This is called once and the result is passed to all models.
    """
    resume_path = DATA_DIR / "resumes" / resume_filename
    if not resume_path.exists():
        raise FileNotFoundError(
            f"Resume not found: {resume_path}\n"
            f"Place the resume file in data/resumes/"
        )
    print(f"\n[Pipeline] Processing: {resume_filename}")
    print("[Pipeline] Running steps 1 (parse) → 2 (segment) → 3 (NER) → 4 (embed)...")
    raw_text = parse_file(str(resume_path))
    parsed   = {"filename": resume_filename, "raw_text": raw_text,
                "file_type": resume_path.suffix.lstrip(".")}
    preprocessed = process_one(parsed)

    print(f"[Pipeline] ✓ query_vector: {len(preprocessed['embeddings']['query_vector'])} dims")
    print(f"[Pipeline] ✓ skills extracted: {len(preprocessed['entities'].get('skills', []))}")
    print(f"[Pipeline] ✓ sections found: {[k for k,v in preprocessed['sections'].items() if v]}")
    print(f"[Pipeline] ✓ query_string: {preprocessed['embeddings']['query_string'][:100]}...")
    return preprocessed


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison runner
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(
    resume_filename: str,
    top_k: int = 20,
    relevant_job_titles: Optional[List[str]] = None,
    models_to_run: Optional[List[str]] = None,
    k_values: List[int] = [5, 10, 20],
    export_individual: bool = True,
    llm_api_key: Optional[str] = None,
    local_model: str = "mistral",
) -> Dict:
    """
    Run selected models on a resume and produce a comparison table.

    The preprocessing pipeline runs ONCE and results are shared across models.

    Args:
        resume_filename:     e.g. "arjun_sharma_resume.docx"
        top_k:               Recommendations per model
        relevant_job_titles: Ground-truth titles for metric computation (optional)
        models_to_run:       Subset of ALL_MODELS. None = all.
        k_values:            K values for Recall@K / Precision@K / nDCG@K
        export_individual:   Save per-model CSV files
        llm_api_key:         Anthropic API key (for confit_llm)
        local_model:         Ollama model name (for confit_local)
    """
    if models_to_run is None:
        models_to_run = ALL_MODELS

    resume_name = Path(resume_filename).stem

    # ── Run pipeline ONCE ─────────────────────────────────────────────────────
    preprocessed = load_preprocessed(resume_filename)

    all_results: Dict[str, List[Dict]] = {}
    all_metrics: Dict[str, Dict]       = {}

    # ── ConFit v2 (rule-based) ────────────────────────────────────────────────
    if "confit_rule" in models_to_run:
        _print_banner("MODEL 1: ConFit v2 — Rule-Based HRE (step2/step3 data)")
        try:
            engine = ConFitV2Engine(hre_mode="rule")
            recs   = engine.recommend(preprocessed, top_k=top_k)
            all_results["ConFit v2 (rule)"] = recs
            if export_individual:
                engine.export(recs, resume_name=resume_name)
            if relevant_job_titles:
                all_metrics["ConFit v2 (rule)"] = confit_evaluate(recs, relevant_job_titles, k_values)
        except Exception as e:
            print(f"[ConFit rule] ❌ {e}")
            all_results["ConFit v2 (rule)"] = []

    # ── ConFit v2 (LLM) ───────────────────────────────────────────────────────
    if "confit_llm" in models_to_run:
        _print_banner("MODEL 2: ConFit v2 — LLM HRE (Claude API)")
        try:
            engine = ConFitV2Engine(hre_mode="llm", llm_api_key=llm_api_key)
            recs   = engine.recommend(preprocessed, top_k=top_k)
            all_results["ConFit v2 (llm)"] = recs
            if export_individual:
                engine.export(recs, resume_name=resume_name)
            if relevant_job_titles:
                all_metrics["ConFit v2 (llm)"] = confit_evaluate(recs, relevant_job_titles, k_values)
        except Exception as e:
            print(f"[ConFit llm] ❌ {e}")
            all_results["ConFit v2 (llm)"] = []

    # ── ConFit v2 (local) ─────────────────────────────────────────────────────
    if "confit_local" in models_to_run:
        _print_banner(f"MODEL 3: ConFit v2 — Local LLM HRE (Ollama/{local_model})")
        try:
            engine = ConFitV2Engine(hre_mode="local", local_model=local_model)
            recs   = engine.recommend(preprocessed, top_k=top_k)
            all_results["ConFit v2 (local)"] = recs
            if export_individual:
                engine.export(recs, resume_name=resume_name)
            if relevant_job_titles:
                all_metrics["ConFit v2 (local)"] = confit_evaluate(recs, relevant_job_titles, k_values)
        except Exception as e:
            print(f"[ConFit local] ❌ {e}")
            all_results["ConFit v2 (local)"] = []

    # ── ColBERT ───────────────────────────────────────────────────────────────
    if "colbert" in models_to_run:
        _print_banner("MODEL 4: ColBERT — Late Interaction MaxSim")
        try:
            engine = ColBERTEngine()
            recs   = engine.recommend(preprocessed, top_k=top_k)
            all_results["ColBERT"] = recs
            if export_individual:
                engine.export(recs, resume_name=resume_name)
            if relevant_job_titles:
                all_metrics["ColBERT"] = colbert_evaluate(recs, relevant_job_titles, k_values)
        except Exception as e:
            print(f"[ColBERT] ❌ {e}")
            all_results["ColBERT"] = []

    # ── Cross-Encoder ─────────────────────────────────────────────────────────
    if "cross" in models_to_run:
        _print_banner("MODEL 5: Cross-Encoder — Joint Attention Re-ranker")
        try:
            engine = CrossEncoderEngine()
            recs   = engine.recommend(preprocessed, top_k=top_k)
            all_results["CrossEncoder"] = recs
            if export_individual:
                engine.export(recs, resume_name=resume_name)
            if relevant_job_titles:
                all_metrics["CrossEncoder"] = ce_evaluate(recs, relevant_job_titles, k_values)
        except Exception as e:
            print(f"[CrossEncoder] ❌ {e}")
            all_results["CrossEncoder"] = []

    # ── Build comparison table ────────────────────────────────────────────────
    rows = []
    for model_name, recs in all_results.items():
        row = {"model": model_name, "n_results": len(recs)}
        if recs:
            scores = [r.get("score", 0) for r in recs]
            row["avg_score"] = round(float(np.mean(scores)), 4)
            row["max_score"] = round(float(np.max(scores)), 4)
            row["score_std"] = round(float(np.std(scores)), 4)
            row["top5"]      = " | ".join(r.get("title","")[:25] for r in recs[:5])
        if model_name in all_metrics:
            m = all_metrics[model_name]
            for k in k_values:
                row[f"recall@{k}"]    = m.get(f"recall@{k}", "N/A")
                row[f"precision@{k}"] = m.get(f"precision@{k}", "N/A")
                row[f"ndcg@{k}"]      = m.get(f"ndcg@{k}", "N/A")
            row["mrr"] = m.get("mrr", "N/A")
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    _print_comparison(comparison_df, k_values, relevant_job_titles is not None)

    # ── Export ────────────────────────────────────────────────────────────────
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_path = RECOMMENDATIONS_DIR / f"evaluation_comparison_{resume_name}_{ts}.csv"
    comparison_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Comparison saved → {comp_path}")

    return {
        "results":         all_results,
        "metrics":         all_metrics,
        "comparison_df":   comparison_df,
        "comparison_path": str(comp_path),
        "preprocessed":    preprocessed,
    }


def _print_banner(title: str):
    print("\n" + "━"*60)
    print(f"  {title}")
    print("━"*60)


def _print_comparison(df, k_values, has_metrics):
    print("\n" + "═"*75)
    print("  COMPARISON RESULTS")
    print("═"*75)

    print(f"\n{'Model':<25} {'N':>5} {'Avg Score':>10} {'Max Score':>10} {'Std':>8}")
    print("  " + "-"*58)
    for _, row in df.iterrows():
        print(f"  {row['model']:<23} {row.get('n_results',0):>5} "
              f"{row.get('avg_score','N/A'):>10} {row.get('max_score','N/A'):>10} "
              f"{row.get('score_std','N/A'):>8}")

    if has_metrics:
        for k in k_values:
            print(f"\n  @ k={k}:  {'Model':<23} {'Recall':>8} {'Precision':>10} {'nDCG':>8}")
            print("  " + "-"*52)
            for _, row in df.iterrows():
                print(f"           {row['model']:<23} "
                      f"{row.get(f'recall@{k}','N/A'):>8} "
                      f"{row.get(f'precision@{k}','N/A'):>10} "
                      f"{row.get(f'ndcg@{k}','N/A'):>8}")
        print(f"\n  MRR: ", end="")
        for _, row in df.iterrows():
            print(f"{row['model']}={row.get('mrr','N/A')}  ", end="")
        print()

    print("\n  Top-5 per model:")
    for _, row in df.iterrows():
        print(f"  [{row['model']}]")
        for i, t in enumerate(str(row.get('top5','')).split(" | "), 1):
            print(f"    {i}. {t}")

    print("\n" + "═"*75)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare all models — pipeline runs once, shared across models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--resume", required=True,
                        help="Resume filename in data/resumes/ (e.g. arjun_sharma_resume.docx)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=ALL_MODELS,
        help=(
            "Models to run (default: all):\n"
            "  confit_rule  — ConFit v2 with rule-based HRE\n"
            "  confit_llm   — ConFit v2 with Claude API HRE\n"
            "  confit_local — ConFit v2 with local Ollama HRE\n"
            "  colbert      — ColBERT MaxSim\n"
            "  cross        — Cross-Encoder"
        ),
    )
    parser.add_argument("--relevant-titles", nargs="+", default=None,
                        help="Known relevant job titles for metric computation")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--api-key", default=None, help="Anthropic API key (confit_llm)")
    parser.add_argument("--local-model", default="mistral", help="Ollama model (confit_local)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip per-model CSV exports")
    args = parser.parse_args()

    print(f"\n🚀 Models: {args.models or ALL_MODELS}")
    print(f"   Resume: {args.resume} | Top-K: {args.top_k}")

    run_comparison(
        resume_filename    = args.resume,
        top_k              = args.top_k,
        relevant_job_titles= args.relevant_titles,
        models_to_run      = args.models,
        k_values           = args.k_values,
        export_individual  = not args.no_export,
        llm_api_key        = args.api_key,
        local_model        = args.local_model,
    )
