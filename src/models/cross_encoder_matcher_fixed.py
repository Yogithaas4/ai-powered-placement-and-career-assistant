"""
cross_encoder_matcher_fixed.py
------------------------------
Resume-Job Matching using a Cross-Encoder re-ranker.

KEY CHANGE FROM ORIGINAL:
This version does NOT re-preprocess the resume. It accepts a preprocessed dict
(output of main_pipeline.process_one()) and uses the values directly:
    - query_vector from embeddings (no re-encoding)
    - sections from segmentation (no re-segmentation)
    - query_string from embeddings (no reconstruction)

The model performs:
    Stage 1 — ChromaDB recall using precomputed query_vector (zero computation)
    Stage 2 — Cross-encoder re-ranking on condensed resume (only novel work)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError("pip install sentence-transformers")

try:
    import chromadb
except ImportError:
    raise ImportError("pip install chromadb")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, RECOMMENDATIONS_DIR
from resume_processing.step4_embeddings import build_query_string
from models.base_matcher import BaseMatcher


# ══════════════════════════════════════════════════════════════════════════════
#  Resume Condenser (uses already-segmented sections)
# ══════════════════════════════════════════════════════════════════════════════

def build_condensed_resume(preprocessed: dict, max_chars: int = 700) -> str:
    """
    Build a condensed resume string for cross-encoder input.

    Uses step2 sections (already segmented) and step4 query_string
    (already structured). No re-segmentation or re-extraction.

    Priority order: query_string → summary → skills → experience head
    """
    sections = preprocessed.get("sections", {})
    query_string = preprocessed["embeddings"].get("query_string", "")

    parts = []
    budget = max_chars

    # Lead with the structured query string from step4 (most information-dense)
    if query_string and budget > 0:
        chunk = query_string[:budget]
        parts.append(chunk)
        budget -= len(chunk)

    # Add summary section from step2
    for key in ["summary", "skills", "experience"]:
        text = sections.get(key, "").strip()
        if text and budget > 50:
            chunk = text[:budget]
            parts.append(chunk)
            budget -= len(chunk)

    return "\n".join(parts) if parts else query_string


def _safe_job_index_ce(metadata: dict) -> int:
    j = metadata.get("job_index", -1)
    try:
        v = int(j)
        return v if v >= 0 else -1
    except (TypeError, ValueError):
        return -1


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-Encoder Re-ranker
# ══════════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """MS-MARCO cross-encoder for relevance scoring. Sigmoid-normalised output."""

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        print(f"[CrossEncoder] Loading: {self.model_name}")
        self.model = CrossEncoder(self.model_name, max_length=512)

    def score_pairs(self, query: str, passages: List[str],
                    batch_size: int = 32) -> np.ndarray:
        """Score (query, passage) pairs. Returns sigmoid-normalised scores."""
        pairs = [(query, p) for p in passages]
        raw_scores = self.model.predict(pairs, batch_size=batch_size,
                                        show_progress_bar=False)
        # Sigmoid normalization
        return 1.0 / (1.0 + np.exp(-np.array(raw_scores)))


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-Encoder Engine
# ══════════════════════════════════════════════════════════════════════════════

class CrossEncoderEngine(BaseMatcher):
    """
    Two-stage cross-encoder recommendation engine.

    Stage 1 — ChromaDB Recall:
        Uses precomputed query_vector from embeddings (step4)
        NO re-encoding, NO re-processing
    
    Stage 2 — Cross-Encoder Re-ranking:
        Scores (condensed_resume, job_text) pairs
        Condenses resume from sections (no re-parsing)
    
    Final score = blend of cross-encoder + bi-encoder scores.
    """

    MODEL_NAME = "CrossEncoder"

    # Score blending weights (stronger bi-encoder anchor → closer to shared Chroma order)
    ALPHA_CE = 0.62  # Cross-encoder weight
    ALPHA_BI = 0.38  # Bi-encoder (recall stage) weight

    def __init__(self, jobs_db_dir: Optional[str] = None,
                 jobs_csv: Optional[str] = None,
                 cross_encoder_model: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize CrossEncoder engine.
        
        Args:
            jobs_db_dir: Path to job ChromaDB
            jobs_csv: Path to jobs CSV for metadata
            cross_encoder_model: Custom CE model name
            batch_size: Batch size for CE inference
        """
        super().__init__(jobs_db_dir=jobs_db_dir, jobs_csv=jobs_csv)

        self.jobs_db_dir = jobs_db_dir or str(DATA_DIR / "jobs_db")
        self.jobs_csv = jobs_csv or str(DATA_DIR / "jobs" / "cs_engineering_jobs.csv")
        self.batch_size = batch_size

        # Load job ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.jobs_db_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name="jobs",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[CrossEncoder] Loaded jobs collection ({self.collection.count()} jobs)")
        except Exception as e:
            print(f"[CrossEncoder] Warning: Could not load jobs ChromaDB: {e}")
            self.collection = None

        # Load jobs dataframe for metadata
        self.jobs_df = self._load_jobs_df()
        print(f"[CrossEncoder] Loaded {len(self.jobs_df)} jobs metadata")

        # Initialize cross-encoder
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model)
        print("[CrossEncoder] Engine ready")

    def _load_jobs_df(self) -> pd.DataFrame:
        """Load jobs dataframe from CSV."""
        p = Path(self.jobs_csv)
        return pd.read_csv(p, encoding="utf-8-sig").fillna("") if p.exists() else pd.DataFrame()

    def _build_job_text(self, metadata: dict, job_idx: int) -> str:
        """Build job text for cross-encoding from metadata + CSV description."""
        title = metadata.get("title", "")
        company = metadata.get("company", "")
        domain = metadata.get("domain", "")
        skills = metadata.get("skills", "")
        level = metadata.get("experience_level", "")

        header = f"Job: {title}. Company: {company}. Domain: {domain}. "
        if level:
            header += f"Level: {level}. "
        if skills:
            header += f"Skills: {skills[:200]}. "

        raw_desc = ""
        if 0 <= job_idx < len(self.jobs_df):
            raw_desc = str(self.jobs_df.iloc[job_idx].get("Job Description", ""))[:400]

        return (header + raw_desc)[:700]

    def recommend(self, preprocessed: dict, top_k: int = 20,
                  recall_multiplier: int = 5,
                  stage1_n_results: Optional[int] = None,
                  **kwargs) -> List[Dict]:
        """
        Recommend jobs using cross-encoder re-ranking.

        Args:
            preprocessed: Output of main_pipeline.process_one()
            top_k: Number of final recommendations
            recall_multiplier: Stage 1 pool = top_k * this
            stage1_n_results: If set, fixed Chroma pool size (overrides multiplier)

        Returns:
            List of top_k job recommendations with scores
        """
        # Validate input
        if not self.validate_preprocessed(preprocessed):
            return []

        resume_name = preprocessed.get("filename", "resume")

        # Stage 1 — ChromaDB recall using precomputed query_vector (no re-encoding)
        resume_vec = np.array(preprocessed["embeddings"]["query_vector"])
        print(f"\n[CrossEncoder] Stage 1: ChromaDB recall for {resume_name}")
        print(f"               Vector shape: {resume_vec.shape}")

        if not self.collection:
            print("[CrossEncoder] Error: No jobs collection loaded")
            return []

        if stage1_n_results is not None:
            n_recall = min(int(stage1_n_results), 500)
        else:
            n_recall = min(top_k * recall_multiplier, 300)
        results = self.collection.query(
            query_embeddings=[resume_vec.tolist()],
            n_results=n_recall,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["metadatas"]:
            print("[CrossEncoder] No results from ChromaDB")
            return []

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Build condensed resume from step2 sections (no re-parsing)
        condensed_resume = build_condensed_resume(preprocessed)
        print(f"[CrossEncoder] Condensed resume: {len(condensed_resume)} chars")
        print(f"               Built from step2 sections + step4 query_string")

        # Build job texts and candidate metadata
        job_texts = []
        cand_meta = []
        for metadata, distance in zip(metadatas, distances):
            job_idx = _safe_job_index_ce(metadata)
            job_text = self._build_job_text(metadata, job_idx)
            job_texts.append(job_text)
            cand_meta.append({
                "metadata": metadata,
                "job_idx": job_idx,
                "bi_score": round(1.0 - distance, 4),
            })

        # Stage 2 — Cross-encoder re-ranking
        print(f"[CrossEncoder] Stage 2: Cross-encoding {len(job_texts)} candidates...")
        ce_scores = self.reranker.score_pairs(
            condensed_resume, job_texts, batch_size=self.batch_size
        )

        # Combine scores: CE-dominant with BI-encoder as secondary signal
        final_candidates = []
        for cand, ce_score in zip(cand_meta, ce_scores):
            metadata = cand["metadata"]
            job_idx = int(cand["job_idx"])
            ensemble = (self.ALPHA_CE * float(ce_score) +
                       self.ALPHA_BI * cand["bi_score"])

            desc = ""
            if 0 <= job_idx < len(self.jobs_df):
                desc = str(self.jobs_df.iloc[job_idx].get("Job Description", ""))[:500]

            final_candidates.append({
                "raw_score": ensemble,
                "score": round(ensemble, 4),
                "ce_score": round(float(ce_score), 4),
                "bi_score": cand["bi_score"],
                "job_index": job_idx,
                "title": metadata.get("title", ""),
                "company": metadata.get("company", ""),
                "domain": metadata.get("domain", ""),
                "skills": metadata.get("skills", ""),
                "experience_level": metadata.get("experience_level", ""),
                "location": metadata.get("location", ""),
                "work_type": metadata.get("work_type", ""),
                "salary": metadata.get("salary", ""),
                "source": metadata.get("source", ""),
                "description": desc,
            })

        # Sort and rank
        final_candidates.sort(key=lambda x: x["raw_score"], reverse=True)
        final = []
        for rank, r in enumerate(final_candidates[:top_k], 1):
            r_copy = dict(r)
            r_copy["rank"] = rank
            final.append(r_copy)

        print(f"[CrossEncoder] Done — {len(final)} recommendations")
        return final

    def export(self, recommendations: List[Dict],
               output_path: Optional[str] = None) -> str:
        """Export recommendations to CSV."""
        if not recommendations:
            return ""

        df = pd.DataFrame(recommendations).drop(
            columns=["raw_score"], errors="ignore"
        )

        if output_path is None:
            RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RECOMMENDATIONS_DIR / f"cross_encoder_{ts}.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[CrossEncoder] Saved → {output_path}")
        return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CrossEncoder — accepts preprocessed resume",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--preprocessed-json", required=True,
                        help="Path to preprocessed JSON")
    parser.add_argument("--resume-index", type=int, default=0,
                        help="Index in JSON array")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load preprocessed data
    with open(args.preprocessed_json, "r") as f:
        all_preprocessed = json.load(f)

    if args.resume_index >= len(all_preprocessed):
        print(f"Error: Index {args.resume_index} out of range")
        exit(1)

    preprocessed = all_preprocessed[args.resume_index]
    print(f"[+] Loaded: {preprocessed['filename']}")

    # Initialize engine
    engine = CrossEncoderEngine()

    # Get recommendations
    recs = engine.recommend(preprocessed, top_k=args.top_k)

    # Display
    print("\n" + "=" * 70)
    print(f"  CrossEncoder — TOP {min(10, len(recs))} RECOMMENDATIONS")
    print("=" * 70)
    for r in recs[:10]:
        print(f"  {r['rank']:2}. [{r['score']:.4f}] {r['title']} @ {r['company']}")
        print(f"      Domain: {r['domain']} | Level: {r['experience_level']}")

    # Export
    engine.export(recs, output_path=args.output)
