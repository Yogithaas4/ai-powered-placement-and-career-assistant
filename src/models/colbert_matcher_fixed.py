"""
colbert_matcher_fixed.py
------------------------
Resume-Job Matching using ColBERT-style Late Interaction.

Based on: "ColBERT: Efficient and Effective Passage Search via Contextualized
Late Interaction over BERT" ArXiv: https://arxiv.org/abs/2004.12832

KEY CHANGE FROM ORIGINAL:
This version does NOT re-preprocess the resume. It accepts a preprocessed dict
(output of main_pipeline.process_one()) and uses the values directly:
    - query_vector from embeddings (no re-encoding)
    - query_string from embeddings (no reconstruction)
    
The model performs:
    Stage 1 — ChromaDB recall using precomputed query_vector (zero computation)
    Stage 2 — ColBERT token-level MaxSim re-ranking (only novel work)
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    raise ImportError("pip install torch transformers")

try:
    import chromadb
except ImportError:
    raise ImportError("pip install chromadb")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, RECOMMENDATIONS_DIR
from resume_processing.step4_embeddings import build_query_string
from models.base_matcher import BaseMatcher

# After MaxSim, re-anchor to Chroma bi-encoder similarity so top lists track the
# same vector retrieval signal as ConFit / CrossEncoder (higher cross-model Jaccard).
COLBERT_BI_BLEND = 0.35


def _safe_job_index_colbert(metadata: dict) -> int:
    j = metadata.get("job_index", -1)
    try:
        v = int(j)
        return v if v >= 0 else -1
    except (TypeError, ValueError):
        return -1


# ══════════════════════════════════════════════════════════════════════════════
#  ColBERT Token Encoder
# ══════════════════════════════════════════════════════════════════════════════

class ColBERTEncoder:
    """
    Per-token BERT embeddings for ColBERT late interaction.

    Prepends [unused0] for queries (resume), [unused1] for documents (jobs)
    as in the original ColBERT paper. Vectors are L2-normalised per token.
    """

    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 180
    QUERY_MAX_LEN = 64   # resume query is short

    def __init__(self):
        print(f"[ColBERT] Loading BERT: {self.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[ColBERT] Device: {self.device}")

    @torch.no_grad()
    def encode(self, text: str, max_length: int = MAX_LENGTH,
               is_query: bool = False) -> np.ndarray:
        """
        Encode text → (n_tokens, 768) L2-normalised token matrix.
        is_query=True uses [unused0] marker (resume side).
        is_query=False uses [unused1] marker (job side).
        """
        marker = "[unused0]" if is_query else "[unused1]"
        enc = self.tokenizer(
            f"{marker} {text}", max_length=max_length,
            truncation=True, padding=False, return_tensors="pt",
        ).to(self.device)

        out = self.model(**enc)
        vecs = out.last_hidden_state[0].cpu().numpy()
        input_ids = enc["input_ids"][0].cpu().numpy()
        mask = input_ids != self.tokenizer.pad_token_id
        vecs = vecs[mask]

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return vecs / norms


# ══════════════════════════════════════════════════════════════════════════════
#  MaxSim Scoring
# ══════════════════════════════════════════════════════════════════════════════

def maxsim_score(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """
    ColBERT MaxSim: Σ_i max_j cosine_sim(q_i, d_j)
    Vectors are already L2-normalised so dot product = cosine sim.
    """
    sim = np.dot(query_vecs, doc_vecs.T)   # (M, N)
    return float(sim.max(axis=1).sum())


# ══════════════════════════════════════════════════════════════════════════════
#  Job Token Index (pre-computed, cached)
# ══════════════════════════════════════════════════════════════════════════════

class JobTokenIndex:
    """Pre-computes and caches ColBERT token matrices for all jobs (offline indexing)."""

    CACHE_FILE = DATA_DIR / "processed" / "colbert_job_token_index.pkl"

    def __init__(self, encoder: ColBERTEncoder, jobs_df: pd.DataFrame):
        self.encoder = encoder
        self.jobs_df = jobs_df
        self.index: Dict[int, np.ndarray] = {}

    def build(self, max_jobs: int = 2000, force_rebuild: bool = False):
        """Build token index for all jobs (or load from cache)."""
        if self.CACHE_FILE.exists() and not force_rebuild:
            print(f"[ColBERT Index] Loading cache: {self.CACHE_FILE}")
            with open(self.CACHE_FILE, "rb") as f:
                self.index = pickle.load(f)
            print(f"[ColBERT Index] {len(self.index)} jobs loaded from cache")
            return

        n = min(max_jobs, len(self.jobs_df))
        print(f"[ColBERT Index] Building token index for {n} jobs...")
        for i in range(n):
            row = self.jobs_df.iloc[i]
            title = str(row.get("Job Title", ""))
            desc = str(row.get("Job Description", ""))[:1000]
            skills = str(row.get("Skills Required", ""))
            text = f"{title}. {skills}. {desc}"
            try:
                self.index[i] = self.encoder.encode(text, is_query=False)
            except Exception as e:
                print(f"[ColBERT Index] Warning: job {i}: {e}")
            if (i + 1) % 200 == 0:
                print(f"[ColBERT Index]   {i + 1}/{n}...")

        self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CACHE_FILE, "wb") as f:
            pickle.dump(self.index, f)
        print(f"[ColBERT Index] Saved → {self.CACHE_FILE}")

    def get(self, job_idx: int) -> Optional[np.ndarray]:
        """Retrieve pre-computed token vectors for a job."""
        return self.index.get(job_idx)


# ══════════════════════════════════════════════════════════════════════════════
#  ColBERT Engine
# ══════════════════════════════════════════════════════════════════════════════

class ColBERTEngine(BaseMatcher):
    """
    Two-stage ColBERT resume-to-job matching engine.

    Stage 1 — ChromaDB Recall:
        Uses precomputed query_vector from embeddings (step4)
        NO re-encoding, NO re-processing
    
    Stage 2 — ColBERT MaxSim Re-ranking:
        Encodes resume query text (from embeddings["query_string"])
        Scores against job token matrices using MaxSim
    
    Final scoring combines both stages.
    """

    MODEL_NAME = "ColBERT"

    def __init__(self, jobs_db_dir: Optional[str] = None,
                 jobs_csv: Optional[str] = None,
                 max_index_jobs: int = 2000,
                 force_rebuild_index: bool = False):
        """
        Initialize ColBERT engine.
        
        Args:
            jobs_db_dir: Path to job ChromaDB
            jobs_csv: Path to jobs CSV for metadata
            max_index_jobs: Max jobs to index (for performance)
            force_rebuild_index: Force rebuild of token index
        """
        super().__init__(jobs_db_dir=jobs_db_dir, jobs_csv=jobs_csv)

        # Load job ChromaDB
        self.jobs_db_dir = jobs_db_dir or str(DATA_DIR / "jobs_db")
        self.jobs_csv = jobs_csv or str(DATA_DIR / "jobs" / "all_jobs_v3_fixed.csv")

        try:
            self.chroma_client = chromadb.PersistentClient(path=self.jobs_db_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name="jobs",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[ColBERT] Loaded jobs collection ({self.collection.count()} jobs)")
        except Exception as e:
            print(f"[ColBERT] Warning: Could not load jobs ChromaDB: {e}")
            self.collection = None

        # Load jobs dataframe for metadata
        self.jobs_df = self._load_jobs_df()
        print(f"[ColBERT] Loaded {len(self.jobs_df)} jobs metadata")

        # Initialize BERT encoder and build job token index
        self.col_encoder = ColBERTEncoder()
        self.job_index = JobTokenIndex(self.col_encoder, self.jobs_df)
        self.job_index.build(max_jobs=max_index_jobs, force_rebuild=force_rebuild_index)

        print("[ColBERT] Engine ready")

    def _load_jobs_df(self) -> pd.DataFrame:
        """Load jobs dataframe from CSV."""
        p = Path(self.jobs_csv)
        return pd.read_csv(p, encoding="utf-8-sig").fillna("") if p.exists() else pd.DataFrame()

    def recommend(self, preprocessed: dict, top_k: int = 20,
                  recall_multiplier: int = 5,
                  stage1_n_results: Optional[int] = None,
                  **kwargs) -> List[Dict]:
        """
        Recommend jobs using ColBERT late interaction.

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
        print(f"\n[ColBERT] Stage 1: ChromaDB recall for {resume_name}")
        print(f"          Vector shape: {resume_vec.shape}")

        if not self.collection:
            print("[ColBERT] Error: No jobs collection loaded")
            return []

        if stage1_n_results is not None:
            n_recall = min(int(stage1_n_results), 500)
        else:
            n_recall = min(top_k * recall_multiplier, 500)
        results = self.collection.query(
            query_embeddings=[resume_vec.tolist()],
            n_results=n_recall,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["metadatas"]:
            print("[ColBERT] No results from ChromaDB")
            return []

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Stage 2 — ColBERT token-level re-ranking
        # Use query_string from embeddings (already structured by step4)
        query_text = preprocessed["embeddings"].get("query_string", "")
        if not query_text:
            # Fallback: rebuild from entities + sections
            query_text = build_query_string(
                preprocessed["entities"],
                preprocessed["sections"]
            )

        print(f"[ColBERT] Stage 2: MaxSim re-ranking on {len(metadatas)} candidates")
        print(f"          Query: {query_text[:100]}...")

        # Encode resume query tokens (only once)
        resume_token_vecs = self.col_encoder.encode(
            query_text, max_length=ColBERTEncoder.QUERY_MAX_LEN, is_query=True
        )
        print(f"          Resume tokens: {resume_token_vecs.shape[0]}")

        # Score each job using MaxSim
        reranked = []
        for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
            job_idx = _safe_job_index_colbert(metadata)
            job_token_vecs = self.job_index.get(job_idx)

            if job_token_vecs is None:
                # Fallback to bi-encoder score if token index missing
                colbert_raw = 1.0 - distance
            else:
                # ColBERT MaxSim score
                raw = maxsim_score(resume_token_vecs, job_token_vecs)
                colbert_raw = raw / max(len(resume_token_vecs), 1)

            # Get job description
            desc = ""
            if 0 <= job_idx < len(self.jobs_df):
                desc = str(self.jobs_df.iloc[job_idx].get("Job Description", ""))[:500]

            bi = float(1.0 - distance)
            reranked.append({
                "_colbert_raw": colbert_raw,
                "bi_score": round(bi, 4),
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
                "job_index": job_idx,
            })

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(metadatas)}]")

        if not reranked:
            return []

        raws = [r["_colbert_raw"] for r in reranked]
        mn, mx = min(raws), max(raws)
        span = (mx - mn) or 1.0
        w_bi = COLBERT_BI_BLEND
        for r in reranked:
            norm_c = (r["_colbert_raw"] - mn) / span
            combined = (1.0 - w_bi) * norm_c + w_bi * r["bi_score"]
            r["raw_score"] = combined
            r["score"] = round(combined, 4)
            r["maxsim_norm"] = round(norm_c, 4)
            del r["_colbert_raw"]

        # Sort and rank
        reranked.sort(key=lambda x: x["raw_score"], reverse=True)
        final = []
        for rank, r in enumerate(reranked[:top_k], 1):
            r_copy = dict(r)
            r_copy["rank"] = rank
            final.append(r_copy)

        print(f"[ColBERT] Done — {len(final)} recommendations")
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
            output_path = RECOMMENDATIONS_DIR / f"colbert_{ts}.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[ColBERT] Saved → {output_path}")
        return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ColBERT — accepts preprocessed resume",
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
    engine = ColBERTEngine()

    # Get recommendations
    recs = engine.recommend(preprocessed, top_k=args.top_k)

    # Display
    print("\n" + "=" * 70)
    print(f"  ColBERT — TOP {min(10, len(recs))} RECOMMENDATIONS")
    print("=" * 70)
    for r in recs[:10]:
        print(f"  {r['rank']:2}. [{r['score']:.4f}] {r['title']} @ {r['company']}")
        print(f"      Domain: {r['domain']} | Level: {r['experience_level']}")

    # Export
    engine.export(recs, output_path=args.output)
