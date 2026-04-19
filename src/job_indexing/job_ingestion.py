"""
job_ingestion.py
----------------
Handles ingestion, cleaning, and embedding of job data.
Converts job listings into embeddings stored in ChromaDB for matching.

Functions:
    - load_jobs_from_csv(): Load jobs from CSV file
    - clean_job_text(): Clean and normalize job text
    - extract_job_keywords(): Extract key terms from job descriptions
    - build_job_embeddings(): Generate embeddings for each job
    - store_jobs_in_chromadb(): Save jobs to ChromaDB for matching
    - get_job_stats(): Get statistics about stored jobs
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ sentence-transformers not installed. Install via: pip install sentence-transformers")
    SentenceTransformer = None

# For ChromaDB
try:
    import chromadb
except ImportError:
    print("⚠️ chromadb not installed. Install via: pip install chromadb")
    chromadb = None

from config import JOBS_DIR, DATA_DIR

warnings.filterwarnings("ignore")


class JobIngestionEngine:
    """
    Ingests, processes, and embeds job data for matching with resumes.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", persist_dir: Optional[str] = None):
        """
        Initialize the job ingestion engine.

        Args:
            model_name: Name of the embedding model to use
            persist_dir: Directory to persist ChromaDB (if None, uses memory)
        """
        self.model_name = model_name
        self.persist_dir = persist_dir or str(DATA_DIR / "jobs_db")

        # Initialize embedding model
        if SentenceTransformer:
            print(f"📦 Loading embedding model: {model_name}")
            self.embedder = SentenceTransformer(model_name)
        else:
            self.embedder = None
            print("⚠️ Embedding model unavailable. Some features will be disabled.")

        # Initialize ChromaDB client (PersistentClient matches matchers / evaluation)
        if chromadb:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="jobs",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.client = None
            self.collection = None
            print("⚠️ ChromaDB unavailable. Some features will be disabled.")

        self.jobs_df = None
        self.jobs_metadata = {}

    def load_jobs_from_csv(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load job listings from CSV file.

        Args:
            csv_path: Path to CSV file (if None, looks in data/jobs/)

        Returns:
            DataFrame with job listings
        """
        if csv_path is None:
            csv_path = JOBS_DIR / "cs_engineering_jobs.csv"
        else:
            csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Job CSV not found: {csv_path}")

        print(f"📖 Loading jobs from {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # Ensure required columns
        required_cols = ["Title", "Company", "Job Description", "Skills", "Domain"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Fill NaN values
        df = df.fillna("")

        self.jobs_df = df
        print(f"✅ Loaded {len(df)} jobs")
        return df

    def clean_job_text(self, text: str, max_length: int = 2000) -> str:
        """
        Clean and normalize job text.

        Args:
            text: Raw job text
            max_length: Maximum characters to keep

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove HTML/XML tags
        import re
        text = re.sub(r'<[^>]+>', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep spaces and punctuation
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)

        # Truncate to max length
        text = text.strip()[:max_length]

        return text

    def extract_job_keywords(self, job_row: pd.Series) -> List[str]:
        """
        Extract important keywords from a job listing.

        Args:
            job_row: A row from the jobs DataFrame

        Returns:
            List of extracted keywords
        """
        keywords = set()

        # Add domain
        if job_row.get("Domain"):
            keywords.add(job_row["Domain"].lower())

        # Add skills (parsed as comma-separated)
        skills = job_row.get("Skills", "")
        if skills and skills != "Not Listed":
            skill_list = [s.strip() for s in skills.split(",")]
            keywords.update([s.lower() for s in skill_list if s])

        # Add experience level
        exp = job_row.get("Experience Level", "")
        if exp and exp != "Not Specified":
            keywords.add(exp.lower())

        # Add location
        location = job_row.get("Location", "")
        if location:
            keywords.add(location.lower())

        # Add work type
        work_type = job_row.get("Work Type", "")
        if work_type:
            keywords.add(work_type.lower())

        return list(keywords)

    def build_job_embeddings(self, batch_size: int = 32) -> Dict[int, np.ndarray]:
        """
        Generate embeddings for all jobs.

        Args:
            batch_size: Batch size for embedding generation

        Returns:
            Dictionary mapping job index to embedding vector
        """
        if self.jobs_df is None:
            raise ValueError("No jobs loaded. Call load_jobs_from_csv() first.")

        if not self.embedder:
            raise RuntimeError("Embedder not available. Install sentence-transformers.")

        embeddings = {}
        total = len(self.jobs_df)

        print(f"🔧 Generating embeddings for {total} jobs (batch_size={batch_size})...")

        for idx in range(0, total, batch_size):
            batch_end = min(idx + batch_size, total)
            batch_jobs = self.jobs_df.iloc[idx:batch_end]

            # Create job text: title + description + skills
            texts = []
            for _, job in batch_jobs.iterrows():
                text = f"{job['Title']} {job['Domain']} {job['Job Description']} {job['Skills']}"
                text = self.clean_job_text(text)
                texts.append(text)

            # Embed batch
            batch_embeddings = self.embedder.encode(texts, convert_to_numpy=True)

            for i, emb in enumerate(batch_embeddings):
                embeddings[idx + i] = emb

            if (batch_end) % (batch_size * 5) == 0 or batch_end == total:
                print(f"  ✓ Processed {batch_end}/{total} jobs")

        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings

    def store_jobs_in_chromadb(self, embeddings: Dict[int, np.ndarray]) -> int:
        """
        Store jobs and their embeddings in ChromaDB.

        Args:
            embeddings: Dictionary mapping job index to embedding vector

        Returns:
            Number of jobs stored
        """
        if not self.collection:
            raise RuntimeError("ChromaDB not available.")

        if self.jobs_df is None:
            raise ValueError("No jobs loaded. Call load_jobs_from_csv() first.")

        print(f"💾 Storing {len(embeddings)} jobs in ChromaDB...")

        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for idx, emb in embeddings.items():
            job = self.jobs_df.iloc[idx]

            job_id = f"job_{idx}"
            ids.append(job_id)

            # Document: combine title + description for text search
            doc = f"{job['Title']} {job['Job Description']}"
            documents.append(self.clean_job_text(doc))

            # Metadata: structured job information
            metadata = {
                "title": str(job["Title"])[:500],
                "company": str(job["Company"])[:200],
                "domain": str(job["Domain"])[:100],
                "skills": str(job["Skills"])[:500],
                "experience_level": str(job.get("Experience Level", ""))[:100],
                "salary": str(job.get("Salary", ""))[:100],
                "location": str(job.get("Location", ""))[:100],
                "work_type": str(job.get("Work Type", ""))[:50],
                "source": str(job.get("Source", ""))[:50],
                "job_index": idx
            }
            metadatas.append(metadata)

            embeddings_list.append(emb)

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )

        print(f"✅ Stored {len(ids)} jobs in ChromaDB")

        # Store metadata for later retrieval
        self.jobs_metadata = {i: m for i, m in zip(ids, metadatas)}

        return len(ids)

    def ingest_pipeline(self, csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the full ingestion pipeline: load → clean → embed → store.

        Args:
            csv_path: Path to jobs CSV

        Returns:
            Tuple of (jobs DataFrame, stored count)
        """
        print("\n" + "="*60)
        print("  JOB INGESTION PIPELINE")
        print("="*60)

        # Load
        df = self.load_jobs_from_csv(csv_path)

        # Embed
        embeddings = self.build_job_embeddings()

        # Store
        stored = self.store_jobs_in_chromadb(embeddings)

        # Stats
        stats = self.get_job_stats()

        print(f"\n{'='*60}")
        print(f"  ✅ Ingestion complete: {stored} jobs ready for matching")
        print(f"{'='*60}\n")

        return df, stats

    def get_job_stats(self) -> Dict:
        """
        Get statistics about stored jobs.

        Returns:
            Dictionary with job statistics
        """
        if self.jobs_df is None:
            return {}

        stats = {
            "total_jobs": len(self.jobs_df),
            "by_domain": self.jobs_df["Domain"].value_counts().to_dict(),
            "by_source": self.jobs_df.get("Source", pd.Series()).value_counts().to_dict(),
            "by_experience": self.jobs_df.get("Experience Level", pd.Series()).value_counts().to_dict(),
            "by_location": self.jobs_df.get("Location", pd.Series()).value_counts().to_dict(),
        }

        print("\n📊 Job Statistics:")
        print(f"  Total jobs: {stats['total_jobs']}")
        print(f"\n  By Domain:")
        for domain, count in stats["by_domain"].items():
            print(f"    • {domain}: {count}")
        print(f"\n  By Source:")
        for source, count in stats["by_source"].items():
            print(f"    • {source}: {count}")

        return stats


# ===================== STANDALONE FUNCTIONS =====================

def quick_ingest(csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, JobIngestionEngine]:
    """
    Quick ingestion function for CLI use.

    Usage:
        df, engine = quick_ingest()  # Uses default path
        df, engine = quick_ingest("path/to/jobs.csv")
    """
    engine = JobIngestionEngine()
    df, stats = engine.ingest_pipeline(csv_path)
    return df, engine


# ===================== CLI =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Job data ingestion pipeline")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to jobs CSV file (default: data/jobs/cs_engineering_jobs.csv)"
    )
    parser.add_argument(
        "--model", type=str, default="BAAI/bge-large-en-v1.5",
        help="Embedding model name"
    )
    parser.add_argument(
        "--persist-dir", type=str, default=None,
        help="ChromaDB persistence directory"
    )

    args = parser.parse_args()

    engine = JobIngestionEngine(model_name=args.model, persist_dir=args.persist_dir)
    df, stats = engine.ingest_pipeline(args.csv)
