"""
job_recommendation.py
---------------------
Matches resumes to jobs using semantic similarity and produces ranked recommendations.

Key Functions:
    - RecommendationEngine: Main class for resume-to-job matching
    - recommend_jobs(): Get top N job matches for a resume
    - export_recommendations(): Save recommendations to CSV

Example Usage:
    engine = RecommendationEngine()
    recommendations = engine.recommend_for_resume("resume_text", top_k=20)
    engine.export_recommendations(recommendations, "output.csv")
"""

import json
import platform
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ sentence-transformers not installed. Install via: pip install sentence-transformers")
    SentenceTransformer = None

try:
    import chromadb
except ImportError:
    print("⚠️ chromadb not installed. Install via: pip install chromadb")
    chromadb = None

from config import DATA_DIR, RECOMMENDATIONS_DIR
from job_indexing.job_ingestion import JobIngestionEngine

warnings.filterwarnings("ignore")


class RecommendationEngine:
    """
    Matches resumes to jobs using semantic embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        jobs_db_dir: Optional[str] = None,
        jobs_csv: Optional[str] = None
    ):
        """
        Initialize the recommendation engine.

        Args:
            model_name: Embedding model name
            jobs_db_dir: Directory where job embeddings are stored
            jobs_csv: Path to jobs CSV (if None, uses default)
        """
        self.model_name = model_name
        self.jobs_db_dir = jobs_db_dir or str(DATA_DIR / "jobs_db")
        self.jobs_csv = jobs_csv

        # Initialize embedding model
        if SentenceTransformer:
            device = "cpu" if platform.system() == "Darwin" else None
            print(f"📦 Loading embedding model: {model_name} on {'cpu' if device else 'default device'}")
            self.embedder = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        else:
            self.embedder = None
            raise RuntimeError("SentenceTransformer not available")

        # Initialize ChromaDB client
        if chromadb:
            try:
                # Try new API (ChromaDB 0.4+)
                self.client = chromadb.PersistentClient(path=self.jobs_db_dir)
                print(f"✅ Using ChromaDB PersistentClient")
            except (TypeError, AttributeError):
                # Fallback to in-memory client
                print(f"⚠️ Using in-memory ChromaDB client")
                self.client = chromadb.Client()
            
            try:
                self.collection = self.client.get_collection(name="jobs")
                print(f"✅ Loaded ChromaDB collection with jobs")
            except Exception as e:
                print(f"⚠️ Could not load jobs collection: {e}")
                print(f"   Running ingestion pipeline first...")
                self._ingest_jobs()
        else:
            raise RuntimeError("ChromaDB not available")

        self.jobs_df = None
        self._load_jobs_dataframe()

    def _ingest_jobs(self):
        """Ingest jobs if not already done."""
        engine = JobIngestionEngine(
            model_name=self.model_name,
            persist_dir=self.jobs_db_dir
        )
        self.jobs_df, _ = engine.ingest_pipeline(self.jobs_csv)
        self.collection = engine.collection

    def _load_jobs_dataframe(self):
        """Load jobs DataFrame from CSV."""
        if self.jobs_csv is None:
            self.jobs_csv = DATA_DIR / "jobs" / "cs_engineering_jobs.csv"

        if Path(self.jobs_csv).exists():
            try:
                self.jobs_df = pd.read_csv(self.jobs_csv, encoding="utf-8-sig")
                self.jobs_df = self.jobs_df.fillna("")
                print(f"✅ Loaded {len(self.jobs_df)} jobs from CSV")
            except Exception as e:
                print(f"⚠️ Could not load jobs CSV: {e}")
                self.jobs_df = pd.DataFrame()

    def extract_resume_text(self, resume_dict: Dict) -> str:
        """
        Extract relevant text from resume data structure.

        Args:
            resume_dict: Resume data (from step5_storage or similar)

        Returns:
            Combined resume text
        """
        text_parts = []

        # Extract from entities
        if "entities" in resume_dict:
            entities = resume_dict["entities"]
            if isinstance(entities, dict):
                for key, value in entities.items():
                    if isinstance(value, list):
                        text_parts.extend([str(v) for v in value if v])
                    elif value:
                        text_parts.append(str(value))

        # Extract from sections
        if "sections" in resume_dict:
            sections = resume_dict["sections"]
            if isinstance(sections, dict):
                for key, value in sections.items():
                    if value:
                        text_parts.append(str(value)[:500])  # Limit per section

        # Extract raw text if available
        if "raw_text" in resume_dict:
            text_parts.append(resume_dict["raw_text"][:2000])

        combined = " ".join(text_parts)
        return combined

    def embed_resume(self, resume_text: str) -> np.ndarray:
        """
        Generate embedding for resume text.

        Args:
            resume_text: Resume text to embed

        Returns:
            Embedding vector
        """
        if len(resume_text.strip()) < 10:
            raise ValueError("Resume text too short")

        embedding = self.embedder.encode(resume_text, convert_to_numpy=True)
        return embedding

    def recommend_for_resume(
        self,
        resume_input: str,
        top_k: int = 20,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Get top K job recommendations for a resume.

        Args:
            resume_input: Either resume text or path to resume JSON
            top_k: Number of recommendations to return
            min_score: Minimum similarity score (0-1) to include

        Returns:
            List of recommendations, sorted by match score (highest first)
        """
        print(f"\n🔍 Finding top {top_k} job matches...")

        # If it's a file path, load the JSON
        if Path(resume_input).exists() and resume_input.endswith(".json"):
            with open(resume_input, "r") as f:
                resume_dict = json.load(f)
            resume_text = self.extract_resume_text(resume_dict)
        else:
            # Treat as direct text
            resume_text = resume_input

        # Generate resume embedding
        resume_embedding = self.embed_resume(resume_text)

        # Query ChromaDB for similar jobs
        results = self.collection.query(
            query_embeddings=[resume_embedding],
            n_results=min(top_k * 2, 500),  # Get more than needed, then filter
            include=["documents", "metadatas", "distances"]
        )

        # Process results
        recommendations = []

        if results and results["metadatas"] and len(results["metadatas"]) > 0:
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for metadata, distance in zip(metadatas, distances):
                # Convert distance to similarity score (lower distance = higher similarity)
                # For cosine distance: similarity = 1 - distance
                similarity_score = 1 - distance

                if similarity_score < min_score:
                    continue

                job_idx = metadata.get("job_index", -1)
                job_row = self.jobs_df.iloc[job_idx] if job_idx >= 0 else None

                rec = {
                    "rank": len(recommendations) + 1,
                    "score": round(similarity_score, 4),
                    "title": metadata.get("title", "Unknown"),
                    "company": metadata.get("company", "Unknown"),
                    "domain": metadata.get("domain", ""),
                    "skills": metadata.get("skills", ""),
                    "experience_level": metadata.get("experience_level", ""),
                    "location": metadata.get("location", ""),
                    "work_type": metadata.get("work_type", ""),
                    "salary": metadata.get("salary", ""),
                    "source": metadata.get("source", ""),
                }

                # Add description if we have the job row
                if job_row is not None:
                    desc = str(job_row.get("Job Description", ""))[:500]
                    rec["description"] = desc

                recommendations.append(rec)

                if len(recommendations) >= top_k:
                    break

        print(f"✅ Found {len(recommendations)} matching jobs")

        return recommendations

    def export_recommendations(
        self,
        recommendations: List[Dict],
        output_path: Optional[str] = None,
        resume_name: str = "unknown"
    ) -> str:
        """
        Export recommendations to CSV.

        Args:
            recommendations: List of recommendation dicts
            output_path: Where to save (if None, uses recommendations/)
            resume_name: Name of the resume for the filename

        Returns:
            Path to saved file
        """
        if not recommendations:
            print("⚠️ No recommendations to export")
            return ""

        # Create DataFrame
        df = pd.DataFrame(recommendations)

        # Determine output path
        if output_path is None:
            RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recommendations_{resume_name}_{timestamp}.csv"
            output_path = RECOMMENDATIONS_DIR / filename
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"💾 Recommendations saved → {output_path}")

        return str(output_path)

    def get_recommendation_summary(self, recommendations: List[Dict]) -> Dict:
        """
        Get summary statistics for recommendations.

        Args:
            recommendations: List of recommendation dicts

        Returns:
            Summary dictionary
        """
        if not recommendations:
            return {}

        df = pd.DataFrame(recommendations)

        summary = {
            "total_matches": len(df),
            "avg_score": round(df["score"].mean(), 4),
            "max_score": round(df["score"].max(), 4),
            "min_score": round(df["score"].min(), 4),
            "by_domain": df["domain"].value_counts().to_dict(),
            "by_location": df["location"].value_counts().to_dict(),
            "by_experience": df["experience_level"].value_counts().to_dict(),
        }

        return summary

    def print_recommendations(self, recommendations: List[Dict], top: int = 10):
        """
        Pretty print recommendations.

        Args:
            recommendations: List of recommendation dicts
            top: Number of top recommendations to display
        """
        print("\n" + "="*80)
        print(f"  TOP {min(top, len(recommendations))} JOB RECOMMENDATIONS")
        print("="*80)

        for rec in recommendations[:top]:
            print(f"\n  {rec['rank']}. {rec['title']} @ {rec['company']}")
            print(f"     Domain: {rec['domain']}")
            print(f"     Match Score: {rec['score']*100:.1f}%")
            print(f"     Skills: {rec['skills'][:80]}...")
            print(f"     Location: {rec['location']} ({rec['work_type']})")
            print(f"     Experience: {rec['experience_level']}")
            if rec.get("salary") and rec["salary"] != "Not Disclosed":
                print(f"     Salary: {rec['salary']}")
            print(f"     Source: {rec['source']}")

        print("\n" + "="*80)


# ===================== STANDALONE FUNCTIONS =====================

def recommend_jobs(
    resume_text_or_path: str,
    top_k: int = 20,
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[List[Dict], RecommendationEngine]:
    """
    Quick function to get recommendations for a resume.

    Args:
        resume_text_or_path: Resume text or path to JSON file
        top_k: Number of recommendations
        model_name: Embedding model

    Returns:
        Tuple of (recommendations, engine)
    """
    engine = RecommendationEngine(model_name=model_name)
    recommendations = engine.recommend_for_resume(resume_text_or_path, top_k=top_k)
    return recommendations, engine


# ===================== CLI =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Job recommendation engine")
    parser.add_argument(
        "--resume", type=str, required=True,
        help="Path to resume JSON or text file"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of recommendations (default: 20)"
    )
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (if None, auto-generated)"
    )
    parser.add_argument(
        "--print-top", type=int, default=10,
        help="Number of top recommendations to print"
    )

    args = parser.parse_args()

    # Load and recommend
    engine = RecommendationEngine(model_name=args.model)
    recommendations = engine.recommend_for_resume(args.resume, top_k=args.top_k)

    # Print summary
    engine.print_recommendations(recommendations, top=args.print_top)

    # Export
    if args.output or recommendations:
        resume_name = Path(args.resume).stem
        output_file = engine.export_recommendations(
            recommendations,
            output_path=args.output,
            resume_name=resume_name
        )

        # Print summary stats
        summary = engine.get_recommendation_summary(recommendations)
        print("\n📊 Recommendation Summary:")
        print(f"   Total matches: {summary['total_matches']}")
        print(f"   Avg score: {summary['avg_score']}")
        print(f"   Best match: {summary['max_score']*100:.1f}%")
