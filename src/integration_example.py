"""
integration_example.py
---------------------
Complete example showing how to use the job ingestion and recommendation modules.

This script demonstrates:
    1. Loading and ingesting job data
    2. Processing resumes
    3. Generating job recommendations
    4. Exporting results

Usage:
    python integration_example.py                    # Process all resumes
    python integration_example.py --resume-name "john_doe"
    python integration_example.py --top-k 50         # Get top 50 recommendations
"""

import json
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd

from config import RESUMES_DIR, RECOMMENDATIONS_DIR, PROCESSED_DIR
from job_indexing.job_ingestion import JobIngestionEngine
from job_recommendation import RecommendationEngine


class JobMatchingPipeline:
    """
    Complete pipeline for matching resumes to jobs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the pipeline."""
        self.model_name = model_name
        self.ingestion_engine = None
        self.recommendation_engine = None

    def run_full_pipeline(
        self,
        ingest_jobs: bool = True,
        resume_name: Optional[str] = None,
        top_k: int = 20
    ) -> Dict:
        """
        Run the complete pipeline.

        Args:
            ingest_jobs: Whether to re-ingest job data
            resume_name: Specific resume to process (if None, processes first resume)
            top_k: Number of recommendations

        Returns:
            Dictionary with results
        """
        results = {}

        # ──────────────────────────────────────────────────────────
        # STEP 1: Ingest Jobs
        # ──────────────────────────────────────────────────────────
        if ingest_jobs:
            print("\n" + "="*70)
            print("STEP 1: INGESTING JOB DATA")
            print("="*70)

            self.ingestion_engine = JobIngestionEngine(model_name=self.model_name)
            jobs_df, stats = self.ingestion_engine.ingest_pipeline()

            results["jobs_ingested"] = len(jobs_df)
            results["job_stats"] = stats

        # ──────────────────────────────────────────────────────────
        # STEP 2: Initialize Recommendation Engine
        # ──────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("STEP 2: INITIALIZING RECOMMENDATION ENGINE")
        print("="*70)

        self.recommendation_engine = RecommendationEngine(model_name=self.model_name)

        # ──────────────────────────────────────────────────────────
        # STEP 3: Get Resumes to Process
        # ──────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("STEP 3: LOADING RESUMES")
        print("="*70)

        if resume_name:
            # Process specific resume
            resume_path = PROCESSED_DIR / f"{resume_name}_preprocessed.json"
            if not resume_path.exists():
                # Try alternate path
                resume_path = PROCESSED_DIR / "output_preprocessed.json"
                if not resume_path.exists():
                    print(f"❌ Resume not found: {resume_name}")
                    return results

            resumes_to_process = [resume_path]
        else:
            # Process all resumes from output_preprocessed.json
            resume_path = PROCESSED_DIR / "output_preprocessed.json"
            if resume_path.exists():
                with open(resume_path) as f:
                    resume_data = json.load(f)
                    if isinstance(resume_data, list):
                        resumes_to_process = [resume_path]  # Use single file with all
                    else:
                        resumes_to_process = [resume_path]
            else:
                print(f"❌ No processed resumes found at {resume_path}")
                return results

        print(f"✅ Found {len(resumes_to_process)} resume(s) to process")

        # ──────────────────────────────────────────────────────────
        # STEP 4: Generate Recommendations
        # ──────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("STEP 4: GENERATING RECOMMENDATIONS")
        print("="*70)

        all_recommendations = {}

        for resume_path in resumes_to_process:
            with open(resume_path) as f:
                resume_data = json.load(f)

            # Handle both single resume and list of resumes
            if isinstance(resume_data, list):
                resume_list = resume_data
            else:
                resume_list = [resume_data]

            for resume_entry in resume_list:
                filename = resume_entry.get("filename", "unknown")
                resume_name_clean = Path(filename).stem

                print(f"\n  Processing: {filename}")

                try:
                    recommendations = self.recommendation_engine.recommend_for_resume(
                        resume_entry,
                        top_k=top_k
                    )

                    all_recommendations[resume_name_clean] = recommendations

                    # Print top recommendations
                    self.recommendation_engine.print_recommendations(
                        recommendations,
                        top=5
                    )

                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {e}")
                    continue

        # ──────────────────────────────────────────────────────────
        # STEP 5: Export Results
        # ──────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("STEP 5: EXPORTING RESULTS")
        print("="*70)

        export_results = {}
        for resume_name, recommendations in all_recommendations.items():
            output_file = self.recommendation_engine.export_recommendations(
                recommendations,
                resume_name=resume_name
            )
            export_results[resume_name] = output_file

        # ──────────────────────────────────────────────────────────
        # STEP 6: Summary
        # ──────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)

        results["resumes_processed"] = len(all_recommendations)
        results["recommendations"] = all_recommendations
        results["export_files"] = export_results

        self.print_summary(results)

        return results

    def process_single_resume_text(
        self,
        resume_text: str,
        resume_name: str = "custom_resume",
        top_k: int = 20
    ) -> List[Dict]:
        """
        Process a single resume given as text.

        Args:
            resume_text: Raw resume text
            resume_name: Name for the resume
            top_k: Number of recommendations

        Returns:
            List of recommendations
        """
        if not self.recommendation_engine:
            self.recommendation_engine = RecommendationEngine(
                model_name=self.model_name
            )

        print(f"\n🔍 Processing resume: {resume_name}")

        recommendations = self.recommendation_engine.recommend_for_resume(
            resume_text,
            top_k=top_k
        )

        self.recommendation_engine.print_recommendations(recommendations)

        # Export
        output_file = self.recommendation_engine.export_recommendations(
            recommendations,
            resume_name=resume_name
        )

        return recommendations

    def print_summary(self, results: Dict):
        """Print summary of pipeline results."""
        print(f"\n📊 SUMMARY:")
        print(f"   Jobs ingested: {results.get('jobs_ingested', 'N/A')}")
        print(f"   Resumes processed: {results.get('resumes_processed', 0)}")
        print(f"\n📁 Outputs saved to: {RECOMMENDATIONS_DIR}")

        for resume_name, file_path in results.get("export_files", {}).items():
            print(f"   ✅ {resume_name}: {file_path}")


# ===================== EXAMPLE FUNCTIONS =====================

def example_1_basic_usage():
    """Example 1: Basic usage with defaults."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)

    pipeline = JobMatchingPipeline()
    results = pipeline.run_full_pipeline(
        ingest_jobs=True,
        top_k=20
    )

    return results


def example_2_custom_resume():
    """Example 2: Process a custom resume text."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Resume Text")
    print("="*70)

    resume_text = """
    Software Engineer with 5 years of experience

    SKILLS: Python, Java, C++, React, Node.js, AWS, Docker, Kubernetes

    EXPERIENCE:
    - Backend Engineer at TechCorp (2020-2024)
      - Built microservices using Java and Spring Boot
      - Deployed to AWS using Kubernetes
      - Led team of 3 engineers
    
    - Junior Developer at StartupXYZ (2019-2020)
      - Full-stack development with Python and React
      - Database optimization with PostgreSQL
    
    EDUCATION:
    - BS in Computer Science, State University (2019)

    CERTIFICATIONS:
    - AWS Solutions Architect
    - Kubernetes Administrator
    """

    pipeline = JobMatchingPipeline()
    # First ingest jobs
    pipeline.ingestion_engine = JobIngestionEngine()
    pipeline.ingestion_engine.ingest_pipeline()
    # Then get recommendations
    recommendations = pipeline.process_single_resume_text(
        resume_text,
        resume_name="example_backend_engineer",
        top_k=20
    )

    return recommendations


def example_3_top_k_comparison():
    """Example 3: Compare different top_k values."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing Different Top-K Values")
    print("="*70)

    resume_text = """
    Data Scientist with ML expertise
    Skills: Python, Machine Learning, TensorFlow, PyTorch, SQL, Spark
    Experience: 3 years in AI/ML at various companies
    """

    pipeline = JobMatchingPipeline()
    pipeline.recommendation_engine = RecommendationEngine()

    for k in [10, 20, 50]:
        print(f"\n--- Top {k} Recommendations ---")
        recs = pipeline.recommendation_engine.recommend_for_resume(
            resume_text,
            top_k=k
        )
        summary = pipeline.recommendation_engine.get_recommendation_summary(recs)
        print(f"Average score: {summary['avg_score']}")
        print(f"Best match: {summary['max_score']*100:.1f}%")


# ===================== CLI =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Job matching pipeline for resumes"
    )
    parser.add_argument(
        "--mode", type=str, choices=["full", "single", "custom"],
        default="full",
        help="Pipeline mode"
    )
    parser.add_argument(
        "--resume-name", type=str, default=None,
        help="Specific resume to process (for 'single' mode)"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of recommendations"
    )
    parser.add_argument(
        "--no-ingest", action="store_true",
        help="Skip job ingestion (use existing data)"
    )
    parser.add_argument(
        "--example", type=int, choices=[1, 2, 3], default=None,
        help="Run example (1, 2, or 3)"
    )

    args = parser.parse_args()

    if args.example:
        if args.example == 1:
            example_1_basic_usage()
        elif args.example == 2:
            example_2_custom_resume()
        elif args.example == 3:
            example_3_top_k_comparison()
    else:
        pipeline = JobMatchingPipeline()
        results = pipeline.run_full_pipeline(
            ingest_jobs=not args.no_ingest,
            resume_name=args.resume_name,
            top_k=args.top_k
        )
