"""
base_matcher.py
---------------
Unified interface for all matching models.

All models (ConFit v2, ColBERT, CrossEncoder) inherit from BaseMatcher
and implement the same interface:
    - recommend(preprocessed: dict, top_k: int) → List[Dict]
    - export(recommendations: List[Dict], output_path: Optional[str]) → str

This eliminates code duplication and ensures consistent usage patterns.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

from config import RECOMMENDATIONS_DIR


class BaseMatcher(ABC):
    """
    Abstract base class for resume-job matching models.
    
    All concrete models (ConFit v2, ColBERT, CrossEncoder) inherit from this
    and implement recommend() and optionally override export().
    """
    
    MODEL_NAME = "BaseMatcher"  # Override in subclasses
    
    def __init__(self, jobs_db_dir: Optional[str] = None, 
                 jobs_csv: Optional[str] = None, **kwargs):
        """
        Initialize the matcher.
        
        Args:
            jobs_db_dir: Path to job ChromaDB directory (used for retrieval)
            jobs_csv: Path to job CSV (used for metadata/details)
            **kwargs: Model-specific parameters
        """
        self.jobs_db_dir = jobs_db_dir
        self.jobs_csv = jobs_csv
        self._load_jobs_if_available()
    
    def _load_jobs_if_available(self):
        """Load jobs dataset if provided. Can be overridden by subclasses."""
        self.jobs_df = None
        if self.jobs_csv:
            try:
                import pandas as pd
                self.jobs_df = pd.read_csv(self.jobs_csv)
            except Exception as e:
                print(f"[{self.MODEL_NAME}] Warning: Could not load jobs CSV: {e}")
    
    @abstractmethod
    def recommend(self, preprocessed: dict, top_k: int = 20,
                  **kwargs) -> List[Dict]:
        """
        Generate job recommendations for a preprocessed resume.
        
        Args:
            preprocessed: Output of main_pipeline.process_one() containing:
                - filename: str
                - file_type: str
                - raw_text: str
                - sections: dict (skills, experience, education, etc.)
                - entities: dict (name, email, skills, years_exp, etc.)
                - embeddings: dict (query_vector, query_string, section_vectors)
            
            top_k: Number of recommendations to return
            **kwargs: Model-specific parameters
        
        Returns:
            List of recommendation dicts, each containing at minimum:
            {
                "rank": int,
                "score": float,
                "title": str,
                "company": str,
                "domain": str,
                "skills": str,
                "experience_level": str,
                "location": str,
                "work_type": str,
                "salary": str,
                "source": str,
                "description": str
            }
        """
        raise NotImplementedError
    
    def export(self, recommendations: List[Dict], 
               output_path: Optional[str] = None) -> str:
        """
        Export recommendations to CSV.
        
        Args:
            recommendations: List of recommendation dicts from recommend()
            output_path: Path to save CSV. If None, auto-generates timestamped path.
        
        Returns:
            Path to the saved CSV file (as string)
        """
        if not recommendations:
            return ""
        
        df = pd.DataFrame(recommendations)
        # Drop internal fields that shouldn't be exported
        columns_to_drop = ["raw_score", "hre_mode", "description"]
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns],
                     errors="ignore")
        
        if output_path is None:
            RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            resume_stem = "resume"  # Can be overridden by subclasses
            output_path = RECOMMENDATIONS_DIR / f"{self.MODEL_NAME}_{ts}.csv"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[{self.MODEL_NAME}] Saved recommendations → {output_path}")
        return str(output_path)
    
    def validate_preprocessed(self, preprocessed: dict) -> bool:
        """
        Validate that preprocessed dict has required fields.
        
        Returns:
            True if valid, False otherwise.
        """
        required_keys = {
            "filename", "file_type", "raw_text", 
            "sections", "entities", "embeddings"
        }
        if not all(k in preprocessed for k in required_keys):
            print(f"[{self.MODEL_NAME}] Error: Missing required keys in preprocessed dict")
            print(f"  Found: {set(preprocessed.keys())}")
            print(f"  Expected: {required_keys}")
            return False
        
        # Check embeddings has query_vector
        if "query_vector" not in preprocessed.get("embeddings", {}):
            print(f"[{self.MODEL_NAME}] Error: No query_vector in embeddings")
            return False
        
        return True
    
    def __repr__(self) -> str:
        return f"<{self.MODEL_NAME} matcher>"
