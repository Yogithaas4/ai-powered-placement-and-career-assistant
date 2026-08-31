"""Project paths resolved from the repository root (parent of ``src``)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
PROCESSED_DIR = DATA_DIR / "processed"
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"
RESUMES_DIR = DATA_DIR / "resumes"

CHROMA_PATH = str(DATA_DIR / "chroma_db")
DEFAULT_JOBS_CSV = str(JOBS_DIR / "all_jobs_v3_fixed.csv")