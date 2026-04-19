"""
Shim: job ingestion lives in ``job_indexing``.

    python src/job_ingestion.py --csv data/jobs/cs_engineering_jobs.csv
"""

from job_indexing.job_ingestion import JobIngestionEngine, quick_ingest

__all__ = ["JobIngestionEngine", "quick_ingest"]

import runpy
import sys
from pathlib import Path

# Ensure ``src`` is importable when this file is executed directly
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __name__ == "__main__":
    raise SystemExit(runpy.run_module("job_indexing.job_ingestion", run_name="__main__"))
