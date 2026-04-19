"""
Shim: resume CLI and imports live in ``resume_processing``.

Run (from repo root, with ``src`` on ``PYTHONPATH`` or as here):

    python src/main_pipeline.py --folder data/resumes
    python -m resume_processing.main_pipeline --folder data/resumes --test
"""

from resume_processing.main_pipeline import process_one, run_pipeline

__all__ = ["process_one", "run_pipeline"]


def _cli() -> None:
    import argparse

    from config import RESUMES_DIR
    from resume_processing.main_pipeline import run_pipeline

    parser = argparse.ArgumentParser(description="Resume preprocessing pipeline")
    parser.add_argument(
        "--folder",
        default=str(RESUMES_DIR),
        help="Folder containing resume files (.docx / .pdf)",
    )
    parser.add_argument("--test", action="store_true", help="Test mode: first 2 resumes only")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of resumes")
    args = parser.parse_args()
    limit = 2 if args.test else args.limit
    run_pipeline(args.folder, limit=limit)


if __name__ == "__main__":
    _cli()
