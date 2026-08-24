"""
run_full_pipeline.py
------------------------
THE command to run after editing questions_clean.json or
questions_with_image.json. Chains every step that needs to happen for
the DB to correctly reflect the current source JSON:

    1. sync_ingest_questions.py  -- upsert + remove questions to match JSON exactly
    2. schema_extended.sql       -- ensure all lookup/concept/mastery tables exist
                                     (no-op if already applied; safe to re-run)
    3. build_canonical_mapping.py + build_practice_category_mapping.py
                                     -- regenerate the taxonomy reports from the
                                        CURRENT record counts (so they don't drift
                                        if question counts per subject changed)
    4. validate_mapping_files.py -- refuse to proceed if the two mapping files
                                     disagree (catches drift before it's applied)
    5. apply_canonical_mapping.py -- populate lookup tables + derive
                                      concepts/DAG from the now-current data

Every step here is idempotent (verified) -- re-running this after a
small JSON edit is safe and cheap; nothing needs to be run manually
in between.

Usage:
    python run_full_pipeline.py --clean questions_clean.json --images questions_with_image.json
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)


def run_step(description, cmd):
    print(f"\n{'=' * 70}\n{description}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n!! STOPPED: '{description}' failed (exit code {result.returncode}). "
              f"Fix the issue above before re-running the pipeline.")
        sys.exit(result.returncode)


def run_sql_step(description, sql_path):
    print(f"\n{'=' * 70}\n{description}\n{'=' * 70}")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5434")
    user = os.environ.get("POSTGRES_USER", "adaptive_user")
    db = os.environ.get("POSTGRES_DB", "adaptive_aptitude")
    cmd = ["psql", "-h", host, "-p", port, "-U", user, "-d", db, "-f", sql_path, "-v", "ON_ERROR_STOP=1"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n!! STOPPED: '{description}' failed. If psql isn't on PATH, apply "
              f"{sql_path} manually (e.g. via docker exec) and re-run this pipeline.")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="questions_clean.json")
    ap.add_argument("--images", default="questions_with_image.json")
    ap.add_argument("--report-dir", default=os.path.join(BASE_DIR, "reports"))
    ap.add_argument("--skip-schema", action="store_true",
                     help="Skip re-applying schema_extended.sql (use if you know it's already current)")
    args = ap.parse_args()

    py = sys.executable

    if not args.skip_schema:
        run_sql_step("Step 1/5: Ensure extended schema is up to date",
                     os.path.join(BASE_DIR, "db", "schema_extended.sql"))
    else:
        print("Skipping schema step (--skip-schema)")

    run_step("Step 2/5: Sync questions table to current source JSON",
              [py, "sync_ingest_questions.py", "--clean", args.clean, "--images", args.images])

    run_step("Step 3/5: Regenerate canonical subject mapping report",
              [py, "build_canonical_mapping.py", "--clean", args.clean, "--images", args.images,
               "--out-dir", args.report_dir])

    run_step("Step 4/5: Regenerate practice-category mapping report",
              [py, "build_practice_category_mapping.py",
               "--mapping", os.path.join(args.report_dir, "canonical_subject_mapping.json"),
               "--review-csv", os.path.join(args.report_dir, "canonical_mapping_review.csv"),
               "--out-dir", args.report_dir])

    run_step("Step 4b/5: Validate the two mapping files agree",
              [py, "validate_mapping_files.py",
               "--mapping", os.path.join(args.report_dir, "canonical_subject_mapping.json"),
               "--practice-category-mapping", os.path.join(args.report_dir, "practice_category_mapping.json")])

    run_step("Step 5/5: Apply canonical mapping, derive concepts + DAG",
              [py, "apply_canonical_mapping.py",
               "--mapping", os.path.join(args.report_dir, "canonical_subject_mapping.json"),
               "--practice-category-mapping", os.path.join(args.report_dir, "practice_category_mapping.json"),
               "--report-dir", args.report_dir])

    print(f"\n{'=' * 70}\nPipeline complete. `questions_resolved` view now reflects the current JSON.\n{'=' * 70}")


if __name__ == "__main__":
    main()
