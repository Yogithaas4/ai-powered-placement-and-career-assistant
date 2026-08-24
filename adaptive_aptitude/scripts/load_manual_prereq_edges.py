"""
load_manual_prereq_edges.py
-------------------------------
Loads human-reviewed cross-subject prerequisite edges from
manual_prereq_edges_template.csv into concept_dependencies with
confidence='manual' (vs. the auto-derived same-topic edges, which are
confidence='heuristic').

Validates every concept_id against the actual concepts table before
inserting -- a typo'd concept_id would otherwise either fail with an
opaque FK error or (if you'd used a non-FK design) silently insert a
dangling edge that never shows up anywhere. Rows with an unknown
concept_id are reported and skipped, not silently dropped or forced in.

Safe to re-run: upserts on (prereq_concept_id, dependent_concept_id),
so editing/adding rows in the CSV and re-running just updates what's
already loaded.

Usage:
    python load_manual_prereq_edges.py --csv reports/manual_prereq_edges_template.csv
"""

import argparse
import csv
import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = os.environ.get("POSTGRES_PORT", "5434")
PG_USER = os.environ.get("POSTGRES_USER", "adaptive_user")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "adaptive_pass")
PG_DB = os.environ.get("POSTGRES_DB", "adaptive_aptitude")


def get_conn():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER,
                             password=PG_PASSWORD, dbname=PG_DB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(BASE_DIR, "reports", "manual_prereq_edges_template.csv"))
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Drop the placeholder example row if it's still in there
    rows = [r for r in rows if not r["prereq_concept_id"].strip().endswith("::...")]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT concept_id FROM concepts")
            known_ids = {r[0] for r in cur.fetchall()}

        valid, invalid = [], []
        for r in rows:
            p, d = r["prereq_concept_id"].strip(), r["dependent_concept_id"].strip()
            if p not in known_ids:
                invalid.append((p, "prereq_concept_id", r))
            elif d not in known_ids:
                invalid.append((d, "dependent_concept_id", r))
            elif p == d:
                invalid.append((p, "prereq == dependent (self-loop, not allowed)", r))
            else:
                valid.append((p, d))

        if invalid:
            print(f"WARNING: {len(invalid)} row(s) reference an unknown concept_id or are otherwise "
                  f"invalid -- NOT loaded:")
            for bad_id, field, r in invalid:
                print(f"  {field}: {bad_id!r}  (note: {r.get('note', '')})")

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO concept_dependencies (prereq_concept_id, dependent_concept_id, confidence)
                VALUES (%s, %s, 'manual')
                ON CONFLICT (prereq_concept_id, dependent_concept_id) DO UPDATE SET
                    confidence = 'manual'
            """, valid, page_size=200)
        conn.commit()
        print(f"\nLoaded {len(valid)} manually-reviewed edge(s) into concept_dependencies (confidence='manual').")

        with conn.cursor() as cur:
            cur.execute("SELECT confidence, COUNT(*) FROM concept_dependencies GROUP BY confidence")
            print("\nconcept_dependencies now:")
            for confidence, n in cur.fetchall():
                print(f"  {n:5d}  {confidence}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
