"""
repair_duplicate_generic_mappings.py
---------------------------------------
One-time repair for databases that hit the duplicate-generic-row bug
(running apply_canonical_mapping.py more than once before the fix in
db/schema_extended.sql / apply_canonical_mapping.py v4). Symptom: totals
from questions_resolved come out as an exact multiple of the true
question count (e.g. 18196 = 2 x 9098).

What this does:
  1. Reports how many duplicate generic rows exist per raw_subject.
  2. Deletes the duplicates (keeps the lowest-id row per raw_subject).
  3. Creates the partial unique index that prevents this from recurring
     (same index schema_extended.sql now creates -- safe/no-op if you
     re-apply the updated schema file instead).
  4. Re-derives concepts (question_count was inflated) and re-checks
     questions_resolved totals.

After running this, re-run:
    python scripts/apply_canonical_mapping.py --mapping ... --practice-category-mapping ...
to refresh concepts/practice-category assignments cleanly (safe now that
re-running won't reintroduce the duplicates).

Usage:
    python repair_duplicate_generic_mappings.py
"""

import os

import psycopg2
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
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT raw_subject, COUNT(*) FROM subject_topic_canonical_map
                WHERE raw_topic IS NULL GROUP BY raw_subject HAVING COUNT(*) > 1
                ORDER BY 2 DESC
            """)
            dupes = cur.fetchall()

        if not dupes:
            print("No duplicate generic rows found -- nothing to repair.")
        else:
            print(f"Found {len(dupes)} raw subject(s) with duplicate generic rows:")
            for raw_subject, n in dupes:
                print(f"  {n}x  {raw_subject}")

            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM subject_topic_canonical_map a
                    USING subject_topic_canonical_map b
                    WHERE a.raw_topic IS NULL
                      AND b.raw_topic IS NULL
                      AND a.raw_subject = b.raw_subject
                      AND a.id > b.id
                """)
                n_deleted = cur.rowcount
            conn.commit()
            print(f"Deleted {n_deleted} duplicate row(s).")

        with conn.cursor() as cur:
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_subj_topic_map_generic_unique
                    ON subject_topic_canonical_map (raw_subject)
                    WHERE raw_topic IS NULL
            """)
        conn.commit()
        print("Ensured partial unique index exists (prevents this from recurring).")

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM questions")
            n_questions = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM questions_resolved")
            n_resolved = cur.fetchone()[0]

        print(f"\nquestions: {n_questions}   questions_resolved: {n_resolved}")
        if n_questions != n_resolved:
            print("WARNING: still mismatched -- check for other duplicate join sources "
                  "(subject_practice_category_map, question_type_canonical_map, concepts) "
                  "before proceeding.")
        else:
            print("Counts match -- repair successful. Now re-run apply_canonical_mapping.py "
                  "to refresh concepts/question_count with correct numbers.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
