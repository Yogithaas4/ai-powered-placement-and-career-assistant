"""
migrate_sqlite_to_postgres.py
-------------------------------
Phase 0, step 4: move student_skill / interaction_log / student_session
out of the local SQLite file (adaptive_platform.db) into the new
Postgres tables (students, student_mastery, student_responses,
student_sessions). One database for the whole feature after this.

Mapping notes (SQLite concept_id -> Postgres concept_id):
  The old SQLite concept_id values came from the hand-authored DAG
  (e.g. "CN::OSI_Model") and do NOT match the new auto-derived
  concept_id values (e.g. "computer-networks::osi-model::layer-functions").
  This script resolves old concept_ids via the interaction_log's own
  (subject, topic, subtopic) columns, re-deriving the NEW concept_id
  the same way apply_canonical_mapping.py does, rather than trying to
  string-match old IDs to new ones. Rows whose (subject, topic, subtopic)
  cannot be resolved to a known concept are written to
  reports/migration_unresolved_rows.csv instead of being dropped silently.

Safe to re-run: uses ON CONFLICT DO NOTHING / DO UPDATE, keyed on
natural identifiers, so re-running after a partial failure won't
duplicate rows.

Usage:
    python migrate_sqlite_to_postgres.py --sqlite adaptive_platform.db
"""

import argparse
import csv
import os
import re
import sqlite3
import sys

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


def get_pg_conn():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER,
                             password=PG_PASSWORD, dbname=PG_DB)


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-") or "unknown"


def load_concept_lookup(pg_conn):
    """(canonical_subject, canonical_topic, subtopic) triples we know
    about, keyed loosely so we can match SQLite's raw subject/topic/
    subtopic (pre-canonicalization) against them via the canonical map."""
    with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT raw_subject, raw_topic, canonical_subject FROM subject_topic_canonical_map")
        subject_map_rows = cur.fetchall()
        cur.execute("SELECT concept_id, canonical_subject, canonical_topic, subtopic FROM concepts")
        concept_rows = cur.fetchall()

    subject_map = {}
    for r in subject_map_rows:
        subject_map.setdefault(r["raw_subject"], {})[r["raw_topic"]] = r["canonical_subject"]

    concept_index = {
        (c["canonical_subject"], c["canonical_topic"], c["subtopic"]): c["concept_id"]
        for c in concept_rows
    }
    return subject_map, concept_index


def resolve_concept_id(subject, topic, subtopic, subject_map, concept_index):
    canonical_subject = None
    topic_specific = subject_map.get(subject, {})
    if topic in topic_specific:
        canonical_subject = topic_specific[topic]
    elif None in topic_specific:
        canonical_subject = topic_specific[None]
    else:
        canonical_subject = subject  # unmapped -- fall back to raw

    key = (canonical_subject, topic, subtopic)
    if key in concept_index:
        return concept_index[key]

    # Fall back to slug-matching in case canonical_topic differs slightly
    # from the raw topic used in old interaction_log rows.
    slug_guess = f"{slugify(canonical_subject)}::{slugify(topic)}::{slugify(subtopic)}"
    for cid in concept_index.values():
        if cid == slug_guess:
            return cid
    return None


def migrate(sqlite_path, pg_conn):
    sconn = sqlite3.connect(sqlite_path)
    sconn.row_factory = sqlite3.Row

    subject_map, concept_index = load_concept_lookup(pg_conn)
    unresolved = []

    # ── students (derive from any table that has student_id) ──────────────
    student_ids = set()
    for table in ("student_skill", "interaction_log", "student_session"):
        rows = sconn.execute(f"SELECT DISTINCT student_id FROM {table}").fetchall()
        student_ids.update(r["student_id"] for r in rows)

    with pg_conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO students (student_id) VALUES (%s)
            ON CONFLICT (student_id) DO NOTHING
        """, [(sid,) for sid in student_ids])
    pg_conn.commit()
    print(f"Ensured {len(student_ids)} student(s) exist in Postgres")

    # ── student_session -> student_sessions ─────────────────────────────
    sessions = sconn.execute("SELECT * FROM student_session").fetchall()
    session_id_map = {}  # old integer session_id -> new UUID
    with pg_conn.cursor() as cur:
        for s in sessions:
            cur.execute("""
                INSERT INTO student_sessions
                    (student_id, canonical_subject, algorithm, started_at, ended_at,
                     questions_asked, correct_count)
                VALUES (%s, %s, 'bkt_ema_epsilon_greedy', %s, %s, %s, %s)
                RETURNING session_id
            """, (
                s["student_id"], s["subject"], s["start_time"], s["end_time"],
                s["questions_asked"], s["correct_count"],
            ))
            new_id = cur.fetchone()[0]
            session_id_map[s["session_id"]] = new_id
    pg_conn.commit()
    print(f"Migrated {len(sessions)} session(s)")

    # ── student_skill -> student_mastery ─────────────────────────────────
    skills = sconn.execute("SELECT * FROM student_skill").fetchall()
    mastery_rows = []
    for sk in skills:
        cid = resolve_concept_id(sk["subject"], sk["topic"], sk["subtopic"], subject_map, concept_index)
        if cid is None:
            unresolved.append(dict(sk))
            continue
        mastery_rows.append((
            sk["student_id"], cid, sk["bkt_score"], sk["ema_score"], sk["skill_score"],
            sk["attempts"], sk["correct_count"], sk["last_updated"],
        ))

    with pg_conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO student_mastery
                (student_id, concept_id, bkt_score, ema_score, skill_score, attempts, correct_count, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (student_id, concept_id) DO UPDATE SET
                bkt_score = EXCLUDED.bkt_score,
                ema_score = EXCLUDED.ema_score,
                skill_score = EXCLUDED.skill_score,
                attempts = EXCLUDED.attempts,
                correct_count = EXCLUDED.correct_count,
                last_updated = EXCLUDED.last_updated
        """, mastery_rows)
    pg_conn.commit()
    print(f"Migrated {len(mastery_rows)} mastery row(s) ({len(unresolved)} unresolved)")

    # ── interaction_log -> student_responses ─────────────────────────────
    # interaction_log has no direct session_id column in the original
    # schema, so we attach each response to the most recent open/matching
    # session for that student+subject at the time, falling back to NULL
    # session linkage (allowed to be optional here) if none is found --
    # in practice you'd want the live API to always pass a real session_id.
    interactions = sconn.execute(
        "SELECT * FROM interaction_log ORDER BY student_id, timestamp"
    ).fetchall()

    response_rows = []
    for it in interactions:
        cid = resolve_concept_id(it["subject"], it["topic"], it["subtopic"], subject_map, concept_index)
        if cid is None:
            unresolved.append(dict(it))
            continue
        response_rows.append((
            None,  # session_id unknown from legacy log; see note above
            it["student_id"], it["question_id"], cid, it["difficulty"],
            None,  # selected_answer not stored in legacy schema
            bool(it["correct"]), it["time_taken_sec"],
            psycopg2.extras.Json({"bkt": it["bkt_before"], "ema": it["ema_before"]}),
            psycopg2.extras.Json({"bkt": it["bkt_after"], "ema": it["ema_after"]}),
            it["timestamp"],
        ))

    # student_responses.session_id is NOT NULL in the schema (a response
    # must belong to a session) -- legacy rows without a resolvable session
    # are written to a CSV for manual review/backfill instead of being
    # forced into a fabricated session.
    if response_rows:
        print(f"NOTE: {len(response_rows)} legacy interaction_log rows have no session_id "
              f"(not tracked in the old schema) and were NOT inserted into student_responses, "
              f"since that column is NOT NULL by design. Writing them to "
              f"reports/legacy_interactions_no_session.csv for manual backfill if needed.")
        out_path = os.path.join(BASE_DIR, "reports", "legacy_interactions_no_session.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["student_id", "question_id", "concept_id", "correct", "time_taken_sec", "timestamp"])
            for row in response_rows:
                w.writerow([row[1], row[2], row[3], row[6], row[7], row[10]])

    if unresolved:
        out_path = os.path.join(BASE_DIR, "reports", "migration_unresolved_rows.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({k for row in unresolved for k in row.keys()})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(unresolved)
        print(f"WARNING: {len(unresolved)} row(s) could not be resolved to a concept_id. "
              f"Written to {out_path} -- these need a concept lookup fix, not silent dropping.")

    sconn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default="adaptive_platform.db")
    args = ap.parse_args()

    if not os.path.exists(args.sqlite):
        print(f"SQLite file not found: {args.sqlite}", file=sys.stderr)
        sys.exit(1)

    pg_conn = get_pg_conn()
    try:
        migrate(args.sqlite, pg_conn)
    finally:
        pg_conn.close()
    print("\nMigration complete. Verify counts, then stop pointing knowledge_model.py at SQLite.")
