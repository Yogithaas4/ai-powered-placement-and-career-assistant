"""
sync_ingest_questions.py
----------------------------
Full-sync ingestion: makes the `questions` table match
questions_clean.json + questions_with_image.json EXACTLY -- upserts
everything present, and removes anything in the DB that's no longer in
the JSON (e.g. the 2 unresolvable fill_blank questions that got deleted
from the source). This is the first step of the full pipeline (see
run_full_pipeline.py) meant to be re-run every time the JSON changes.

Deletion safety: if a question_id that's being removed already has
student_responses recorded against it (a real student answered it),
the FK (student_responses.question_id REFERENCES questions) blocks the
delete rather than silently losing response history. Those are reported
separately, not silently skipped or forced through.

Usage:
    python sync_ingest_questions.py --clean questions_clean.json --images questions_with_image.json
"""

import argparse
import json
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


def load_json_tolerant(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("//")]
        return json.loads("\n".join(lines))


def to_row(q):
    return (
        q.get("question_id"), q.get("question_type"), q.get("question"),
        q.get("option_a"), q.get("option_b"), q.get("option_c"), q.get("option_d"),
        q.get("correct_answer"),
        json.dumps(q.get("left_items")) if q.get("left_items") is not None else None,
        json.dumps(q.get("right_items")) if q.get("right_items") is not None else None,
        json.dumps(q.get("correct_mapping")) if q.get("correct_mapping") is not None else None,
        bool(q.get("has_image", False)),
        q.get("image_key"),
        json.dumps(q.get("image_info")) if q.get("image_info") is not None else None,
        q.get("subject"), q.get("topic"), q.get("subtopic"), q.get("difficulty"),
        q.get("time_expected_minutes"),
        q.get("source"), q.get("validation_status"),
    )


def sync(conn, clean_path, images_path):
    clean = load_json_tolerant(clean_path)
    images = load_json_tolerant(images_path)
    all_q = clean + images
    json_ids = {q["question_id"] for q in all_q}

    rows = [to_row(q) for q in all_q]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO questions (question_id, question_type, question, option_a, option_b, option_c, option_d,
                correct_answer, left_items, right_items, correct_mapping, has_image, image_key, image_meta,
                subject, topic, subtopic, difficulty, time_expected_minutes, source, validation_status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (question_id) DO UPDATE SET
                question_type = EXCLUDED.question_type,
                question = EXCLUDED.question,
                option_a = EXCLUDED.option_a, option_b = EXCLUDED.option_b,
                option_c = EXCLUDED.option_c, option_d = EXCLUDED.option_d,
                correct_answer = EXCLUDED.correct_answer,
                left_items = EXCLUDED.left_items, right_items = EXCLUDED.right_items,
                correct_mapping = EXCLUDED.correct_mapping,
                has_image = EXCLUDED.has_image, image_key = EXCLUDED.image_key, image_meta = EXCLUDED.image_meta,
                subject = EXCLUDED.subject, topic = EXCLUDED.topic, subtopic = EXCLUDED.subtopic,
                difficulty = EXCLUDED.difficulty, time_expected_minutes = EXCLUDED.time_expected_minutes,
                source = EXCLUDED.source, validation_status = EXCLUDED.validation_status,
                updated_at = now()
        """, rows, page_size=500)
    conn.commit()
    print(f"Upserted {len(rows)} question(s) ({len(clean)} clean + {len(images)} with_image)")

    with conn.cursor() as cur:
        cur.execute("SELECT question_id FROM questions")
        db_ids = {r[0] for r in cur.fetchall()}

    to_remove = db_ids - json_ids
    if not to_remove:
        print("No questions to remove -- DB already matches source JSON.")
        return

    print(f"{len(to_remove)} question(s) in DB no longer present in source JSON, attempting removal...")
    removed, blocked = [], []
    with conn.cursor() as cur:
        for qid in to_remove:
            try:
                cur.execute("SAVEPOINT sp")
                cur.execute("DELETE FROM questions WHERE question_id = %s", (qid,))
                cur.execute("RELEASE SAVEPOINT sp")
                removed.append(qid)
            except psycopg2.errors.ForeignKeyViolation:
                cur.execute("ROLLBACK TO SAVEPOINT sp")
                blocked.append(qid)
    conn.commit()

    print(f"  Removed: {len(removed)}")
    for qid in removed:
        print(f"    - {qid}")
    if blocked:
        print(f"  BLOCKED (has recorded student responses, not deleted): {len(blocked)}")
        for qid in blocked:
            print(f"    - {qid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="questions_clean.json")
    ap.add_argument("--images", default="questions_with_image.json")
    args = ap.parse_args()

    conn = get_conn()
    try:
        sync(conn, args.clean, args.images)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM questions")
            print(f"\nquestions table now has {cur.fetchone()[0]} row(s).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
