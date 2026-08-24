"""
ingest_questions.py
--------------------
One-shot (re-runnable) loader that:

  1. Reads questions_clean.json + questions_with_image.json from the
     data-prep pipeline output.
  2. For any question that references an image, uploads the file from
     the local `images/` folder into the MinIO bucket (S3-compatible),
     keyed as "questions/<filename>".
  3. Upserts every question into the Postgres `questions` table, with
     `image_key` pointing at the MinIO object (or null for questions
     with no raster image).

Safe to re-run: images are uploaded idempotently (same key overwrites
the same object) and rows are upserted on question_id.

Usage:
    python scripts/ingest_questions.py
"""

import json
import os
import sys

import boto3
import psycopg2
import psycopg2.extras
from botocore.client import Config
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

QUESTIONS_CLEAN_JSON = os.environ["QUESTIONS_CLEAN_JSON"]
QUESTIONS_WITH_IMAGE_JSON = os.environ["QUESTIONS_WITH_IMAGE_JSON"]
QUESTION_IMAGES_DIR = os.environ["QUESTION_IMAGES_DIR"]

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ROOT_USER = os.environ["MINIO_ROOT_USER"]
MINIO_ROOT_PASSWORD = os.environ["MINIO_ROOT_PASSWORD"]
MINIO_BUCKET = os.environ["MINIO_BUCKET"]
MINIO_PUBLIC_READ = os.environ.get("MINIO_PUBLIC_READ", "false").lower() == "true"

PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = os.environ["POSTGRES_PORT"]
PG_USER = os.environ["POSTGRES_USER"]
PG_PASSWORD = os.environ["POSTGRES_PASSWORD"]
PG_DB = os.environ["POSTGRES_DB"]


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(s3):
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if MINIO_BUCKET not in existing:
        s3.create_bucket(Bucket=MINIO_BUCKET)
        print(f"Created bucket: {MINIO_BUCKET}")

    if MINIO_PUBLIC_READ:
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{MINIO_BUCKET}/*"],
                }
            ],
        }
        s3.put_bucket_policy(Bucket=MINIO_BUCKET, Policy=json.dumps(policy))


def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    )


def load_json(path):
    with open(path, encoding="utf-8") as f:
        # The consolidated image-backed export currently includes a leading
        # JavaScript-style provenance comment.  Accept it while keeping the
        # remaining file standard JSON, so re-seeding stays reproducible.
        content = "\n".join(
            line for line in f if not line.lstrip().startswith("//")
        )
    return json.loads(content)


UPSERT_SQL = """
INSERT INTO questions (
    question_id, question_type, question,
    option_a, option_b, option_c, option_d, correct_answer,
    left_items, right_items, correct_mapping,
    has_image, image_key, image_meta,
    subject, topic, subtopic, difficulty, time_expected_minutes,
    source, validation_status, raw_tags, notes, extra_fields,
    updated_at
) VALUES (
    %(question_id)s, %(question_type)s, %(question)s,
    %(option_a)s, %(option_b)s, %(option_c)s, %(option_d)s, %(correct_answer)s,
    %(left_items)s, %(right_items)s, %(correct_mapping)s,
    %(has_image)s, %(image_key)s, %(image_meta)s,
    %(subject)s, %(topic)s, %(subtopic)s, %(difficulty)s, %(time_expected_minutes)s,
    %(source)s, %(validation_status)s, %(raw_tags)s, %(notes)s, %(extra_fields)s,
    now()
)
ON CONFLICT (question_id) DO UPDATE SET
    question_type = EXCLUDED.question_type,
    question = EXCLUDED.question,
    option_a = EXCLUDED.option_a,
    option_b = EXCLUDED.option_b,
    option_c = EXCLUDED.option_c,
    option_d = EXCLUDED.option_d,
    correct_answer = EXCLUDED.correct_answer,
    left_items = EXCLUDED.left_items,
    right_items = EXCLUDED.right_items,
    correct_mapping = EXCLUDED.correct_mapping,
    has_image = EXCLUDED.has_image,
    image_key = EXCLUDED.image_key,
    image_meta = EXCLUDED.image_meta,
    subject = EXCLUDED.subject,
    topic = EXCLUDED.topic,
    subtopic = EXCLUDED.subtopic,
    difficulty = EXCLUDED.difficulty,
    time_expected_minutes = EXCLUDED.time_expected_minutes,
    source = EXCLUDED.source,
    validation_status = EXCLUDED.validation_status,
    raw_tags = EXCLUDED.raw_tags,
    notes = EXCLUDED.notes,
    extra_fields = EXCLUDED.extra_fields,
    updated_at = now();
"""


def prepare_row(q, image_key):
    ii = q.get("image_info")
    if not isinstance(ii, dict):
        ii = None
    image_meta = None
    if ii:
        image_meta = {k: v for k, v in ii.items() if k != "image_reference"}
        if not any(v for v in image_meta.values()):
            image_meta = None

    return {
        "question_id": q["question_id"],
        "question_type": q.get("question_type"),
        "question": q["question"],
        "option_a": q.get("option_a"),
        "option_b": q.get("option_b"),
        "option_c": q.get("option_c"),
        "option_d": q.get("option_d"),
        "correct_answer": q.get("correct_answer"),
        "left_items": json.dumps(q["left_items"]) if q.get("left_items") else None,
        "right_items": json.dumps(q["right_items"]) if q.get("right_items") else None,
        "correct_mapping": json.dumps(q["correct_mapping"]) if q.get("correct_mapping") else None,
        "has_image": bool(q.get("has_image")),
        "image_key": image_key,
        "image_meta": json.dumps(image_meta) if image_meta else None,
        "subject": q.get("subject"),
        "topic": q.get("topic"),
        "subtopic": q.get("subtopic"),
        "difficulty": q.get("difficulty"),
        "time_expected_minutes": q.get("time_expected_minutes"),
        "source": q.get("source"),
        "validation_status": q.get("validation_status"),
        "raw_tags": q.get("raw_tags"),
        "notes": q.get("notes"),
        "extra_fields": json.dumps(q["extra_fields"]) if q.get("extra_fields") else None,
    }


def main():
    print("Loading JSON files...")
    clean = load_json(QUESTIONS_CLEAN_JSON)
    with_image = load_json(QUESTIONS_WITH_IMAGE_JSON)
    print(f"  questions_clean.json: {len(clean)}")
    print(f"  questions_with_image.json: {len(with_image)}")

    print("Connecting to MinIO...")
    s3 = get_s3_client()
    ensure_bucket(s3)

    uploaded = 0
    upload_failures = []
    rows = []

    for q in clean:
        rows.append(prepare_row(q, image_key=None))

    for q in with_image:
        ii = q.get("image_info") or {}
        ref = ii.get("image_reference")
        image_key = None
        if ref:
            filename = os.path.basename(ref)
            local_path = os.path.join(QUESTION_IMAGES_DIR, filename)
            if os.path.exists(local_path):
                image_key = f"questions/{filename}"
                s3.upload_file(local_path, MINIO_BUCKET, image_key)
                uploaded += 1
            else:
                upload_failures.append(q["question_id"])
        rows.append(prepare_row(q, image_key=image_key))

    print(f"Uploaded {uploaded} images to bucket '{MINIO_BUCKET}'")
    if upload_failures:
        print(f"WARNING: {len(upload_failures)} questions referenced an image file that "
              f"wasn't found locally (kept in DB with image_key=null): {upload_failures[:10]}...")

    print(f"Connecting to Postgres at {PG_HOST}:{PG_PORT}/{PG_DB}...")
    conn = get_pg_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, UPSERT_SQL, rows, page_size=500)
        print(f"Upserted {len(rows)} questions into 'questions' table.")
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
