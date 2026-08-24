"""
data/db_loader.py
------------------
Replaces the old CSV-based question loading (data/dataset_loader.py's
load_question_dataset) with a Postgres-backed loader, now that the
full cleaned question bank lives in the `questions` table.

QuestionSelector still expects a pandas DataFrame with a
"question_index" column as a stable string id -- we alias question_id
to question_index so the rest of the engine (concept_dag,
question_selector) needs zero changes.
"""

import os

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_USER = os.environ.get("POSTGRES_USER", "adaptive_user")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "adaptive_pass")
PG_DB = os.environ.get("POSTGRES_DB", "adaptive_aptitude")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "question-images")


def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    )


def get_engine():
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(url)


def image_url_for_key(image_key):
    if not image_key:
        return None
    return f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{image_key}"


def load_question_dataset() -> pd.DataFrame:
    """Load the full question bank from Postgres into a DataFrame.

    Adds:
      - question_index: alias of question_id (kept for compatibility
        with the existing QuestionSelector / concept_dag code, which
        was originally written against the CSV loader).
      - image_url: ready-to-use URL for the frontend, or None.
    """
    engine = get_engine()
    df = pd.read_sql_query("SELECT * FROM questions", engine)

    df["question_index"] = df["question_id"].astype(str)
    df["image_url"] = df["image_key"].apply(image_url_for_key)

    print(f"Loaded {len(df)} questions from Postgres across {df['subject'].nunique()} subjects")
    print(df.groupby("subject").size().to_string())
    return df
