# Adaptive Aptitude Platform — Implementation Status

This document records the current implementation, local setup, and the next planned work for the adaptive aptitude-test feature.

## Purpose

The feature delivers adaptive practice over GATE-style questions. It tracks a learner's concept mastery and chooses the next question using a concept-prerequisite DAG, Bayesian Knowledge Tracing (BKT), exponential moving average (EMA), and epsilon-greedy exploration.

## Current architecture

```text
Consolidated JSON question bank + local image files
        |
        | ingest_questions.py
        v
PostgreSQL: questions table <----> MinIO: question-images bucket
        |
        v
Pandas question loader -> adaptive selector -> FastAPI prototype
                              |
                              v
                 SQLite mastery/session prototype (to be migrated)
```

## Implemented

- Docker Compose runs PostgreSQL, MinIO, and Adminer locally.
- The `questions` Postgres table stores all question content, options, answers, matching data, taxonomy, difficulty, expected time, and image metadata.
- `scripts/ingest_questions.py` bulk-upserts the consolidated question files and uploads referenced raster images to MinIO.
- The loader exposes a frontend-ready `image_url` from each stored object key.
- The adaptive prototype includes:
  - BKT mastery updates;
  - EMA smoothing;
  - a combined mastery score;
  - epsilon-greedy question selection;
  - a hand-authored prerequisite DAG for core CS subjects;
  - FastAPI endpoints for sessions, next questions, answers, summaries, skills, and history.
- Phase 0 taxonomy/DAG work is now applied in Postgres: canonical labels resolve through
  `questions_resolved`, with 2,539 active concepts, 409 regenerated heuristic same-topic
  edges, and 10 manually reviewed cross-subject edges. The validated graph is acyclic.

## Data status

- Question bank: 9,098 usable questions.
- Standard questions: 8,856.
- Image-backed records in Postgres: 243, each with a linked MinIO object as of the latest ingestion run.
- Backlogs remain outside the active bank: `questions_with_images_unfixed.json` and `questions_to_review.json`.

## Important current limitations

1. Question bank data is in Postgres, but `student_skill`, `interaction_log`, and `student_session` are still stored in the local SQLite file `adaptive_platform.db`.
2. The database schema does not yet contain persistent concepts, prerequisite edges, learner mastery, responses, sessions, or experiment-run tables.
3. The answer API accepts a client-supplied `correct_answer`. This must be replaced with server-side lookup by `question_id` before any real deployment.
4. Subject labels are canonicalized, but topics/subtopics still retain source vocabulary. The
   30 punctuation-only merges in `reports/concept_subtopic_merges.csv` need one human sanity pass;
   broader semantic topic/subtopic normalization remains future work.
5. There is no Streamlit UI yet.
6. The current evaluation script uses SQLite and simulated learners. It is useful for engineering regression checks but cannot alone support claims of real learning improvement.

## Manual prerequisite edges

`reports/manual_prereq_edges_template.csv` now holds reviewed cross-subject
prerequisites. Run `python scripts/load_manual_prereq_edges.py` after editing
it. The loader validates concept IDs and refuses cycles; manual edges are
preserved when `apply_canonical_mapping.py` regenerates heuristic edges.

## Local operation

From this directory:

```powershell
docker compose up -d
python scripts/ingest_questions.py
uvicorn api.main:app --reload --port 8000
```

The service uses `.env` for Postgres, MinIO, dataset, and image-directory paths. Do not commit `.env`; use `.env.example` as the template.

## Next implementation order

1. Normalize subject/topic/subtopic names and verify the question-type/image metadata.
2. Extend Postgres with concepts, dependencies, sessions, responses, mastery, and experiment tables.
3. Migrate BKT/EMA persistence from SQLite to Postgres and make answer checking server-side and type-aware.
4. Build a common experimental harness to compare random, EMA, BKT, BKT+EMA+epsilon-greedy, and a contextual-bandit baseline.
5. Build the Streamlit practice flow and dashboard, including MCQ, fill-blank, numerical, match-following, and image renderers.
6. Collect pilot interactions and report predictive calibration, ROC-AUC/PR-AUC, Brier score, questions-to-mastery, coverage, and post-test learning gains.
