# Phase 0 — Complete Execution Guide

Run everything from inside `ai-powered-placement-and-career-assistant/adaptive_aptitude/`
(the repo folder that already has `docker-compose.yml`, `.env`, `db/schema.sql`, etc).

All commands below are one-time setup or one-shot scripts — nothing here
needs to stay running except the Docker containers themselves.

---

## 0. One-time environment setup (skip if already done)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
**What this does:** creates an isolated Python environment and installs
`fastapi`, `psycopg2-binary`, `python-dotenv`, etc. — everything the
existing repo and the new Phase 0 scripts need. No new packages were
introduced; `psycopg2` and `python-dotenv` (used by the new scripts) are
already in your `requirements.txt`.

```bash
cp .env.example .env   # only if .env doesn't already exist
```
**What this does:** `.env` holds Postgres/MinIO credentials and ports.
You already have one — leave it as is.

---

## 1. Start the containers

```bash
docker compose up -d
```
**What this does:** starts three containers in the background —
`adaptive_postgres` (Postgres 16, port `5434→5432`), `adaptive_minio`
(object storage, ports `9000`/`9001`), `adaptive_adminer` (Postgres web
UI, port `8081`). On **first-ever** run, Postgres also auto-executes
`db/schema.sql` (mounted into `/docker-entrypoint-initdb.d/`) to create
the base `questions` table — but since you've already ingested 9,098
questions, this has already happened; `docker compose up -d` on an
existing volume just starts the containers, it does **not** re-run
`schema.sql`.

```bash
docker compose ps
```
**What this does:** confirms all three containers show `healthy`/`running`.
If `postgres` isn't healthy yet, wait a few seconds and re-check.

---

## 2. Sanity-check your existing data is still there

```bash
docker exec -it adaptive_postgres psql -U adaptive_user -d adaptive_aptitude -c "SELECT COUNT(*) FROM questions;"
```
**What this does:** runs a query directly inside the Postgres container
(no local `psql` install needed). Should return **9098**. This is just
a checkpoint before touching anything — if this doesn't match, stop and
figure out why before proceeding.

---

## 3. Copy the Phase 0 files into the repo

From the unzipped `phase0_deliverables/` folder:

```bash
cp phase0_deliverables/db/schema_extended.sql       adaptive_aptitude/db/
cp phase0_deliverables/scripts/*.py                 adaptive_aptitude/scripts/
cp -r phase0_deliverables/reports                   adaptive_aptitude/
```
**What this does:** merges the new schema file and scripts into your
existing `db/` and `scripts/` folders (no filename collisions with
what's already there), and adds a new `reports/` folder holding the
audit/mapping outputs already generated against your real data.

---

## 4. Review the two mapping files (do this before step 6)

Open and skim:
- `reports/canonical_subject_mapping.json` — 7 entries have `"needs_review": true`. Edit the `"canonical_subject"` value for any you disagree with.
- `reports/practice_category_mapping.json` — the 5-category grouping (edit `PRACTICE_CATEGORY_MAP` in `scripts/build_practice_category_mapping.py` and re-run it if you want different boundaries, then it rewrites this JSON).

**If you hand-edit either file, run the validator before touching Postgres:**
```bash
python scripts/validate_mapping_files.py --mapping reports/canonical_subject_mapping.json --practice-category-mapping reports/practice_category_mapping.json
```
**What this does:** checks the two files agree with each other — every
canonical subject the mapping file can produce has a matching entry in
the practice-category file. Catches drift (like a canonical subject
getting renamed in one file but not the other) before it silently
produces NULL `practice_category` values at query time. Exits non-zero
if there's a real problem.

**What this does:** nothing is applied to the database yet — these are
plain JSON files you can hand-edit. Step 6 is what actually writes them
into Postgres.

*(Optional, only if you've corrected any of the 264 flagged questions
since the export was generated)*:
```bash
python scripts/export_review_issues.py --clean "D:\notes\capstone\project\adaptive_assessment_dataset\created\consolidated\questions_clean.json" --images "D:\notes\capstone\project\adaptive_assessment_dataset\created\consolidated\questions_with_image.json" --out-dir reports
```
**What this does:** re-scans your two JSON files and regenerates
`reports/issues_for_manual_review.json`/`.csv` so you can see what's
still outstanding.

---

## 5. Apply the extended schema

```bash
docker exec -i adaptive_postgres psql -U adaptive_user -d adaptive_aptitude < db/schema_extended.sql
```
**What this does:** creates the 9 new tables (`concepts`,
`concept_dependencies`, `subject_topic_canonical_map`,
`subject_practice_category_map`, `question_type_canonical_map`,
`students`, `student_sessions`, `student_responses`, `student_mastery`,
`experiment_runs`, `experiment_metrics`) and one view,
`questions_resolved`. **`questions` itself is not modified at all** — no
new columns, nothing to keep in sync. Canonical subject/topic/question-type/
practice-category/concept-id resolution all happens at read time when you
query `questions_resolved`, which joins `questions` against the small
lookup tables. Everything is `IF NOT EXISTS`/`CREATE OR REPLACE`, so it's
safe to re-run.

Verify it worked:
```bash
docker exec -it adaptive_postgres psql -U adaptive_user -d adaptive_aptitude -c "\dt"
docker exec -it adaptive_postgres psql -U adaptive_user -d adaptive_aptitude -c "\dv"
```
**What this does:** `\dt` lists tables — you should see the 9 new ones
alongside your original `questions`. `\dv` lists views — you should see
`questions_resolved`.

---

## 6. Apply the canonical mapping + derive the concept graph

```bash
python scripts/apply_canonical_mapping.py --mapping reports/canonical_subject_mapping.json --practice-category-mapping reports/practice_category_mapping.json
```
**What this does, in order (all against the live Postgres):**
1. Upserts your subject-mapping decisions into `subject_topic_canonical_map`.
2. Upserts the 5-category grouping into `subject_practice_category_map`. (`question_type_canonical_map` is static and already seeded by the schema file — nothing to do here for that one.)
3. Groups every distinct `(canonical_subject, topic, subtopic)` actually present in your 9,098 questions — resolved via a join against `subject_topic_canonical_map`, not any stored column — into a `concepts` row. This is what replaces the old hand-authored, Computer-Networks-only DAG.
4. Derives same-topic, difficulty-ordered prerequisite edges into `concept_dependencies` (marked `confidence='heuristic'`), reading through the `questions_resolved` view, and writes `reports/manual_prereq_edges_template.csv` as a starting point for any cross-subject prerequisites you want to add by hand later.
5. Runs a sanity check by querying `questions_resolved` directly and reports how many rows have no resolved `canonical_subject` / `practice_category` / `concept_id` — should be 0 / 0 / a small number (only rows with empty `subtopic`).

Verify it worked:
```bash
docker exec -it adaptive_postgres psql -U adaptive_user -d adaptive_aptitude -c "SELECT practice_category, COUNT(*) FROM questions_resolved GROUP BY practice_category ORDER BY 2 DESC;"
```
**What this does:** should show your 5 practice categories with the
counts from the earlier report (Core CS 3,524 / Aptitude 2,287 /
Programming & DSA 1,637 / Engineering Mathematics 1,478 / Data Science
& AI 172) — computed live via the view, nothing stored on `questions`.

```bash
docker exec -it adaptive_postgres psql -U adaptive_user -d adaptive_aptitude -c "SELECT COUNT(*) FROM concepts;"
```
**What this does:** confirms the concept graph was derived — expect
several hundred concepts (one per real `subject/topic/subtopic`
combination), a big jump from the old DAG that only covered CN.

---

## 7. Migrate mastery/session/interaction data off SQLite

```bash
python scripts/migrate_sqlite_to_postgres.py --sqlite adaptive_platform.db
```
**What this does:** reads `student_skill`, `interaction_log`, and
`student_session` from your local `adaptive_platform.db` file, resolves
each old row to the new concept graph (not string-matching old
hand-authored concept_ids, since those don't exist anymore — it
re-derives the concept from the row's own subject/topic/subtopic), and
inserts into the new Postgres `students`/`student_mastery`/
`student_sessions` tables. Anything it can't resolve is written to
`reports/migration_unresolved_rows.csv` instead of being dropped
silently. In your current snapshot this table is essentially empty (0
interactions), so this step is mostly a no-op today — it matters once
`core/knowledge_model.py` has been pointed at Postgres for a while and
you're migrating real history later, or if you have an older
`adaptive_platform.db` with real data sitting somewhere.

---

## 8. What's NOT done yet (deliberately out of scope for Phase 0)

- The FastAPI backend and Streamlit UI need to be updated to query the `questions_resolved` view (or `concepts`/lookup tables directly) instead of the raw `questions` table when they need canonical subject/topic/practice-category/concept — that wiring is Phase 2/3 work, not created here.
- `core/knowledge_model.py` still *writes new* mastery data to SQLite — this migration only moved *existing* data. Repointing the knowledge model to Postgres going forward is Phase 2 (backend) work.
- The 264 flagged data-quality issues in `reports/issues_for_manual_review.csv` aren't auto-fixed — hand-correct them in your source JSON, then re-ingest.
- No answer-checking security fix yet (`correct_answer` still trusted from the client in `api/main.py`) — Phase 2.
- No category-picker → sub-subject selection logic yet (how the engine picks which of ~13 subjects inside "Core CS" to serve from) — flagged for Phase 2 design.
