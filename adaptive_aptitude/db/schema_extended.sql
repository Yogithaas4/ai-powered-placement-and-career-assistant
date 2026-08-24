-- Adaptive Aptitude Platform -- Phase 0 schema extension.
-- Adds everything needed to move off SQLite and to run/compare
-- multiple adaptive-selection algorithms for the research paper.
--
-- Depends on the existing `questions` table (db/schema.sql).
-- Safe to re-run (IF NOT EXISTS everywhere).
--
-- DESIGN NOTE (v2): `questions` is NOT modified at all -- no ALTER TABLE,
-- no new columns. Canonical subject/topic/question-type/practice-category/
-- concept resolution all happens at READ TIME via the `questions_resolved`
-- view at the bottom of this file, which joins the small lookup tables
-- below. This trades a small per-query join (cheap at this table size,
-- ~9k rows, all join keys indexed) for zero risk of the derived labels
-- drifting out of sync with `questions` -- there is nothing to keep in
-- sync. Every consumer (API, Streamlit, evaluation scripts) should query
-- `questions_resolved` instead of re-deriving the mapping itself, so the
-- resolution logic still lives in exactly one place.
--
-- Run after db/schema.sql, e.g.:
--   psql -h localhost -p 5434 -U adaptive_user -d adaptive_aptitude -f db/schema_extended.sql

-- gen_random_uuid() is built into Postgres core since v13, but pgcrypto
-- provides it too on older setups -- harmless to ensure it's available.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ─────────────────────────────────────────────────────────────────────────
-- 1. Canonical taxonomy lookup tables (source of truth -- nothing on
--    `questions` mirrors these; they're joined against at read time)
-- ─────────────────────────────────────────────────────────────────────────

-- Raw (subject, topic) as they appear in `questions` -> canonical subject.
-- raw_topic = NULL means "applies to this raw_subject regardless of topic"
-- (the generic case); a row with raw_topic set overrides the generic row
-- for that specific topic (used for the Mathematics -> Discrete Mathematics
-- vs Engineering Mathematics split). Populated by
-- scripts/apply_canonical_mapping.py from reports/canonical_subject_mapping.json.
CREATE TABLE IF NOT EXISTS subject_topic_canonical_map (
    id                  SERIAL PRIMARY KEY,
    raw_subject         TEXT NOT NULL,
    raw_topic           TEXT,
    canonical_subject   TEXT NOT NULL,
    rule                TEXT,               -- human-readable justification, kept for auditability
    needs_review        BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (raw_subject, raw_topic)
);

CREATE INDEX IF NOT EXISTS idx_subj_topic_map_raw_subject ON subject_topic_canonical_map (raw_subject);

-- IMPORTANT: standard Postgres UNIQUE constraints never treat two NULLs as
-- equal, so the UNIQUE (raw_subject, raw_topic) above does NOT prevent two
-- "generic" rows (raw_topic IS NULL) for the same raw_subject -- re-running
-- the upsert script would silently insert a duplicate every time instead of
-- updating in place. A PARTIAL unique index (only over rows where raw_topic
-- IS NULL) is what actually enforces "at most one generic mapping per raw
-- subject", and is what apply_canonical_mapping.py's ON CONFLICT target
-- uses for those rows.
--
-- If you're applying this to a database that already has duplicates from
-- before this fix (multiple raw_topic IS NULL rows for the same
-- raw_subject), the DELETE below removes them first (keeps the
-- lowest-id row per raw_subject), so this index creation won't fail.
DELETE FROM subject_topic_canonical_map a
USING subject_topic_canonical_map b
WHERE a.raw_topic IS NULL
  AND b.raw_topic IS NULL
  AND a.raw_subject = b.raw_subject
  AND a.id > b.id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_subj_topic_map_generic_unique
    ON subject_topic_canonical_map (raw_subject)
    WHERE raw_topic IS NULL;

-- Broad practice-category grouping (Aptitude / Engineering Mathematics /
-- Programming & DSA / Core CS / Data Science & AI) so the student picker
-- shows ~5 buckets instead of 24 canonical subjects.
CREATE TABLE IF NOT EXISTS subject_practice_category_map (
    canonical_subject   TEXT PRIMARY KEY,
    practice_category   TEXT NOT NULL
);

-- Question-type consolidation: table_based / graph_based / diagram_based
-- (single-digit counts each) collapse into image_based for SELECTION
-- purposes. The raw question_type in `questions` is never touched -- the
-- renderer can still branch on the raw type / image_info.type to decide
-- whether to draw a table vs. a graph vs. a plain picture.
CREATE TABLE IF NOT EXISTS question_type_canonical_map (
    raw_question_type        TEXT PRIMARY KEY,
    canonical_question_type  TEXT NOT NULL
);

INSERT INTO question_type_canonical_map (raw_question_type, canonical_question_type) VALUES
    ('mcq', 'mcq'),
    ('numerical', 'numerical'),
    ('multi_select', 'multi_select'),
    ('fill_blank', 'fill_blank'),
    ('match_following', 'match_following'),
    ('image_based', 'image_based'),
    ('graph_based', 'image_based'),
    ('table_based', 'image_based'),
    ('diagram_based', 'image_based')
ON CONFLICT (raw_question_type) DO UPDATE SET canonical_question_type = EXCLUDED.canonical_question_type;


-- ─────────────────────────────────────────────────────────────────────────
-- 2. Concepts + prerequisite DAG
-- ─────────────────────────────────────────────────────────────────────────
-- Concepts are derived from the actual (canonical_subject, topic, subtopic)
-- combinations present in the data -- NOT hand-authored, so coverage
-- matches the real question bank instead of only Computer Networks.
-- This IS real relational data (a graph with edges, referenced by FKs
-- from mastery/response tables below) -- it has to be stored, unlike the
-- subject/topic/question-type label mappings above.

CREATE TABLE IF NOT EXISTS concepts (
    concept_id          TEXT PRIMARY KEY,      -- e.g. "operating-systems::process-scheduling::sjf"
    canonical_subject   TEXT NOT NULL,
    canonical_topic     TEXT NOT NULL,         -- == raw `questions.topic`; no separate topic renaming is done
    subtopic            TEXT NOT NULL,         -- == raw `questions.subtopic`, unchanged
    question_count      INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (canonical_subject, canonical_topic, subtopic)
);

CREATE INDEX IF NOT EXISTS idx_concepts_subject ON concepts (canonical_subject);

-- Maps EVERY raw (canonical_subject, topic, subtopic) spelling seen in the
-- data to its concept_id -- not just the one representative spelling stored
-- on `concepts`. Without this, differently-punctuated variants that get
-- merged into the same concept (e.g. "B Tree" and "B-Tree") would only
-- resolve via an exact string match against whichever spelling `concepts`
-- happened to keep -- silently orphaning every question using the other
-- spelling. Populated by scripts/apply_canonical_mapping.py.
CREATE TABLE IF NOT EXISTS subtopic_concept_map (
    canonical_subject   TEXT NOT NULL,
    canonical_topic     TEXT NOT NULL,
    raw_subtopic        TEXT NOT NULL,
    concept_id          TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    PRIMARY KEY (canonical_subject, canonical_topic, raw_subtopic)
);

CREATE TABLE IF NOT EXISTS concept_dependencies (
    prereq_concept_id      TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    dependent_concept_id   TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    confidence              TEXT DEFAULT 'heuristic',  -- 'heuristic' (auto-derived) | 'manual' (expert-reviewed)
    PRIMARY KEY (prereq_concept_id, dependent_concept_id),
    CHECK (prereq_concept_id <> dependent_concept_id)
);


-- ─────────────────────────────────────────────────────────────────────────
-- 3. Students, sessions, responses, mastery (replaces SQLite)
-- ─────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS students (
    student_id      TEXT PRIMARY KEY,
    display_name    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS student_sessions (
    session_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id           TEXT NOT NULL REFERENCES students(student_id),
    canonical_subject     TEXT NOT NULL,
    algorithm             TEXT NOT NULL DEFAULT 'bkt_ema_epsilon_greedy',  -- selection strategy used this session
    experiment_run_id      UUID,        -- FK to experiment_runs added below, after that table exists
    started_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at                TIMESTAMPTZ,
    questions_asked          INTEGER NOT NULL DEFAULT 0,
    correct_count             INTEGER NOT NULL DEFAULT 0
);

-- NOTE on student_responses.concept_id: this is NOT the same kind of
-- redundancy as the columns we removed from `questions`. This is an
-- event/fact table -- it stores the concept a response was recorded
-- against AT THE TIME it was answered. If you later edit the canonical
-- mapping or re-derive concepts, historical responses should NOT silently
-- change meaning -- that's standard practice for immutable log/fact
-- tables (store the resolved dimension key at event time), unlike
-- `questions` which is reference data that should always reflect the
-- current mapping via the view.
CREATE TABLE IF NOT EXISTS student_responses (
    response_id         BIGSERIAL PRIMARY KEY,
    session_id           UUID NOT NULL REFERENCES student_sessions(session_id) ON DELETE CASCADE,
    student_id            TEXT NOT NULL REFERENCES students(student_id),
    question_id            TEXT NOT NULL REFERENCES questions(question_id),
    concept_id              TEXT REFERENCES concepts(concept_id),
    difficulty_at_ask         TEXT,
    selected_answer            TEXT,           -- raw student response (letter set, number, mapping JSON, etc.)
    is_correct                  BOOLEAN NOT NULL,
    time_taken_sec                REAL,
    mastery_before                 JSONB,       -- {"bkt":.., "ema":.., "skill_score":..}
    mastery_after                   JSONB,
    answered_at                      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_responses_student_concept ON student_responses (student_id, concept_id);
CREATE INDEX IF NOT EXISTS idx_responses_session ON student_responses (session_id);

CREATE TABLE IF NOT EXISTS student_mastery (
    student_id      TEXT NOT NULL REFERENCES students(student_id),
    concept_id       TEXT NOT NULL REFERENCES concepts(concept_id),
    bkt_score         REAL NOT NULL DEFAULT 0.3,
    ema_score          REAL NOT NULL DEFAULT 0.5,
    skill_score          REAL NOT NULL DEFAULT 0.3,
    attempts               INTEGER NOT NULL DEFAULT 0,
    correct_count            INTEGER NOT NULL DEFAULT 0,
    last_updated               TIMESTAMPTZ,
    PRIMARY KEY (student_id, concept_id)
);


-- ─────────────────────────────────────────────────────────────────────────
-- 4. Experiment tracking (for the research-paper model comparison)
-- ─────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    algorithm            TEXT NOT NULL,          -- 'random' | 'rule_weakest_topic' | 'ema_only' | 'bkt' |
                                                    -- 'bkt_ema_epsilon_greedy' | 'thompson_sampling' | 'linucb' | 'irt_2pl'
    description             TEXT,
    hyperparameters           JSONB,
    started_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at                  TIMESTAMPTZ,
    notes                          TEXT
);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    run_id                UUID NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    metric_name             TEXT NOT NULL,   -- 'roc_auc' | 'pr_auc' | 'brier_score' | 'log_loss' |
                                                -- 'calibration_error' | 'questions_to_mastery' | 'coverage' |
                                                -- 'diversity' | 'repeat_rate' | 'difficulty_match_rate' | 'post_test_gain'
    metric_value              DOUBLE PRECISION NOT NULL,
    computed_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    context                       JSONB       -- e.g. {"subject": "...", "n_students": .., "n_sessions": ..}
);

CREATE INDEX IF NOT EXISTS idx_experiment_metrics_run ON experiment_metrics (run_id, metric_name);

-- student_sessions.experiment_run_id references experiment_runs, which is
-- declared after it above -- add the FK now that both tables exist.
ALTER TABLE student_sessions
    DROP CONSTRAINT IF EXISTS student_sessions_experiment_run_id_fkey;
ALTER TABLE student_sessions
    ADD CONSTRAINT student_sessions_experiment_run_id_fkey
    FOREIGN KEY (experiment_run_id) REFERENCES experiment_runs(run_id);


-- ─────────────────────────────────────────────────────────────────────────
-- 5. Read-time resolution view -- THIS is the single place canonical
--    subject/topic/question-type/practice-category/concept resolution
--    logic lives. `questions` itself is never modified.
-- ─────────────────────────────────────────────────────────────────────────
-- Precedence for subject_topic_canonical_map: a topic-specific row
-- (raw_topic = questions.topic) wins over the generic row for that
-- raw_subject (raw_topic IS NULL) -- this is what makes the Mathematics
-- split (Discrete Mathematics vs Engineering Mathematics, by topic) work.

CREATE OR REPLACE VIEW questions_resolved AS
SELECT
    q.*,
    COALESCE(specific_map.canonical_subject, generic_map.canonical_subject, q.subject) AS canonical_subject,
    q.topic AS canonical_topic,
    COALESCE(qtm.canonical_question_type, q.question_type) AS canonical_question_type,
    pcm.practice_category,
    scm.concept_id
FROM questions q
LEFT JOIN subject_topic_canonical_map specific_map
       ON specific_map.raw_subject = q.subject AND specific_map.raw_topic = q.topic
LEFT JOIN subject_topic_canonical_map generic_map
       ON generic_map.raw_subject = q.subject AND generic_map.raw_topic IS NULL
LEFT JOIN question_type_canonical_map qtm
       ON qtm.raw_question_type = q.question_type
LEFT JOIN subject_practice_category_map pcm
       ON pcm.canonical_subject = COALESCE(specific_map.canonical_subject, generic_map.canonical_subject, q.subject)
LEFT JOIN subtopic_concept_map scm
       ON scm.canonical_subject = COALESCE(specific_map.canonical_subject, generic_map.canonical_subject, q.subject)
      AND scm.canonical_topic = q.topic
      AND scm.raw_subtopic = q.subtopic;

-- Example usage (this is what the API/Streamlit/eval scripts should do
-- instead of re-deriving the mapping themselves):
--   SELECT * FROM questions_resolved WHERE practice_category = 'Core CS (Systems & Theory)' LIMIT 10;
--   SELECT * FROM questions_resolved WHERE concept_id = 'operating-systems::process-scheduling::sjf';

-- ─────────────────────────────────────────────────────────────────────────
-- 6. Phase 1 correction: sessions are scoped by practice_category, not
--    canonical_subject.
-- ─────────────────────────────────────────────────────────────────────────
-- A real session ("student taps 'Core CS (Systems & Theory)'") pulls
-- questions from EVERY canonical_subject that rolls up into that category
-- (Computer Networks, OS, Databases, Digital Logic, ... all in one pool).
-- `student_sessions.canonical_subject TEXT NOT NULL` above assumed one
-- subject per session, which is wrong for that flow -- it would make a
-- "Core CS" session impossible to log correctly.
--
-- Fix: add `practice_category` as the primary session-scoping column and
-- make `canonical_subject` nullable (kept only for any future subject-only
-- session mode; real category-scoped sessions leave it NULL). Per-response
-- canonical_subject is NOT duplicated here -- it's already correctly
-- resolvable at read time via student_responses.concept_id -> concepts.
-- canonical_subject, consistent with the read-time-resolution rule this
-- schema already follows everywhere else. Safe to re-run.

ALTER TABLE student_sessions ADD COLUMN IF NOT EXISTS practice_category TEXT;
ALTER TABLE student_sessions ALTER COLUMN canonical_subject DROP NOT NULL;
