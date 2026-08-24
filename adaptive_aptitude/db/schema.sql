-- Adaptive Aptitude Platform -- question bank schema.
-- Runs automatically on first Postgres container start (mounted into
-- /docker-entrypoint-initdb.d/). Safe to re-run manually too.

CREATE TABLE IF NOT EXISTS questions (
    question_id             TEXT PRIMARY KEY,
    question_type           TEXT,
    question                TEXT NOT NULL,

    option_a                TEXT,
    option_b                TEXT,
    option_c                TEXT,
    option_d                TEXT,
    correct_answer          TEXT,

    left_items              JSONB,
    right_items             JSONB,
    correct_mapping         JSONB,

    has_image                BOOLEAN NOT NULL DEFAULT FALSE,
    image_key                TEXT,        -- object key in MinIO, null if no raster image
    image_meta                JSONB,       -- description / graph_description / table_data / tree_structure / page

    subject                  TEXT,
    topic                    TEXT,
    subtopic                 TEXT,
    difficulty                TEXT,
    time_expected_minutes     INTEGER,

    source                    TEXT,
    validation_status         TEXT,
    raw_tags                  TEXT,
    notes                      TEXT,
    extra_fields               JSONB,

    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_questions_subject_topic ON questions (subject, topic);
CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions (difficulty);
CREATE INDEX IF NOT EXISTS idx_questions_has_image ON questions (has_image);
