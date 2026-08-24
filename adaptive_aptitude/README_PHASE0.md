# Phase 0 — Audit, Data Quality, Taxonomy, Migration

This delivers Phase 0 from the plan: dataset audit, canonical
subject/topic taxonomy, extended Postgres schema, and the
SQLite → Postgres migration for mastery/session/interaction data.
Everything here was built and run against your actual
`questions_clean.json` (8,855) + `questions_with_image.json` (243) =
9,098 records — not assumed from the docs.

## What's in this delivery

```
reports/
  audit_report.md / .json                -- full dataset audit
  issues_for_manual_review.json / .csv   -- question_ids needing hand-fixes, by category
  canonical_subject_mapping.json         -- 36 raw subjects -> 24 canonical subjects
  canonical_mapping_review.csv           -- per-record preview of the mapping, for review
  practice_category_mapping.json         -- 24 canonical subjects -> 5 practice categories
db/
  schema_extended.sql              -- concepts, DAG, sessions, responses, mastery,
                                       experiment tracking, practice-category tables
                                       (extends db/schema.sql)
scripts/
  audit_dataset.py                       -- regenerates the audit report
  export_review_issues.py                -- regenerates the manual-review export
  build_canonical_mapping.py             -- regenerates the subject canonical mapping
  build_practice_category_mapping.py     -- regenerates the practice-category grouping
  apply_canonical_mapping.py             -- applies both mappings + question-type
                                             consolidation to Postgres, and auto-derives
                                             the concept graph from real data (replaces
                                             the old hand-authored, CN-only DAG)
  migrate_sqlite_to_postgres.py          -- migrates adaptive_platform.db -> Postgres
```

## Updates from your review

- **Duplicate-text groups**: noted, not touched. Since you already ran a dedup
  pass and some flagged groups may be same-template/different-numbers problems
  (not true dupes), I did not add any automatic deduplication here — flagging
  it in the audit was informational, not a to-do.
- **`multi_select_suspicious_single_answer`** and **`match_following_missing_mapping_data`**:
  confirmed as non-issues (multi_select can legitimately have one correct
  option; match_following resolves via the lettered option, not needing
  `left_items`/`right_items` populated) — **removed from the manual-review
  export**, they were never real problems.
- **`numerical_answer_not_parseable_as_number`**, **missing image/graph/table
  data**, **missing correct_answer**, **option-dependent-with-no-options**:
  exported to `reports/issues_for_manual_review.json` / `.csv`, one row per
  `(question_id, category)`, with enough context (subject, topic, snippet,
  current `correct_answer`) to fix by hand. **264 distinct question_ids**
  flagged across 5 categories. Re-run `export_review_issues.py` any time
  after you've made fixes to see what's left.
- **Question-type consolidation**: `table_based` (6), `graph_based` (1), and
  `diagram_based` (6) collapse into `image_based` for *selection* purposes —
  13 records total. This is now a static lookup table
  (`question_type_canonical_map`, seeded by the schema itself), not a
  column on `questions`. Raw `question_type` is completely untouched — the
  renderer can still branch on `image_info.type` (`"graph"`, `"table"`, etc.)
  to decide whether to draw a table vs. show a picture vs. render a graph.
- **Practice categories for the subject picker**: your 24 canonical subjects
  group into **5 practice categories** (not 24, not showing raw subjects to
  the student):

  | Category | Questions | Canonical subjects included |
  |---|---|---|
  | Core CS (Systems & Theory) | 3,524 | Databases, Computer Networks, Operating Systems, Digital Logic, Computer Organization and Architecture, Theory of Computation, Compiler Design, Computer Science (Misc), Cloud Computing, Web Technologies, Software Engineering, Cybersecurity, System Software |
  | Aptitude | 2,287 | General Aptitude |
  | Programming & DSA | 1,637 | Programming and Data Structures, Algorithms |
  | Engineering Mathematics | 1,478 | Discrete Mathematics, Engineering Mathematics, Physics |
  | Data Science & AI | 172 | Artificial Intelligence, Machine Learning, Operations Research, Data Mining and Warehousing, Big Data Systems |

  Proposed default in `scripts/build_practice_category_mapping.py`, a plain
  editable dict. "Core CS" is the biggest bucket — split it into "Systems"
  (OS/CN/COA/Digital Logic) vs. "Theory & Databases" (TOC/Compiler/Databases)
  for 6 categories instead of 5 if you want.

## Schema design: zero redundancy on `questions` (v2)

**`questions` is never modified — no `ALTER TABLE`, no new columns, nothing
to keep in sync.** All canonical-subject / canonical-topic /
canonical-question-type / practice-category / concept resolution happens at
**read time** through a single view, `questions_resolved`, defined at the
bottom of `db/schema_extended.sql`. It joins `questions` against three small
lookup tables (`subject_topic_canonical_map`, `subject_practice_category_map`,
`question_type_canonical_map`) and `concepts`. Query it like a table:

```sql
SELECT * FROM questions_resolved WHERE practice_category = 'Aptitude' LIMIT 10;
SELECT * FROM questions_resolved WHERE concept_id = 'operating-systems::process-scheduling::sjf';
```

Every consumer (FastAPI backend, Streamlit, evaluation/research scripts)
should query this view instead of re-implementing the mapping logic
themselves — that's what keeps the resolution logic in exactly one place
without storing it redundantly.

**One exception, and it's a different kind of thing:** `student_responses.concept_id`
*is* still a stored column, on purpose. That table is an event/fact log —
it needs to record which concept a response was scored against *at the time
it was answered*, so that if you later edit the canonical mapping, historical
responses don't silently change meaning. That's standard practice for
immutable log tables and isn't the same "redundant cache of static reference
data" problem `questions` would have had.

## Key audit findings (see reports/audit_report.md for full detail)

- **9,098 total records**, 3 missing `correct_answer`, 135 option-dependent
  questions with no options at all — small enough to hand-fix, listed in the report.
- **135 type-specific issues** worth checking before ingestion: 23 `match_following`
  questions missing `left_items`/`right_items`/`correct_mapping`, 92 `numerical`
  answers that don't parse as a number (likely units/ranges baked into the string),
  36 visual-type questions with no `image_info` at all, 22 `multi_select` questions
  whose `correct_answer` looks like a single letter instead of a set.
- **518 duplicate-question-text groups covering 1,099 records** — not previously
  flagged. Likely the same GATE/NET question re-extracted from overlapping PDF
  batches. **Recommend deciding a dedup policy before ingestion** (keep first by
  question_id, keep the one with richer `image_info`, or keep both if from genuinely
  different source papers) — I did not dedup automatically since a wrong merge
  loses data permanently.
- **36 raw subject labels, 329 raw topics** — far more fragmented than just
  "Operating System" vs "Operating Systems". Real duplication includes Data
  Structures content spread across 5 different subject labels, and a
  "Mathematics" bucket that mixes Discrete-Math topics with Calculus/Probability/
  Linear-Algebra topics under one label.

## Canonical taxonomy: what I did and why

`build_canonical_mapping.py` maps all 36 raw subjects → 24 canonical subjects.
Every merge decision has a `rule` string recorded (see
`canonical_subject_mapping.json`) so nothing was silently guessed. **6 buckets
are flagged `needs_review: true`** — these are genuine judgment calls, not
mechanical renames:

| Raw subject | Proposed canonical | Why it's flagged |
|---|---|---|
| Data Structures and Algorithms (46) | Algorithms | Majority topics are algorithmic, but "Programming Fundamentals" is its single largest topic |
| Computer Science (118) | Computer Science (Miscellaneous) | Genuine grab-bag: Software Engineering, OOP, Java, Computer Graphics, Web, Cloud — not one GATE subject |
| Digital Electronics (4) | Digital Logic | Same GATE syllabus section, but tiny sample |
| Physics (2), Stack (1), Binary Tree (1) | various | Single/near-single stray records |

**Before running `apply_canonical_mapping.py` against your real Postgres, open
`reports/canonical_subject_mapping.json` and either accept or edit these 6
entries** — it's a plain JSON dict, editable by hand.

The "Mathematics" bucket is handled specially: it's split by *topic*, not
merged wholesale — topics like Set Theory/Mathematical Logic/Graph Theory route
to "Discrete Mathematics", while Calculus/Probability/Linear Algebra route to
"Engineering Mathematics" (this logic lives in `TOPIC_OVERRIDES` in both
`build_canonical_mapping.py` and `apply_canonical_mapping.py`).

## Concept graph: auto-derived, not hand-authored

The old `core/concept_dag.py` only really covered Computer Networks. Instead of
hand-authoring more, `apply_canonical_mapping.py` derives one `concept_id` per
distinct `(canonical_subject, canonical_topic, subtopic)` actually present in
your 9,098 questions — so DAG coverage automatically matches your real question
bank across all 24 subjects.

Prerequisite edges are auto-derived **only within the same topic**, ordering
concepts by average question difficulty as a weak proxy for "taught before" —
all inserted with `confidence='heuristic'`. Cross-subject prerequisites (e.g.
"Programming and Data Structures" before "Algorithms") are **not** guessed;
a template CSV (`reports/manual_prereq_edges_template.csv`) is generated for
you or a domain reviewer to fill in properly. Treat the auto-derived DAG as a
reasonable starting point for engineering/testing, not as a validated
pedagogical ordering to cite in the paper.

## Run order (against your real Docker Postgres)

```bash
# 1. Copy these files into your repo (adaptive_aptitude/) alongside the
#    existing db/schema.sql, keeping the same relative layout.

# 2. Re-run the audit + mapping against your live data if it's changed
#    since this snapshot (optional -- already run once and included above).
python scripts/audit_dataset.py --clean questions_clean.json --images questions_with_image.json --out-dir reports
python scripts/build_canonical_mapping.py --clean questions_clean.json --images questions_with_image.json --out-dir reports

# 3. Review reports/canonical_subject_mapping.json (6 flagged entries) and
#    reports/practice_category_mapping.json (5-category grouping), edit if needed.
#    Optionally regenerate the manual-review export after fixing data issues:
python scripts/export_review_issues.py --clean questions_clean.json --images questions_with_image.json --out-dir reports

# 4. Apply the extended schema (needs docker compose up -d already running)
psql -h localhost -p 5434 -U adaptive_user -d adaptive_aptitude -f db/schema_extended.sql

# 5. Apply the canonical mapping + question-type consolidation + practice
#    categories, and derive concepts/DAG, all in Postgres
python scripts/apply_canonical_mapping.py \
    --mapping reports/canonical_subject_mapping.json \
    --practice-category-mapping reports/practice_category_mapping.json

# 6. Migrate mastery/session/interaction data off SQLite
python scripts/migrate_sqlite_to_postgres.py --sqlite adaptive_platform.db
```

## Known limitation surfaced during migration (not papered over)

The old SQLite `interaction_log` has no `session_id` column, but the new
`student_responses.session_id` is `NOT NULL` by design (a response must
belong to a tracked session — this was one of the original security/integrity
gaps). Legacy interaction rows without a resolvable session are written to
`reports/legacy_interactions_no_session.csv` instead of being force-fit into
a fabricated session. In your current snapshot `interaction_log` is empty (0
rows) so this doesn't bite yet, but will the first time you migrate a
SQLite file with real history — decide then whether to backfill a synthetic
session per student or just start responses tracking fresh from the
Postgres-only phase.

## Not done in Phase 0 (belongs to later phases)

- Deduplication of the 1,099 duplicate-text records (needs your dedup policy decision).
- Fixing the 135 type-specific data issues (small enough to hand-fix or re-run
  targeted extraction on).
- Server-side, type-aware answer checking in the API (Phase 2 — this is the
  security bug where `correct_answer` currently comes from the client).
- The actual model-comparison experiments (Phase 1) — `experiment_runs` /
  `experiment_metrics` tables are ready to receive results once you build the
  experiment harness.
