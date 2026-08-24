# Data Repair + One-Command Pipeline

## What I checked

I re-audited your **current** `questions_clean.json` / `questions_with_image.json`
(8,855 + 241 = 9,096 records — I confirmed the 2 fewer from before are the
two `fill_blank` questions with unanswerable open-ended prompts you deleted,
not a data-loss bug).

**Your clarification applied correctly:** everything in `questions_with_image.json`
has `has_image=True` and is rendered as an image — so I only checked answer
presence/validity there, not text completeness. Result: **0 missing answers**,
6 legitimate semicolon-delimited multi-select answers (not a defect — see
below), and 1 genuinely broken record.

**`questions_clean.json` got the full audit** since it's rendered from text.

## Genuine corrections applied (`final_repair.py`, already run on the files in `data/`)

| Count | Fix |
|---|---|
| 35 | `image_based`/`graph_based`/`table_based`/`diagram_based` → `mcq` — complete text MCQ, no image needed, `has_image=False` |
| 55 | `numerical` → `mcq` — answer was a letter (A–D) with complete options, mislabeled |
| 1 | `mcq` → `fill_blank` (`em::2.37.1`) — symbolic integral answer, not a lettered choice |
| 7 | `mcq` with `True`/`False` answer, no options → added True/False options, remapped answer to A/B |
| 3 | Unicode minus sign (`−` U+2212) normalized to ASCII hyphen (`-`) in numerical answers — this is exactly the "negative sign" false-positive you flagged earlier; not broken data, just an encoding mismatch with a plain `float()` parser |

**Verified idempotent** — ran the script twice, second run made 0 changes.
Backups written as `questions_clean.json.bak` / `questions_with_image.json.bak`
before any edit.

## Still genuinely broken (need manual content lookup, not auto-fixable)

- **`isro_w_cover::38.16.3`**: `correct_answer='X'`, not a valid option letter. Needs the actual tree diagrams checked against the source PDF.
- **`da::3.20.49`**: `correct_answer='TBA'`, a placeholder that was never filled in.

## Not a defect, but worth knowing for the answer-checker

Multi-select answers are **semicolon-delimited in 184 records** (`"A;C;D"`)
vs. **comma-delimited in 33** (`"A,C,D"`). Semicolon is the dominant
convention. Whatever builds the type-aware answer checker (Phase 2) needs
to accept both.

## The one-command pipeline (your second question)

You're right that re-running everything by hand every time the JSON
changes doesn't scale. `scripts/run_full_pipeline.py` chains the whole
thing into one command:

```bash
python scripts/run_full_pipeline.py --clean "D:\notes\capstone\project\ai-powered-placement-and-career-assistant\adaptive_aptitude\data\questions_clean.json" --images "D:\notes\capstone\project\ai-powered-placement-and-career-assistant\adaptive_aptitude\data\questions_with_image.json"
```

What it does, in order:
1. **Applies `db/schema_extended.sql`** (no-op if already current)
2. **`sync_ingest_questions.py`** — new script, full-sync not just insert: upserts everything in the JSON, **and removes from Postgres anything no longer in the JSON** (handles cases like your 2 deleted questions automatically). If a question being removed already has real student responses recorded against it, the delete is safely blocked (FK-protected) and reported rather than silently destroying response history.
3. **Regenerates `canonical_subject_mapping.json`** from the current record counts (so the taxonomy report doesn't quietly go stale if subject distribution shifts)
4. **Regenerates `practice_category_mapping.json`** to match
5. **Validates the two mapping files agree** (catches drift automatically — this is the exact class of bug that bit us earlier — refuses to proceed if they don't)
6. **Applies the mapping + derives concepts/DAG** from the now-current data

**Tested for real, not just written:** ran this against a completely blank
database with your actual repaired 9,096-question data, twice in a row.
Both runs gave identical results — 2,539 concepts, 0 unresolved
canonical_subject/practice_category/concept_id. Every step is idempotent,
so running this after a small manual JSON edit is cheap and safe.

Any time you edit either JSON file — one fix, a batch of fixes, whatever —
this is the only command you need to run afterward.
