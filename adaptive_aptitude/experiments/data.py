"""
experiments/data.py
--------------------
Loads the question pool and the REAL concept DAG for the experiment harness.

This replaces two things the live prototype (api/main.py) currently does
that Phase 1 experiments must NOT do:

  1. `data/db_loader.py.load_question_dataset()` reads the raw `questions`
     table. We read `questions_resolved` instead, so every question carries
     its canonical_subject / concept_id / practice_category (the Phase 0
     work) rather than the raw, occasionally-messy `subject`/`topic`
     strings.

  2. `core/concept_dag.build_default_dag()` hand-authors ~17 Computer
     Networks concepts. We build the ConceptDAG from the `concepts` +
     `concept_dependencies` tables instead, so experiments run over the
     full 2,539-concept graph spanning every subject.

SCOPING: real sessions (and therefore experiments) are scoped by
`practice_category` (5 broad, student-facing buckets: "Core CS (Systems &
Theory)", "Aptitude", "Programming & DSA", "Engineering Mathematics",
"Data Science & AI") -- NOT by canonical_subject (22). A "Core CS" session
draws questions from Computer Networks, OS, Databases, Digital Logic, etc.
all in one pool; canonical_subject is tracked per-question underneath that
for mastery rollups/dashboard reporting, it is not the pool boundary.

Reuses connection config from data/db_loader.py so there's exactly one
place .env / POSTGRES_* values are read from.
"""

import sys
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.db_loader import get_engine, image_url_for_key  # noqa: E402
from core.concept_dag import ConceptDAG  # noqa: E402


def load_questions_resolved(practice_category: str = None, canonical_subject: str = None) -> pd.DataFrame:
    """
    Load the question pool via `questions_resolved`, i.e. WITH canonical
    labels, practice_category, and concept_id already joined in.

    Args:
        practice_category: filter to one of the 5 broad buckets. This is
            the normal way to scope a real session/experiment.
        canonical_subject: filter to one of the 22 mid-level subjects.
            Mainly useful for subject-level diagnostics/reporting, not for
            scoping a real session -- use practice_category for that.
        Pass at most one of these; passing both raises, since they express
        two different (nested) scoping intents and combining them silently
        would hide which one the caller meant.
    """
    if practice_category and canonical_subject:
        raise ValueError(
            "Pass at most one of practice_category / canonical_subject, not both -- "
            "canonical_subject is a subset of practice_category, so combining them "
            "silently picks whichever filter is more restrictive, which hides intent."
        )

    engine = get_engine()
    query = "SELECT * FROM questions_resolved"
    params = None
    if practice_category:
        query += " WHERE practice_category = %(scope)s"
        params = {"scope": practice_category}
    elif canonical_subject:
        query += " WHERE canonical_subject = %(scope)s"
        params = {"scope": canonical_subject}

    df = pd.read_sql_query(query, engine, params=params)
    df["image_url"] = df["image_key"].apply(image_url_for_key)

    n_null_concept = df["concept_id"].isna().sum()
    if n_null_concept:
        print(
            f"WARNING: {n_null_concept}/{len(df)} rows have concept_id IS NULL "
            f"(subtopic not yet mapped in subtopic_concept_map) -- these are "
            f"excluded from concept-aware selection."
        )
    n_null_category = df["practice_category"].isna().sum()
    if n_null_category:
        print(
            f"WARNING: {n_null_category}/{len(df)} rows have practice_category IS NULL "
            f"(canonical_subject not yet mapped in subject_practice_category_map) -- "
            f"these are invisible to any practice_category-scoped session."
        )

    print(
        f"Loaded {len(df)} questions from questions_resolved "
        f"across {df['practice_category'].nunique()} practice categories, "
        f"{df['canonical_subject'].nunique()} canonical subjects"
    )
    return df


def load_concept_dag(engine=None) -> ConceptDAG:
    """
    Build a ConceptDAG (the same in-memory class core/question_selector.py
    already knows how to use) from `concepts` + `concept_dependencies` +
    `subject_practice_category_map`, instead of build_default_dag()'s
    hand-authored CN-only graph.

    Each node's `.subject` = canonical_subject, `.practice_category` = the
    broad bucket it rolls up into -- so callers can scope selection by
    practice_category (real sessions) while still rolling mastery up by
    canonical_subject (dashboard reporting).
    """
    if engine is None:
        engine = get_engine()

    concepts_df = pd.read_sql_query(
        """
        SELECT c.concept_id, c.canonical_subject, c.canonical_topic, c.subtopic,
               c.question_count, pcm.practice_category
        FROM concepts c
        LEFT JOIN subject_practice_category_map pcm
               ON pcm.canonical_subject = c.canonical_subject
        """,
        engine,
    )
    edges_df = pd.read_sql_query(
        "SELECT prereq_concept_id, dependent_concept_id FROM concept_dependencies",
        engine,
    )

    n_null_category = concepts_df["practice_category"].isna().sum()
    if n_null_category:
        print(
            f"WARNING: {n_null_category}/{len(concepts_df)} concepts have no "
            f"practice_category mapping -- they are unreachable from any "
            f"category-scoped session until subject_practice_category_map is fixed."
        )

    dag = ConceptDAG()
    for row in concepts_df.itertuples(index=False):
        dag.add_concept(
            concept_id=row.concept_id,
            subject=row.canonical_subject,
            topic=row.canonical_topic,
            subtopic=row.subtopic,
            practice_category=row.practice_category,
        )
    for row in edges_df.itertuples(index=False):
        dag.add_prerequisite(row.prereq_concept_id, row.dependent_concept_id)

    print(
        f"Loaded concept DAG: {len(dag.nodes)} concepts, {len(edges_df)} prerequisite edges, "
        f"{concepts_df['practice_category'].nunique()} practice categories"
    )
    return dag


def load_experiment_data(practice_category: str = None):
    """Convenience: returns (questions_df, dag) scoped to one
    practice_category (or the whole bank if None), in one call."""
    engine = get_engine()
    questions_df = load_questions_resolved(practice_category=practice_category)
    dag = load_concept_dag(engine=engine)
    return questions_df, dag
