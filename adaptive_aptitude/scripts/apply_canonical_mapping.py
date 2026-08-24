"""
apply_canonical_mapping.py
---------------------------
Phase 0, Step 5.

Applies the canonical subject/practice-category mappings and derives:
    1. subject_topic_canonical_map
    2. subject_practice_category_map
    3. concepts
    4. subtopic_concept_map
    5. heuristic concept prerequisite DAG edges

This version combines:

- Previous version's learner-history-safe reconciliation:
    * removes stale subtopic mappings
    * removes obsolete concepts only when they have never been used
    * preserves concepts referenced by student_responses/student_mastery
    * replaces heuristic DAG edges on every run
    * preserves manually/expert-reviewed DAG edges
    * preserves the manual prerequisite template

- New version's reporting strategy:
    * accepts --report-dir
    * writes all generated reports to that directory
    * reports concept-label merges
    * writes the prerequisite template to that directory

BUGFIX (this pass): stale-concept deletion previously only checked
student_responses/student_mastery before deleting a concept no longer
derivable from current data. Since concept_dependencies has
ON DELETE CASCADE back to concepts, a concept carrying a MANUAL
prerequisite edge could be silently deleted -- destroying that manual
edge with no warning. Reproduced and confirmed against a real database
before fixing. Now also checks for manual edges (heuristic edges don't
need this protection -- they're regenerated fresh every run anyway).

Usage:

    python apply_canonical_mapping.py \
        --mapping reports/canonical_subject_mapping.json \
        --practice-category-mapping reports/practice_category_mapping.json \
        --report-dir reports
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

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


DIFFICULTY_RANK = {
    "Easy": 0,
    "Medium": 1,
    "Hard": 2,
}


def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB,
    )


def slugify(s: str) -> str:
    """
    Create a deterministic slug for concept_id components.

    '+' is converted to '-plus-' BEFORE punctuation is removed, so:
        B Tree  -> b-tree
        B-Tree  -> b-tree
        B+ Tree -> b-plus-tree
    This prevents B-tree and B+-tree from being accidentally merged.
    """
    s = (s or "").strip().lower()
    s = s.replace("+", "-plus-")
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or "unknown"


DISCRETE_MATH_TOPICS = {
    "Discrete Mathematics", "Set Theory", "Mathematical Logic",
    "Propositional Logic", "Relations", "Functions", "Predicate Logic",
    "Graph Theory", "Combinatorics", "Number Theory",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def step1_upsert_subject_map(conn, subject_map):
    with conn.cursor() as cur:
        for raw_subject, entry in subject_map.items():
            cur.execute("""
                INSERT INTO subject_topic_canonical_map
                    (raw_subject, raw_topic, canonical_subject, rule, needs_review)
                VALUES (%s, NULL, %s, %s, %s)
                ON CONFLICT (raw_subject) WHERE raw_topic IS NULL
                DO UPDATE SET
                    canonical_subject = EXCLUDED.canonical_subject,
                    rule = EXCLUDED.rule,
                    needs_review = EXCLUDED.needs_review
            """, (raw_subject, entry["canonical_subject"], entry["rule"], entry["needs_review"]))

        for topic in DISCRETE_MATH_TOPICS:
            cur.execute("""
                INSERT INTO subject_topic_canonical_map
                    (raw_subject, raw_topic, canonical_subject, rule, needs_review)
                VALUES ('Mathematics', %s, 'Discrete Mathematics',
                        'topic content is discrete-math, not calculus/probability/linear-algebra', FALSE)
                ON CONFLICT (raw_subject, raw_topic)
                DO UPDATE SET
                    canonical_subject = EXCLUDED.canonical_subject,
                    rule = EXCLUDED.rule,
                    needs_review = EXCLUDED.needs_review
            """, (topic,))
    conn.commit()
    print(f"Upserted {len(subject_map) + len(DISCRETE_MATH_TOPICS)} rows into subject_topic_canonical_map")


def step2_upsert_practice_categories(conn, practice_category_path):
    practice_map_raw = load_json(practice_category_path)
    with conn.cursor() as cur:
        for canonical_subject, entry in practice_map_raw.items():
            cur.execute("""
                INSERT INTO subject_practice_category_map (canonical_subject, practice_category)
                VALUES (%s, %s)
                ON CONFLICT (canonical_subject) DO UPDATE SET practice_category = EXCLUDED.practice_category
            """, (canonical_subject, entry["practice_category"]))
    conn.commit()
    print(f"Upserted {len(practice_map_raw)} rows into subject_practice_category_map")


def step3_derive_concepts(conn, report_dir):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COALESCE(specific_map.canonical_subject, generic_map.canonical_subject, q.subject) AS canonical_subject,
                q.topic AS canonical_topic,
                q.subtopic,
                COUNT(*) AS n,
                MODE() WITHIN GROUP (ORDER BY q.difficulty) AS typical_difficulty
            FROM questions q
            LEFT JOIN subject_topic_canonical_map specific_map
                   ON specific_map.raw_subject = q.subject AND specific_map.raw_topic = q.topic
            LEFT JOIN subject_topic_canonical_map generic_map
                   ON generic_map.raw_subject = q.subject AND generic_map.raw_topic IS NULL
            WHERE q.subtopic IS NOT NULL AND q.subtopic <> ''
            GROUP BY 1, 2, 3
        """)
        groups = cur.fetchall()

    by_concept_id = defaultdict(list)
    for group in groups:
        concept_id = f"{slugify(group['canonical_subject'])}::{slugify(group['canonical_topic'])}::{slugify(group['subtopic'])}"
        by_concept_id[concept_id].append(group)

    merges_log = []
    concept_rows = []
    subtopic_map_rows = []

    for concept_id, members in by_concept_id.items():
        total_question_count = sum(member["n"] for member in members)
        members_sorted = sorted(members, key=lambda member: -member["n"])
        representative = members_sorted[0]

        concept_rows.append((
            concept_id, representative["canonical_subject"], representative["canonical_topic"],
            representative["subtopic"], total_question_count,
        ))

        for member in members:
            subtopic_map_rows.append((
                member["canonical_subject"], member["canonical_topic"], member["subtopic"], concept_id,
            ))

        if len(members) > 1:
            merges_log.append({
                "concept_id": concept_id,
                "canonical_subject": representative["canonical_subject"],
                "canonical_topic": representative["canonical_topic"],
                "merged_subtopic_labels": " | ".join(sorted(set(member["subtopic"] for member in members))),
                "kept_label": representative["subtopic"],
                "total_question_count": total_question_count,
            })

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO concepts (concept_id, canonical_subject, canonical_topic, subtopic, question_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (concept_id) DO UPDATE SET
                subtopic = EXCLUDED.subtopic,
                question_count = EXCLUDED.question_count
        """, concept_rows, page_size=500)

        psycopg2.extras.execute_batch(cur, """
            INSERT INTO subtopic_concept_map (canonical_subject, canonical_topic, raw_subtopic, concept_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (canonical_subject, canonical_topic, raw_subtopic)
            DO UPDATE SET concept_id = EXCLUDED.concept_id
        """, subtopic_map_rows, page_size=500)

        # Remove stale spelling mappings
        current_map_keys = {(row[0], row[1], row[2]) for row in subtopic_map_rows}
        cur.execute("SELECT canonical_subject, canonical_topic, raw_subtopic FROM subtopic_concept_map")
        stale_map_keys = [(row[0], row[1], row[2]) for row in cur.fetchall()
                           if (row[0], row[1], row[2]) not in current_map_keys]

        if stale_map_keys:
            psycopg2.extras.execute_batch(cur, """
                DELETE FROM subtopic_concept_map
                WHERE canonical_subject = %s AND canonical_topic = %s AND raw_subtopic = %s
            """, stale_map_keys, page_size=500)

        # Identify stale concepts
        current_concept_ids = set(by_concept_id.keys())
        cur.execute("SELECT concept_id FROM concepts")
        stale_concept_ids = [row[0] for row in cur.fetchall() if row[0] not in current_concept_ids]

        deleted_stale = 0
        retained_stale_learner_data = 0
        retained_stale_manual_edges = 0

        # BUGFIX: also protect concepts carrying a MANUAL prerequisite
        # edge. concept_dependencies has ON DELETE CASCADE back to
        # concepts, so without this a manual edge could be silently
        # destroyed the moment its concept becomes stale. Heuristic
        # edges don't need this -- they're wiped and regenerated fresh
        # every run in step4 regardless.
        if stale_concept_ids:
            cur.execute("""
                DELETE FROM concepts c
                WHERE c.concept_id = ANY(%s)
                  AND NOT EXISTS (SELECT 1 FROM student_responses sr WHERE sr.concept_id = c.concept_id)
                  AND NOT EXISTS (SELECT 1 FROM student_mastery sm WHERE sm.concept_id = c.concept_id)
                  AND NOT EXISTS (
                      SELECT 1 FROM concept_dependencies cd
                      WHERE cd.confidence = 'manual'
                        AND (cd.prereq_concept_id = c.concept_id OR cd.dependent_concept_id = c.concept_id)
                  )
                RETURNING c.concept_id
            """, (stale_concept_ids,))

            deleted_ids = {row[0] for row in cur.fetchall()}
            deleted_stale = len(deleted_ids)
            still_stale = [cid for cid in stale_concept_ids if cid not in deleted_ids]

            if still_stale:
                cur.execute("""
                    SELECT DISTINCT c.concept_id
                    FROM concepts c
                    JOIN concept_dependencies cd
                      ON cd.confidence = 'manual'
                     AND (cd.prereq_concept_id = c.concept_id OR cd.dependent_concept_id = c.concept_id)
                    WHERE c.concept_id = ANY(%s)
                """, (still_stale,))
                manual_protected = {row[0] for row in cur.fetchall()}
                retained_stale_manual_edges = len(manual_protected)
                retained_stale_learner_data = len(still_stale) - retained_stale_manual_edges

    conn.commit()

    print(f"Derived/updated {len(concept_rows)} concepts and {len(subtopic_map_rows)} subtopic-spelling mappings from actual data.")

    if stale_map_keys or deleted_stale or retained_stale_learner_data or retained_stale_manual_edges:
        print(f"Reconciled {len(stale_map_keys)} stale spelling mapping(s).")
        print(f"Removed {deleted_stale} obsolete, unreferenced concept(s).")

    if retained_stale_learner_data:
        print(f"WARNING: retained {retained_stale_learner_data} obsolete concept(s) because learner history references them.")

    if retained_stale_manual_edges:
        print(f"WARNING: retained {retained_stale_manual_edges} obsolete concept(s) because a MANUAL prerequisite "
              f"edge references them. Review reports/manual_prereq_edges_template.csv -- these concepts no longer "
              f"come from current question data, so the edge may need updating to point at whatever concept the "
              f"subtopic now resolves to.")

    if merges_log:
        os.makedirs(report_dir, exist_ok=True)
        merge_report_path = os.path.join(report_dir, "concept_subtopic_merges.csv")
        with open(merge_report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "concept_id", "canonical_subject", "canonical_topic",
                "merged_subtopic_labels", "kept_label", "total_question_count",
            ])
            writer.writeheader()
            writer.writerows(merges_log)
        print(f"NOTE: {len(merges_log)} concept(s) were formed by merging differently-punctuated subtopic labels.")
        print(f"Full merge report written to:\n  {merge_report_path}")
    else:
        print("No subtopic-label merges detected.")

    return len(concept_rows)


def step4_derive_prereq_edges(conn, report_dir):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT c.concept_id, c.canonical_subject, c.canonical_topic, c.subtopic
            FROM concepts c
            WHERE EXISTS (SELECT 1 FROM subtopic_concept_map scm WHERE scm.concept_id = c.concept_id)
        """)
        concepts = cur.fetchall()

        cur.execute("""
            SELECT concept_id, difficulty, COUNT(*) AS n
            FROM questions_resolved
            WHERE concept_id IS NOT NULL
            GROUP BY concept_id, difficulty
        """)
        diff_rows = cur.fetchall()

    diff_by_concept = {}
    for row in diff_rows:
        diff_by_concept.setdefault(row["concept_id"], {})[row["difficulty"]] = row["n"]

    def avg_difficulty_rank(concept_id):
        distribution = diff_by_concept.get(concept_id, {})
        total = sum(distribution.values())
        if not total:
            return 1.0
        return sum(DIFFICULTY_RANK.get(d, 1) * n for d, n in distribution.items()) / total

    by_topic = {}
    for concept in concepts:
        key = (concept["canonical_subject"], concept["canonical_topic"])
        by_topic.setdefault(key, []).append(concept["concept_id"])

    edges = []
    for key, concept_ids in by_topic.items():
        if len(concept_ids) < 2:
            continue
        ranked = sorted(concept_ids, key=avg_difficulty_rank)
        for prereq, dependent in zip(ranked, ranked[1:]):
            difficulty_gap = avg_difficulty_rank(dependent) - avg_difficulty_rank(prereq)
            if difficulty_gap > 0.15:
                edges.append((prereq, dependent))

    with conn.cursor() as cur:
        cur.execute("DELETE FROM concept_dependencies WHERE confidence = 'heuristic'")
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO concept_dependencies (prereq_concept_id, dependent_concept_id, confidence)
            VALUES (%s, %s, 'heuristic')
            ON CONFLICT DO NOTHING
        """, edges, page_size=500)
    conn.commit()
    print(f"Derived {len(edges)} heuristic prerequisite edges (same-topic, difficulty-ordered).")

    os.makedirs(report_dir, exist_ok=True)
    template_path = os.path.join(report_dir, "manual_prereq_edges_template.csv")

    if not os.path.exists(template_path):
        with open(template_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prereq_concept_id", "dependent_concept_id", "note"])
            writer.writerow(["programming-and-data-structures::...", "algorithms::...",
                              "EXAMPLE cross-subject edge -- fill in real ones and load separately"])
        print(f"Created manual prerequisite template:\n  {template_path}")
    else:
        print(f"Preserved existing manual prerequisite template:\n  {template_path}")


def sanity_check(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM questions")
        n_questions = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM questions_resolved WHERE canonical_subject IS NULL")
        n_unresolved_subject = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM questions_resolved WHERE practice_category IS NULL")
        n_unresolved_category = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM questions_resolved WHERE concept_id IS NULL")
        n_unresolved_concept = cur.fetchone()[0]

    print(f"\nSanity check via questions_resolved view ({n_questions} total questions):")
    print(f"  unresolved canonical_subject: {n_unresolved_subject}")
    print(f"  unresolved practice_category: {n_unresolved_category}")
    print(f"  unresolved concept_id: {n_unresolved_concept} (expected >0 -- rows with empty/NULL subtopic have no concept)")

    if n_unresolved_category > 0:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT canonical_subject, COUNT(*) AS n
                FROM questions_resolved
                WHERE practice_category IS NULL
                GROUP BY canonical_subject ORDER BY n DESC
            """)
            rows = cur.fetchall()
        print("\n  canonical_subject values with NO practice_category match:")
        for row in rows:
            print(f"    {row['n']:5d}  {row['canonical_subject']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default=os.path.join(BASE_DIR, "reports", "canonical_subject_mapping.json"))
    parser.add_argument("--practice-category-mapping",
                         default=os.path.join(BASE_DIR, "reports", "practice_category_mapping.json"))
    parser.add_argument("--report-dir", default=None,
                         help="Directory for generated reports. Defaults to the directory containing --mapping.")
    args = parser.parse_args()

    report_dir = args.report_dir or os.path.dirname(os.path.abspath(args.mapping))
    os.makedirs(report_dir, exist_ok=True)

    subject_map = load_json(args.mapping)
    conn = get_conn()
    try:
        step1_upsert_subject_map(conn, subject_map)
        step2_upsert_practice_categories(conn, args.practice_category_mapping)
        n_concepts = step3_derive_concepts(conn, report_dir)
        step4_derive_prereq_edges(conn, report_dir)
        sanity_check(conn)
        print(f"\nDone. {n_concepts} concepts now backing the DAG.")
        print("`questions` table was never modified directly -- query `questions_resolved` for canonical labels.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
