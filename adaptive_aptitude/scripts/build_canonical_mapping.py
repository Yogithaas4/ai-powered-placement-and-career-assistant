"""
build_canonical_mapping.py  (v2 -- fixes a display bug in v1)
----------------------------------------------------------------
Phase 0, step 2: build a canonical subject taxonomy for the question
bank.

BUGFIX from v1: the JSON summary used to be built by scanning records
and keeping whichever record was seen FIRST for each raw subject. For a
topic-conditional subject like "Mathematics" (most topics fall back to
"Engineering Mathematics", but Discrete-Math-flavored topics override to
"Discrete Mathematics" -- see TOPIC_OVERRIDES), this made the displayed
canonical_subject essentially random depending on file order, even
though the actual per-record resolution (used by apply_canonical_mapping.py
and the CSV preview below) was always correct. Fixed by reading
canonical_subject_mapping.json's summary directly from SUBJECT_MAP
(the static, authoritative fallback table) instead of from per-record
scanning. The per-record CSV preview still uses the topic-aware
resolver, since that one SHOULD vary per record.

Output: reports/canonical_subject_mapping.json
    { raw_subject: {"canonical_subject": ..., "rule": ..., "needs_review": bool,
                     "record_count": int} }
  -- canonical_subject here is always the FALLBACK value for that raw
  subject. If TOPIC_OVERRIDES has entries for this raw_subject, the rule
  text says so explicitly ("fallback; see TOPIC_OVERRIDES...") -- check
  reports/canonical_mapping_review.csv for the true per-record resolution.

Also emits reports/canonical_mapping_review.csv, a flat per-record
preview (subject, topic -> canonical_subject) for manual review before
this is applied at ingestion time -- this one DOES use the topic-aware
resolver, so it's the accurate source if you want to see exactly what
a given record resolves to.

Usage:
    python build_canonical_mapping.py --clean questions_clean.json \
        --images questions_with_image.json --out-dir reports
"""

import argparse
import csv
import json
import os
from collections import Counter

from audit_dataset import load_json_tolerant


# ── Rule 1: direct subject -> canonical subject remapping ──────────────────
# Every raw subject label in the dataset must appear here exactly once.
# "rule" documents WHY, so a human reviewer doesn't have to re-derive it.

SUBJECT_MAP = {
    # --- straightforward case/pluralization duplicates ---
    "Operating System":                    ("Operating Systems", "case/pluralization dupe", False),
    "Operating Systems":                   ("Operating Systems", "canonical form", False),

    "Programming and Data Structure":      ("Programming and Data Structures", "case/pluralization dupe", False),
    "Programming and Data Structures":     ("Programming and Data Structures", "canonical form", False),
    "Data Structures":                     ("Programming and Data Structures", "same GATE syllabus section", False),
    "Programming":                         ("Programming and Data Structures", "100/104 topics are 'Programming in C'", False),
    "Programming Fundamentals":            ("Programming and Data Structures", "all 9 records are Programming Fundamentals topic", False),
    "Stack":                               ("Programming and Data Structures", "single stray record, data-structure topic", True),
    "Binary Tree":                         ("Programming and Data Structures", "single stray record, data-structure topic", True),

    "Algorithms":                          ("Algorithms", "canonical form (kept distinct from PDS per GATE syllabus)", False),
    "Data Structures and Algorithms":      ("Algorithms", "majority topics (graph algo/sort/search/hash) are algorithmic", True),

    # --- General Aptitude absorbs its own sub-buckets ---
    "General Aptitude":                    ("General Aptitude", "canonical form", False),
    "Quantitative Aptitude":               ("General Aptitude", "already a topic inside General Aptitude (n=605)", False),
    "Verbal Aptitude":                     ("General Aptitude", "already a topic inside General Aptitude (n=593)", False),

    # --- Mathematics: this is the FALLBACK only. Topics in
    #     TOPIC_OVERRIDES below (Set Theory, Graph Theory, etc.) route to
    #     "Discrete Mathematics" instead, regardless of this row. Every
    #     other topic under raw subject "Mathematics" (Calculus,
    #     Probability, Linear Algebra, Geometry, Algebra, Numerical
    #     Methods, ...) uses this fallback. ---
    "Mathematics":                         ("Engineering Mathematics",
                                             "fallback for non-discrete-math topics; "
                                             "see TOPIC_OVERRIDES for the Discrete-Math split", True),
    "Engineering Mathematics":             ("Engineering Mathematics", "canonical form", False),
    "Discrete Mathematics":                ("Discrete Mathematics", "canonical form, kept distinct (large, coherent bucket)", False),

    # --- subjects already clean / large / unambiguous ---
    "Databases":                           ("Databases", "canonical form", False),
    "Computer Networks":                   ("Computer Networks", "canonical form", False),
    "Digital Logic":                       ("Digital Logic", "canonical form", False),
    "Digital Electronics":                 ("Digital Logic", "same GATE syllabus section as Digital Logic", True),
    "Computer Organization and Architecture": ("Computer Organization and Architecture", "canonical form", False),
    "Theory of Computation":               ("Theory of Computation", "canonical form", False),
    "Compiler Design":                     ("Compiler Design", "canonical form", False),
    "Artificial Intelligence":             ("Artificial Intelligence", "canonical form (GATE DA)", False),
    "Machine Learning":                    ("Artificial Intelligence", "merged into Artificial Intelligence (small bucket, closely related GATE DA topic)", False),
    "Operations Research":                 ("Operations Research", "canonical form (GATE DA)", False),
    "Software Engineering":                ("Software Engineering", "canonical form", False),
    "Cybersecurity":                       ("Cybersecurity", "canonical form", False),
    "System Software":                     ("System Software", "canonical form", False),
    "Data Mining and Warehousing":         ("Data Mining and Warehousing", "canonical form", False),
    "Big Data Systems":                    ("Big Data Systems", "canonical form", False),
    "Cloud Computing":                     ("Cloud Computing", "canonical form", False),
    "Web Technologies":                    ("Web Technologies", "canonical form", False),
    "Physics":                             ("Engineering Mathematics", "single stray record, closest to Eng. Math content", True),

    # --- grab-bag: genuinely mixed, kept as its own bucket pending review ---
    "Computer Science":                    ("Computer Science (Miscellaneous)",
                                             "mixed grab-bag: Software Eng/OOP/Java/Graphics/Web/Cloud topics; "
                                             "not force-merged into any single GATE subject", True),
}


# ── Rule 2: topic-conditional overrides ─────────────────────────────────────
# For a raw subject where the correct canonical subject depends on the
# topic (currently only "Mathematics"), list which topics route to
# which canonical subject. Anything not listed falls back to SUBJECT_MAP.

DISCRETE_MATH_TOPICS = {
    "Discrete Mathematics", "Set Theory", "Mathematical Logic",
    "Propositional Logic", "Relations", "Functions", "Predicate Logic",
    "Graph Theory", "Combinatorics", "Number Theory",
}

TOPIC_OVERRIDES = {
    "Mathematics": {
        topic: ("Discrete Mathematics", "topic content is discrete-math, not calculus/probability/linear-algebra")
        for topic in DISCRETE_MATH_TOPICS
    }
}


def resolve_canonical_subject(raw_subject: str, raw_topic: str):
    """Per-RECORD resolution (topic-aware). Used for the CSV preview
    only -- the JSON summary uses SUBJECT_MAP directly, see build_mapping()."""
    overrides = TOPIC_OVERRIDES.get(raw_subject, {})
    if raw_topic in overrides:
        canonical, rule = overrides[raw_topic]
        return canonical, rule, False

    if raw_subject in SUBJECT_MAP:
        return SUBJECT_MAP[raw_subject]

    return raw_subject, "UNMAPPED subject -- not seen when mapping was built, needs manual entry", True


def build_mapping(clean_path, images_path, out_dir):
    clean = load_json_tolerant(clean_path)
    images = load_json_tolerant(images_path)
    all_records = clean + images

    subject_counts = Counter(r.get("subject") for r in all_records)

    # ── JSON summary: read directly from SUBJECT_MAP (the static,
    # authoritative fallback), NOT by scanning records. This is the fix --
    # a subject's summary entry no longer depends on which record happens
    # to be seen first. ──
    subject_summary = {}
    unmapped = set()
    for raw_subject, count in subject_counts.items():
        if raw_subject in SUBJECT_MAP:
            canonical, rule, needs_review = SUBJECT_MAP[raw_subject]
        else:
            canonical, rule, needs_review = raw_subject, "UNMAPPED subject -- needs manual entry", True
            unmapped.add(raw_subject)
        subject_summary[raw_subject] = {
            "canonical_subject": canonical,
            "rule": rule,
            "needs_review": needs_review,
            "record_count": count,
        }

    # ── CSV preview: per-record, topic-aware resolution (this one DOES
    # vary within a subject, e.g. Mathematics) ──
    rows = []
    for r in all_records:
        raw_subject = r.get("subject")
        raw_topic = r.get("topic")
        canonical, rule, needs_review = resolve_canonical_subject(raw_subject, raw_topic)
        rows.append({
            "question_id": r.get("question_id"),
            "raw_subject": raw_subject,
            "raw_topic": raw_topic,
            "canonical_subject": canonical,
            "needs_review": needs_review,
            "rule": rule,
        })

    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "canonical_subject_mapping.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(subject_summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "canonical_mapping_review.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "raw_subject", "raw_topic",
                                                "canonical_subject", "needs_review", "rule"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")

    if unmapped:
        print(f"WARNING: {len(unmapped)} raw subject(s) had no mapping rule: {sorted(unmapped)}")

    topic_conditional = [s for s in subject_summary if s in TOPIC_OVERRIDES]
    if topic_conditional:
        print(f"\nNOTE: these raw subjects are topic-conditional -- the JSON entry below shows "
              f"only the FALLBACK canonical_subject; some records route elsewhere based on topic "
              f"(see canonical_mapping_review.csv for the true per-record value):")
        for s in topic_conditional:
            v = subject_summary[s]
            print(f"  - '{s}': fallback -> '{v['canonical_subject']}' "
                  f"({len(TOPIC_OVERRIDES[s])} topic(s) override to a different subject)")

    needs_review_subjects = [k for k, v in subject_summary.items() if v["needs_review"]]
    print(f"\n{len(needs_review_subjects)} raw subject(s) flagged needs_review (review before ingestion):")
    for k in needs_review_subjects:
        v = subject_summary[k]
        print(f"  - '{k}' ({v['record_count']} records) -> '{v['canonical_subject']}'  [{v['rule']}]")

    canonical_subjects = sorted(set(v["canonical_subject"] for v in subject_summary.values()))
    print(f"\nResult: {len(subject_summary)} raw subjects -> {len(canonical_subjects)} canonical subjects "
          f"(fallback-only count; actual distinct values after topic overrides may include "
          f"'Discrete Mathematics' pulled from the 'Mathematics' fallback too)")
    for cs in canonical_subjects:
        total = sum(v["record_count"] for v in subject_summary.values() if v["canonical_subject"] == cs)
        print(f"  {total:5d}  {cs}")

    return subject_summary, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="questions_clean.json")
    ap.add_argument("--images", default="questions_with_image.json")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    build_mapping(args.clean, args.images, args.out_dir)
