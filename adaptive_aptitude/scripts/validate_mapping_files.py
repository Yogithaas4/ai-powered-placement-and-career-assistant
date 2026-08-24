"""
validate_mapping_files.py
----------------------------
Run this BEFORE apply_canonical_mapping.py, especially after hand-editing
either mapping JSON. Pure Python, no database needed -- catches the class
of bug that's come up a few times now: canonical_subject_mapping.json and
practice_category_mapping.json getting edited independently and drifting
out of sync with each other.

Checks:
  1. Every canonical_subject value produced by canonical_subject_mapping.json
     has a matching key in practice_category_mapping.json. Any that don't
     would resolve to practice_category = NULL at query time -- reported
     as an ERROR.
  2. Any keys in practice_category_mapping.json that don't correspond to a
     canonical_subject value anymore -- harmless (never looked up) but
     reported as a WARNING, since it usually means the file wasn't
     regenerated after an edit and is worth cleaning up.
  3. For topic-conditional raw subjects (currently "Mathematics"), confirms
     the fallback canonical_subject differs from any TOPIC_OVERRIDES value
     for that subject and warns if they're identical (usually a sign the
     JSON's displayed value is the override, not the true fallback -- the
     bug that caused a previous mix-up).

Usage:
    python validate_mapping_files.py \
        --mapping reports/canonical_subject_mapping.json \
        --practice-category-mapping reports/practice_category_mapping.json
"""

import argparse
import json
import sys

DISCRETE_MATH_TOPICS = {
    "Discrete Mathematics", "Set Theory", "Mathematical Logic",
    "Propositional Logic", "Relations", "Functions", "Predicate Logic",
    "Graph Theory", "Combinatorics", "Number Theory",
}
TOPIC_OVERRIDE_SUBJECTS = {"Mathematics": "Discrete Mathematics"}


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--practice-category-mapping", required=True)
    args = ap.parse_args()

    subj = load(args.mapping)
    prac = load(args.practice_category_mapping)

    canonical_values = {e["canonical_subject"] for e in subj.values()}
    # Topic overrides also introduce canonical values not necessarily
    # equal to any raw subject's fallback -- add them explicitly.
    canonical_values |= set(TOPIC_OVERRIDE_SUBJECTS.values())
    practice_keys = set(prac.keys())

    errors = []
    warnings = []

    missing_in_practice = canonical_values - practice_keys
    if missing_in_practice:
        errors.append(
            f"{len(missing_in_practice)} canonical subject(s) have NO entry in "
            f"practice_category_mapping.json -- these would get practice_category=NULL:\n"
            + "\n".join(f"    - {s}" for s in sorted(missing_in_practice))
        )

    orphan_in_practice = practice_keys - canonical_values
    if orphan_in_practice:
        warnings.append(
            f"{len(orphan_in_practice)} key(s) in practice_category_mapping.json don't match "
            f"any current canonical_subject -- harmless (never looked up) but usually means the "
            f"file is stale, worth regenerating:\n"
            + "\n".join(f"    - {s}" for s in sorted(orphan_in_practice))
        )

    for raw_subject, override_target in TOPIC_OVERRIDE_SUBJECTS.items():
        entry = subj.get(raw_subject)
        if entry and entry["canonical_subject"] == override_target:
            warnings.append(
                f"'{raw_subject}' entry shows canonical_subject='{override_target}', which is the "
                f"TOPIC OVERRIDE target, not the fallback. If this is meant to be the fallback for "
                f"non-discrete-math topics under '{raw_subject}', it's probably wrong -- check "
                f"build_canonical_mapping.py's SUBJECT_MAP for the intended fallback value."
            )

    print(f"Checked {len(subj)} raw-subject entries -> {len(canonical_values)} canonical subjects, "
          f"against {len(prac)} practice-category entries.\n")

    if errors:
        print(f"❌ {len(errors)} ERROR(S) -- fix before running apply_canonical_mapping.py:\n")
        for e in errors:
            print(e + "\n")
    else:
        print("✅ No missing practice_category mappings -- safe to apply.\n")

    if warnings:
        print(f"⚠️  {len(warnings)} WARNING(S) (non-blocking):\n")
        for w in warnings:
            print(w + "\n")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
