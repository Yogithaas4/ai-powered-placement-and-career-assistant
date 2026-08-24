"""
build_practice_category_mapping.py
-------------------------------------
Groups the 24 canonical subjects into a small set of broad
"practice categories" for the student-facing subject picker (you don't
want to show 24 checkboxes). This is a PROPOSED default -- read the
rationale for each grouping and edit PRACTICE_CATEGORY_MAP below if you
want different boundaries before wiring it into ingestion.

Design (5 categories, chosen to mirror how GATE CSE/DA itself is
usually studied in blocks, and to keep no category wildly bigger than
the others except where the source material is genuinely lopsided):

    Aptitude                    -- General Aptitude (verbal + quant, GATE's own bucket)
    Engineering Mathematics     -- Discrete Math + Engineering Math (+ stray Physics)
    Programming & DSA           -- Programming and Data Structures + Algorithms
    Core CS (Systems & Theory)  -- OS, CN, COA, Digital Logic, TOC, Compiler Design,
                                    Databases, Software Engineering, Cybersecurity,
                                    System Software, Cloud Computing, Web Tech,
                                    Computer Science (Misc)
    Data Science & AI           -- AI, ML, Operations Research, Data Mining, Big Data

Record counts under this grouping (from your actual 9,098 questions):
    Aptitude                    2,287
    Engineering Mathematics     1,478
    Programming & DSA           1,637
    Core CS (Systems & Theory)  3,524
    Data Science & AI             172

If you'd rather split "Core CS" further (it's the biggest bucket) --
e.g. separate "Systems" (OS/CN/COA/Digital Logic) from "Theory & DB"
(TOC/Compiler/Databases) -- edit PRACTICE_CATEGORY_MAP and re-run.

Usage:
    python build_practice_category_mapping.py --mapping reports/canonical_subject_mapping.json --out-dir reports
"""

import argparse
import csv
import json
import os
from collections import defaultdict

PRACTICE_CATEGORY_MAP = {
    "General Aptitude":                        "Aptitude",

    "Discrete Mathematics":                    "Engineering Mathematics",
    "Engineering Mathematics":                 "Engineering Mathematics",

    "Programming and Data Structures":         "Programming & DSA",
    "Algorithms":                              "Programming & DSA",

    "Operating Systems":                       "Core CS (Systems & Theory)",
    "Computer Networks":                       "Core CS (Systems & Theory)",
    "Computer Organization and Architecture":  "Core CS (Systems & Theory)",
    "Digital Logic":                           "Core CS (Systems & Theory)",
    "Theory of Computation":                   "Core CS (Systems & Theory)",
    "Compiler Design":                         "Core CS (Systems & Theory)",
    "Databases":                               "Core CS (Systems & Theory)",
    "Software Engineering":                    "Core CS (Systems & Theory)",
    "Cybersecurity":                           "Core CS (Systems & Theory)",
    "System Software":                         "Core CS (Systems & Theory)",
    "Cloud Computing":                         "Core CS (Systems & Theory)",
    "Web Technologies":                        "Core CS (Systems & Theory)",
    "Computer Science (Miscellaneous)":        "Core CS (Systems & Theory)",

    "Artificial Intelligence":                 "Data Science & AI",
    "Operations Research":                     "Data Science & AI",
    "Data Mining and Warehousing":              "Data Science & AI",
    "Big Data Systems":                        "Data Science & AI",
}


def run(mapping_path, out_dir, review_csv_path=None):
    with open(mapping_path, "r", encoding="utf-8") as f:
        subject_map = json.load(f)

    canonical_totals = defaultdict(int)
    if review_csv_path and os.path.exists(review_csv_path):
        # Prefer TRUE per-record totals (post topic-override) from the CSV
        # preview -- summing subject_map's record_count directly would
        # misreport topic-conditional subjects like "Mathematics" (its
        # record_count is fallback-only, undercounting Discrete Mathematics
        # and overcounting Engineering Mathematics -- the category-level
        # total still comes out right since both feed the same practice
        # category, but the per-subject breakdown in the report would be wrong).
        with open(review_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                canonical_totals[row["canonical_subject"]] += 1
    else:
        for raw, entry in subject_map.items():
            canonical_totals[entry["canonical_subject"]] += entry["record_count"]

    unmapped = [cs for cs in canonical_totals if cs not in PRACTICE_CATEGORY_MAP]
    if unmapped:
        print(f"WARNING: {len(unmapped)} canonical subject(s) have no practice_category "
              f"assignment yet -- add them to PRACTICE_CATEGORY_MAP: {unmapped}")

    result = {
        cs: {
            "practice_category": PRACTICE_CATEGORY_MAP.get(cs, "UNASSIGNED"),
            "record_count": canonical_totals[cs],
        }
        for cs in canonical_totals
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "practice_category_mapping.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}\n")

    category_totals = defaultdict(int)
    for v in result.values():
        category_totals[v["practice_category"]] += v["record_count"]

    print("Practice categories (proposed):")
    for cat, total in sorted(category_totals.items(), key=lambda x: -x[1]):
        print(f"\n  {cat}  ({total} questions)")
        for cs, v in sorted(result.items(), key=lambda x: -x[1]["record_count"]):
            if v["practice_category"] == cat:
                print(f"      {v['record_count']:5d}  {cs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="reports/canonical_subject_mapping.json")
    ap.add_argument("--review-csv", default="reports/canonical_mapping_review.csv",
                     help="Per-record preview CSV (from build_canonical_mapping.py) used for "
                          "TRUE post-topic-override counts. Falls back to the JSON's fallback-only "
                          "counts if not found.")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    run(args.mapping, args.out_dir, args.review_csv)
