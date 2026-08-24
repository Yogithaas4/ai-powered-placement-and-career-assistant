"""
export_review_issues.py
-------------------------
Read-only export of question_ids that need manual correction, grouped
by issue category, so you can go fix them in your own editor/pipeline
without me touching questions_clean.json / questions_with_image.json
directly.

Categories exported (only the ones worth hand-fixing -- multi_select
single-answer and match_following-missing-mapping were confirmed as
non-issues and are intentionally NOT included):
    - numerical_answer_not_parseable_as_number
    - visual_type_without_image_info
    - image_based_missing_image_reference
    - graph_based_missing_graph_description
    - table_based_missing_table_data
    - missing_correct_answer
    - option_dependent_type_with_no_options

Output:
    reports/issues_for_manual_review.json
        { category: [ {question_id, origin_file, subject, topic,
                        question_type, snippet}, ... ] }
    reports/issues_for_manual_review.csv
        flat version, one row per (question_id, category) -- a question
        can appear in more than one category

Usage:
    python export_review_issues.py --clean questions_clean.json \
        --images questions_with_image.json --out-dir reports
"""

import argparse
import csv
import json
import os

from audit_dataset import load_json_tolerant

OPTION_DEPENDENT_TYPES = {"mcq", "multi_select", "fill_blank", "match_following"}


def snippet(text, n=100):
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def classify(records):
    issues = {
        "numerical_answer_not_parseable_as_number": [],
        "visual_type_without_image_info": [],
        "image_based_missing_image_reference": [],
        "graph_based_missing_graph_description": [],
        "table_based_missing_table_data": [],
        "missing_correct_answer": [],
        "option_dependent_type_with_no_options": [],
    }

    for r in records:
        qid = r.get("question_id")
        qtype = r.get("question_type")
        info = r.get("image_info") or {}
        base = {
            "question_id": qid,
            "origin_file": r.get("_origin_file"),
            "subject": r.get("subject"),
            "topic": r.get("topic"),
            "question_type": qtype,
            "snippet": snippet(r.get("question")),
        }

        if not r.get("correct_answer"):
            issues["missing_correct_answer"].append(base)

        if qtype in OPTION_DEPENDENT_TYPES:
            opts = [r.get("option_a"), r.get("option_b"), r.get("option_c"), r.get("option_d")]
            if all(o in (None, "") for o in opts):
                issues["option_dependent_type_with_no_options"].append(base)

        if qtype == "numerical":
            ca = r.get("correct_answer")
            try:
                float(str(ca).strip())
            except (TypeError, ValueError):
                row = dict(base)
                row["correct_answer"] = ca
                issues["numerical_answer_not_parseable_as_number"].append(row)

        if qtype in ("image_based", "graph_based", "table_based", "diagram_based"):
            if not r.get("has_image") and not r.get("image_info"):
                issues["visual_type_without_image_info"].append(base)
            if qtype == "image_based" and not info.get("image_reference"):
                issues["image_based_missing_image_reference"].append(base)
            if qtype == "graph_based" and not info.get("graph_description"):
                issues["graph_based_missing_graph_description"].append(base)
            if qtype == "table_based" and not info.get("table_data"):
                issues["table_based_missing_table_data"].append(base)

    return {k: v for k, v in issues.items() if v}  # drop empty categories


def run(clean_path, images_path, out_dir):
    clean = load_json_tolerant(clean_path)
    images = load_json_tolerant(images_path)
    for r in clean:
        r["_origin_file"] = "questions_clean.json"
    for r in images:
        r["_origin_file"] = "questions_with_image.json"

    issues = classify(clean + images)

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "issues_for_manual_review.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "issues_for_manual_review.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "question_id", "origin_file", "subject", "topic",
                          "question_type", "correct_answer", "snippet"])
        for category, rows in issues.items():
            for row in rows:
                writer.writerow([
                    category, row["question_id"], row["origin_file"], row["subject"],
                    row["topic"], row["question_type"], row.get("correct_answer", ""),
                    row["snippet"],
                ])

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}\n")
    total_flagged_ids = len({row["question_id"] for rows in issues.values() for row in rows})
    print(f"{total_flagged_ids} distinct question_id(s) flagged across {len(issues)} categories:")
    for k, v in issues.items():
        print(f"  {len(v):5d}  {k}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="questions_clean.json")
    ap.add_argument("--images", default="questions_with_image.json")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    run(args.clean, args.images, args.out_dir)
