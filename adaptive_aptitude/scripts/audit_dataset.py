"""
audit_dataset.py
-----------------
Phase 0, step 1: produce a full audit of the consolidated question bank
(questions_clean.json + questions_with_image.json) before any DB work
happens. Never mutates the source files.

Usage:
    python audit_dataset.py \
        --clean questions_clean.json \
        --images questions_with_image.json \
        --out-dir reports/

Produces (in --out-dir):
    audit_report.json   -- full machine-readable audit
    audit_report.md      -- human-readable summary
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict


def load_json_tolerant(path: str):
    """Load a JSON array, tolerating a leading //-style comment line
    (the issue ChatGPT flagged in questions_with_image.json). If the
    file is already clean JSON this is a no-op."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip leading // comment lines (not valid JSON, but was observed
    # in earlier pipeline output) and retry.
    lines = raw.splitlines()
    cleaned = [ln for ln in lines if not ln.strip().startswith("//")]
    return json.loads("\n".join(cleaned))


VALID_QUESTION_TYPES = {
    "mcq", "numerical", "multi_select", "fill_blank", "match_following",
    "graph_based", "table_based", "image_based", "diagram_based",
}

# Fields expected to be non-null on every record regardless of type.
CORE_REQUIRED_FIELDS = [
    "question_id", "question", "question_type", "correct_answer",
    "subject", "topic", "subtopic", "difficulty", "time_expected_minutes",
]


def normalize_label(label: str) -> str:
    """Lightweight normalization used ONLY to detect likely-duplicate
    labels for the audit report (case, whitespace, trailing plural 's',
    punctuation). This does NOT decide the canonical mapping -- that is
    a separate, reviewable step (build_canonical_mapping.py)."""
    if not label:
        return ""
    s = label.strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    if s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


def audit_field_completeness(records):
    total = len(records)
    missing = {}
    for field in CORE_REQUIRED_FIELDS:
        missing[field] = sum(1 for r in records if not r.get(field))

    # options: only meaningful for types that need them
    option_dependent_types = {"mcq", "multi_select", "fill_blank", "match_following"}
    missing_options = 0
    for r in records:
        if r.get("question_type") in option_dependent_types:
            opts = [r.get("option_a"), r.get("option_b"), r.get("option_c"), r.get("option_d")]
            if all(o in (None, "") for o in opts):
                missing_options += 1

    return {
        "total_records": total,
        "missing_core_fields": missing,
        "records_of_option_dependent_type_with_no_options": missing_options,
    }


def audit_type_specific(records):
    issues = defaultdict(list)
    for r in records:
        qid = r.get("question_id")
        qtype = r.get("question_type")

        if qtype not in VALID_QUESTION_TYPES:
            issues["invalid_question_type"].append(qid)

        if qtype == "match_following":
            if not r.get("left_items") or not r.get("right_items") or not r.get("correct_mapping"):
                issues["match_following_missing_mapping_data"].append(qid)

        if qtype == "multi_select":
            ca = r.get("correct_answer") or ""
            if "," not in ca and len(ca) <= 1:
                # single-letter answer on a multi_select is suspicious
                issues["multi_select_suspicious_single_answer"].append(qid)

        if qtype in ("image_based", "graph_based", "table_based", "diagram_based"):
            if not r.get("has_image") and not r.get("image_info"):
                issues["visual_type_without_image_info"].append(qid)
            info = r.get("image_info") or {}
            if qtype == "image_based" and not info.get("image_reference"):
                issues["image_based_missing_image_reference"].append(qid)
            if qtype == "graph_based" and not info.get("graph_description"):
                issues["graph_based_missing_graph_description"].append(qid)
            if qtype == "table_based" and not info.get("table_data"):
                issues["table_based_missing_table_data"].append(qid)

        if qtype == "numerical":
            ca = r.get("correct_answer")
            try:
                float(str(ca).strip())
            except (TypeError, ValueError):
                issues["numerical_answer_not_parseable_as_number"].append(qid)

    return {k: {"count": len(v), "example_ids": v[:10]} for k, v in issues.items()}


def audit_duplicates(records):
    ids = [r.get("question_id") for r in records]
    id_counts = Counter(ids)
    dup_ids = {k: v for k, v in id_counts.items() if v > 1}

    # Near-duplicate question text (exact match after whitespace normalization
    # -- catches copy-paste dupes across batches, not fuzzy near-dupes)
    text_norm_counts = Counter(
        re.sub(r"\s+", " ", (r.get("question") or "")).strip().lower() for r in records
    )
    dup_texts = {k: v for k, v in text_norm_counts.items() if v > 1 and k}

    return {
        "duplicate_question_ids": {"count": len(dup_ids), "examples": dict(list(dup_ids.items())[:10])},
        "duplicate_question_text_groups": {
            "count": len(dup_texts),
            "total_records_involved": sum(dup_texts.values()),
        },
    }


def audit_taxonomy(records):
    subjects = Counter(r.get("subject") for r in records)
    subject_topic = Counter((r.get("subject"), r.get("topic")) for r in records)

    # Group raw subject labels by normalized form to surface likely dupes.
    norm_to_raw = defaultdict(set)
    for s in subjects:
        norm_to_raw[normalize_label(s)].add(s)
    likely_duplicate_subjects = {
        norm: sorted(raws) for norm, raws in norm_to_raw.items() if len(raws) > 1
    }

    # Same, for topics -- but topic collisions are only meaningful within
    # a given canonical-subject bucket, so group by (normalized_subject, normalized_topic).
    topic_norm_to_raw = defaultdict(set)
    for (s, t) in subject_topic:
        key = (normalize_label(s), normalize_label(t))
        topic_norm_to_raw[key].add((s, t))
    likely_duplicate_topics = {
        f"{k[0]} :: {k[1]}": sorted(f"{s} :: {t}" for s, t in v)
        for k, v in topic_norm_to_raw.items() if len(v) > 1
    }

    return {
        "unique_subjects": len(subjects),
        "subject_counts": dict(subjects.most_common()),
        "likely_duplicate_subject_labels": likely_duplicate_subjects,
        "unique_subject_topic_pairs": len(subject_topic),
        "likely_duplicate_topic_labels_sample": dict(list(likely_duplicate_topics.items())[:30]),
        "likely_duplicate_topic_label_groups_total": len(likely_duplicate_topics),
    }


def audit_distribution(records):
    return {
        "by_question_type": dict(Counter(r.get("question_type") for r in records).most_common()),
        "by_difficulty": dict(Counter(r.get("difficulty") for r in records).most_common()),
        "by_validation_status": dict(Counter(r.get("validation_status") for r in records).most_common()),
        "by_source": dict(Counter(r.get("source") for r in records).most_common(20)),
    }


def run_audit(clean_path, images_path, out_dir):
    clean = load_json_tolerant(clean_path)
    images = load_json_tolerant(images_path)
    all_records = clean + images

    for r in clean:
        r["_origin_file"] = "questions_clean.json"
    for r in images:
        r["_origin_file"] = "questions_with_image.json"

    report = {
        "source_files": {
            "clean_count": len(clean),
            "with_image_count": len(images),
            "total_count": len(all_records),
        },
        "field_completeness": audit_field_completeness(all_records),
        "type_specific_issues": audit_type_specific(all_records),
        "duplicates": audit_duplicates(all_records),
        "taxonomy": audit_taxonomy(all_records),
        "distribution": audit_distribution(all_records),
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "audit_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(out_dir, "audit_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return report


def render_markdown(report) -> str:
    lines = []
    lines.append("# Question Bank Audit Report\n")
    sf = report["source_files"]
    lines.append(f"- Clean questions: **{sf['clean_count']}**")
    lines.append(f"- Image-backed questions: **{sf['with_image_count']}**")
    lines.append(f"- Total: **{sf['total_count']}**\n")

    lines.append("## Field completeness\n")
    fc = report["field_completeness"]
    for field, n in fc["missing_core_fields"].items():
        lines.append(f"- `{field}` missing: {n}")
    lines.append(f"- Option-dependent questions with no options at all: "
                  f"{fc['records_of_option_dependent_type_with_no_options']}\n")

    lines.append("## Type-specific issues\n")
    ts = report["type_specific_issues"]
    if not ts:
        lines.append("None found.\n")
    for k, v in ts.items():
        lines.append(f"- **{k}**: {v['count']} (e.g. {v['example_ids'][:5]})")
    lines.append("")

    lines.append("## Duplicates\n")
    d = report["duplicates"]
    lines.append(f"- Duplicate question_id occurrences: {d['duplicate_question_ids']['count']}")
    lines.append(f"- Duplicate question-text groups: {d['duplicate_question_text_groups']['count']} "
                  f"(covering {d['duplicate_question_text_groups']['total_records_involved']} records)\n")

    lines.append("## Taxonomy\n")
    tax = report["taxonomy"]
    lines.append(f"- Unique subject labels: {tax['unique_subjects']}")
    lines.append(f"- Unique (subject, topic) pairs: {tax['unique_subject_topic_pairs']}")
    lines.append(f"- Likely-duplicate subject-label groups: {len(tax['likely_duplicate_subject_labels'])}")
    for norm, raws in tax["likely_duplicate_subject_labels"].items():
        lines.append(f"  - `{norm}` <- {raws}")
    lines.append(f"- Likely-duplicate topic-label groups (within same normalized subject): "
                  f"{tax['likely_duplicate_topic_label_groups_total']}\n")

    lines.append("## Distribution\n")
    dist = report["distribution"]
    lines.append("### By question type")
    for k, v in dist["by_question_type"].items():
        lines.append(f"- {k}: {v}")
    lines.append("\n### By difficulty")
    for k, v in dist["by_difficulty"].items():
        lines.append(f"- {k}: {v}")
    lines.append("\n### By validation status")
    for k, v in dist["by_validation_status"].items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="questions_clean.json")
    ap.add_argument("--images", default="questions_with_image.json")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    run_audit(args.clean, args.images, args.out_dir)
