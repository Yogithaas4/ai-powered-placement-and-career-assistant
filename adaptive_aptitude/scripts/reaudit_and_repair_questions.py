"""Re-audit the current consolidated question JSON and apply safe repairs.

This intentionally makes only deterministic corrections.  It does not invent
answers, remove duplicate questions, or downgrade a visual question unless
the extracted text already contains a complete A-D MCQ.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


OPTION_KEYS = ("option_a", "option_b", "option_c", "option_d")
VISUAL_TYPES = {"image_based", "graph_based", "table_based", "diagram_based"}


def load_with_comments(path: Path):
    raw = path.read_text(encoding="utf-8")
    comments = [line for line in raw.splitlines() if line.lstrip().startswith("//")]
    content = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("//"))
    return json.loads(content), comments


def write_with_comments(path: Path, records, comments):
    prefix = ("\n".join(comments) + "\n") if comments else ""
    path.write_text(prefix + json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def has_complete_options(question):
    return all(question.get(key) not in (None, "") for key in OPTION_KEYS)


def is_option_letter(value):
    return bool(re.fullmatch(r"\s*[A-Da-d]\s*", str(value or "")))


def normalized_question_text(question):
    return re.sub(r"\s+", " ", (question.get("question") or "").strip().lower())


def apply_safe_repairs(records):
    changes = []
    for question in records:
        old_type = question.get("question_type")
        answer = question.get("correct_answer")

        # A visual type without an image or structured visual data is only
        # changed when it is already a complete A-D text MCQ.  This covers
        # extraction-origin labels where equations/code are now present in
        # text and no visual asset is needed by the frontend.
        if (old_type in VISUAL_TYPES and not question.get("has_image")
                and not question.get("image_info") and has_complete_options(question)
                and is_option_letter(answer)):
            question["question_type"] = "mcq"
            changes.append((question["question_id"], old_type, "mcq", "complete text MCQ; no visual asset required"))
            continue

        # A letter answer plus complete A-D options is MCQ interaction, even
        # if the stem asks for a numerical computation.
        if old_type == "numerical" and has_complete_options(question) and is_option_letter(answer):
            question["question_type"] = "mcq"
            changes.append((question["question_id"], old_type, "mcq", "A-D answer key with complete options"))
            continue

        # These matching questions already encode the possible mappings as
        # A-D options.  With no complete mapping object they must be rendered
        # and scored as MCQs, not as drag-and-drop matching questions.
        if (old_type == "match_following" and not question.get("correct_mapping")
                and has_complete_options(question) and is_option_letter(answer)):
            question["question_type"] = "mcq"
            changes.append((question["question_id"], old_type, "mcq", "mapping absent; answer is an A-D mapping choice"))
            continue

        # Two true/false stems have no A-D fields.  Add the standard choices
        # and translate their literal answer to the stored option letter.
        if (old_type == "mcq" and not any(question.get(key) for key in OPTION_KEYS)
                and str(answer).strip().lower() in {"true", "false"}):
            literal = str(answer).strip().lower()
            question.update({"option_a": "True", "option_b": "False", "option_c": None, "option_d": None,
                             "correct_answer": "A" if literal == "true" else "B"})
            changes.append((question["question_id"], "mcq", "mcq", "standardized true/false options"))

    return changes


def remaining_issues(records, image_dir: Path):
    unresolved = defaultdict(list)
    for question in records:
        qid = question["question_id"]
        qtype = question.get("question_type")
        answer = question.get("correct_answer")
        info = question.get("image_info")
        # A few legacy records use a boolean/null here.  Treat those as no
        # structured metadata rather than failing the audit.
        if not isinstance(info, dict):
            info = {}
        if not str(answer or "").strip():
            unresolved["missing_correct_answer"].append(qid)
        if qtype in {"mcq", "multi_select"} and not any(question.get(key) for key in OPTION_KEYS):
            unresolved["choice_question_without_options"].append(qid)
        if qtype == "match_following" and not (question.get("left_items") and question.get("right_items") and question.get("correct_mapping")):
            unresolved["match_following_without_complete_mapping"].append(qid)
        if qtype in VISUAL_TYPES and question.get("has_image") and not info.get("image_reference"):
            unresolved["visual_question_without_image_reference"].append(qid)
        ref = info.get("image_reference")
        if ref and not (image_dir / Path(ref).name).is_file():
            unresolved["missing_local_referenced_image"].append(qid)

    duplicate_groups = defaultdict(list)
    for question in records:
        text = normalized_question_text(question)
        if text:
            duplicate_groups[text].append(question)
    duplicates = [group for group in duplicate_groups.values() if len(group) > 1]
    conflicting_duplicates = [
        group for group in duplicates
        if len({str(q.get("correct_answer")) for q in group}) > 1
    ]

    return {
        "unresolved": {key: sorted(value) for key, value in unresolved.items()},
        "duplicate_text_groups": len(duplicates),
        "duplicate_records": sum(len(group) for group in duplicates),
        "duplicate_groups_with_conflicting_answers": len(conflicting_duplicates),
        "conflicting_duplicate_examples": [
            {"question_ids": [q["question_id"] for q in group],
             "answers": sorted({str(q.get("correct_answer")) for q in group})}
            for group in conflicting_duplicates[:25]
        ],
        "question_type_counts": dict(Counter(q.get("question_type") for q in records)),
    }


def render_report(total, changes, issues):
    lines = ["# Re-audit and Safe Repair Report", "", f"- Records scanned: **{total}**", f"- Safe source corrections applied: **{len(changes)}**", ""]
    by_reason = Counter(change[3] for change in changes)
    lines.extend(["## Applied corrections", ""])
    for reason, count in by_reason.items():
        lines.append(f"- {count}: {reason}")
    lines.extend(["", "## Remaining manual-review items", ""])
    for key, ids in issues["unresolved"].items():
        lines.append(f"- `{key}`: {len(ids)}")
    lines.extend(["", "## Duplicate text", "", f"- Exact normalized-text groups: {issues['duplicate_text_groups']}", f"- Records in duplicate groups: {issues['duplicate_records']}", f"- Groups with conflicting stored answers: {issues['duplicate_groups_with_conflicting_answers']}", "", "Duplicates are reported, not automatically deleted: the same exam question can legitimately occur in multiple source papers."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clean_path, image_path = Path(args.clean), Path(args.images)
    clean, clean_comments = load_with_comments(clean_path)
    images, image_comments = load_with_comments(image_path)
    records = clean + images
    changes = apply_safe_repairs(records)
    issues = remaining_issues(records, Path(args.image_dir))

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "re_audit_report.json").write_text(json.dumps({
        "records_scanned": len(records),
        "changes": [{"question_id": qid, "from_type": old, "to_type": new, "reason": reason}
                    for qid, old, new, reason in changes],
        **issues,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "re_audit_report.md").write_text(render_report(len(records), changes, issues), encoding="utf-8")

    if not args.dry_run:
        write_with_comments(clean_path, clean, clean_comments)
        write_with_comments(image_path, images, image_comments)

    print(f"Scanned {len(records)} records; applied {len(changes)} safe correction(s).")
    print(f"Reports: {report_dir / 're_audit_report.md'} and {report_dir / 're_audit_report.json'}")


if __name__ == "__main__":
    main()
