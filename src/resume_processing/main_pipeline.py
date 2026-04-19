"""
main_pipeline.py
----------------
Runs the full resume preprocessing pipeline.

Usage:
    python src/main_pipeline.py --folder data/resumes          # all resumes
    python -m resume_processing.main_pipeline --folder data/resumes --test
"""

import argparse
import json
import os
from tqdm import tqdm

from config import PROCESSED_DIR, RESUMES_DIR
from .step1_parser import parse_all_resumes
from .step2_segmentation import segment_resume
from .step3_ner import extract_all_entities
from .step4_embeddings import build_embeddings
from .step5_storage import store_resume, get_stats


def process_one(parsed: dict) -> dict:
    """Run steps 2–4 on a single parsed resume dict."""
    raw  = parsed["raw_text"]
    secs = segment_resume(raw)
    ents = extract_all_entities(raw, secs)
    embs = build_embeddings(ents, secs)
    return {
        "filename" : parsed["filename"],
        "file_type": parsed["file_type"],
        "raw_text" : raw,
        "sections" : secs,
        "entities" : ents,
        "embeddings": embs,
    }


def run_pipeline(folder: str, limit: int = None):
    print("=" * 52)
    print("   RESUME PREPROCESSING PIPELINE")
    print("=" * 52)

    # ── Step 1: Parse files ──────────────────────────────────
    print("\n[Step 1] Parsing resume files...")
    parsed_list = parse_all_resumes(folder)

    if not parsed_list:
        print("[!] No resumes to process. Exiting.")
        return

    if limit:
        parsed_list = parsed_list[:limit]
        print(f"  Running on first {limit} resume(s) only")

    # ── Steps 2-4: Segment + NER + Embed ─────────────────────
    print("\n[Steps 2-4] Segmenting, extracting entities, embedding...")
    processed, failed = [], []

    for parsed in tqdm(parsed_list, desc="Processing"):
        try:
            result = process_one(parsed)
            processed.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {parsed['filename']}: {e}")
            failed.append(parsed["filename"])

    print(f"\n  Processed : {len(processed)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print(f"  Failed files: {failed}")

    # ── Step 5: Store in ChromaDB ─────────────────────────────
    print("\n[Step 5] Storing in ChromaDB...")
    stored_ids = []
    for resume in tqdm(processed, desc="Storing"):
        try:
            doc_id = store_resume(resume)
            if doc_id:
                stored_ids.append(doc_id)
        except Exception as e:
            print(f"\n  [ERROR] Storing {resume['filename']}: {e}")

    stats = get_stats()
    print(f"\n  ChromaDB stats : {stats}")

    # ── Save inspection JSON ──────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "output_preprocessed.json"
    inspection = []
    for r in processed:
        inspection.append({
            "filename"    : r["filename"],
            "file_type"   : r["file_type"],
            "entities"    : r["entities"],
            "sections"    : {k: v[:200] for k, v in r["sections"].items() if v},
            "query_string": r["embeddings"]["query_string"],
            "vector_dims" : len(r["embeddings"].get("query_vector", [])),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2)

    print(f"\n  Inspection JSON saved → {out_path.as_posix()}")
    print(f"\n{'='*52}")
    print(f"  Done! {len(stored_ids)} resume(s) indexed in ChromaDB.")
    print(f"  Check '{out_path.as_posix()}' to verify extraction quality.")
    print(f"{'='*52}")

    return processed


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume preprocessing pipeline")
    parser.add_argument("--folder", default=str(RESUMES_DIR),
                        help="Folder containing resume files (.docx / .pdf)")
    parser.add_argument("--test",  action="store_true",
                        help="Test mode: process first 2 resumes only")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of resumes to process")
    args = parser.parse_args()

    limit = 2 if args.test else args.limit
    run_pipeline(args.folder, limit=limit)
