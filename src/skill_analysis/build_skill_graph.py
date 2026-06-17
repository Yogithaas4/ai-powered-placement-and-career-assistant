"""
build_skill_graph.py
--------------------
Builds a reusable skill-gap graph payload from an already preprocessed resume
and an existing recommendation export.

Examples
--------
python -m skill_analysis.build_skill_graph ^
  --resume-json data/processed/resume_full/arjun_sharma_resume.json ^
  --recommendations-json data/recommendations/sample_recs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from skill_analysis.analysis import export_graph_payload


def _load_recommendations(path: Path) -> list:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("Recommendation JSON must contain a list of recommendation objects.")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
        return df.to_dict(orient="records")

    raise ValueError("Recommendations file must be .json or .csv")


def main():
    parser = argparse.ArgumentParser(description="Build skill graph payload from existing pipeline artifacts.")
    parser.add_argument("--resume-json", required=True, help="Path to full preprocessed resume JSON.")
    parser.add_argument("--recommendations-json", default=None, help="Path to recommendations JSON.")
    parser.add_argument("--recommendations-csv", default=None, help="Path to recommendations CSV.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top recommendations to analyze.")
    parser.add_argument("--output", default="data/processed/skill_graph.json", help="Output JSON path.")
    args = parser.parse_args()

    rec_path = args.recommendations_json or args.recommendations_csv
    if not rec_path:
        raise SystemExit("Provide either --recommendations-json or --recommendations-csv.")

    resume_path = Path(args.resume_json)
    with open(resume_path, "r", encoding="utf-8") as f:
        preprocessed = json.load(f)

    recommendations = _load_recommendations(Path(rec_path))
    output_path = export_graph_payload(preprocessed, recommendations, args.output, top_k=args.top_k)
    print(f"Saved skill graph payload -> {output_path}")


if __name__ == "__main__":
    main()
