"""
scripts/check_bkt_param_coverage.py
--------------------------------------
Catches exactly the bug that shipped silently for a while: SUBJECT_BKT_PARAMS
keys in core/knowledge_model.py drifting out of sync with the real
canonical_subject values in Postgres (e.g. "Operating System" vs the real
"Operating Systems"). SUBJECT_BKT_PARAMS.get(subject, BKTParams()) never
raises on a mismatch -- it just silently uses the generic default -- so
this has to be checked explicitly rather than relying on something crashing.

Run this after ANY change to SUBJECT_BKT_PARAMS or to
reports/canonical_subject_mapping.json / the subject_topic_canonical_map
table, and periodically as a sanity check.

Usage:
    python scripts/check_bkt_param_coverage.py
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.db_loader import get_engine  # noqa: E402
from core.knowledge_model import SUBJECT_BKT_PARAMS  # noqa: E402

import pandas as pd


def main():
    engine = get_engine()
    real_subjects = set(
        pd.read_sql_query("SELECT DISTINCT canonical_subject FROM concepts", engine)
        ["canonical_subject"].dropna().tolist()
    )
    tuned_subjects = set(SUBJECT_BKT_PARAMS.keys())

    stale_keys = tuned_subjects - real_subjects
    untuned_subjects = real_subjects - tuned_subjects

    print(f"Real canonical subjects in DB: {len(real_subjects)}")
    print(f"Subjects with tuned BKT params: {len(tuned_subjects)}")
    print()

    if stale_keys:
        print(f"❌ {len(stale_keys)} SUBJECT_BKT_PARAMS key(s) don't match any real "
              f"canonical_subject -- these are silently using the generic default "
              f"RIGHT NOW even though someone intended to tune them:")
        for s in sorted(stale_keys):
            print(f"    {s!r}")
        print()
    else:
        print("✓ Every SUBJECT_BKT_PARAMS key matches a real canonical_subject.")
        print()

    if untuned_subjects:
        print(f"ℹ {len(untuned_subjects)} real subject(s) have no tuned params (using the "
              f"generic default -- may be intentional, see the comment in knowledge_model.py):")
        for s in sorted(untuned_subjects):
            print(f"    {s}")

    if stale_keys:
        sys.exit(1)  # non-zero exit so this can be wired into a pre-run check / CI


if __name__ == "__main__":
    main()
