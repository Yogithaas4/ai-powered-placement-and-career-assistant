"""
Relevance sets for ranking metrics.

- **Chroma top-N**: job_index values from the jobs collection using the resume
  query_vector only (no reranker, no pseudo-union across models). This avoids
  the trivial Precision=1 leakage from defining relevance from model outputs.
- **User indices**: optional comma/newline-separated ground-truth job_index list.
"""

from __future__ import annotations

import re
from typing import Set

import chromadb

from config import DATA_DIR

JOBS_DB_PATH = str(DATA_DIR / "jobs_db")


def job_indices_from_chroma_topn(
    preprocessed: dict,
    n: int = 80,
    collection_name: str = "jobs",
) -> Set[int]:
    """Indices of jobs in the top-N Chroma hits for this resume embedding."""
    vec = (preprocessed.get("embeddings") or {}).get("query_vector")
    if not vec:
        return set()
    try:
        client = chromadb.PersistentClient(path=JOBS_DB_PATH)
        col = client.get_collection(name=collection_name)
    except Exception:
        return set()
    try:
        cnt = col.count()
    except Exception:
        return set()
    if cnt <= 0:
        return set()
    k = min(max(1, int(n)), cnt)
    try:
        res = col.query(
            query_embeddings=[list(vec)],
            n_results=k,
            include=["metadatas"],
        )
    except Exception:
        return set()
    out: Set[int] = set()
    for m in (res.get("metadatas") or [[]])[0]:
        if not m:
            continue
        ji = m.get("job_index")
        try:
            v = int(ji)
            if v >= 0:
                out.add(v)
        except (TypeError, ValueError):
            continue
    return out


def jobs_collection_count(collection_name: str = "jobs") -> int:
    try:
        client = chromadb.PersistentClient(path=JOBS_DB_PATH)
        return int(client.get_collection(name=collection_name).count())
    except Exception:
        return 0


def parse_user_job_indices(text: str) -> Set[int]:
    """Parse '12, 34' or newline-separated job_index values."""
    if not text or not str(text).strip():
        return set()
    parts = re.split(r"[\s,;]+", str(text).strip())
    out: Set[int] = set()
    for p in parts:
        if not p:
            continue
        try:
            v = int(p.strip())
            if v >= 0:
                out.add(v)
        except ValueError:
            continue
    return out
