"""
step5_storage.py
----------------
Stores preprocessed resumes in local ChromaDB.
Uses 5 separate collections — one per vector type — since
ChromaDB only supports one vector per document.

Collections:
    resumes            → full query vector (overall match)
    resumes_skills     → skills section vector
    resumes_experience → experience section vector
    resumes_education  → education section vector
    resumes_projects   → projects section vector

Deduplicates by filename — re-running overwrites, not duplicates.
"""

import chromadb
import uuid
import os

from config import CHROMA_PATH

DB_PATH = CHROMA_PATH

# All 5 collection names
COLLECTIONS = {
    "query"     : "resumes",
    "skills"    : "resumes_skills",
    "experience": "resumes_experience",
    "education" : "resumes_education",
    "projects"  : "resumes_projects",
}

_client      = None
_collections = {}


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
    return _client


def get_collection(key: str):
    """Get or create a named collection. key is one of COLLECTIONS keys."""
    global _collections
    if key not in _collections:
        client = get_client()
        name   = COLLECTIONS[key]
        _collections[key] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collections[key]


def _get_existing_id(filename: str) -> str | None:
    """Check if a resume with this filename already exists. Returns ID or None."""
    col      = get_collection("query")
    existing = col.get(where={"filename": filename})
    return existing["ids"][0] if existing["ids"] else None


def store_resume(preprocessed: dict) -> str:
    """
    Store all 5 vectors for a resume across 5 ChromaDB collections.
    Deduplicates by filename using upsert.
    Returns the shared document ID used across all collections.
    """
    filename = preprocessed.get("filename", "")
    ev       = preprocessed.get("embeddings", {})
    entities = preprocessed.get("entities", {})

    query_vector    = ev.get("query_vector", [])
    query_string    = ev.get("query_string", "")
    section_vectors = ev.get("section_vectors", {})

    if not query_vector:
        print(f"  [SKIP] No embedding for {filename}")
        return ""

    # Reuse existing ID if file was stored before (deduplication)
    existing_id = _get_existing_id(filename)
    doc_id      = existing_id if existing_id else str(uuid.uuid4())

    # Shared metadata stored in every collection for easy lookup
    metadata = {
        "filename"    : filename,
        "file_type"   : preprocessed.get("file_type", ""),
        "name"        : entities.get("name", ""),
        "email"       : entities.get("email", ""),
        "phone"       : entities.get("phone", ""),
        "location"    : entities.get("location", ""),
        "years_exp"   : entities.get("years_exp", ""),
        "skills"      : ", ".join(entities.get("skills", [])),
        "organizations": ", ".join(entities.get("organizations", [])),
        "query_string": query_string,
    }

    # ── 1. Full query vector (main collection) ──────────────
    get_collection("query").upsert(
        ids        = [doc_id],
        embeddings = [query_vector],
        documents  = [query_string],
        metadatas  = [metadata],
    )

    # ── 2-5. Section vectors (only if section exists) ───────
    section_keys = ["skills", "experience", "education", "projects"]
    for key in section_keys:
        vec = section_vectors.get(key, [])
        if vec:
            # Document text = the section text for context
            section_text = preprocessed.get("sections", {}).get(key, query_string)
            get_collection(key).upsert(
                ids        = [doc_id],
                embeddings = [vec],
                documents  = [section_text[:500]],
                metadatas  = [metadata],
            )

    return doc_id


def search_resumes(query_vector: list, top_k: int = 5,
                   section: str = "query") -> list:
    """
    Search for similar resumes using a specific vector type.
    section: one of 'query', 'skills', 'experience', 'education', 'projects'

    For best results call this for each section and combine scores.
    """
    col   = get_collection(section)
    count = col.count()
    if count == 0:
        return []

    results = col.query(
        query_embeddings = [query_vector],
        n_results        = min(top_k, count),
        include          = ["metadatas", "distances", "documents"]
    )

    return [
        {
            "id"      : doc_id,
            "score"   : round(1 - dist, 4),
            "section" : section,
            "metadata": meta,
        }
        for doc_id, dist, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]


def search_resumes_combined(query_vectors: dict, top_k: int = 5) -> list:
    """
    Search across all section vectors and combine scores with weights.

    query_vectors: dict with keys matching section names
        e.g. {"query": [...], "skills": [...], "experience": [...]}

    Weights reflect importance of each section for job matching:
        skills     → 40%  (most important)
        experience → 30%
        query      → 20%  (overall resume)
        projects   → 5%
        education  → 5%
    """
    WEIGHTS = {
        "skills"    : 0.40,
        "experience": 0.30,
        "query"     : 0.20,
        "projects"  : 0.05,
        "education" : 0.05,
    }

    # Collect scores per doc_id across all sections
    score_map = {}

    for section, weight in WEIGHTS.items():
        vec = query_vectors.get(section, [])
        if not vec:
            continue

        results = search_resumes(vec, top_k=top_k * 2, section=section)
        for r in results:
            did = r["id"]
            if did not in score_map:
                score_map[did] = {"weighted_score": 0.0, "metadata": r["metadata"]}
            score_map[did]["weighted_score"] += r["score"] * weight

    # Sort by combined weighted score
    ranked = sorted(
        score_map.values(),
        key=lambda x: x["weighted_score"],
        reverse=True
    )[:top_k]

    return [
        {
            "score"   : round(r["weighted_score"], 4),
            "metadata": r["metadata"],
        }
        for r in ranked
    ]


def get_stats() -> dict:
    """Return count of records in each collection."""
    stats = {"db_path": os.path.abspath(DB_PATH)}
    for key, name in COLLECTIONS.items():
        try:
            col          = get_collection(key)
            stats[name]  = col.count()
        except Exception:
            stats[name]  = 0
    return stats


def reset_all():
    """Wipe all 5 collections. Useful during testing."""
    global _collections
    client = get_client()
    for name in COLLECTIONS.values():
        try:
            client.delete_collection(name)
            print(f"  [RESET] Deleted collection: {name}")
        except Exception:
            pass
    _collections = {}
    print("[!] All collections reset")


# ── quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    VECTOR_SIZE  = 1024
    dummy_vector = [0.01] * VECTOR_SIZE

    test = {
        "filename" : "test_resume.docx",
        "file_type": "docx",
        "entities" : {
            "name"         : "Test User",
            "email"        : "test@example.com",
            "phone"        : "+91-9999999999",
            "location"     : "Bengaluru",
            "skills"       : ["Python", "ML"],
            "organizations": ["Infosys"],
            "years_exp"    : "3 years",
        },
        "sections": {
            "skills"    : "Python, Machine Learning, FastAPI",
            "experience": "Software Engineer at Infosys",
            "education" : "B.Tech VIT University",
            "projects"  : "Resume parser using NLP",
        },
        "embeddings": {
            "query_string"   : "[SKILLS] Python, ML  [EXP] 3 years",
            "query_vector"   : dummy_vector,
            "section_vectors": {
                "skills"    : dummy_vector,
                "experience": dummy_vector,
                "education" : dummy_vector,
                "projects"  : dummy_vector,
            }
        }
    }

    doc_id = store_resume(test)
    print(f"\nStored ID : {doc_id}")
    print(f"\nStats:")
    for k, v in get_stats().items():
        print(f"  {k}: {v}")