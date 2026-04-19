"""
inspect_embeddings.py
---------------------
Shows what is stored across all 5 ChromaDB collections.
Run after main_pipeline.py has processed at least one resume.

Usage:
    python src/inspect_embeddings.py
"""

import chromadb

from config import CHROMA_PATH

DB_PATH = CHROMA_PATH

COLLECTIONS = {
    "query"     : "resumes",
    "skills"    : "resumes_skills",
    "experience": "resumes_experience",
    "education" : "resumes_education",
    "projects"  : "resumes_projects",
}

WEIGHTS = {
    "query"     : 0.20,
    "skills"    : 0.40,
    "experience": 0.30,
    "education" : 0.05,
    "projects"  : 0.05,
}


def inspect():
    client = chromadb.PersistentClient(path=DB_PATH)

    # ── Collection summary ────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ChromaDB — collection summary")
    print(f"{'='*55}")
    for key, name in COLLECTIONS.items():
        try:
            col   = client.get_or_create_collection(name)
            count = col.count()
            weight = int(WEIGHTS[key] * 100)
            print(f"  {name:<25} {count} record(s)   weight: {weight}%")
        except Exception as e:
            print(f"  {name:<25} ERROR: {e}")

    # ── Per-resume detail from main collection ────────────────
    main_col = client.get_or_create_collection(COLLECTIONS["query"])
    total    = main_col.count()

    if total == 0:
        print("\n  No resumes stored yet. Run src/main_pipeline.py first.")
        return

    results = main_col.get(include=["embeddings", "metadatas", "documents"])

    for i, (doc_id, embedding, metadata, document) in enumerate(zip(
        results["ids"],
        results["embeddings"],
        results["metadatas"],
        results["documents"],
    )):
        print(f"\n{'─'*55}")
        print(f"  Resume #{i+1}  —  ID: {doc_id[:16]}...")
        print(f"{'─'*55}")
        print(f"  File        : {metadata.get('filename')}")
        print(f"  Name        : {metadata.get('name') or 'not detected'}")
        print(f"  Email       : {metadata.get('email') or 'not detected'}")
        print(f"  Location    : {metadata.get('location') or 'not detected'}")
        print(f"  Years exp   : {metadata.get('years_exp') or 'not detected'}")

        skills = metadata.get("skills", "")
        print(f"  Skills      : {skills[:100] + '...' if len(skills) > 100 else skills}")

        print(f"\n  Query string:")
        print(f"    {document}")

        # ── Vector info per collection ────────────────────────
        print(f"\n  Vectors stored:")
        for key, name in COLLECTIONS.items():
            try:
                col     = client.get_or_create_collection(name)
                section = col.get(ids=[doc_id], include=["embeddings"])
                if section["ids"]:
                    vec    = section["embeddings"][0]
                    dims   = len(vec)
                    sample = [round(float(v), 6) for v in vec[:12]]
                    weight = int(WEIGHTS[key] * 100)
                    print(f"    {key:<12} {dims} dims   weight: {weight}%")
                    print(f"               first 12: {sample}")
                else:
                    print(f"    {key:<12} NOT STORED (section was empty in resume)")
            except Exception as e:
                print(f"    {key:<12} ERROR: {e}")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    inspect()