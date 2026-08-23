"""
step4_embeddings.py
-------------------
Builds a structured query string and generates embedding vectors.
Uses  : BAAI/bge-large-en-v1.5 (1024-dim, best open-source embedder)
Output: {query_string, query_vector, section_vectors}
"""

from sentence_transformers import SentenceTransformer

_model = None


def _get_model():
    global _model
    if _model is None:
        print("[+] Loading embedding model (bge-large-en-v1.5)...")
        _model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _model


# ── query string ─────────────────────────────────────────────

def build_query_string(entities: dict, sections: dict) -> str:
    """
    Structured query string that mirrors how job descriptions
    are written — improves BM25 / keyword matching quality.
    Format: [SKILLS] ... [EXP] ... [ROLE] ... [EDU] ...
    """
    parts = []

    if entities.get("skills"):
        parts.append("[SKILLS] " + ", ".join(entities["skills"][:20]))

    exp_parts = []
    if entities.get("years_exp"):
        exp_parts.append(entities["years_exp"])
    if entities.get("organizations"):
        exp_parts.append("at " + ", ".join(entities["organizations"][:2]))
    if exp_parts:
        parts.append("[EXP] " + " ".join(exp_parts))

    exp_text = sections.get("experience", "")
    if exp_text:
        first_line = exp_text.split("\n")[0].strip()
        if first_line:
            parts.append("[ROLE] " + first_line[:80])

    edu_text = sections.get("education", "")
    if edu_text:
        first_line = edu_text.split("\n")[0].strip()
        if first_line:
            parts.append("[EDU] " + first_line[:80])

    return "  ".join(parts)


# ── embedding generation ─────────────────────────────────────

def embed(text: str) -> list:
    """Embed a single text string. Returns list of floats."""
    if not text.strip():
        return []
    model  = _get_model()
    vector = model.encode(
        f"Represent this resume for job matching: {text}",
        normalize_embeddings=True
    )
    return vector.tolist()


def build_section_embeddings(sections: dict) -> dict:
    """
    Generate one embedding vector per resume section.
    Used for targeted section-level matching at retrieval time.
    """
    model = _get_model()
    result = {}
    for key in ["skills", "experience", "education", "projects"]:
        text = sections.get(key, "").strip()
        if text:
            v = model.encode(
                f"Resume {key} section: {text[:1000]}",
                normalize_embeddings=True
            )
            result[key] = v.tolist()
        else:
            result[key] = []
    return result


# ── master builder ────────────────────────────────────────────

def build_embeddings(entities: dict, sections: dict) -> dict:
    query_string    = build_query_string(entities, sections)
    query_vector    = embed(query_string)
    section_vectors = build_section_embeddings(sections)

    return {
        "query_string"   : query_string,
        "query_vector"   : query_vector,
        "section_vectors": section_vectors,
    }


# ── quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from .step2_segmentation import segment_resume
    from .step3_ner import extract_all_entities

    sample = """Arjun Sharma
arjun.sharma@gmail.com | Bengaluru

Skills
Python, FastAPI, PostgreSQL, Docker, Machine Learning, TensorFlow

Experience
Software Engineer — Infosys (2022–Present)
3 years of experience in backend development.

Education
B.Tech Computer Science — VIT University, 2022
"""
    sections   = segment_resume(sample)
    entities   = extract_all_entities(sample, sections)
    embeddings = build_embeddings(entities, sections)

    print("=== Query String ===")
    print(embeddings["query_string"])
    print("\n=== Vector Sizes ===")
    print(f"  query_vector    : {len(embeddings['query_vector'])} dims")
    for k, v in embeddings["section_vectors"].items():
        status = f"{len(v)} dims" if v else "empty (section not found)"
        print(f"  {k:<16} : {status}")