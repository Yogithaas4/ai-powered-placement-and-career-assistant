"""
step3_ner.py
------------
Extracts structured entities from resume text.
Uses  : spaCy en_core_web_lg  (name, location, orgs, dates)
        skillNer               (skills with normalization)
        regex                  (email, phone, years of experience)
Output: entities dict
"""

import re
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor


# ── lazy globals — loaded once, reused for all resumes ──────
_nlp             = None
_skill_extractor = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        print("[+] Loading spaCy model (en_core_web_lg)...")
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


def _get_skill_extractor():
    global _skill_extractor
    if _skill_extractor is None:
        print("[+] Loading skillNer...")
        nlp = _get_nlp()
        _skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return _skill_extractor


# ── regex extractors ─────────────────────────────────────────

def extract_email(text: str) -> str:
    m = re.search(r"[\w\.\+\-]+@[\w\.\-]+\.\w{2,}", text)
    return m.group(0) if m else ""


def extract_phone(text: str) -> str:
    m = re.search(
        r"(\+?\d{1,3}[\s\-\.]?)?(\(?\d{3}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4}", text
    )
    return m.group(0).strip() if m else ""


def extract_years_experience(text: str) -> str:
    patterns = [
        r"(\d+)\+?\s*years?\s+of\s+experience",
        r"(\d+)\+?\s*yrs?\s+of\s+experience",
        r"experience\s+of\s+(\d+)\+?\s*years?",
        r"over\s+(\d+)\s+years?",
        r"(\d+)\+?\s*years?\s+experience",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1) + " years"
    return ""


# ── spaCy NER ────────────────────────────────────────────────

def extract_spacy_entities(text: str) -> dict:
    """
    Extract: PERSON (name), GPE/LOC (location),
             ORG (companies), DATE (years)
    Only processes first 5000 chars for speed.
    """
    nlp = _get_nlp()
    doc = nlp(text[:5000])

    result = {
        "name"         : "",
        "location"     : "",
        "organizations": [],
        "dates"        : [],
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not result["name"]:
            result["name"] = ent.text.strip()
        elif ent.label_ in ("GPE", "LOC") and not result["location"]:
            result["location"] = ent.text.strip()
        elif ent.label_ == "ORG":
            if ent.text.strip() not in result["organizations"]:
                result["organizations"].append(ent.text.strip())
        elif ent.label_ == "DATE":
            result["dates"].append(ent.text.strip())

    return result


# ── skillNer skill extraction ────────────────────────────────

def extract_skills(text: str) -> list:
    """
    Extract and deduplicate skills using skillNer.
    Runs on skills + experience + projects sections combined.
    """
    if not text.strip():
        return []

    extractor = _get_skill_extractor()
    try:
        annotations = extractor.annotate(text)
        skills = []

        for match in annotations.get("results", {}).get("full_matches", []):
            skills.append(match["doc_node_value"])

        for match in annotations.get("results", {}).get("ngram_scored", []):
            if match.get("score", 0) >= 0.8:
                skills.append(match["doc_node_value"])

        # Deduplicate preserving order
        seen, unique = set(), []
        for s in skills:
            key = s.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique

    except Exception as e:
        print(f"  [WARN] skillNer error: {e}")
        return []


# ── master extractor ─────────────────────────────────────────

def extract_all_entities(raw_text: str, sections: dict) -> dict:
    """
    Combines regex + spaCy + skillNer into one clean entity dict.
    """
    # Regex — fast and reliable for structured fields
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)
    years = extract_years_experience(
        sections.get("summary", "") + " " + sections.get("experience", "")
    )

    # spaCy on the top portion of the resume (header has most info)
    spacy_ents = extract_spacy_entities(raw_text)

    # skillNer on skills + experience + projects combined
    skill_text = " ".join([
        sections.get("skills", ""),
        sections.get("experience", ""),
        sections.get("projects", ""),
    ])
    skills = extract_skills(skill_text)

    return {
        "name"         : spacy_ents["name"],
        "email"        : email,
        "phone"        : phone,
        "location"     : spacy_ents["location"],
        "skills"       : skills,
        "organizations": spacy_ents["organizations"],
        "dates"        : spacy_ents["dates"],
        "years_exp"    : years,
    }


# ── quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from .step2_segmentation import segment_resume

    sample = """Arjun Sharma
arjun.sharma@gmail.com | +91-9845012345 | Bengaluru, India

Summary
Software engineer with 3 years of experience in backend development.

Skills
Python, FastAPI, PostgreSQL, Docker, Machine Learning, TensorFlow

Experience
Software Engineer — Infosys, Bengaluru (2022–Present)
Built REST APIs and ML pipelines using Python and TensorFlow.

Education
B.Tech Computer Science — VIT University, 2022
"""

    sections = segment_resume(sample)
    entities = extract_all_entities(sample, sections)

    print("=== Extracted Entities ===")
    for k, v in entities.items():
        print(f"  {k:<15} : {v}")
