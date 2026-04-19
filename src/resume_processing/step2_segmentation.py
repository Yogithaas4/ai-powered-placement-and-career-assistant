"""
step2_segmentation.py
---------------------
Splits resume raw text into labelled sections.
Output: dict → {summary, skills, experience, education,
                projects, certifications, other}
"""

import re


SECTION_HEADINGS = {
    "summary": [
        "summary", "objective", "profile", "about me",
        "professional summary", "career objective", "overview",
        "about", "professional profile"
    ],
    "skills": [
        "skills", "technical skills", "core competencies",
        "technologies", "tech stack", "tools", "expertise",
        "key skills", "competencies", "areas of expertise",
        "technical expertise", "technology stack"
    ],
    "experience": [
        "experience", "work experience", "employment history",
        "professional experience", "work history", "internship",
        "internships", "career history", "job experience",
        "employment", "professional background"
    ],
    "education": [
        "education", "academic background", "educational qualification",
        "qualifications", "academic qualifications", "educational details",
        "academic details", "educational background"
    ],
    "projects": [
        "projects", "personal projects", "academic projects",
        "key projects", "project experience", "project work",
        "notable projects", "selected projects"
    ],
    "certifications": [
        "certifications", "certificates", "courses",
        "training", "achievements", "awards", "accomplishments",
        "licenses", "credentials", "certifications and awards"
    ],
}


def _build_pattern() -> re.Pattern:
    all_kw = []
    for kws in SECTION_HEADINGS.values():
        all_kw.extend(kws)
    all_kw.sort(key=len, reverse=True)
    escaped = [re.escape(k) for k in all_kw]
    pattern = r"^(?:" + "|".join(escaped) + r")\s*[:\-]?\s*$"
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


def _map_heading(heading: str) -> str:
    h = heading.strip().lower()
    for section, kws in SECTION_HEADINGS.items():
        if h in kws:
            return section
    return "other"


def segment_resume(raw_text: str) -> dict:
    """
    Split resume text into sections.
    Returns dict: {section_name → text}
    """
    sections = {k: "" for k in SECTION_HEADINGS}
    sections["other"] = ""

    pattern = _build_pattern()
    lines   = raw_text.split("\n")

    current = "other"
    buffer  = []

    for line in lines:
        stripped = line.strip()
        if pattern.match(stripped):
            # Flush buffer into current section
            if buffer:
                sections[current] += "\n".join(buffer).strip() + "\n"
                buffer = []
            current = _map_heading(stripped)
        else:
            if stripped:
                buffer.append(stripped)

    # Flush remaining
    if buffer:
        sections[current] += "\n".join(buffer).strip()

    return {k: v.strip() for k, v in sections.items()}


def section_summary(sections: dict) -> dict:
    return {
        k: f"{len(v)} chars" if v else "NOT FOUND"
        for k, v in sections.items()
    }


# ── quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    sample = """Arjun Sharma
arjun@gmail.com | Bengaluru

Summary
Software engineer with 3 years of experience.

Skills
Python, FastAPI, Docker, TensorFlow

Experience
Software Engineer — Infosys (2022–Present)
Built REST APIs and ML pipelines.

Education
B.Tech Computer Science — VIT University, 2022

Projects
Resume Parser — spaCy + BERT pipeline

Certifications
AWS Certified Developer – 2023
"""
    secs = segment_resume(sample)
    print("=== Section Summary ===")
    for k, v in section_summary(secs).items():
        print(f"  {k:<20} {v}")
    print("\n=== Skills Section ===")
    print(secs["skills"])
