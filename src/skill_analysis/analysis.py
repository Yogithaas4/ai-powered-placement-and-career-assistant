from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from config import DEFAULT_JOBS_CSV

SKILL_ALIASES = {
    "amazon web services": "AWS",
    "aws ecs": "AWS",
    "aws ec2": "AWS",
    "aws lambda": "AWS",
    "ci cd": "CI/CD",
    "ci/cd": "CI/CD",
    "github actions": "GitHub Actions",
    "rest apis": "REST API",
    "rest api": "REST API",
    "apis": "API",
    "machine learn": "Machine Learning",
    "deep learn": "Deep Learning",
    "scikit learn": "Scikit-learn",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "fast api": "FastAPI",
    "node.js": "Node.js",
    "node": "Node.js",
    "js": "JavaScript",
    "ts": "TypeScript",
    "c sharp": "C#",
    "c plus plus": "C++",
    "nlp": "NLP",
    "llms": "LLM",
    "llm": "LLM",
    "k8s": "Kubernetes",
    "gcp": "GCP",
}

KNOWN_SKILLS = [
    "Python", "Java", "C++", "C#", "C", "Go", "Rust", "R", "JavaScript", "TypeScript",
    "React", "Angular", "Vue", "Node.js", "Next.js", "HTML", "CSS", "REST API",
    "GraphQL", "Django", "Flask", "FastAPI", "Spring Boot", "SQL", "NoSQL", "MongoDB",
    "PostgreSQL", "MySQL", "Redis", "Machine Learning", "Deep Learning", "TensorFlow",
    "PyTorch", "Scikit-learn", "NLP", "Computer Vision", "LLM", "Spark", "Hadoop",
    "Kafka", "Airflow", "dbt", "Pandas", "NumPy", "AWS", "Azure", "GCP", "Docker",
    "Kubernetes", "Terraform", "Ansible", "Jenkins", "CI/CD", "GitHub Actions", "Linux",
    "Nginx", "Helm", "Prometheus", "Grafana", "Datadog", "Git", "Selenium", "Cypress",
    "Postman", "REST Assured", "JMeter", "JUnit", "PyTest", "Unity", "Unreal Engine",
    "Photon", "ROS", "Tableau", "Power BI", "Azure ML", "Microservices", "API",
]

SKILL_CATEGORIES = {
    "Python": "Backend / ML",
    "Java": "Backend / API",
    "Go": "Backend / API",
    "Rust": "Backend / Systems",
    "FastAPI": "Backend / API",
    "Django": "Backend / API",
    "Flask": "Backend / API",
    "Spring Boot": "Backend / API",
    "Node.js": "Backend / API",
    "REST API": "Backend / API",
    "GraphQL": "Backend / API",
    "SQL": "Data Engineering",
    "PostgreSQL": "Data Engineering",
    "MySQL": "Data Engineering",
    "MongoDB": "Data Engineering",
    "Spark": "Data Engineering",
    "Hadoop": "Data Engineering",
    "Kafka": "Data Engineering",
    "Airflow": "Data Engineering",
    "dbt": "Data Engineering",
    "TensorFlow": "ML / AI",
    "PyTorch": "ML / AI",
    "Scikit-learn": "ML / AI",
    "Machine Learning": "ML / AI",
    "Deep Learning": "ML / AI",
    "NLP": "ML / AI",
    "Computer Vision": "ML / AI",
    "LLM": "ML / AI",
    "Docker": "Cloud & DevOps",
    "Kubernetes": "Cloud & DevOps",
    "AWS": "Cloud & DevOps",
    "Azure": "Cloud & DevOps",
    "GCP": "Cloud & DevOps",
    "Terraform": "Cloud & DevOps",
    "Ansible": "Cloud & DevOps",
    "Jenkins": "Cloud & DevOps",
    "CI/CD": "Cloud & DevOps",
    "GitHub Actions": "Cloud & DevOps",
    "Linux": "Cloud & DevOps",
    "Helm": "Cloud & DevOps",
    "Prometheus": "Cloud & DevOps",
    "Grafana": "Cloud & DevOps",
    "Datadog": "Cloud & DevOps",
    "React": "Frontend / Web",
    "Angular": "Frontend / Web",
    "Vue": "Frontend / Web",
    "HTML": "Frontend / Web",
    "CSS": "Frontend / Web",
    "Unity": "Game Development",
    "Unreal Engine": "Game Development",
    "Photon": "Game Development",
    "Selenium": "QA / Testing",
    "Cypress": "QA / Testing",
    "Postman": "QA / Testing",
    "REST Assured": "QA / Testing",
    "JMeter": "QA / Testing",
    "PyTest": "QA / Testing",
    "JUnit": "QA / Testing",
    "Tableau": "Analytics / BI",
    "Power BI": "Analytics / BI",
}

LEARNING_PREREQS = {
    "Docker": ["Linux", "Python", "Node.js"],
    "Kubernetes": ["Docker", "Linux"],
    "AWS": ["Docker", "Linux", "Python", "Terraform"],
    "Terraform": ["AWS", "Azure", "GCP"],
    "Airflow": ["Python", "SQL"],
    "Spark": ["Python", "SQL"],
    "PyTorch": ["Python", "Machine Learning"],
    "TensorFlow": ["Python", "Machine Learning"],
    "FastAPI": ["Python", "REST API"],
    "GraphQL": ["REST API", "JavaScript", "TypeScript"],
    "React": ["JavaScript", "TypeScript", "HTML", "CSS"],
    "Azure ML": ["Python", "Machine Learning", "Azure"],
}

TITLE_ROLE_HINTS = {
    "backend": "Backend Engineer",
    "api": "Backend Engineer",
    "machine learning": "ML Engineer",
    "ml ": "ML Engineer",
    "data scientist": "Data Scientist",
    "data engineer": "Data Engineer",
    "frontend": "Frontend Engineer",
    "full stack": "Full Stack Engineer",
    "devops": "DevOps Engineer",
    "cloud": "Cloud Engineer",
    "qa": "QA Engineer",
    "test": "QA Engineer",
    "game": "Game Developer",
}

LOW_SIGNAL_SKILLS = {"API"}


def _clean_token(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#./-]+", " ", text.lower())).strip()


def normalize_skill(skill: str) -> str:
    raw = _clean_token(skill)
    if not raw:
        return ""
    if raw in SKILL_ALIASES:
        return SKILL_ALIASES[raw]

    for known in KNOWN_SKILLS:
        if raw == _clean_token(known):
            return known

    if len(raw) <= 2 and raw not in {"c", "r"}:
        return ""

    return " ".join(part.capitalize() if part not in {"api", "ci/cd"} else part.upper() for part in raw.split())


@lru_cache(maxsize=1)
def load_jobs_dataframe() -> pd.DataFrame:
    path = Path(DEFAULT_JOBS_CSV)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig").fillna("")


@lru_cache(maxsize=1)
def known_skill_patterns() -> List[Tuple[str, re.Pattern[str]]]:
    patterns = []
    for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
        token = re.escape(skill)
        patterns.append((skill, re.compile(rf"(?<![a-z0-9]){token}(?![a-z0-9])", re.IGNORECASE)))
    return patterns


def _split_skills(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[,|\n;/]+", str(text))
    return [p.strip(" -:.") for p in parts if p.strip(" -:.")]


def extract_known_skills(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    seen = set()
    for skill, pattern in known_skill_patterns():
        if pattern.search(text) and skill not in seen:
            seen.add(skill)
            found.append(skill)
    return found


def canonicalize_skills(skills: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for skill in skills:
        normalized = normalize_skill(skill)
        if not normalized:
            continue
        if len(normalized) == 1:
            continue
        if normalized in LOW_SIGNAL_SKILLS:
            continue
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def extract_resume_skills(preprocessed: dict) -> List[str]:
    entities = preprocessed.get("entities", {})
    sections = preprocessed.get("sections", {})
    combined = []
    combined.extend(entities.get("skills") or [])
    for section_name in ("skills", "experience", "projects", "certifications", "summary"):
        combined.extend(_split_skills(sections.get(section_name, "")))
        combined.extend(extract_known_skills(sections.get(section_name, "")))
    return canonicalize_skills(combined)


def job_row_from_rec(rec: dict) -> dict:
    jobs_df = load_jobs_dataframe()
    idx = rec.get("job_index", -1)
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = -1
    if 0 <= idx < len(jobs_df):
        return jobs_df.iloc[idx].to_dict()
    return {}


def extract_job_skills(rec: dict) -> List[str]:
    row = job_row_from_rec(rec)
    combined = []
    combined.extend(_split_skills(rec.get("skills", "")))
    combined.extend(extract_known_skills(rec.get("description", "")))
    combined.extend(extract_known_skills(rec.get("title", "")))
    normalized = canonicalize_skills(combined)
    if len(normalized) >= 3:
        return normalized

    fallback = list(combined)
    fallback.extend(_split_skills(row.get("Skills", "")))
    fallback.extend(extract_known_skills(row.get("Job Description", "")))
    return canonicalize_skills(fallback)


def infer_role_label(rec: dict) -> str:
    title = str(rec.get("title", "")).lower()
    domain = str(rec.get("domain", "")).strip()
    for needle, label in TITLE_ROLE_HINTS.items():
        if needle in title:
            return label
    if domain:
        return domain
    return "Recommended Roles"


def infer_learning_reason(target_skill: str, resume_skills: List[str], dominant_domains: List[str], occurrences: int) -> str:
    normalized_resume = set(resume_skills)
    prereqs = LEARNING_PREREQS.get(target_skill, [])
    for prereq in prereqs:
        if prereq in normalized_resume:
            return f"Since you already know {prereq}, {target_skill} is a practical next step."

    category = SKILL_CATEGORIES.get(target_skill, "")
    if category:
        return f"{target_skill} keeps showing up across your recommended {category.lower()} roles."

    if dominant_domains:
        return f"{target_skill} is repeatedly requested in your top matches, especially around {dominant_domains[0]} roles."

    if occurrences > 1:
        return f"{target_skill} appears in multiple recommended jobs, so it is worth prioritizing."

    return f"{target_skill} helps close one of the clearest gaps in this recommendation."


def build_learning_path(missing_skills: List[str], resume_skills: List[str], dominant_domains: List[str], global_missing: Counter) -> List[Dict]:
    path = []
    resume_skill_set = set(resume_skills)
    for skill in missing_skills:
        prereq = next((p for p in LEARNING_PREREQS.get(skill, []) if p in resume_skill_set), None)
        start = prereq or SKILL_CATEGORIES.get(skill, "Foundations")
        path.append(
            {
                "from": start,
                "to": skill,
                "reason": infer_learning_reason(skill, resume_skills, dominant_domains, global_missing.get(skill, 0)),
            }
        )
    return path


def build_skill_graph(preprocessed: dict, recommendations: List[dict], top_k: int = 10) -> Dict:
    resume_skills = extract_resume_skills(preprocessed)
    selected = recommendations[:top_k]
    role_counter = Counter()
    missing_counter = Counter()
    prerequisite_edges = Counter()
    demand_edges = Counter()
    nodes: Dict[str, Dict] = {}

    def add_node(node_id: str, label: str, node_type: str, weight: int = 1):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "label": label, "type": node_type, "weight": weight}
        else:
            nodes[node_id]["weight"] = max(nodes[node_id]["weight"], weight)

    for skill in resume_skills[:12]:
        add_node(f"resume::{skill}", skill, "resume_skill", 1)

    for rec in selected:
        role = infer_role_label(rec)
        role_counter[role] += 1
        add_node(f"role::{role}", role, "role", role_counter[role])

        job_skills = extract_job_skills(rec)
        missing = [skill for skill in job_skills if skill not in resume_skills][:4]
        for skill in missing:
            missing_counter[skill] += 1
            add_node(f"missing::{skill}", skill, "missing_skill", missing_counter[skill])
            demand_edges[(skill, role)] += 1

            prereq = next((p for p in LEARNING_PREREQS.get(skill, []) if p in resume_skills), None)
            if prereq:
                prerequisite_edges[(prereq, skill)] += 1

    edges = []
    for (skill, role), weight in demand_edges.items():
        edges.append(
            {
                "source": f"missing::{skill}",
                "target": f"role::{role}",
                "label": f"{weight} jobs",
                "weight": weight,
                "type": "demand",
            }
        )

    for (source_skill, target_skill), weight in prerequisite_edges.items():
        add_node(f"resume::{source_skill}", source_skill, "resume_skill", 1)
        add_node(f"missing::{target_skill}", target_skill, "missing_skill", weight)
        edges.append(
            {
                "source": f"resume::{source_skill}",
                "target": f"missing::{target_skill}",
                "label": "learn next",
                "weight": weight,
                "type": "path",
            }
        )

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "resume_skills": resume_skills,
        "top_missing_skills": [{"skill": skill, "count": count} for skill, count in missing_counter.most_common(8)],
        "top_roles": [{"role": role, "count": count} for role, count in role_counter.most_common(6)],
    }


def build_graphviz_dot(graph: Dict) -> str:
    palette = {
        "resume_skill": "#dbeafe",
        "missing_skill": "#fee2e2",
        "role": "#dcfce7",
    }

    lines = [
        "digraph SkillGraph {",
        '  rankdir=LR;',
        '  graph [bgcolor="white", pad="0.25"];',
        '  node [shape=box, style="rounded,filled", color="#94a3b8", fontname="Helvetica"];',
        '  edge [color="#64748b", fontname="Helvetica"];',
    ]

    for node in graph.get("nodes", []):
        fill = palette.get(node.get("type"), "#f8fafc")
        lines.append(
            f'  "{node["id"]}" [label="{node["label"]}", fillcolor="{fill}"];'
        )

    seen = set()
    for edge in graph.get("edges", []):
        key = (edge["source"], edge["target"])
        if key in seen:
            continue
        seen.add(key)
        label = edge.get("label", "")
        lines.append(
            f'  "{edge["source"]}" -> "{edge["target"]}" [label="{label}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def summarize_focus_track(missing_counter: Counter) -> str:
    if not missing_counter:
        return "Profile Fit Looks Strong"
    track_counts = Counter()
    for skill, count in missing_counter.items():
        track_counts[SKILL_CATEGORIES.get(skill, "Core Engineering")] += count
    return track_counts.most_common(1)[0][0]


def closest_roles_summary(recommendations: List[dict]) -> List[str]:
    counter = Counter()
    for rec in recommendations:
        counter[infer_role_label(rec)] += 1
    return [role for role, _ in counter.most_common(2)]


def analyze_recommendations(preprocessed: dict, recommendations: List[dict], top_k: int = 10) -> Dict:
    resume_skills = extract_resume_skills(preprocessed)
    selected = recommendations[:top_k]
    global_missing = Counter()
    dominant_domains = Counter(str(rec.get("domain", "")).strip() or "Recommended roles" for rec in selected)
    enriched_jobs = []

    for rec in selected:
        job_skills = extract_job_skills(rec)
        overlap = [skill for skill in job_skills if skill in resume_skills]
        missing = [skill for skill in job_skills if skill not in resume_skills]
        trimmed_missing = missing[:4]
        global_missing.update(trimmed_missing)

        enriched_jobs.append(
            {
                **rec,
                "job_skills": job_skills,
                "matching_skills": overlap[:8],
                "missing_skills": trimmed_missing,
            }
        )

    domain_labels = [domain for domain, _ in dominant_domains.most_common(3)]
    for rec in enriched_jobs:
        rec["learning_path"] = build_learning_path(
            rec.get("missing_skills", []),
            resume_skills,
            domain_labels,
            global_missing,
        )

    focus_track = summarize_focus_track(global_missing)
    closest_roles = closest_roles_summary(selected)
    recommended_path = []
    path_seen = set()
    for rec in enriched_jobs:
        for step in rec.get("learning_path", []):
            key = (step["from"], step["to"])
            if key not in path_seen:
                path_seen.add(key)
                recommended_path.append(f'{step["from"]} -> {step["to"]}')
            if len(recommended_path) >= 4:
                break
        if len(recommended_path) >= 4:
            break

    return {
        "resume_skills": resume_skills,
        "jobs": enriched_jobs,
        "global": {
            "closest_roles": closest_roles,
            "focus_track": focus_track,
            "top_missing_skills": [
                {"skill": skill, "count": count}
                for skill, count in global_missing.most_common(8)
            ],
            "recommended_path": recommended_path,
            "dominant_domains": domain_labels,
            "summary": (
                f"Based on your profile and the current top matches, you are closest to "
                f"{', '.join(closest_roles) if closest_roles else 'these roles'}, and the strongest improvement area is {focus_track}."
            ),
        },
    }


def export_graph_payload(preprocessed: dict, recommendations: List[dict], output_path: str | Path, top_k: int = 10) -> Path:
    payload = analyze_recommendations(preprocessed, recommendations, top_k=top_k)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
