"""
confit_v2_fixed.py
------------------
Resume-Job Matching using ConFit v2 approach.

Based on: "ConFit v2: Improving Resume-Job Matching using Hypothetical Resume
Embedding and Runner-Up Hard-Negative Mining"
ArXiv: https://arxiv.org/abs/2502.12361

KEY CHANGE FROM ORIGINAL:
This version does NOT re-preprocess the resume. It accepts a preprocessed dict
(output of main_pipeline.process_one()) and uses the values directly:
    - query_vector from embeddings (no re-encoding)
    - entities from extraction (no re-extraction)
    - sections from segmentation (no re-segmentation)

This eliminates all redundant computation and allows the UI to call it directly
without re-running the entire pipeline.

Integration with the preprocessing pipeline
-------------------------------------------
Consumes the preprocessed dict with keys:
    filename, file_type, raw_text, sections, entities, embeddings

HRE Generator Modes:
    "rule"  — uses entities + sections already extracted
    "llm"   — sends the query_string to Claude API for high-quality HRE
    "local" — uses local Ollama model for offline HRE
    "groq"  — uses Groq's hosted OpenAI-compatible API (free tier) for HRE
"""

import json
import time
import hashlib
import pickle
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Literal
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    import chromadb
except ImportError:
    raise ImportError("pip install chromadb")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, RECOMMENDATIONS_DIR
from resume_processing.step4_embeddings import embed
from models.base_matcher import BaseMatcher

HREMode = Literal["rule", "llm", "local", "groq"]
HRE_CACHE_FILE = DATA_DIR / "processed" / "hre_cache.pkl"

# Blend Chroma cosine similarity into final score so rankings stay anchored to the
# same retrieval signal as other matchers (improves cross-model list agreement).
RETRIEVAL_ANCHOR_WEIGHT = 0.32


def _safe_job_index(metadata: dict) -> int:
    j = metadata.get("job_index", -1)
    try:
        v = int(j)
        return v if v >= 0 else -1
    except (TypeError, ValueError):
        return -1


# ══════════════════════════════════════════════════════════════════════════════
#  Base HRE Generator
# ══════════════════════════════════════════════════════════════════════════════

class BaseHREGenerator:
    def generate(self, preprocessed: dict, job_title: str, job_description: str) -> str:
        raise NotImplementedError

    def _cache_key(self, job_title: str, job_description: str) -> str:
        raw = f"{job_title}||{job_description[:500]}"
        return hashlib.md5(raw.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  Option A — Rule-Based HRE
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedHREGenerator(BaseHREGenerator):
    """
    Builds a hypothetical resume from already-extracted pipeline data.

    Uses:
        entities["skills"] — from step3_ner
        entities["years_exp"] — from step3_ner
        entities["organizations"] — from step3_ner
        sections["experience"] — from step2_segmentation
        sections["education"] — from step2_segmentation

    Zero re-extraction. Pure template-based using pipeline output.
    """

    def generate(self, preprocessed: dict, job_title: str, job_description: str) -> str:
        entities = preprocessed.get("entities", {})
        sections = preprocessed.get("sections", {})

        # Use skills already extracted by step3_ner
        skills = entities.get("skills", [])
        skills_str = ", ".join(skills[:15]) if skills else "software engineering"

        # Use experience info from step3_ner
        years_exp = entities.get("years_exp", "3+ years")
        orgs = entities.get("organizations", [])
        org_str = f" at {orgs[0]}" if orgs else ""

        # Use experience section from step2_segmentation
        exp_section = sections.get("experience", "")
        role_line = exp_section.split("\n")[0].strip() if exp_section else ""

        # Use education section from step2_segmentation
        edu_section = sections.get("education", "")
        edu_line = edu_section.split("\n")[0].strip() if edu_section else "Computer Science"

        # Build resume-style text
        title_clean = job_title.strip()
        hypo_resume = (
            f"Experienced {title_clean} with {years_exp} of experience{org_str}. "
            f"Technical expertise: {skills_str}. "
        )
        if role_line:
            hypo_resume += f"Most recent role: {role_line}. "
        hypo_resume += (
            f"Educational background: {edu_line}. "
            f"Proven track record delivering results in similar roles."
        )
        return hypo_resume.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Option B — LLM-Based HRE (Claude API)
# ══════════════════════════════════════════════════════════════════════════════

class LLMHREGenerator(BaseHREGenerator):
    """
    Generates hypothetical resume using Claude API.

    Receives query_string (already structured by step4) and job description,
    asks LLM to write an ideal resume for the job.

    Caches results to disk.
    """

    SYSTEM_PROMPT = (
        "You are an expert resume writer. Given a job description and a candidate's "
        "profile summary, write a concise hypothetical ideal resume (150-200 words) "
        "for the ideal candidate. Write in resume style: implicit first-person, "
        "realistic skills and experience descriptions matching the exact tools in "
        "the job posting. Output resume text only — no preamble, no explanation."
    )

    USER_PROMPT_TEMPLATE = (
        "Candidate Profile:\n{profile}\n\n"
        "Job Title: {title}\n\n"
        "Job Description:\n{description}\n\n"
        "Write a 150-200 word hypothetical resume for the ideal candidate."
    )

    def __init__(self, api_key: Optional[str] = None, cache: bool = True):
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required.\n"
                "Pass api_key= or set ANTHROPIC_API_KEY env var."
            )
        self.cache_enabled = cache
        self._cache: Dict[str, str] = {}
        self._load_cache()

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._use_sdk = True
            print("[ConFit v2 / LLM] Using anthropic SDK")
        except ImportError:
            self._use_sdk = False
            print("[ConFit v2 / LLM] SDK not found — using raw HTTP")

    def _load_cache(self):
        if self.cache_enabled and HRE_CACHE_FILE.exists():
            try:
                with open(HRE_CACHE_FILE, "rb") as f:
                    self._cache = pickle.load(f)
                print(f"[ConFit v2 / LLM] {len(self._cache)} cached HRE entries loaded")
            except Exception:
                self._cache = {}

    def _save_cache(self):
        if self.cache_enabled:
            HRE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HRE_CACHE_FILE, "wb") as f:
                pickle.dump(self._cache, f)

    def _call_sdk(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _call_http(self, prompt: str) -> str:
        payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 400,
            "system": self.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data["content"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"API error {e.code}: {e.read().decode()}")

    def generate(self, preprocessed: dict, job_title: str, job_description: str) -> str:
        key = self._cache_key(job_title, job_description)
        if key in self._cache:
            return self._cache[key]

        # Use query_string from step4 (already structured) as the profile
        profile = preprocessed.get("embeddings", {}).get("query_string", "")

        prompt = self.USER_PROMPT_TEMPLATE.format(
            profile=profile,
            title=job_title,
            description=job_description[:2000],
        )

        hypo = ""
        for attempt in range(3):
            try:
                hypo = self._call_sdk(prompt) if self._use_sdk else self._call_http(prompt)
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 2 ** attempt
                    print(f"[LLM] Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[LLM] Error (attempt {attempt+1}): {e}")
                if attempt == 2:
                    print("[LLM] Falling back to rule-based")
                    hypo = RuleBasedHREGenerator().generate(preprocessed, job_title, job_description)

        if hypo:
            self._cache[key] = hypo
            self._save_cache()
        return hypo


# ══════════════════════════════════════════════════════════════════════════════
#  Option C — Local LLM HRE (Ollama)
# ══════════════════════════════════════════════════════════════════════════════

class LocalLLMHREGenerator(BaseHREGenerator):
    """
    Generates hypothetical resumes using a local Ollama model.

    Setup: ollama pull mistral
    Docs:  https://ollama.com/download
    """

    SYSTEM_PROMPT = (
        "You are an expert resume writer. Write a 150-200 word hypothetical resume "
        "for the ideal candidate for this job. Resume-style language only, no preamble."
    )

    USER_PROMPT_TEMPLATE = (
        "Candidate Profile:\n{profile}\n\n"
        "Job Title: {title}\n\nJob Description:\n{description}\n\n"
        "Write a 150-200 word hypothetical resume for the ideal candidate."
    )

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, preprocessed: dict, job_title: str, job_description: str) -> str:
        profile = preprocessed.get("embeddings", {}).get("query_string", "")

        prompt = self.USER_PROMPT_TEMPLATE.format(
            profile=profile,
            title=job_title,
            description=job_description[:2000],
        )

        try:
            import requests
            payload = {
                "model": self.model,
                "prompt": f"{self.SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
            }
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"[LocalLLM] Error: {e}")
            print("[LocalLLM] Falling back to rule-based")
            return RuleBasedHREGenerator().generate(preprocessed, job_title, job_description)


# ══════════════════════════════════════════════════════════════════════════════
#  Option D — Groq HRE (OpenAI-compatible API, free tier)
# ══════════════════════════════════════════════════════════════════════════════

class GroqHREGenerator(BaseHREGenerator):
    """Generates hypothetical resume using Groq (free, OpenAI-compatible endpoint)."""

    SYSTEM_PROMPT = (
        "You are an expert resume writer. Given a job description and a candidate's "
        "profile summary, write a concise hypothetical ideal resume (150-200 words) "
        "for the ideal candidate. Write in resume style: implicit first-person, "
        "realistic skills and experience matching the exact tools in the posting. "
        "Output resume text only — no preamble, no explanation."
    )

    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: Optional[str] = None, cache: bool = True):
        import os
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self.model = model
        self.cache_enabled = cache
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_enabled and HRE_CACHE_FILE.exists():
            try:
                with open(HRE_CACHE_FILE, "rb") as f:
                    self._cache = pickle.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        if self.cache_enabled:
            HRE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HRE_CACHE_FILE, "wb") as f:
                pickle.dump(self._cache, f)

    def generate(self, preprocessed: dict, job_title: str, job_description: str) -> str:
        key = self._cache_key(job_title, job_description)
        if key in self._cache:
            return self._cache[key]

        profile = preprocessed.get("embeddings", {}).get("query_string", "")
        user_prompt = (
            f"Candidate Profile:\n{profile}\n\n"
            f"Job Title: {job_title}\n\nJob Description:\n{job_description[:2000]}\n\n"
            f"Write a 150-200 word hypothetical resume for the ideal candidate."
        )

        import requests

        hypo = ""
        for attempt in range(3):
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 400,
                    },
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    print(f"[Groq HRE] Rate limited — waiting {wait}s (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                hypo = resp.json()["choices"][0]["message"]["content"].strip()
                break
            except Exception as e:
                print(f"[Groq HRE] Error: {e} — falling back to rule-based")
                hypo = RuleBasedHREGenerator().generate(preprocessed, job_title, job_description)
                break

        if not hypo:
            print("[Groq HRE] Exhausted retries — falling back to rule-based")
            hypo = RuleBasedHREGenerator().generate(preprocessed, job_title, job_description)

        if hypo:
            self._cache[key] = hypo
            self._save_cache()
        return hypo


# ══════════════════════════════════════════════════════════════════════════════
#  ConFit v2 Engine
# ══════════════════════════════════════════════════════════════════════════════

class ConFitV2Engine(BaseMatcher):
    """
    Two-stage ConFit v2 matching engine.

    Stage 1 — ChromaDB Recall:
        Uses precomputed query_vector from embeddings (step4)
        Retrieves top_k * fetch_multiplier candidates

    Stage 2 — HRE Re-scoring:
        Generates hypothetical resumes (rule/llm/local/groq)
        Re-scores candidates using HRE embeddings

    Runner-up penalty applied to smooth score gaps.
    """

    MODEL_NAME = "ConFit v2"

    def __init__(self, jobs_db_dir: Optional[str] = None,
                 jobs_csv: Optional[str] = None,
                 hre_mode: HREMode = "rule",
                 llm_api_key: Optional[str] = None,
                 local_model: str = "mistral",
                 local_base_url: str = "http://localhost:11434",
                 hre_alpha: float = 0.65,
                 groq_model: Optional[str] = None,
                 groq_api_key: Optional[str] = None):
        """
        Initialize ConFit v2 engine.

        Args:
            jobs_db_dir: Path to job ChromaDB
            jobs_csv: Path to jobs CSV for metadata
            hre_mode: "rule", "llm", "local", or "groq"
            llm_api_key: Anthropic API key (required for llm mode)
            local_model: Ollama model name (for local mode)
            local_base_url: Ollama base URL
            hre_alpha: Weight for HRE score in final ranking (0-1)
            groq_model: Groq model name override (for groq mode)
            groq_api_key: Groq API key override (for groq mode; defaults to GROQ_API_KEY env var)
        """
        super().__init__(jobs_db_dir=jobs_db_dir, jobs_csv=jobs_csv)

        self.hre_mode = hre_mode
        self.hre_alpha = hre_alpha

        # Initialize job ChromaDB collection
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=jobs_db_dir or str(DATA_DIR / "jobs_db"))
            self.collection = self._client.get_or_create_collection(
                name="jobs",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[ConFit v2] Loaded jobs collection ({self.collection.count()} jobs)")
        except Exception as e:
            print(f"[ConFit v2] Warning: Could not load jobs ChromaDB: {e}")
            self.collection = None

        # Initialize HRE generator based on mode
        if hre_mode == "rule":
            self.hre_gen = RuleBasedHREGenerator()
        elif hre_mode == "llm":
            self.hre_gen = LLMHREGenerator(api_key=llm_api_key)
        elif hre_mode == "local":
            self.hre_gen = LocalLLMHREGenerator(model=local_model, base_url=local_base_url)
        elif hre_mode == "groq":
            self.hre_gen = GroqHREGenerator(
                model=groq_model or "openai/gpt-oss-120b",
                api_key=groq_api_key,
            )
        else:
            raise ValueError(f"Unknown hre_mode: {hre_mode}")

        print(f"[ConFit v2] Initialized with HRE mode: {hre_mode}")

    def _encode_job_with_hre(self, preprocessed: dict, job_title: str,
                             job_description: str, job_snippet: str = "") -> np.ndarray:
        """
        Generate HRE for a job and return its embedding.

        Combines: α * HRE_embedding + (1-α) * job_snippet_embedding
        """
        # Generate hypothetical resume for this job
        hre = self.hre_gen.generate(preprocessed, job_title, job_description)
        if not hre:
            return np.zeros(1024)

        # Embed the HRE
        hre_vec = np.array(embed(hre))

        # Combine with job snippet if available (α blend)
        if job_snippet:
            snippet_vec = np.array(embed(job_snippet))
            combined = (self.hre_alpha * hre_vec) + ((1 - self.hre_alpha) * snippet_vec)
            return combined / np.linalg.norm(combined)

        return hre_vec

    def _runner_up_rerank(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Runner-up hard-negative penalty (from ConFit v2 paper §3.2)."""
        if len(candidates) < 2:
            return candidates

        scores = np.array([c["raw_score"] for c in candidates])
        adjusted = scores.copy()

        for i in range(1, len(scores)):
            gap = scores[i-1] - scores[i]
            if gap < 0.03:  # Small gap → penalize
                adjusted[i] -= 0.005 * (1 - gap / 0.03)

        order = np.argsort(-adjusted)
        reranked = []
        for rank, idx in enumerate(order[:top_k], 1):
            c = dict(candidates[idx])
            c["rank"] = rank
            c["score"] = round(float(adjusted[idx]), 4)
            reranked.append(c)
        return reranked

    def recommend(self, preprocessed: dict, top_k: int = 20,
                  fetch_multiplier: int = 3,
                  stage1_n_results: Optional[int] = None,
                  **kwargs) -> List[Dict]:
        """
        Recommend jobs for a preprocessed resume.

        Args:
            preprocessed: Output of main_pipeline.process_one()
            top_k: Number of final recommendations
            fetch_multiplier: Multiplier for initial ChromaDB retrieval
            stage1_n_results: If set, fixed Chroma pool size (overrides multiplier)

        Returns:
            List of top_k job recommendations with scores
        """
        # Validate input
        if not self.validate_preprocessed(preprocessed):
            return []

        # Extract resume info for logging
        resume_emb = np.array(preprocessed["embeddings"]["query_vector"])
        resume_name = preprocessed.get("filename", "resume")

        print(f"\n[ConFit v2] Processing: {resume_name}")
        print(f"[ConFit v2] Entities — skills: {len(preprocessed['entities'].get('skills', []))}, "
              f"years_exp: {preprocessed['entities'].get('years_exp', 'N/A')}")

        # Stage 1 — ChromaDB recall using precomputed query_vector
        if not self.collection:
            print("[ConFit v2] Error: No jobs collection loaded")
            return []

        if stage1_n_results is not None:
            n_retrieve = min(int(stage1_n_results), 500)
        else:
            n_retrieve = min(top_k * fetch_multiplier, 500)
        print(f"[ConFit v2] Stage 1: Retrieving {n_retrieve} candidates from ChromaDB...")

        results = self.collection.query(
            query_embeddings=[resume_emb.tolist()],
            n_results=n_retrieve,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["metadatas"]:
            print("[ConFit v2] No results from ChromaDB")
            return []

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]

        # Stage 2 — HRE re-scoring
        print(f"[ConFit v2] Stage 2: HRE re-scoring {len(metadatas)} candidates...")
        candidates = []

        for i, (metadata, distance, doc_text) in enumerate(zip(metadatas, distances, documents)):
            job_title = metadata.get("title", "")
            job_idx = _safe_job_index(metadata)

            # Get full job description
            full_desc = ""
            if self.jobs_df is not None and 0 <= job_idx < len(self.jobs_df):
                full_desc = str(self.jobs_df.iloc[job_idx].get("Job Description", ""))
            else:
                full_desc = doc_text or ""

            # Generate HRE and score
            hre_emb = self._encode_job_with_hre(preprocessed, job_title, full_desc, doc_text or "")
            hre_score = float(np.dot(resume_emb, hre_emb))
            bi_score = float(1.0 - distance)
            w = RETRIEVAL_ANCHOR_WEIGHT
            combined = (1.0 - w) * hre_score + w * bi_score

            candidates.append({
                "raw_score": combined,
                "score": round(combined, 4),
                "hre_score": round(hre_score, 4),
                "bi_score": round(bi_score, 4),
                "hre_mode": self.hre_mode,
                "job_index": job_idx,
                "title": job_title,
                "company": metadata.get("company", ""),
                "domain": metadata.get("domain", ""),
                "skills": metadata.get("skills", ""),
                "experience_level": metadata.get("experience_level", ""),
                "location": metadata.get("location", ""),
                "work_type": metadata.get("work_type", ""),
                "salary": metadata.get("salary", ""),
                "source": metadata.get("source", ""),
                "description": full_desc[:500],
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(metadatas)}]")

        candidates.sort(key=lambda x: x["raw_score"], reverse=True)

        print("[ConFit v2] Applying runner-up re-ranking...")
        final = self._runner_up_rerank(candidates, top_k=top_k)
        print(f"[ConFit v2] Done — {len(final)} recommendations")
        return final

    def export(self, recommendations: List[Dict], output_path: Optional[str] = None) -> str:
        """Export recommendations to CSV."""
        if not recommendations:
            return ""

        df = pd.DataFrame(recommendations).drop(columns=["raw_score"], errors="ignore")

        if output_path is None:
            RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RECOMMENDATIONS_DIR / f"confit_v2_{self.hre_mode}_{ts}.csv"

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[ConFit v2] Saved → {output_path}")
        return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ConFit v2 — accepts preprocessed resume",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--preprocessed-json", required=True,
                        help="Path to preprocessed JSON (output of main_pipeline)")
    parser.add_argument("--resume-index", type=int, default=0,
                        help="Index in JSON array to use")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hre-mode", default="rule", choices=["rule", "llm", "local", "groq"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--local-model", default="mistral")
    parser.add_argument("--groq-model", default=None)
    parser.add_argument("--groq-api-key", default=None)
    parser.add_argument("--hre-alpha", type=float, default=0.65)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load preprocessed data
    with open(args.preprocessed_json, "r") as f:
        all_preprocessed = json.load(f)

    if args.resume_index >= len(all_preprocessed):
        print(f"Error: Index {args.resume_index} out of range")
        exit(1)

    preprocessed = all_preprocessed[args.resume_index]
    print(f"[+] Loaded: {preprocessed['filename']}")

    # Initialize engine
    engine = ConFitV2Engine(
        hre_mode=args.hre_mode,
        llm_api_key=args.api_key,
        local_model=args.local_model,
        groq_model=args.groq_model,
        groq_api_key=args.groq_api_key,
        hre_alpha=args.hre_alpha,
    )

    # Get recommendations
    recs = engine.recommend(preprocessed, top_k=args.top_k)

    # Display
    print("\n" + "=" * 70)
    print(f"  ConFit v2 [{args.hre_mode.upper()}] — TOP {min(10, len(recs))} RECOMMENDATIONS")
    print("=" * 70)
    for r in recs[:10]:
        print(f"  {r['rank']:2}. [{r['score']*100:.1f}%] {r['title']} @ {r['company']}")
        print(f"      Domain: {r['domain']} | Level: {r['experience_level']}")

    # Export
    engine.export(recs, output_path=args.output)