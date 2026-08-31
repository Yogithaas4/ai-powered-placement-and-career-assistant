"""
hybrid_pipeline_matcher.py
---------------------------
Final Hybrid Resume-Job Recommendation Model.

Four-stage staged pipeline (NOT Reciprocal Rank Fusion -- see
models/hybrid_matcher.py for the RRF-based "Hybrid (RRF)" combiner, which is
a different, already-existing architecture kept unchanged):

    Stage 1 -- Dense Bi-Encoder Retrieval (ChromaDB, precomputed query_vector)
    Stage 2 -- Skill-Level ColBERT MaxSim reranking/filtering
    Stage 3 -- Experience Compatibility filtering
    Stage 4 -- Cross-Encoder reranking (primary final signal)

Design notes
------------
- No re-preprocessing: consumes the preprocessed dict from
  resume_processing.main_pipeline.process_one() exactly like ConFit v2 /
  ColBERT / CrossEncoder do (query_vector, query_string, sections, entities).
- Reuses existing model wrappers instead of duplicating them:
    ColBERTEncoder + maxsim_score   from models.colbert_matcher_fixed
    CrossEncoderReranker,
    build_condensed_resume         from models.cross_encoder_matcher_fixed
- No fine-tuning anywhere in this file -- pretrained models only. The
  CrossEncoder call is isolated in _crossencoder_rerank() so a fine-tuned
  checkpoint can be swapped in later without touching the rest of the
  pipeline.
- All intermediate scores (dense_score, colbert_skill_score,
  experience_compatibility, crossencoder_score) are preserved on every
  returned recommendation for analysis/evaluation -- they are not silently
  collapsed into a single number unless the caller explicitly asks for a
  weighted blend via `final_score_weights`.
"""

import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import chromadb
except ImportError:
    raise ImportError("pip install chromadb")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, RECOMMENDATIONS_DIR                    # noqa: E402
from models.base_matcher import BaseMatcher                          # noqa: E402
from models.colbert_matcher_fixed import ColBERTEncoder, maxsim_score  # noqa: E402
from models.cross_encoder_matcher_fixed import (                     # noqa: E402
    CrossEncoderReranker,
    build_condensed_resume,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Small shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_job_index(metadata: dict) -> int:
    j = metadata.get("job_index", -1)
    try:
        v = int(j)
        return v if v >= 0 else -1
    except (TypeError, ValueError):
        return -1


def _split_skills(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[,;/\n]+", str(text))
    return [p.strip(" -:.") for p in parts if p.strip(" -:.")]


# ── Experience-level vocabulary (must match scraper.py / job_ingestion.py) ──
EXPERIENCE_LEVEL_ORDER = {
    "Entry Level": 0,
    "Junior (1-3 yrs)": 1,
    "Mid Level (3-5 yrs)": 2,
    "Senior (5+ yrs)": 3,
    "Lead/Principal (8+ yrs)": 4,
}
UNSPECIFIED_LEVELS = {"", "not specified", "not disclosed", "n/a", "na", "none"}


def _parse_resume_years(years_exp_text: str) -> Optional[float]:
    """Parse '3 years' -> 3.0. Returns None if unavailable/unparseable."""
    if not years_exp_text:
        return None
    match = re.match(r"(\d+(?:\.\d+)?)", str(years_exp_text).strip())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _resume_level_from_years(years: Optional[float]) -> Optional[str]:
    """Bucket numeric years of experience into the same level vocabulary jobs use."""
    if years is None:
        return None
    if years < 1:
        return "Entry Level"
    if years < 3:
        return "Junior (1-3 yrs)"
    if years < 5:
        return "Mid Level (3-5 yrs)"
    if years < 8:
        return "Senior (5+ yrs)"
    return "Lead/Principal (8+ yrs)"


def compute_experience_compatibility(
    resume_years: Optional[float],
    job_level_text: str,
    tolerance: int = 1,
) -> Dict:
    """
    Determine whether a candidate satisfies a job's experience requirement.

    Never invents missing information:
        - Unknown/unspecified job requirement  -> always compatible (neutral score 1.0)
        - Unknown resume experience             -> always compatible (neutral score 0.5,
                                                    flagged so it's visible in logs/exports)
    Only filters out candidates that are CLEARLY under-qualified once both
    sides are known, using a configurable `tolerance` (number of levels the
    candidate is allowed to fall short by and still pass).

    Returns:
        {
            "compatible": bool,
            "score": float (0-1, higher = better fit; informational even when compatible),
            "reason": str,
            "resume_level": Optional[str],
            "job_level": Optional[str],
        }
    """
    job_level_raw = (job_level_text or "").strip()
    job_level = job_level_raw if job_level_raw in EXPERIENCE_LEVEL_ORDER else None

    if job_level is None:
        return {
            "compatible": True,
            "score": 1.0,
            "reason": "Job experience requirement unspecified — not filtered",
            "resume_level": _resume_level_from_years(resume_years),
            "job_level": None,
        }

    resume_level = _resume_level_from_years(resume_years)
    if resume_level is None:
        return {
            "compatible": True,
            "score": 0.5,
            "reason": "Resume years of experience unavailable — not filtered",
            "resume_level": None,
            "job_level": job_level,
        }

    diff = EXPERIENCE_LEVEL_ORDER[resume_level] - EXPERIENCE_LEVEL_ORDER[job_level]
    compatible = diff >= -tolerance
    shortfall = max(0, -diff)
    score = max(0.0, 1.0 - shortfall * 0.25)

    if diff >= 0:
        reason = f"Resume level '{resume_level}' meets or exceeds job level '{job_level}'"
    elif compatible:
        reason = (
            f"Resume level '{resume_level}' is below job level '{job_level}' "
            f"but within tolerance ({tolerance})"
        )
    else:
        reason = (
            f"Resume level '{resume_level}' is below job level '{job_level}' "
            f"beyond tolerance ({tolerance}) — filtered out"
        )

    return {
        "compatible": compatible,
        "score": round(score, 4),
        "reason": reason,
        "resume_level": resume_level,
        "job_level": job_level,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Hybrid Pipeline Engine
# ══════════════════════════════════════════════════════════════════════════════

class HybridPipelineEngine(BaseMatcher):
    """
    Staged hybrid resume-to-job matching engine.

    Stage 1 — Dense Bi-Encoder Retrieval:
        ChromaDB cosine search using the precomputed resume query_vector.
        NO re-encoding of the resume.

    Stage 2 — Skill-Level ColBERT Reranking:
        Token-level MaxSim between resume skills (entities["skills"]) and
        each candidate job's skills (job metadata / CSV "Skills" column).
        Reuses the existing ColBERTEncoder — only run on the Stage-1 pool,
        never the whole job database. Job-side skill token matrices are
        cached in-memory per job_index so repeated calls (e.g. across many
        resumes in a batch run) don't re-encode the same job twice.

    Stage 3 — Experience Compatibility Filtering:
        Compares resume years-of-experience (entities["years_exp"]) against
        each job's "Experience Level" requirement. Only drops candidates
        that are clearly under-qualified beyond a configurable tolerance;
        never invents missing data.

    Stage 4 — Cross-Encoder Reranking:
        Pretrained MS-MARCO cross-encoder (reused from
        models.cross_encoder_matcher_fixed.CrossEncoderReranker) scores the
        surviving candidates. This is the primary final ranking signal.
        No fine-tuning here — swap the checkpoint later without touching
        the rest of the pipeline.
    """

    MODEL_NAME = "HybridPipeline"

    def __init__(
        self,
        jobs_db_dir: Optional[str] = None,
        jobs_csv: Optional[str] = None,
        cross_encoder_model: Optional[str] = None,
        ce_batch_size: int = 32,
        experience_tolerance: int = 1,
    ):
        """
        Args:
            jobs_db_dir: Path to job ChromaDB
            jobs_csv: Path to jobs CSV for metadata / descriptions
            cross_encoder_model: Custom CE model name (defaults to MS-MARCO MiniLM)
            ce_batch_size: Batch size for CE inference
            experience_tolerance: Levels a candidate may fall short of the
                job's required experience level and still be considered
                compatible (default 1 — e.g. Mid-level candidate can still
                pass for a Senior-labelled posting).
        """
        super().__init__(jobs_db_dir=jobs_db_dir, jobs_csv=jobs_csv)

        self.jobs_db_dir = jobs_db_dir or str(DATA_DIR / "jobs_db")
        self.jobs_csv = jobs_csv or str(DATA_DIR / "jobs" / "all_jobs_v3_fixed.csv")
        self.experience_tolerance = experience_tolerance
        self.ce_batch_size = ce_batch_size

        # ── ChromaDB (Stage 1) ──────────────────────────────────────────
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.jobs_db_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name="jobs",
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[HybridPipeline] Loaded jobs collection ({self.collection.count()} jobs)")
        except Exception as e:
            print(f"[HybridPipeline] Warning: Could not load jobs ChromaDB: {e}")
            self.collection = None

        # ── Jobs metadata (for descriptions / skills fallback) ──────────
        self.jobs_df = self._load_jobs_df()
        print(f"[HybridPipeline] Loaded {len(self.jobs_df)} jobs metadata")

        # ── Stage 2: ColBERT encoder (reused, not reimplemented) ────────
        self.col_encoder = ColBERTEncoder()
        self._job_skill_token_cache: Dict[int, Optional[np.ndarray]] = {}

        # ── Stage 4: CrossEncoder (reused, not reimplemented) ───────────
        self.ce_reranker = CrossEncoderReranker(model_name=cross_encoder_model)

        print("[HybridPipeline] Engine ready")

    # ── setup helpers ────────────────────────────────────────────────────

    def _load_jobs_df(self) -> pd.DataFrame:
        p = Path(self.jobs_csv)
        return pd.read_csv(p, encoding="utf-8-sig").fillna("") if p.exists() else pd.DataFrame()

    def _job_skills_text(self, metadata: dict, job_idx: int) -> str:
        skills_text = metadata.get("skills", "")
        if (not skills_text or skills_text == "Not Listed") and 0 <= job_idx < len(self.jobs_df):
            skills_text = str(self.jobs_df.iloc[job_idx].get("Skills", ""))
        if skills_text == "Not Listed":
            skills_text = ""
        return skills_text

    def _get_job_skill_tokens(self, metadata: dict, job_idx: int) -> Optional[np.ndarray]:
        """Encode (and cache) a job's skill text at token level. None if no skills data."""
        if job_idx in self._job_skill_token_cache:
            return self._job_skill_token_cache[job_idx]

        skills_text = self._job_skills_text(metadata, job_idx)
        if not skills_text.strip():
            self._job_skill_token_cache[job_idx] = None
            return None

        try:
            vecs = self.col_encoder.encode(skills_text, max_length=64, is_query=False)
        except Exception as e:
            print(f"[HybridPipeline] Warning: skill encode failed for job {job_idx}: {e}")
            vecs = None

        self._job_skill_token_cache[job_idx] = vecs
        return vecs

    def _build_job_text_for_ce(self, metadata: dict, job_idx: int) -> str:
        """Same shape as CrossEncoderEngine._build_job_text — kept local so this
        engine doesn't need to instantiate a full CrossEncoderEngine just to
        reuse one private method."""
        title = metadata.get("title", "")
        company = metadata.get("company", "")
        domain = metadata.get("domain", "")
        skills = metadata.get("skills", "")
        level = metadata.get("experience_level", "")

        header = f"Job: {title}. Company: {company}. Domain: {domain}. "
        if level:
            header += f"Level: {level}. "
        if skills:
            header += f"Skills: {skills[:200]}. "

        raw_desc = ""
        if 0 <= job_idx < len(self.jobs_df):
            raw_desc = str(self.jobs_df.iloc[job_idx].get("Job Description", ""))[:400]

        return (header + raw_desc)[:700]

    # ── main pipeline ───────────────────────────────────────────────────

    def recommend(
        self,
        preprocessed: dict,
        top_k: int = 20,
        stage1_n_results: int = 1000,
        colbert_top_n: int = 100,
        experience_tolerance: Optional[int] = None,
        final_score_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Run the full 4-stage hybrid pipeline.

        Args:
            preprocessed: Output of main_pipeline.process_one()
            top_k: Number of final recommendations to return
            stage1_n_results: Stage-1 ChromaDB candidate pool size
            colbert_top_n: Candidates kept after Stage-2 skill-level ColBERT reranking
            experience_tolerance: Overrides the instance default for this call
            final_score_weights: Optional dict to blend intermediate scores into
                "score" instead of using the raw crossencoder_score, e.g.
                {"dense": 0.1, "colbert_skill": 0.2, "experience_compatibility": 0.1,
                 "crossencoder": 0.6}. If None (default), "score" == crossencoder_score,
                 since the CrossEncoder is the intended primary final ranking signal.

        Returns:
            List of top_k recommendation dicts (see module docstring for output shape).
        """
        if not self.validate_preprocessed(preprocessed):
            return []

        tolerance = experience_tolerance if experience_tolerance is not None else self.experience_tolerance
        resume_name = preprocessed.get("filename", "resume")
        entities = preprocessed.get("entities", {})
        sections = preprocessed.get("sections", {})

        print(f"\n[HybridPipeline] Processing: {resume_name}")

        # ═══════════════════════════ STAGE 1 ═══════════════════════════
        # Dense bi-encoder retrieval — no re-encoding of the resume.
        if not self.collection:
            print("[HybridPipeline] Error: No jobs collection loaded")
            return []

        resume_vec = np.array(preprocessed["embeddings"]["query_vector"])
        n_recall = min(int(stage1_n_results), 1000)

        results = self.collection.query(
            query_embeddings=[resume_vec.tolist()],
            n_results=n_recall,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["metadatas"] or not results["metadatas"][0]:
            print("[HybridPipeline] No results from ChromaDB — aborting")
            return []

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        candidates = []
        for metadata, distance in zip(metadatas, distances):
            job_idx = _safe_job_index(metadata)
            candidates.append({
                "metadata": metadata,
                "job_idx": job_idx,
                "dense_score": round(1.0 - distance, 4),
            })

        print(f"[HybridPipeline] Stage 1 (Dense Retrieval): {len(candidates)} candidates")

        # ═══════════════════════════ STAGE 2 ═══════════════════════════
        # Skill-level ColBERT MaxSim reranking on the Stage-1 pool only.
        resume_skills = list(entities.get("skills") or [])
        if not resume_skills:
            resume_skills = _split_skills(sections.get("skills", ""))
        resume_skills_text = ", ".join(resume_skills)

        resume_skill_tokens = None
        if resume_skills_text.strip():
            try:
                resume_skill_tokens = self.col_encoder.encode(
                    resume_skills_text, max_length=48, is_query=True
                )
            except Exception as e:
                print(f"[HybridPipeline] Warning: resume skill encode failed: {e}")
                resume_skill_tokens = None

        for cand in candidates:
            metadata = cand["metadata"]
            job_idx = cand["job_idx"]

            if resume_skill_tokens is None:
                # No resume skills available — don't invent a score, fall back
                # to the dense retrieval signal so Stage 2 is a no-op ranking-wise.
                cand["colbert_skill_score"] = None
                cand["_stage2_sort_key"] = cand["dense_score"]
                continue

            job_skill_tokens = self._get_job_skill_tokens(metadata, job_idx)
            if job_skill_tokens is None or len(job_skill_tokens) == 0:
                cand["colbert_skill_score"] = None
                cand["_stage2_sort_key"] = cand["dense_score"]
                continue

            raw = maxsim_score(resume_skill_tokens, job_skill_tokens)
            skill_score = raw / max(len(resume_skill_tokens), 1)
            cand["colbert_skill_score"] = round(float(skill_score), 4)
            cand["_stage2_sort_key"] = skill_score

        candidates.sort(key=lambda c: c["_stage2_sort_key"], reverse=True)
        kept_n = min(int(colbert_top_n), len(candidates))
        candidates = candidates[:kept_n]

        print(f"[HybridPipeline] Stage 2 (Skill ColBERT): kept top {len(candidates)} candidates")

        # ═══════════════════════════ STAGE 3 ═══════════════════════════
        # Experience compatibility filtering — never invents missing data.
        resume_years = _parse_resume_years(entities.get("years_exp", ""))

        survivors = []
        removed_count = 0
        for cand in candidates:
            metadata = cand["metadata"]
            job_level_text = metadata.get("experience_level", "")
            compat = compute_experience_compatibility(resume_years, job_level_text, tolerance=tolerance)
            cand["experience_compatibility"] = compat["score"]
            cand["experience_compatible"] = compat["compatible"]
            cand["experience_reason"] = compat["reason"]

            if compat["compatible"]:
                survivors.append(cand)
            else:
                removed_count += 1

        if not survivors:
            print(
                "[HybridPipeline] Warning: experience filtering removed all candidates — "
                "falling back to unfiltered Stage-2 pool to avoid an empty result set"
            )
            survivors = candidates

        print(
            f"[HybridPipeline] Stage 3 (Experience Filter): removed {removed_count}, "
            f"{len(survivors)} candidates survive"
        )

        # ═══════════════════════════ STAGE 4 ═══════════════════════════
        # Cross-encoder reranking — primary final ranking signal. Pretrained only.
        condensed_resume = build_condensed_resume(preprocessed)
        job_texts = [
            self._build_job_text_for_ce(cand["metadata"], cand["job_idx"])
            for cand in survivors
        ]

        if job_texts:
            ce_scores = self.ce_reranker.score_pairs(
                condensed_resume, job_texts, batch_size=self.ce_batch_size
            )
        else:
            ce_scores = []

        final_candidates = []
        for cand, ce_score in zip(survivors, ce_scores):
            metadata = cand["metadata"]
            job_idx = cand["job_idx"]

            dense_score = cand["dense_score"]
            colbert_skill_score = cand.get("colbert_skill_score")
            experience_compatibility = cand.get("experience_compatibility")
            crossencoder_score = round(float(ce_score), 4)

            if final_score_weights:
                w = final_score_weights
                parts, weight_sum = [], 0.0
                for key, val in (
                    ("dense", dense_score),
                    ("colbert_skill", colbert_skill_score),
                    ("experience_compatibility", experience_compatibility),
                    ("crossencoder", crossencoder_score),
                ):
                    weight = w.get(key)
                    if weight and val is not None:
                        parts.append(weight * val)
                        weight_sum += weight
                final_score = sum(parts) / weight_sum if weight_sum > 0 else crossencoder_score
            else:
                # CrossEncoder is the intended primary final ranking signal.
                final_score = crossencoder_score

            desc = ""
            if 0 <= job_idx < len(self.jobs_df):
                desc = str(self.jobs_df.iloc[job_idx].get("Job Description", ""))[:500]

            final_candidates.append({
                "raw_score": final_score,
                "score": round(final_score, 4),
                "dense_score": dense_score,
                "colbert_skill_score": colbert_skill_score,
                "experience_compatibility": experience_compatibility,
                "crossencoder_score": crossencoder_score,
                "job_index": job_idx,
                "title": metadata.get("title", ""),
                "company": metadata.get("company", ""),
                "domain": metadata.get("domain", ""),
                "skills": metadata.get("skills", ""),
                "experience_level": metadata.get("experience_level", ""),
                "location": metadata.get("location", ""),
                "work_type": metadata.get("work_type", ""),
                "salary": metadata.get("salary", ""),
                "source": metadata.get("source", ""),
                "description": desc,
            })

        final_candidates.sort(key=lambda x: x["raw_score"], reverse=True)
        final = []
        for rank, r in enumerate(final_candidates[:top_k], 1):
            r_copy = dict(r)
            r_copy["rank"] = rank
            final.append(r_copy)

        print(f"[HybridPipeline] Stage 4 (CrossEncoder): final ranking of {len(final)} recommendations")
        return final

    # ── export ───────────────────────────────────────────────────────────

    def export(self, recommendations: List[Dict], output_path: Optional[str] = None) -> str:
        """Export recommendations to CSV, consistent with the other matchers."""
        if not recommendations:
            return ""

        df = pd.DataFrame(recommendations).drop(columns=["raw_score"], errors="ignore")

        if output_path is None:
            RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RECOMMENDATIONS_DIR / f"hybrid_pipeline_{ts}.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[HybridPipeline] Saved → {output_path}")
        return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — consistent with ConFit v2 / ColBERT / CrossEncoder CLIs
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Hybrid Pipeline (Dense -> Skill-ColBERT -> Experience Filter -> CrossEncoder)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--preprocessed-json", required=True,
                        help="Path to preprocessed JSON")
    parser.add_argument("--resume-index", type=int, default=0,
                        help="Index in JSON array")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--stage1-n-results", type=int, default=1000)
    parser.add_argument("--colbert-top-n", type=int, default=100)
    parser.add_argument("--experience-tolerance", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.preprocessed_json, "r") as f:
        all_preprocessed = json.load(f)

    if args.resume_index >= len(all_preprocessed):
        print(f"Error: Index {args.resume_index} out of range")
        raise SystemExit(1)

    preprocessed = all_preprocessed[args.resume_index]
    print(f"[+] Loaded: {preprocessed['filename']}")

    engine = HybridPipelineEngine(experience_tolerance=args.experience_tolerance)

    recs = engine.recommend(
        preprocessed,
        top_k=args.top_k,
        stage1_n_results=args.stage1_n_results,
        colbert_top_n=args.colbert_top_n,
    )

    print("\n" + "=" * 70)
    print(f"  HybridPipeline — TOP {min(10, len(recs))} RECOMMENDATIONS")
    print("=" * 70)
    for r in recs[:10]:
        print(f"  {r['rank']:2}. [{r['score']:.4f}] {r['title']} @ {r['company']}")
        print(
            f"      dense={r['dense_score']} colbert_skill={r['colbert_skill_score']} "
            f"exp_compat={r['experience_compatibility']} ce={r['crossencoder_score']}"
        )

    engine.export(recs, output_path=args.output)
