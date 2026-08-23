"""
Streamlit UI for resume upload, preprocessing, recommendation, career guidance,
and evaluation.

Resume preprocessing happens once per upload. The saved preprocessed bundle is
then reused by every matcher, and job details are reused from the indexed job
artifacts already stored in Chroma/CSV.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, PROCESSED_DIR, RECOMMENDATIONS_DIR, RESUMES_DIR
from evaluation.internal_quality import evaluate_internal_quality
from evaluation.ranking_only import compare_model_rankings
from evaluation.relevance import jobs_collection_count, parse_user_job_indices
from models.colbert_matcher_fixed import ColBERTEngine
from models.confit_v2_fixed import ConFitV2Engine
from models.cross_encoder_matcher_fixed import CrossEncoderEngine
from resume_processing.main_pipeline import process_one
from resume_processing.step1_parser import parse_file
from resume_processing.step5_storage import get_stats, store_resume
from skill_analysis.analysis import analyze_recommendations
from skill_analysis.llm_explainer import explain_analysis, llm_is_configured
from tailoring_resume.resume_tailoring import generate_tailored_resume
from ui.match_evaluation import aggregate_metrics_for_models
from models.hybrid_matcher import HybridMatcher

MODEL_ORDER = ("ConFit v2", "ColBERT", "CrossEncoder")
HYBRID_NAME = "Hybrid (RRF)"

@st.cache_resource
def _engine_confit(hre_mode: str, hre_alpha: float) -> ConFitV2Engine:
    return ConFitV2Engine(hre_mode=hre_mode, hre_alpha=hre_alpha)


@st.cache_resource
def _engine_colbert() -> ColBERTEngine:
    return ColBERTEngine()


@st.cache_resource
def _engine_cross(batch_size: int) -> CrossEncoderEngine:
    return CrossEncoderEngine(batch_size=batch_size)


def _init_session() -> None:
    defaults = {
        "preprocessed": None,
        "upload_sig": None,
        "process_error": None,
        "recs_by_model": {},
        "last_model": MODEL_ORDER[0],
        "tailored_resumes": {},
    }
    for key, value in defaults.items():
        if key in st.session_state:
            continue
        if key in {"recs_by_model", "tailored_resumes"}:
            st.session_state[key] = {}
        else:
            st.session_state[key] = value


def _save_full_preprocessed_json(preprocessed: dict) -> Path:
    stem = Path(preprocessed.get("filename", "resume")).stem
    out_dir = PROCESSED_DIR / "resume_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.json"
    serializable = {
        "filename": preprocessed.get("filename"),
        "file_type": preprocessed.get("file_type"),
        "raw_text": preprocessed.get("raw_text"),
        "sections": preprocessed.get("sections"),
        "entities": preprocessed.get("entities"),
        "embeddings": preprocessed.get("embeddings"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return out_path


def _append_inspection_json(preprocessed: dict) -> None:
    out_path = PROCESSED_DIR / "output_preprocessed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    filename = preprocessed.get("filename")
    existing = [entry for entry in existing if entry.get("filename") != filename]
    existing.append(
        {
            "filename": filename,
            "file_type": preprocessed.get("file_type"),
            "entities": preprocessed.get("entities", {}),
            "sections": {
                key: (value[:200] if value else "")
                for key, value in preprocessed.get("sections", {}).items()
            },
            "query_string": preprocessed.get("embeddings", {}).get("query_string", ""),
            "vector_dims": len(preprocessed.get("embeddings", {}).get("query_vector") or []),
        }
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def _process_upload(uploaded) -> None:
    st.session_state.process_error = None
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RESUMES_DIR / uploaded.name
    file_path.write_bytes(uploaded.getbuffer())

    raw_text = parse_file(str(file_path))
    parsed = {
        "filename": uploaded.name,
        "raw_text": raw_text,
        "file_type": Path(uploaded.name).suffix.lstrip(".") or "txt",
    }
    preprocessed = process_one(parsed)
    store_resume(preprocessed)
    _save_full_preprocessed_json(preprocessed)
    _append_inspection_json(preprocessed)

    st.session_state.preprocessed = preprocessed
    st.session_state.recs_by_model = {}
    st.session_state.tailored_resumes = {}


def _all_models_ready() -> bool:
    recs_by_model = st.session_state.recs_by_model
    return all(model in recs_by_model and len(recs_by_model[model]) > 0 for model in MODEL_ORDER)


def _run_one_model(
    model: str,
    preprocessed: dict,
    top_k: int,
    hre_mode: str,
    hre_alpha: float,
    chroma_pool: int,
    batch: int,
):
    pool = int(chroma_pool)
    if model == "ConFit v2":
        engine = _engine_confit(hre_mode, float(hre_alpha))
        return engine.recommend(preprocessed, top_k=top_k, stage1_n_results=pool)
    if model == "ColBERT":
        engine = _engine_colbert()
        return engine.recommend(preprocessed, top_k=top_k, stage1_n_results=pool)
    engine = _engine_cross(int(batch))
    return engine.recommend(preprocessed, top_k=top_k, stage1_n_results=pool)


def _available_models() -> list[str]:
    return [model for model in MODEL_ORDER if st.session_state.recs_by_model.get(model)]


def _default_insight_model() -> str:
    available = _available_models()
    if not available:
        return MODEL_ORDER[0]
    if st.session_state.last_model in available:
        return st.session_state.last_model
    return available[0]


def _build_insight_bundle(
    preprocessed: dict,
    model_name: str,
    top_k: int,
    *,
    llm_top_n_jobs: int = 10,
) -> tuple[list, dict, dict]:
    recs = st.session_state.recs_by_model.get(model_name) or []
    analyzed = analyze_recommendations(preprocessed, recs, top_k=min(top_k, 20))
    explained = explain_analysis(analyzed, top_n_jobs=min(llm_top_n_jobs, len(analyzed.get("jobs", []))))
    return recs, analyzed, explained


def _job_key(preprocessed: dict, rec: dict) -> str:
    job_index = rec.get("job_index")
    if job_index not in (None, "", -1):
        return f"{preprocessed.get('filename', 'resume')}::{job_index}"
    return (
        f"{preprocessed.get('filename', 'resume')}::"
        f"{rec.get('title', '')}::{rec.get('company', '')}::{rec.get('rank', '')}"
    )


def _render_learning_path_html(grounded_path: list, ai_path: list) -> str:
    if ai_path:
        html = "".join(
            f"<div><strong>{step.get('step')}</strong> "
            f"<span class='priority-chip priority-{str(step.get('priority', '')).lower()}'>{step.get('priority')}</span><br/>"
            f"<small>{step.get('reason')}</small></div>"
            for step in ai_path[:3]
        )
        if html:
            return html

    html = "".join(
        f"<div><strong>{step.get('from')}</strong> -> <strong>{step.get('to')}</strong><br/>"
        f"<small>{step.get('reason')}</small></div>"
        for step in grounded_path[:3]
    )
    return html or "<small>This job already overlaps well with your current profile.</small>"


def _render_global_learning_path_html(ai_path: list) -> str:
    blocks = []
    for index, step in enumerate(ai_path, 1):
        priority = str(step.get("priority", "")).strip().lower() or "optional"
        blocks.append(
            f"<div class='learning-card learning-card-{priority}'>"
            f"<div class='learning-card-header'>"
            f"<span class='learning-card-index'>{index}.</span>"
            f"<div class='learning-card-title'>{step.get('step', '')}</div>"
            f"<span class='priority-chip priority-{priority}'>{step.get('priority', 'Optional')}</span>"
            f"</div>"
            f"<div class='learning-card-body'>{step.get('reason', '')}</div>"
            f"</div>"
        )
    return "".join(blocks)


_init_session()

st.set_page_config(
    page_title="Resume to Jobs",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 1.2rem; max-width: 1100px; }
  div[data-testid="stMetricValue"] { font-size: 1.35rem; }
  .job-card {
    border: 1px solid rgba(49,51,63,0.12);
    border-radius: 12px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.65rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(14,165,233,0.04));
  }
  .score-pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(99,102,241,0.15);
    color: #4338ca;
  }
  .hero {
    font-size: 1.05rem;
    color: rgba(49,51,63,0.78);
    margin-bottom: 1rem;
  }
  .skill-line {
    margin-top: 0.45rem;
    line-height: 1.5;
  }
  .mini-chip {
    display: inline-block;
    margin: 0.15rem 0.35rem 0.15rem 0;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    font-size: 0.8rem;
    background: rgba(15,23,42,0.06);
  }
  .priority-chip {
    display: inline-block;
    margin-left: 0.35rem;
    padding: 0.16rem 0.48rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid transparent;
  }
  .priority-core {
    background: rgba(34, 197, 94, 0.12);
    color: #166534;
    border-color: rgba(34, 197, 94, 0.18);
  }
  .priority-next {
    background: rgba(59, 130, 246, 0.12);
    color: #1d4ed8;
    border-color: rgba(59, 130, 246, 0.18);
  }
  .priority-optional {
    background: rgba(148, 163, 184, 0.16);
    color: #475569;
    border-color: rgba(148, 163, 184, 0.2);
  }
  .learning-card {
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin: 0.7rem 0;
    border: 1px solid rgba(15, 23, 42, 0.08);
  }
  .learning-card-header {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    flex-wrap: wrap;
  }
  .learning-card-index {
    font-weight: 700;
    color: rgba(15, 23, 42, 0.7);
  }
  .learning-card-title {
    font-weight: 650;
    color: #0f172a;
    flex: 1 1 420px;
    min-width: 0;
  }
  .learning-card-body {
    margin-top: 0.65rem;
    color: rgba(51, 65, 85, 0.88);
    line-height: 1.6;
  }
  .learning-card-core {
    background: linear-gradient(180deg, rgba(34, 197, 94, 0.08), rgba(34, 197, 94, 0.03));
    border-color: rgba(34, 197, 94, 0.16);
  }
  .learning-card-next {
    background: linear-gradient(180deg, rgba(59, 130, 246, 0.08), rgba(59, 130, 246, 0.03));
    border-color: rgba(59, 130, 246, 0.16);
  }
  .learning-card-optional {
    background: linear-gradient(180deg, rgba(148, 163, 184, 0.14), rgba(148, 163, 184, 0.05));
    border-color: rgba(148, 163, 184, 0.18);
  }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Resume to job match")
st.markdown(
    '<p class="hero">Upload a resume once. The app parses it, saves the preprocessed bundle, '
    "stores resume vectors in Chroma, and then every matcher reuses that same preprocessing instead of "
    "recomputing it. Tailored resumes are generated only on demand from your saved resume content plus the "
    "selected job description.</p>",
    unsafe_allow_html=True,
)

tab_match, tab_career, tab_eval = st.tabs(["Match", "Career Direction", "Evaluate"])

with tab_match:
    uploaded = st.file_uploader("Resume (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])

    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.upload_sig != sig:
            with st.spinner("Parsing, embedding, and writing Chroma plus JSON..."):
                try:
                    _process_upload(uploaded)
                    st.session_state.upload_sig = sig
                except Exception as exc:
                    st.session_state.process_error = str(exc)
                    st.session_state.preprocessed = None

        if st.session_state.process_error:
            st.error(st.session_state.process_error)
        elif st.session_state.preprocessed:
            pre = st.session_state.preprocessed
            stats = get_stats()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("File", pre.get("filename", "-"))
            with c2:
                st.metric("Skills (NER)", len(pre.get("entities", {}).get("skills") or []))
            with c3:
                st.metric("Embedding dim", len(pre.get("embeddings", {}).get("query_vector") or []))
            with c4:
                st.metric("Resumes in Chroma (main)", stats.get("resumes", 0))

            with st.expander("Profile and sections (from preprocessing only)", expanded=False):
                st.caption(pre.get("embeddings", {}).get("query_string", "")[:1200])
                st.json(
                    {
                        key: (value[:300] + "..." if len(value) > 300 else value)
                        for key, value in (pre.get("sections") or {}).items()
                        if value
                    }
                )

            st.divider()
            st.subheader("Recommendations")

            cfg1, cfg2, cfg3 = st.columns([1, 1, 1])
            with cfg1:
                top_k = st.slider("Top K", 5, 40, 15, 5, key="match_top_k")
            with cfg2:
                chroma_pool = st.slider(
                    "Chroma pool size (shared)",
                    40,
                    200,
                    100,
                    10,
                    help="Identical Stage-1 candidate count for ConFit, ColBERT, and CrossEncoder.",
                )
            with cfg3:
                batch = st.slider("Cross-encoder batch", 8, 64, 32, 8)

            with st.expander("ConFit v2 options", expanded=False):
                hre_mode = st.selectbox("HRE mode", ["rule", "llm", "local", "groq"], index=0, key="hre_mode")
                hre_alpha = st.slider("HRE alpha (blend)", 0.0, 1.0, 0.65, 0.05, key="hre_alpha")

            try:
                model_index = list(MODEL_ORDER).index(st.session_state.last_model)
            except ValueError:
                model_index = 0

            model_pick = st.selectbox(
                "Matcher",
                list(MODEL_ORDER),
                index=model_index,
                help="Each model uses your saved query_vector for Chroma retrieval and then applies its own reranker.",
            )
            st.session_state.last_model = model_pick

            b1, b2 = st.columns(2)
            with b1:
                run_one = st.button("Run selected matcher", type="primary", use_container_width=True)
            with b2:
                run_all = st.button("Run all three matchers", use_container_width=True)

            if run_one:
                with st.spinner(f"Running {model_pick}..."):
                    try:
                        recs = _run_one_model(
                            model_pick,
                            pre,
                            top_k,
                            hre_mode,
                            hre_alpha,
                            chroma_pool,
                            batch,
                        )
                        st.session_state.recs_by_model[model_pick] = recs
                        st.success(f"{model_pick}: {len(recs)} jobs.")
                    except Exception as exc:
                        st.error(str(exc))

            if run_all:
                status = st.empty()
                bar = st.progress(0)
                for index, model_name in enumerate(MODEL_ORDER):
                    status.markdown(f"Running **{model_name}**...")
                    bar.progress((index + 1) / len(MODEL_ORDER))
                    try:
                        st.session_state.recs_by_model[model_name] = _run_one_model(
                            model_name,
                            pre,
                            top_k,
                            hre_mode,
                            hre_alpha,
                            chroma_pool,
                            batch,
                        )
                    except Exception as exc:
                        st.warning(f"{model_name} failed: {exc}")
                status.markdown("All matchers finished.")
                bar.empty()
                st.success("Finished running all matchers.")

            chips = []
            for model_name in MODEL_ORDER:
                ready = model_name in st.session_state.recs_by_model and len(st.session_state.recs_by_model[model_name]) > 0
                chips.append(f"{'ready' if ready else 'pending'}: {model_name}")
            st.caption(" | ".join(chips))

            if _all_models_ready():
                if st.button("Compute Hybrid (RRF) ranking"):
                    hybrid = HybridMatcher()
                    fused = hybrid.recommend(
                        st.session_state.recs_by_model,
                        top_k=top_k,
                        weights={"CrossEncoder": 1.3, "ColBERT": 1.0, "ConFit v2": 1.0},
                    )
                    st.session_state.recs_by_model[HYBRID_NAME] = fused
                    st.success(f"Hybrid: {len(fused)} fused recommendations.")

            preview_options = list(MODEL_ORDER)
            if HYBRID_NAME in st.session_state.recs_by_model:
                preview_options.append(HYBRID_NAME)
            view = st.radio("Preview list", preview_options, horizontal=True, key="view_model")
            recs = st.session_state.recs_by_model.get(view) or []
            if not recs:
                st.info(f"No results for **{view}** yet. Run that matcher above.")
            else:
                _, analyzed, explained = _build_insight_bundle(
                    pre,
                    view,
                    top_k,
                    llm_top_n_jobs=3,
                )
                explanation_by_rank = {
                    item.get("rank"): item
                    for item in explained.get("jobs", [])
                }

                st.markdown("#### Section 1: Job Recommendations")
                st.caption(
                    "The tailored resume button uses your already uploaded resume plus the selected job description. "
                    "It does not rerun resume preprocessing."
                )
                if explained.get("used_llm"):
                    st.caption(
                        f"AI explanations are generated with {explained.get('model')} for the top "
                        f"{min(3, len(analyzed.get('jobs', [])))} jobs. Match signals stay grounded in preprocessing, "
                        "while Gemini can reorder the learning path and mark skills as optional."
                    )
                else:
                    if llm_is_configured():
                        st.caption("AI explanations fell back to the grounded local summary for this run.")
                    else:
                        st.caption(
                            "AI explanations are currently in grounded fallback mode. Set `GEMINI_API_KEY` to enable "
                            "Gemini rewriting for the top 3 jobs."
                        )

                for job in analyzed.get("jobs", []):
                    title = job.get("title") or "-"
                    company = job.get("company") or ""
                    score = job.get("score")
                    score_txt = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
                    matching = job.get("matching_skills") or []
                    missing = job.get("missing_skills") or []
                    grounded_learning_path = job.get("learning_path") or []
                    ai_job = explanation_by_rank.get(job.get("rank")) or {}
                    ai_learning_path = ai_job.get("learning_path") or []

                    matching_html = "".join(
                        f'<span class="mini-chip">{skill}</span>' for skill in matching[:6]
                    ) or '<span class="mini-chip">Profile overlap not captured</span>'
                    missing_html = "".join(
                        f'<span class="mini-chip">{skill}</span>' for skill in missing[:6]
                    ) or '<span class="mini-chip">No major missing skills surfaced</span>'
                    path_html = _render_learning_path_html(grounded_learning_path, ai_learning_path)

                    card_col, action_col = st.columns([0.78, 0.22])
                    with card_col:
                        st.markdown(
                            f'<div class="job-card"><span class="score-pill">Match {score_txt}</span> '
                            f"<strong>{title}</strong> · <em>{company}</em><br/><small>"
                            f"{(job.get('domain') or '')} · {(job.get('experience_level') or '')} · "
                            f"{(job.get('location') or '')}</small>"
                            f'<div class="skill-line"><strong>Good Match Signals:</strong><br/>{matching_html}</div>'
                            f'<div class="skill-line"><strong>Missing Skills:</strong><br/>{missing_html}</div>'
                            f'<div class="skill-line"><strong>Learning Path:</strong><br/>{path_html}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        explanation = ai_job.get("explanation")
                        if explanation:
                            st.caption(f"AI explanation: {explanation}")

                    with action_col:
                        tailor_key = _job_key(pre, job)
                        artifact = st.session_state.tailored_resumes.get(tailor_key)
                        if st.button(
                            "Tailor resume",
                            key=f"tailor_btn_{view}_{tailor_key}",
                            use_container_width=True,
                        ):
                            with st.spinner("Generating tailored resume..."):
                                try:
                                    artifact = generate_tailored_resume(pre, job)
                                    st.session_state.tailored_resumes[tailor_key] = artifact
                                except Exception as exc:
                                    st.error(str(exc))
                                    artifact = None

                        st.caption("No fake skills. No fake projects. No fake experience.")
                        if artifact and artifact.get("docx_path"):
                            docx_path = Path(artifact["docx_path"])
                            if docx_path.exists():
                                st.download_button(
                                    "Download .docx",
                                    data=docx_path.read_bytes(),
                                    file_name=docx_path.name,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"tailor_download_{view}_{tailor_key}",
                                    use_container_width=True,
                                )
                            if artifact.get("tailoring_summary"):
                                st.caption(artifact["tailoring_summary"])
                            ats_focus = artifact.get("ats_focus") or []
                            if ats_focus:
                                st.caption("ATS focus: " + ", ".join(ats_focus[:4]))

                df = pd.DataFrame(
                    [
                        {
                            "Rank": rec.get("rank"),
                            "Match": rec.get("score"),
                            "job_index": rec.get("job_index"),
                            "Title": rec.get("title"),
                            "Company": rec.get("company"),
                            "Level": rec.get("experience_level"),
                        }
                        for rec in recs
                    ]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

                csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Download CSV (this matcher)",
                    data=csv_bytes,
                    file_name=f"{view.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                )

    else:
        st.info("Choose a resume file to begin. Processing starts automatically after upload.")

with tab_career:
    st.subheader("Career Direction")
    if not st.session_state.preprocessed:
        st.warning("Upload and preprocess a resume on the **Match** tab first.")
    elif not _available_models():
        st.info("Run at least one matcher on the **Match** tab to unlock career guidance.")
    else:
        pre = st.session_state.preprocessed
        available_models = _available_models()
        source_model = st.selectbox(
            "Insight source matcher",
            available_models,
            index=available_models.index(_default_insight_model()),
            key="career_view_model",
            help="Career direction is grounded in the recommendation list from this matcher.",
        )
        top_k = st.session_state.get("match_top_k", 15)
        _, analyzed, explained = _build_insight_bundle(pre, source_model, top_k)
        global_view = analyzed.get("global", {})
        global_text = explained.get("global", {})
        global_learning_path = global_text.get("learning_path") or []

        if explained.get("used_llm"):
            st.caption(
                f"Using Gemini {explained.get('model')} to turn grounded match signals into a strategic learning path. "
                "It can reorder steps, mark items optional, and suggest adjacent next skills tied to what you already know."
            )
        else:
            if llm_is_configured() and explained.get("rate_limited"):
                retry_after = explained.get("retry_after_seconds")
                retry_text = f" Try again in about {retry_after} seconds." if retry_after else ""
                st.info(
                    "Gemini hit its current quota or rate limit, so this view is using the grounded fallback summary."
                    f"{retry_text}"
                )
            elif llm_is_configured():
                st.info("Gemini is configured, but this view is currently using the grounded fallback summary.")
            else:
                st.info(
                    "Gemini guidance is not active yet. Set `GEMINI_API_KEY` to enable smarter career summaries; "
                    "until then this tab shows the grounded fallback summary."
                )
            if explained.get("error") and not explained.get("rate_limited"):
                st.caption(f"LLM fallback reason: {explained.get('error')}")

        st.markdown(f"### {global_text.get('headline', 'Career Direction')}")
        st.write(global_text.get("summary", global_view.get("summary", "")))
        if global_text.get("priority_path"):
            st.markdown(f"**Recommended Path:** {global_text.get('priority_path')}")

        if global_learning_path:
            st.markdown("**Strategic Learning Path**")
            st.markdown(_render_global_learning_path_html(global_learning_path), unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            closest_roles = global_view.get("closest_roles") or ["Recommended roles"]
            st.markdown(f"**You are closest to:** {' / '.join(closest_roles)}")

        with c2:
            top_missing = global_view.get("top_missing_skills") or []
            if top_missing:
                st.markdown("**Top Missing Skills Across Recommended Jobs**")
                for index, item in enumerate(top_missing[:5], 1):
                    st.write(f"{index}. {item['skill']} ({item['count']}/{len(analyzed.get('jobs', []))} jobs)")
            else:
                st.success("Your current profile already covers most surfaced skills in these recommendations.")

        with st.expander("Grounded Inputs Used For The Explanation", expanded=False):
            st.markdown("**Current skills from resume preprocessing**")
            st.write(", ".join(analyzed.get("resume_skills", [])[:20]) or "No extracted skills available")
            st.markdown("**Computed path before Gemini revision**")
            raw_path = global_view.get("recommended_path") or []
            st.write(" -> ".join(raw_path[:4]) if raw_path else "No path needed")
            st.markdown("**Why this stays grounded**")
            st.write(
                "Gemini receives current skills, matching skills, missing skills, closest-role signals, "
                "top missing-skill counts, and the grounded path draft. It can revise prioritization, "
                "but the resume-side match signals still come from preprocessing."
            )

with tab_eval:
    st.subheader("Compare matchers")
    if not st.session_state.preprocessed:
        st.warning("Upload and finish preprocessing on the **Match** tab first.")
    elif not _all_models_ready():
        st.info("Run **all three** matchers on the **Match** tab to unlock evaluation.")
    else:
        pre = st.session_state.preprocessed
        st.markdown("#### Evaluation settings")
        st.caption(f"Jobs in DB (`jobs` collection): **{jobs_collection_count()}**")

        gt_text = st.text_area(
            "Optional ground truth: `job_index` values (comma or space separated)",
            height=68,
            placeholder="e.g. 12, 45, 102 - when non-empty, the tab unlocks proper ranking metrics",
            key="eval_gt_job_indices",
        )
        user_rel = parse_user_job_indices(gt_text)

        agg = aggregate_metrics_for_models(
            pre,
            st.session_state.recs_by_model,
            user_relevant_indices=user_rel if user_rel else None,
        )
        rank_cmp = compare_model_rankings(st.session_state.recs_by_model, top_k=10, rbo_depth=25)
        rank_pw = rank_cmp.get("pairwise") or []

        st.caption(agg.get("evaluation_note", ""))

        diagnostic = agg.get("diagnostic_rows") or []
        if diagnostic:
            st.markdown("#### Diagnostic metrics")
            diagnostic_df = pd.DataFrame(diagnostic)
            st.dataframe(diagnostic_df, use_container_width=True, hide_index=True)
            if "semantic_alignment" in diagnostic_df.columns:
                chart_df = diagnostic_df.set_index("model")[["semantic_alignment"]].dropna()
                if not chart_df.empty:
                    st.bar_chart(chart_df)

        label_rows = agg.get("label_rows") or []
        if label_rows:
            st.markdown("#### Label-based ranking metrics")
            st.caption("These are the primary ranking metrics for this project because they use your explicit relevant `job_index` labels.")
            label_df = pd.DataFrame(label_rows)
            st.dataframe(label_df, use_container_width=True, hide_index=True)
            chart_df = label_df.set_index("model")[["ndcg@10", "recall@10"]].dropna(how="all")
            if not chart_df.empty:
                st.bar_chart(chart_df)

        iq = evaluate_internal_quality(st.session_state.recs_by_model)
        internal_rows = iq.get("internal_quality") or []
        if internal_rows:
            st.markdown("#### Score separation")
            st.caption("Higher `top5_vs_next5_gap` means the model separates its best recommendations more clearly from the next tier.")
            internal_df = pd.DataFrame(internal_rows)
            st.dataframe(internal_df, use_container_width=True, hide_index=True)
            chart_df = internal_df.set_index("model")[["top5_vs_next5_gap"]].dropna()
            if not chart_df.empty:
                st.bar_chart(chart_df)

        if rank_pw:
            st.markdown("#### Cross-model agreement")
            st.caption("`rbo` captures top-weighted ranking similarity, while `jaccard@10` and `overlap@10` show how much the top-10 job sets agree.")
            st.dataframe(pd.DataFrame(rank_pw), use_container_width=True, hide_index=True)

        with st.expander("Reference preview (first 500 chars)", expanded=False):
            st.text(agg.get("reference_preview", ""))

        if st.button("Save evaluation snapshot to disk"):
            RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
            snap = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "resume": pre.get("filename"),
                "relevance_mode": agg.get("relevance_mode"),
                "relevance_size": agg.get("relevance_size"),
                "corpus_job_count": jobs_collection_count(),
                "diagnostic_metrics": diagnostic,
                "label_metrics": label_rows,
                "score_separation": internal_rows,
                "ranking_agreement": rank_pw,
            }
            output_path = RECOMMENDATIONS_DIR / f"ui_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
            output_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
            st.success(str(output_path))

st.markdown("---")
st.caption(f"Data: `{DATA_DIR}` | Jobs DB: `data/jobs_db` | Resume Chroma: `{DATA_DIR / 'chroma_db'}`")
