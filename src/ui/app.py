"""
Streamlit UI — resume upload, automatic pipeline + Chroma + JSON,
then recommendations from three matchers (Chroma recall + preprocessed features).
Evaluation (semantic, ranking @K, agreement; optional BLEU/ROUGE) unlocks after all three matchers run.
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
from evaluation.relevance import jobs_collection_count, parse_user_job_indices
from resume_processing.main_pipeline import process_one
from resume_processing.step1_parser import parse_file
from resume_processing.step5_storage import get_stats, store_resume
from models.confit_v2_fixed import ConFitV2Engine
from models.colbert_matcher_fixed import ColBERTEngine
from models.cross_encoder_matcher_fixed import CrossEncoderEngine

from ui.match_evaluation import aggregate_metrics_for_models

MODEL_ORDER = ("ConFit v2", "ColBERT", "CrossEncoder")


# ── Cached engines (heavy model loads) ──────────────────────────────────────


@st.cache_resource
def _engine_confit(hre_mode: str, hre_alpha: float) -> ConFitV2Engine:
    return ConFitV2Engine(hre_mode=hre_mode, hre_alpha=hre_alpha)


@st.cache_resource
def _engine_colbert() -> ColBERTEngine:
    return ColBERTEngine()


@st.cache_resource
def _engine_cross(batch_size: int) -> CrossEncoderEngine:
    return CrossEncoderEngine(batch_size=batch_size)


def _init_session():
    defaults = {
        "preprocessed": None,
        "upload_sig": None,
        "process_error": None,
        "recs_by_model": {},
        "last_model": MODEL_ORDER[0],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v if k != "recs_by_model" else {}


def _save_full_preprocessed_json(preprocessed: dict) -> Path:
    """Persist full pipeline output (vectors + text) for the current resume."""
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


def _append_inspection_json(preprocessed: dict):
    """Lightweight row for batch inspection (same file as pipeline CLI)."""
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

    fn = preprocessed.get("filename")
    existing = [e for e in existing if e.get("filename") != fn]
    entry = {
        "filename": fn,
        "file_type": preprocessed.get("file_type"),
        "entities": preprocessed.get("entities", {}),
        "sections": {k: (v[:200] if v else "") for k, v in preprocessed.get("sections", {}).items()},
        "query_string": preprocessed.get("embeddings", {}).get("query_string", ""),
        "vector_dims": len(preprocessed.get("embeddings", {}).get("query_vector") or []),
    }
    existing.append(entry)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def _process_upload(uploaded) -> None:
    """Parse → pipeline → Chroma resume store → JSON."""
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


def _all_models_ready() -> bool:
    rb = st.session_state.recs_by_model
    return all(m in rb and len(rb[m]) > 0 for m in MODEL_ORDER)


def _run_one_model(
    model: str,
    preprocessed: dict,
    top_k: int,
    hre_mode: str,
    hre_alpha: float,
    chroma_pool: int,
    batch: int,
):
    """Same Chroma pool size for every matcher so Stage-1 candidates are aligned."""
    pool = int(chroma_pool)
    if model == "ConFit v2":
        eng = _engine_confit(hre_mode, float(hre_alpha))
        return eng.recommend(preprocessed, top_k=top_k, stage1_n_results=pool)
    if model == "ColBERT":
        eng = _engine_colbert()
        return eng.recommend(preprocessed, top_k=top_k, stage1_n_results=pool)
    eng = _engine_cross(int(batch))
    return eng.recommend(preprocessed, top_k=top_k, stage1_n_results=pool)


# ── Page ──────────────────────────────────────────────────────────────────────

_init_session()

st.set_page_config(
    page_title="Resume → Jobs",
    page_icon="✨",
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
    padding: 0.85rem 1rem;
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
</style>
""",
    unsafe_allow_html=True,
)

st.title("Resume → job match")
st.markdown(
    '<p class="hero">Upload a resume once. It runs the full pipeline, saves vectors to Chroma '
    "and a JSON snapshot, then each matcher reuses that preprocessing and queries the <strong>jobs</strong> "
    "collection for retrieval + re-ranking.</p>",
    unsafe_allow_html=True,
)

tab_match, tab_eval = st.tabs(["Match", "Evaluate"])

with tab_match:
    up = st.file_uploader("Resume (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])

    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.upload_sig != sig:
            with st.spinner("Parsing, embedding, writing Chroma + JSON…"):
                try:
                    _process_upload(up)
                    st.session_state.upload_sig = sig
                except Exception as e:
                    st.session_state.process_error = str(e)
                    st.session_state.preprocessed = None

        if st.session_state.process_error:
            st.error(st.session_state.process_error)
        elif st.session_state.preprocessed:
            pre = st.session_state.preprocessed
            stats = get_stats()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("File", pre.get("filename", "—"))
            with c2:
                st.metric("Skills (NER)", len(pre.get("entities", {}).get("skills") or []))
            with c3:
                st.metric("Embedding dim", len(pre.get("embeddings", {}).get("query_vector") or []))
            with c4:
                st.metric("Resumes in Chroma (main)", stats.get("resumes", 0))

            with st.expander("Profile & sections (from preprocessing only)", expanded=False):
                st.caption(pre.get("embeddings", {}).get("query_string", "")[:1200])
                st.json(
                    {
                        k: (v[:300] + "…" if len(v) > 300 else v)
                        for k, v in (pre.get("sections") or {}).items()
                        if v
                    }
                )

            st.divider()
            st.subheader("Recommendations")

            cfg1, cfg2, cfg3 = st.columns([1, 1, 1])
            with cfg1:
                top_k = st.slider("Top K", 5, 40, 15, 5)
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
                hre_mode = st.selectbox("HRE mode", ["rule", "llm", "local"], index=0, key="hre_mode")
                hre_alpha = st.slider("HRE α (blend)", 0.0, 1.0, 0.65, 0.05, key="hre_alpha")

            try:
                _midx = list(MODEL_ORDER).index(st.session_state.last_model)
            except ValueError:
                _midx = 0
            model_pick = st.selectbox(
                "Matcher",
                list(MODEL_ORDER),
                index=_midx,
                help="Each model uses your saved query_vector for Chroma retrieval, then its own re-ranker.",
            )
            st.session_state.last_model = model_pick

            b1, b2 = st.columns(2)
            with b1:
                run_one = st.button("Run selected matcher", type="primary", use_container_width=True)
            with b2:
                run_all = st.button("Run all three matchers", use_container_width=True)

            if run_one:
                with st.spinner(f"Running {model_pick}…"):
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
                    except Exception as e:
                        st.error(str(e))

            if run_all:
                status = st.empty()
                bar = st.progress(0)
                for i, m in enumerate(MODEL_ORDER):
                    status.markdown(f"Running **{m}**…")
                    bar.progress((i + 1) / len(MODEL_ORDER))
                    try:
                        st.session_state.recs_by_model[m] = _run_one_model(
                            m, pre, top_k, hre_mode, hre_alpha, chroma_pool, batch
                        )
                    except Exception as e:
                        st.warning(f"{m} failed: {e}")
                status.markdown("All matchers finished.")
                bar.empty()
                st.success("Finished running all matchers.")

            # Status chips
            chips = []
            for m in MODEL_ORDER:
                ok = m in st.session_state.recs_by_model and len(st.session_state.recs_by_model[m]) > 0
                chips.append(f"{'✅' if ok else '⏳'} {m}")
            st.caption(" · ".join(chips))

            view = st.radio("Preview list", MODEL_ORDER, horizontal=True, key="view_model")
            recs = st.session_state.recs_by_model.get(view) or []
            if not recs:
                st.info(f"No results for **{view}** yet — run that matcher above.")
            else:
                for r in recs[: min(top_k, 20)]:
                    title = r.get("title") or "—"
                    company = r.get("company") or ""
                    score = r.get("score")
                    score_txt = f"{float(score):.4f}" if isinstance(score, (int, float)) else "—"
                    st.markdown(
                        f'<div class="job-card"><span class="score-pill">Match {score_txt}</span> '
                        f"<strong>{title}</strong> · <em>{company}</em><br/><small>"
                        f"{(r.get('domain') or '')} · {(r.get('experience_level') or '')} · "
                        f"{(r.get('location') or '')}</small></div>",
                        unsafe_allow_html=True,
                    )

                df = pd.DataFrame(
                    [
                        {
                            "Rank": r.get("rank"),
                            "Match": r.get("score"),
                            "job_index": r.get("job_index"),
                            "Title": r.get("title"),
                            "Company": r.get("company"),
                            "Level": r.get("experience_level"),
                        }
                        for r in recs
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
        st.info("Choose a resume file to begin — processing starts automatically after upload.")

with tab_eval:
    st.subheader("Compare matchers")
    if not st.session_state.preprocessed:
        st.warning("Upload and finish preprocessing on the **Match** tab first.")
    elif not _all_models_ready():
        st.info("Run **all three** matchers on the **Match** tab to unlock evaluation.")
    else:
        pre = st.session_state.preprocessed
        st.markdown("#### Evaluation settings")
        c_a, c_b = st.columns(2)
        with c_a:
            chroma_rel_n = st.slider(
                "Chroma relevance depth (N)",
                30,
                200,
                70,
                5,
                help="If you do not supply labels: relevance = top-N job_index from Chroma for this resume only (no reranker, no model leakage).",
            )
        with c_b:
            st.caption(f"Jobs in DB (`jobs` collection): **{jobs_collection_count()}**")

        gt_text = st.text_area(
            "Optional ground truth: `job_index` values (comma or space separated)",
            height=68,
            placeholder="e.g. 12, 45, 102  — when non-empty, all ranking metrics use only this set",
            key="eval_gt_job_indices",
        )
        user_rel = parse_user_job_indices(gt_text)

        agg = aggregate_metrics_for_models(
            pre,
            st.session_state.recs_by_model,
            chroma_rel_n=chroma_rel_n,
            user_relevant_indices=user_rel if user_rel else None,
        )
        st.markdown("#### Semantic & ranking metrics")
        st.caption(agg.get("pseudo_relevance_note", ""))

        primary = agg.get("primary_rows") or []
        st.dataframe(pd.DataFrame(primary), use_container_width=True, hide_index=True)
        st.caption(
            "**cosine_embed_sim** / **bertscore_f1**: similarity of resume text to each model’s job bundle. "
            "**Precision@K / Recall@K / MRR / nDCG**: vs the relevance set above. "
            "**success@K**: 1 if any relevant job appears in top-K. **corpus_recall@10**: hits in top-10 ÷ total jobs in DB."
        )

        pw = agg.get("pairwise") or []
        if pw:
            st.markdown("#### Matcher agreement (top 10)")
            st.caption(
                "Prefer **jaccard_job_id_top10** (stable Chroma `job_index`). "
                "Title-only Jaccard is noisy when titles differ slightly."
            )
            st.dataframe(pd.DataFrame(pw), use_container_width=True, hide_index=True)

        with st.expander("Lexical metrics (BLEU / ROUGE) — secondary", expanded=False):
            st.caption("Designed for translation/summarization; kept for curiosity only.")
            lex = agg.get("lexical_rows") or []
            st.dataframe(pd.DataFrame(lex), use_container_width=True, hide_index=True)

        with st.expander("Reference preview (first 500 chars)", expanded=False):
            st.text(agg.get("reference_preview", ""))

        if st.button("Save evaluation snapshot to disk"):
            RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
            snap = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "resume": pre.get("filename"),
                "relevance_mode": agg.get("relevance_mode"),
                "relevance_size": agg.get("relevance_size"),
                "chroma_pool_size": agg.get("chroma_pool_size"),
                "corpus_job_count": agg.get("corpus_job_count"),
                "primary_metrics": primary,
                "lexical_metrics": agg.get("lexical_rows"),
                "pairwise": pw,
            }
            p = RECOMMENDATIONS_DIR / f"ui_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
            p.write_text(json.dumps(snap, indent=2), encoding="utf-8")
            st.success(str(p))

st.markdown("---")
st.caption(f"Data: `{DATA_DIR}` · Jobs DB: `data/jobs_db` · Resume Chroma: `{DATA_DIR / 'chroma_db'}`")
