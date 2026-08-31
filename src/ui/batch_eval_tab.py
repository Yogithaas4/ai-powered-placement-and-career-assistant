"""
batch_eval_tab.py
------------------
Streamlit tab for large-scale dataset evaluation. This is a self-contained
block to paste into src/ui/app.py -- see integration instructions in the
accompanying message for exactly where.

Reuses evaluation.batch_evaluation.run_batch_evaluation() unchanged -- no
duplicate evaluation logic here, just a UI wrapper with progress display and
results viewing.
"""

# ── Add these imports near the top of app.py, alongside your other imports ──
#
# from evaluation.batch_evaluation import run_batch_evaluation, EVAL_RESULTS_DIR
# import pandas as pd  (already imported in app.py)


def _list_past_runs():
    """Return sorted list of (folder_name, path) for existing batch runs, newest first."""
    if not EVAL_RESULTS_DIR.exists():
        return []
    runs = [p for p in EVAL_RESULTS_DIR.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [(p.name, p) for p in runs]


def _load_run_results(run_dir):
    """Load aggregate CSVs + run_summary.json for a completed run, if present."""
    result = {"summary": None, "model_agg": None, "pair_agg": None}
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            result["summary"] = json.load(f)
    model_agg_path = run_dir / "aggregate_model_metrics.csv"
    if model_agg_path.exists():
        result["model_agg"] = pd.read_csv(model_agg_path, encoding="utf-8-sig")
    pair_agg_path = run_dir / "aggregate_pair_metrics.csv"
    if pair_agg_path.exists():
        result["pair_agg"] = pd.read_csv(pair_agg_path, encoding="utf-8-sig")
    return result


def render_batch_evaluation_tab():
    """
    Call this inside `with tab_batch:` in app.py's main tab layout.
    """
    st.subheader("Dataset-level evaluation")
    st.caption(
        "Runs the exact same diagnostic/ranking metrics used in the single-resume "
        "Evaluate tab (semantic alignment, score separation, cross-model agreement) "
        "across an entire folder of resumes automatically -- no manual uploads. "
        "No ground-truth labels are used at this stage; these are diagnostic metrics."
    )

    st.markdown("#### Start a new batch run")

    col1, col2 = st.columns([2, 1])
    with col1:
        resume_root = st.text_input(
            "Resume dataset root folder",
            placeholder=r"C:\Users\Student1\Downloads\archive\data\data",
            key="batch_resume_root",
        )
    with col2:
        hre_mode = st.selectbox(
            "ConFit v2 HRE mode", ["rule", "groq", "local", "llm"], index=0,
            key="batch_hre_mode",
            help="'rule' recommended for large batch runs: free, deterministic, no rate limits.",
        )

    mode = st.radio(
        "Resume selection",
        ["All resumes (random sample)", "Full categories + small sample from others"],
        horizontal=True,
        key="batch_mode",
    )

    full_categories = None
    other_per_category = 3
    limit = None

    if mode == "Full categories + small sample from others":
        full_categories_raw = st.text_input(
            "Categories to include in full (comma-separated, must match folder names)",
            placeholder="ENGINEERING,INFORMATION-TECHNOLOGY",
            key="batch_full_categories",
        )
        full_categories = [c.strip() for c in full_categories_raw.split(",") if c.strip()] or None
        other_per_category = st.number_input(
            "Resumes to sample from every other category", min_value=1, max_value=50,
            value=3, key="batch_other_per_category",
        )
    else:
        limit = st.number_input(
            "Number of resumes to sample (0 = entire dataset)", min_value=0,
            value=50, key="batch_limit",
        )
        limit = None if limit == 0 else int(limit)

    adv = st.expander("Advanced settings", expanded=False)
    with adv:
        top_k = st.slider("Top K per matcher", 5, 30, 15, key="batch_top_k")
        chroma_pool = st.slider("Chroma pool size (shared)", 20, 150, 60, key="batch_chroma_pool")

    run_clicked = st.button("Run batch evaluation", type="primary", disabled=not resume_root)

    if run_clicked:
        if full_categories:
            st.info(f"Selection: ALL resumes from {full_categories}, "
                    f"{other_per_category} sampled from every other category.")
        elif limit:
            st.info(f"Selection: random sample of {limit} resumes from the whole dataset.")
        else:
            st.warning(
                "This will process the ENTIRE dataset. Based on prior runs, budget "
                "roughly 8-13 seconds per resume once models are warmed up, plus a "
                "one-time ~5-10 minute model-loading cost for the first resume."
            )

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        log_box = st.expander("Live log (last 10 resumes)", expanded=True)
        log_lines = []

        final_progress = None
        try:
            for progress in run_batch_evaluation(
                resume_root=resume_root,
                top_k=top_k,
                chroma_pool=chroma_pool,
                hre_mode=hre_mode,
                full_categories=full_categories,
                other_per_category=other_per_category,
                limit=limit,
            ):
                if progress.get("done"):
                    final_progress = progress
                    break

                frac = progress["index"] / progress["total"]
                progress_bar.progress(min(frac, 1.0))
                status_text.markdown(
                    f"**{progress['index']}/{progress['total']}** -- "
                    f"{progress['resume_id']} ({progress['status']})"
                )
                log_lines.append(f"[{progress['status'].upper()}] {progress['resume_id']}")
                log_lines = log_lines[-10:]
                with log_box:
                    st.text("\n".join(log_lines))

        except Exception as exc:
            st.error(f"Batch run failed: {exc}")
            final_progress = None

        if final_progress:
            summary = final_progress["summary"]
            st.success(
                f"Done: {summary['succeeded']} succeeded, {summary['failed']} failed, "
                f"{summary['skipped_already_done']} skipped -- "
                f"{summary['elapsed_seconds'] / 60:.1f} minutes total."
            )
            st.session_state["batch_last_run_dir"] = summary["output_dir"]

    st.markdown("---")
    st.markdown("#### View a completed run")

    past_runs = _list_past_runs()
    if not past_runs:
        st.info("No batch evaluation runs found yet in data/evaluation_results/.")
        return

    default_run = st.session_state.get("batch_last_run_dir")
    run_names = [name for name, _ in past_runs]
    default_index = 0
    if default_run:
        default_name = Path(default_run).name
        if default_name in run_names:
            default_index = run_names.index(default_name)

    selected_name = st.selectbox("Select a run", run_names, index=default_index, key="batch_view_run")
    selected_dir = dict(past_runs)[selected_name]

    results = _load_run_results(selected_dir)

    if results["summary"]:
        s = results["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resumes evaluated", s.get("total_selected", s.get("total_found", "-")))
        c2.metric("Succeeded", s.get("succeeded", "-"))
        c3.metric("Failed", s.get("failed", "-"))
        c4.metric("Duration", f"{s.get('elapsed_seconds', 0) / 60:.1f} min")

        cfg = s.get("config", {})
        st.caption(
            f"HRE mode: `{cfg.get('hre_mode')}` | Top K: {cfg.get('top_k')} | "
            f"Chroma pool: {cfg.get('chroma_pool')}"
            + (f" | Full categories: {cfg.get('full_categories')}" if cfg.get("full_categories") else "")
        )

    if results["model_agg"] is not None and not results["model_agg"].empty:
        st.markdown("##### Per-model aggregate metrics (mean, median, std, 95% CI)")
        st.dataframe(results["model_agg"], width="stretch", hide_index=True)

        gap_df = results["model_agg"][results["model_agg"]["metric"] == "top5_vs_next5_gap"]
        if not gap_df.empty:
            st.markdown("**Score separation by model** (higher = more decisive ranking)")
            chart_df = gap_df.set_index("model")[["mean"]].rename(columns={"mean": "top5_vs_next5_gap"})
            st.bar_chart(chart_df)

        align_df = results["model_agg"][results["model_agg"]["metric"] == "semantic_alignment"]
        if not align_df.empty:
            st.markdown("**Semantic alignment by model**")
            chart_df = align_df.set_index("model")[["mean"]].rename(columns={"mean": "semantic_alignment"})
            st.bar_chart(chart_df)

    if results["pair_agg"] is not None and not results["pair_agg"].empty:
        st.markdown("##### Cross-model agreement (pairwise, aggregated)")
        st.caption("rbo captures top-weighted ranking similarity; jaccard_at_10 / overlap_at_10 show top-10 set agreement.")
        st.dataframe(results["pair_agg"], width="stretch", hide_index=True)

    st.caption(f"Full results on disk: `{selected_dir}`")

with tab_batch:
    render_batch_evaluation_tab()