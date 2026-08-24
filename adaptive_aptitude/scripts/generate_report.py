"""
scripts/generate_report.py
------------------------------
Pulls everything scripts/run_experiments.py logged into `experiment_runs` /
`experiment_metrics` and turns it into paper-ready output: CSV, Markdown
comparison tables, a cross-simulator ranking-agreement table, and PNG bar
charts. This is the missing last step -- until now your only record of a
run was terminal scrollback.

Only the LATEST run per (algorithm, practice_category, simulator) is used,
so re-running scripts/run_experiments.py to fix something doesn't leave
stale duplicate rows polluting the report.

Output, all under reports/phase1/:
  comparison_raw.csv                       -- every (category, simulator, algorithm, metric) row
  comparison_<category>_<simulator>.md      -- one Markdown table per (category, simulator)
  cross_simulator_ranking_<category>.md     -- ranking-agreement table per category (only if
                                                 both bkt_generative and irt_generative are present)
  subject_breakdown_raw.csv                -- per-canonical_subject rollups (appendix material)
  figures/roc_auc_<category>.png            -- bar chart, algorithms x simulators
  figures/calibration_error_<category>.png  -- bar chart, algorithms x simulators

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --practice-category "Core CS (Systems & Theory)"
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless -- this runs from a terminal, not a notebook
import matplotlib.pyplot as plt

from data.db_loader import get_engine  # noqa: E402

TOP_LEVEL_METRICS = [
    "roc_auc", "pr_auc", "brier_score", "log_loss", "calibration_error",
    "questions_to_mastery", "coverage", "diversity", "repeat_rate", "difficulty_match_rate",
]
CHART_METRICS = ["roc_auc", "calibration_error"]  # the two most decision-relevant, per earlier discussion
STEP3_SIMULATORS = ["bkt_generative", "irt_generative"]

OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "phase1")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")


def fetch_raw(engine) -> pd.DataFrame:
    """One row per (run, metric). Extracts practice_category/simulator out
    of experiment_runs.hyperparameters, which run_experiment() always
    populates (see experiments/runner.py)."""
    query = """
        SELECT
            r.run_id,
            r.algorithm,
            r.started_at,
            r.hyperparameters->>'practice_category' AS practice_category,
            r.hyperparameters->>'simulator'          AS simulator,
            m.metric_name,
            m.metric_value,
            m.context
        FROM experiment_runs r
        JOIN experiment_metrics m ON m.run_id = r.run_id
        WHERE r.finished_at IS NOT NULL
        ORDER BY r.started_at DESC
    """
    df = pd.read_sql_query(query, engine)
    if df.empty:
        raise RuntimeError(
            "No finished experiment runs found in experiment_runs/experiment_metrics -- "
            "run scripts/run_experiments.py (without --no-db) first."
        )
    return df


def latest_only(df: pd.DataFrame, subset_cols: list) -> pd.DataFrame:
    """Keep only the most recent run per subset_cols group -- df must
    already be sorted by started_at DESC (fetch_raw does this)."""
    return df.drop_duplicates(subset=subset_cols, keep="first")


def split_top_level_and_subject(df: pd.DataFrame):
    top_level = df[df["metric_name"].isin(TOP_LEVEL_METRICS)].copy()
    subject_level = df[df["metric_name"].str.startswith("subject_")].copy()
    if not subject_level.empty:
        subject_level["canonical_subject"] = subject_level["context"].apply(
            lambda c: (c or {}).get("canonical_subject")
        )
        subject_level["metric_name"] = subject_level["metric_name"].str.replace("subject_", "", regex=False)
    return top_level, subject_level


def write_comparison_tables(top_level: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for (category, simulator), group in top_level.groupby(["practice_category", "simulator"]):
        pivot = group.pivot_table(index="algorithm", columns="metric_name", values="metric_value")
        pivot = pivot.reindex(columns=[m for m in TOP_LEVEL_METRICS if m in pivot.columns])

        safe_cat = category.replace("/", "-").replace(" ", "_")
        path = os.path.join(OUTPUT_DIR, f"comparison_{safe_cat}_{simulator}.md")
        with open(path, "w") as f:
            f.write(f"# {category} -- {simulator}\n\n")
            f.write(pivot.round(4).to_markdown())
            f.write("\n")
        print(f"wrote {path}")


def write_cross_simulator_ranking(top_level: pd.DataFrame):
    """Same ranking-agreement logic as run_experiments.py's
    print_cross_simulator_table, but reading from Postgres so it works
    even for runs from a previous session, not just the one just executed."""
    for category, cat_group in top_level.groupby("practice_category"):
        sims_present = [s for s in STEP3_SIMULATORS if s in cat_group["simulator"].unique()]
        if len(sims_present) < 2:
            continue

        roc = cat_group[cat_group["metric_name"] == "roc_auc"]
        calib = cat_group[cat_group["metric_name"] == "calibration_error"]

        rows = []
        algorithms = sorted(cat_group["algorithm"].unique())
        ranks = {}
        for sim in sims_present:
            sim_roc = roc[roc["simulator"] == sim].set_index("algorithm")["metric_value"].dropna()
            ranked = sim_roc.sort_values(ascending=False)
            ranks[sim] = {algo: i + 1 for i, algo in enumerate(ranked.index)}

        for algo in algorithms:
            row = {"algorithm": algo}
            rank_values = []
            for sim in sims_present:
                sim_roc_val = roc[(roc["simulator"] == sim) & (roc["algorithm"] == algo)]["metric_value"]
                sim_calib_val = calib[(calib["simulator"] == sim) & (calib["algorithm"] == algo)]["metric_value"]
                row[f"roc_auc_{sim}"] = sim_roc_val.iloc[0] if not sim_roc_val.empty else float("nan")
                row[f"calibration_error_{sim}"] = sim_calib_val.iloc[0] if not sim_calib_val.empty else float("nan")
                rank = ranks.get(sim, {}).get(algo)
                row[f"rank_{sim}"] = rank
                rank_values.append(rank)
            valid_ranks = [r for r in rank_values if r is not None]
            if len(valid_ranks) < 2:
                row["agreement"] = "insufficient data"
            elif len(set(valid_ranks)) == 1:
                row["agreement"] = "yes"
            elif max(valid_ranks) - min(valid_ranks) <= 1:
                row["agreement"] = "close"
            else:
                row["agreement"] = "NO -- flips"
            rows.append(row)

        result_df = pd.DataFrame(rows).set_index("algorithm").round(4)
        safe_cat = category.replace("/", "-").replace(" ", "_")
        path = os.path.join(OUTPUT_DIR, f"cross_simulator_ranking_{safe_cat}.md")
        with open(path, "w") as f:
            f.write(f"# Cross-simulator ranking agreement -- {category}\n\n")
            f.write(result_df.to_markdown())
            f.write(
                "\n\n'NO -- flips' means this algorithm's relative ranking changed meaningfully between "
                "the BKT-shaped ground truth and the IRT-shaped ground truth -- its apparent advantage "
                "under one may be an artifact of matching that simulator's assumptions rather than a "
                "real property of the algorithm.\n"
            )
        print(f"wrote {path}")


def write_charts(top_level: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for metric in CHART_METRICS:
        metric_df = top_level[top_level["metric_name"] == metric]
        for category, group in metric_df.groupby("practice_category"):
            pivot = group.pivot_table(index="algorithm", columns="simulator", values="metric_value")
            if pivot.empty:
                continue
            ax = pivot.plot(kind="bar", figsize=(9, 5))
            ax.set_title(f"{metric} by algorithm -- {category}")
            ax.set_ylabel(metric)
            ax.set_xlabel("")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            safe_cat = category.replace("/", "-").replace(" ", "_")
            path = os.path.join(FIGURES_DIR, f"{metric}_{safe_cat}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 1 paper-ready report from logged experiment results")
    parser.add_argument("--practice-category", default=None,
                         help="restrict to one category; default is every category with logged results")
    args = parser.parse_args()

    engine = get_engine()
    raw = fetch_raw(engine)
    if args.practice_category:
        raw = raw[raw["practice_category"] == args.practice_category]
        if raw.empty:
            raise RuntimeError(f"No logged results found for practice_category={args.practice_category!r}")

    raw = latest_only(raw, subset_cols=["algorithm", "practice_category", "simulator", "metric_name"])
    top_level, subject_level = split_top_level_and_subject(raw)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw.drop(columns=["context"]).to_csv(os.path.join(OUTPUT_DIR, "comparison_raw.csv"), index=False)
    print(f"wrote {os.path.join(OUTPUT_DIR, 'comparison_raw.csv')}")
    if not subject_level.empty:
        subject_level.drop(columns=["context"]).to_csv(
            os.path.join(OUTPUT_DIR, "subject_breakdown_raw.csv"), index=False
        )
        print(f"wrote {os.path.join(OUTPUT_DIR, 'subject_breakdown_raw.csv')}")

    write_comparison_tables(top_level)
    write_cross_simulator_ranking(top_level)
    write_charts(top_level)

    print(f"\nDone. Everything is under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
