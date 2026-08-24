"""
scripts/run_experiments.py
-----------------------------
CLI entry point for Phase 1 experiments.

Sessions are scoped by --practice-category, the 5 broad student-facing
buckets ("Core CS (Systems & Theory)", "Aptitude", "Programming & DSA",
"Engineering Mathematics", "Data Science & AI") -- NOT individual subjects.
A run draws from every canonical_subject inside that category in one pool,
exactly like a real session would. Per-subject accuracy/mastery is
reported separately in the "subject breakdown" printed at the end.

Single algorithm, single simulator:
    python scripts/run_experiments.py --practice-category "Core CS (Systems & Theory)" \
        --algorithm bkt_ema_epsilon_greedy --simulator bkt_generative --n-students 200 --n-questions 30

Every algorithm, one simulator:
    python scripts/run_experiments.py --practice-category "Core CS (Systems & Theory)" \
        --algorithm all --simulator irt_generative --n-students 200 --n-questions 30

THE ACTUAL PHASE 1 DELIVERABLE -- every algorithm against BOTH Step 3
simulators, with a cross-simulator ranking-agreement table at the end so
you can see whether an algorithm's apparent advantage holds up under a
structurally different ground truth, or was just an artifact of matching
one simulator's assumptions (see experiments/simulators.py docstring):

    python scripts/run_experiments.py --practice-category all \
        --algorithm all --simulator both --n-students 200 --n-questions 100

Add --no-db to skip Postgres logging (prints only) -- useful for a quick
local sanity check before committing to a full logged run.
"""

import argparse
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.data import load_experiment_data  # noqa: E402
from experiments.selectors import (  # noqa: E402
    RandomSelector,
    RuleWeakestTopicSelector,
    EMAOnlySelector,
    BKTOnlySelector,
    BKTEMAEpsilonGreedySelector,
)
from experiments.simulators import SimpleAbilitySimulator, BKTGenerativeSimulator, IRTGenerativeSimulator  # noqa: E402
from experiments.runner import ExperimentConfig, run_experiment  # noqa: E402

ALGORITHM_REGISTRY = {
    "random": RandomSelector,
    "rule_weakest_topic": RuleWeakestTopicSelector,
    "ema_only": EMAOnlySelector,
    "bkt": BKTOnlySelector,
    "bkt_ema_epsilon_greedy": BKTEMAEpsilonGreedySelector,
    # "thompson_sampling": ...,  # future work
    # "irt_2pl": ...,            # future work (fit-only benchmark)
}

SIMULATOR_REGISTRY = {
    "simple": SimpleAbilitySimulator,          # Step 1/2 placeholder, backward compatibility only
    "bkt_generative": BKTGenerativeSimulator,  # Step 3
    "irt_generative": IRTGenerativeSimulator,  # Step 3
}
# The two Step 3 simulators run together under --simulator both. "simple" is
# deliberately excluded from "both" -- it's legacy, not part of the real comparison.
STEP3_SIMULATORS = ["bkt_generative", "irt_generative"]

TOP_LEVEL_METRIC_ORDER = [
    "roc_auc", "pr_auc", "brier_score", "log_loss", "calibration_error",
    "questions_to_mastery", "coverage", "diversity", "repeat_rate", "difficulty_match_rate",
]


def run_one(selector_name, simulator_name, category, args, questions_df, dag):
    selector_cls = ALGORITHM_REGISTRY[selector_name]
    simulator_cls = SIMULATOR_REGISTRY[simulator_name]
    selector = selector_cls()
    simulator = simulator_cls(seed=args.seed)
    config = ExperimentConfig(
        practice_category=category,
        n_students=args.n_students,
        n_questions_per_student=args.n_questions,
        log_to_db=not args.no_db,
        log_sessions=args.log_sessions,
    )
    return run_experiment(selector, simulator, questions_df, dag, config)


def print_result(selector_name, simulator_name, result):
    print(f"\nrun_id: {result['run_id']}  (simulator={simulator_name})")
    print(f"n_records: {result['n_records']}")
    print("metrics (whole category):")
    for name, value in sorted(result["metrics"].items()):
        print(f"  {name:<24} {value:.4f}" if isinstance(value, float) else f"  {name:<24} {value}")
    print("\nsubject breakdown (within this category):")
    for subject, subj_metrics in sorted(result["subject_breakdown"].items()):
        print(f"  {subject}:")
        for name, value in sorted(subj_metrics.items()):
            print(f"    {name:<20} {value:.4f}" if isinstance(value, float) else f"    {name:<20} {value}")


def print_comparison_table(title: str, results: dict):
    """results: {algorithm_name: result_dict} -- one simulator's worth."""
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    header = f"{'algorithm':<24}" + "".join(f"{m:<16}" for m in TOP_LEVEL_METRIC_ORDER)
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        row = f"{name:<24}"
        for m in TOP_LEVEL_METRIC_ORDER:
            v = result["metrics"].get(m, float("nan"))
            row += f"{v:<16.4f}" if isinstance(v, float) else f"{str(v):<16}"
        print(row)


def print_cross_simulator_table(results_by_sim: dict):
    """
    results_by_sim: {simulator_name: {algorithm_name: result_dict}}

    The actual point of --simulator both: puts roc_auc and
    calibration_error for EVERY algorithm side by side across both
    simulators, plus each algorithm's rank (1 = best) under each, so a
    ranking flip between simulators (an algorithm that looks great under
    one ground truth and mediocre under the other) is visible at a glance
    instead of requiring you to eyeball two separate tables.
    """
    sim_names = [s for s in STEP3_SIMULATORS if s in results_by_sim]
    if len(sim_names) < 2:
        return  # nothing to cross-compare

    algo_names = list(results_by_sim[sim_names[0]].keys())

    # rank (1=best) by roc_auc within each simulator, NaN-safe
    ranks = {}
    for sim in sim_names:
        scored = [
            (a, results_by_sim[sim][a]["metrics"].get("roc_auc", float("nan")))
            for a in algo_names if a in results_by_sim[sim]
        ]
        scored = [(a, v) for a, v in scored if v == v]  # drop NaN
        scored.sort(key=lambda x: -x[1])
        ranks[sim] = {a: i + 1 for i, (a, v) in enumerate(scored)}

    print(f"\n{'=' * 100}\nCROSS-SIMULATOR RANKING AGREEMENT (roc_auc-based rank, 1 = best)\n{'=' * 100}")
    header = f"{'algorithm':<24}"
    for sim in sim_names:
        header += f"{'roc_auc('+sim[:4]+')':<20}{'rank':<7}{'calib_err':<12}"
    header += "rank_agrees?"
    print(header)
    print("-" * len(header))
    for a in algo_names:
        row = f"{a:<24}"
        rank_values = []
        for sim in sim_names:
            r = results_by_sim.get(sim, {}).get(a)
            roc = r["metrics"].get("roc_auc", float("nan")) if r else float("nan")
            calib = r["metrics"].get("calibration_error", float("nan")) if r else float("nan")
            rank = ranks.get(sim, {}).get(a, "-")
            rank_values.append(rank)
            row += f"{roc:<20.4f}{str(rank):<7}{calib:<12.4f}"
        agrees = "yes" if len(set(rank_values)) == 1 and "-" not in rank_values else (
            "close" if all(isinstance(r, int) for r in rank_values) and max(rank_values) - min(rank_values) <= 1
            else "NO -- flips"
        )
        row += agrees
        print(row)
    print(
        "\n'NO -- flips' means this algorithm's relative ranking changed meaningfully between the "
        "BKT-shaped ground truth and the IRT-shaped ground truth -- its apparent advantage under one "
        "may be an artifact of matching that simulator's assumptions rather than a real property of "
        "the algorithm. See experiments/simulators.py module docstring."
    )


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 adaptive-selection experiment(s)")
    parser.add_argument("--practice-category", required=True,
                         help='e.g. "Core CS (Systems & Theory)", or "all" to loop every category')
    parser.add_argument("--algorithm", default="random",
                         choices=list(ALGORITHM_REGISTRY.keys()) + ["all"])
    parser.add_argument("--simulator", default="simple",
                         choices=list(SIMULATOR_REGISTRY.keys()) + ["both"],
                         help="'simple' is the Step 1/2 placeholder (backward compat only). "
                              "'both' runs bkt_generative + irt_generative together and prints "
                              "a cross-simulator ranking-agreement table -- this is the real "
                              "Phase 1 comparison, use this for paper-bound results.")
    parser.add_argument("--n-students", type=int, default=200)
    parser.add_argument("--n-questions", type=int, default=30)
    parser.add_argument("--no-db", action="store_true", help="skip Postgres logging, print only")
    parser.add_argument("--log-sessions", action="store_true",
                         help="also write per-response rows to student_sessions/student_responses")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    category_filter = None if args.practice_category == "all" else args.practice_category
    questions_df, dag = load_experiment_data(practice_category=category_filter)

    categories = (
        sorted(questions_df["practice_category"].dropna().unique().tolist())
        if args.practice_category == "all"
        else [args.practice_category]
    )
    algorithms = list(ALGORITHM_REGISTRY.keys()) if args.algorithm == "all" else [args.algorithm]
    simulators = STEP3_SIMULATORS if args.simulator == "both" else [args.simulator]

    for category in categories:
        print(f"\n{'#' * 100}\n# practice_category={category!r}\n{'#' * 100}")

        results_by_sim = {}
        for sim_name in simulators:
            print(f"\n{'~' * 90}\n~ simulator = {sim_name}\n{'~' * 90}")
            results = {}
            for algo_name in algorithms:
                print(f"\n{'=' * 70}\nRunning {algo_name} vs {sim_name}\n{'=' * 70}")
                try:
                    result = run_one(algo_name, sim_name, category, args, questions_df, dag)
                except ValueError as e:
                    print(f"SKIPPED: {e}")
                    continue
                print_result(algo_name, sim_name, result)
                results[algo_name] = result

            if len(results) > 1:
                print_comparison_table(f"COMPARISON ACROSS ALGORITHMS -- simulator={sim_name}", results)
            results_by_sim[sim_name] = results

        if len(simulators) > 1:
            print_cross_simulator_table(results_by_sim)


if __name__ == "__main__":
    main()
