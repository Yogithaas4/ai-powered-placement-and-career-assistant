"""
scripts/sweep_bkt_params.py
-------------------------------
Answers "what should I change SUBJECT_BKT_PARAMS to?" empirically instead
of by guessing. Runs BKTOnlySelector across a grid of (p_transit, p_slip,
p_guess) values -- using the override_params mechanism added to
experiments/selectors/mastery_based.py's BKTOnlySelector.__init__ -- against
BOTH Step 3 simulators (BKTGenerativeSimulator, IRTGenerativeSimulator),
and prints a table sorted by calibration_error so you can see which
combination actually predicts responses best.

This does NOT edit core/knowledge_model.py's SUBJECT_BKT_PARAMS for you --
it's a measurement tool. Once you see which combination wins, update the
relevant subject's entry in SUBJECT_BKT_PARAMS by hand (that's the one and
only place actually used by the live prototype and by BKTOnlySelector /
BKTEMAEpsilonGreedySelector when NOT overridden).

Usage:
    python scripts/sweep_bkt_params.py --practice-category "Engineering Mathematics" \
        --n-students 100 --n-questions 40

    # customize the grid (comma-separated values, all combinations tried)
    python scripts/sweep_bkt_params.py --practice-category "Engineering Mathematics" \
        --p-transit 0.05,0.10,0.15 --p-slip 0.08,0.15,0.20 --p-guess 0.15,0.25 --p-init 0.30
"""

import argparse
import itertools
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.data import load_experiment_data  # noqa: E402
from experiments.selectors.mastery_based import BKTOnlySelector  # noqa: E402
from experiments.simulators import BKTGenerativeSimulator, IRTGenerativeSimulator  # noqa: E402
from experiments.runner import ExperimentConfig, run_experiment  # noqa: E402
from core.knowledge_model import BKTParams  # noqa: E402


def parse_floats(s):
    return [float(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Sweep BKT parameters against Step 3 simulators")
    parser.add_argument("--practice-category", required=True)
    parser.add_argument("--p-init", default="0.30")
    parser.add_argument("--p-transit", default="0.05,0.10,0.15")
    parser.add_argument("--p-slip", default="0.08,0.15,0.20")
    parser.add_argument("--p-guess", default="0.15,0.20,0.25")
    parser.add_argument("--n-students", type=int, default=100)
    parser.add_argument("--n-questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    p_inits = parse_floats(args.p_init)
    p_transits = parse_floats(args.p_transit)
    p_slips = parse_floats(args.p_slip)
    p_guesses = parse_floats(args.p_guess)

    combos = list(itertools.product(p_inits, p_transits, p_slips, p_guesses))
    print(f"Sweeping {len(combos)} parameter combinations x 2 simulators "
          f"({len(combos) * 2} total runs) for {args.practice_category!r}\n")

    questions_df, dag = load_experiment_data(practice_category=args.practice_category)

    simulators = {
        "bkt_generative": lambda: BKTGenerativeSimulator(seed=args.seed),
        "irt_generative": lambda: IRTGenerativeSimulator(seed=args.seed),
    }

    results = []
    for p_init, p_transit, p_slip, p_guess in combos:
        params = BKTParams(p_init=p_init, p_transit=p_transit, p_slip=p_slip, p_guess=p_guess)
        for sim_name, sim_factory in simulators.items():
            selector = BKTOnlySelector(override_params=params)
            simulator = sim_factory()
            config = ExperimentConfig(
                practice_category=args.practice_category,
                n_students=args.n_students,
                n_questions_per_student=args.n_questions,
                log_to_db=not args.no_db,
            )
            try:
                result = run_experiment(selector, simulator, questions_df, dag, config)
            except ValueError as e:
                print(f"SKIPPED {params} / {sim_name}: {e}")
                continue
            m = result["metrics"]
            results.append({
                "p_init": p_init, "p_transit": p_transit, "p_slip": p_slip, "p_guess": p_guess,
                "simulator": sim_name,
                "roc_auc": m["roc_auc"], "brier_score": m["brier_score"],
                "log_loss": m["log_loss"], "calibration_error": m["calibration_error"],
            })

    results.sort(key=lambda r: r["calibration_error"])

    print(f"\n{'=' * 130}")
    print(f"{'p_init':<8}{'p_transit':<11}{'p_slip':<9}{'p_guess':<9}{'simulator':<18}"
          f"{'roc_auc':<10}{'brier':<10}{'log_loss':<10}{'calib_err':<10}")
    print("-" * 130)
    for r in results:
        print(f"{r['p_init']:<8.2f}{r['p_transit']:<11.2f}{r['p_slip']:<9.2f}{r['p_guess']:<9.2f}"
              f"{r['simulator']:<18}{r['roc_auc']:<10.4f}{r['brier_score']:<10.4f}"
              f"{r['log_loss']:<10.4f}{r['calibration_error']:<10.4f}")

    if results:
        best = results[0]
        print(f"\nBest calibration_error ({best['calibration_error']:.4f}) at "
              f"p_init={best['p_init']}, p_transit={best['p_transit']}, "
              f"p_slip={best['p_slip']}, p_guess={best['p_guess']} "
              f"(against {best['simulator']})")
        print("If this combination looks good against BOTH simulators (check the table above, "
              "not just the top row), update the relevant entry in SUBJECT_BKT_PARAMS in "
              "core/knowledge_model.py to these values.")


if __name__ == "__main__":
    main()
