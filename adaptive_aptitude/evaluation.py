"""
evaluation.py
=============
Complete accuracy evaluation for the Adaptive Test Prep Platform.

Covers 4 levels of evaluation:
  1. BKT Prediction Accuracy   (AUC-ROC, RMSE, Binary Accuracy)
  2. Question Selection Quality (Difficulty-Mastery match rate)
  3. Mastery Convergence Check  (Does mastery move in the right direction?)
  4. A/B Test Simulation        (Adaptive vs Random — needs questions CSV)

Usage:
------
  # After running demo.py or using the API:
  python evaluation.py --db adaptive_platform.db

  # Run A/B test simulation (needs questions CSV):
  python evaluation.py --db adaptive_platform.db --ab --csv questions-enriched.csv

  # Evaluate specific student:
  python evaluation.py --db adaptive_platform.db --student student_001
"""

import os
import sys
import sqlite3
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_interactions(db_path: str, student_id: str = None) -> pd.DataFrame:
    """Load interaction_log from SQLite into a DataFrame."""
    if not os.path.exists(db_path):
        print(f"  ❌ Database not found: {db_path}")
        print(f"     Run demo.py first, or use the API to answer some questions.")
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM interaction_log"
        params = []
        if student_id:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY student_id, concept_id, timestamp"
        df = pd.read_sql(query, conn, params=params)

    return df


def load_skills(db_path: str, student_id: str = None) -> pd.DataFrame:
    """Load student_skill table from SQLite."""
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM student_skill"
        params = []
        if student_id:
            query += " WHERE student_id = ?"
            params.append(student_id)
        df = pd.read_sql(query, conn, params=params)
    return df


def print_header(title: str):
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_section(title: str):
    print(f"\n  ── {title} {'─' * (50 - len(title))}")


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 1 — BKT PREDICTION ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_bkt_accuracy(db_path: str, student_id: str = None) -> dict:
    """
    Measures how accurately BKT predicts whether a student will
    answer correctly BEFORE they answer.

    BKT prediction formula:
      P(correct) = P(known) × (1 - p_slip)  +  (1 - P(known)) × p_guess

    Metrics:
      AUC-ROC  → how well predictions rank correct vs incorrect answers
                 0.5 = random, 1.0 = perfect
      RMSE     → average prediction error (lower = better)
      Accuracy → % of times threshold prediction (>=0.5) matched actual
    """
    print_header("LEVEL 1 — BKT Prediction Accuracy")

    df = load_interactions(db_path, student_id)
    if df.empty:
        print("  ⚠️  No interaction data found.")
        return {}

    if len(df) < 10:
        print(f"  ⚠️  Only {len(df)} interactions. Need at least 10 for meaningful evaluation.")
        print(f"      Answer more questions first.")
        return {}

    # BKT default params (matching knowledge_model.py defaults)
    P_SLIP  = 0.10
    P_GUESS = 0.20

    # Compute predicted P(correct) from bkt_before
    df["p_correct_pred"] = (
        df["bkt_before"] * (1 - P_SLIP) +
        (1 - df["bkt_before"]) * P_GUESS
    )

    y_true = df["correct"].astype(int).values
    y_pred = df["p_correct_pred"].values

    # ── AUC-ROC ──
    try:
        from sklearn.metrics import roc_auc_score, mean_squared_error
        auc  = roc_auc_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    except ImportError:
        # Manual AUC-ROC if sklearn not installed
        pairs = [(y_pred[i], y_pred[j])
                 for i in range(len(y_true)) for j in range(len(y_true))
                 if y_true[i] == 1 and y_true[j] == 0]
        auc  = sum(1 for p, n in pairs if p > n) / len(pairs) if pairs else 0.5
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    binary_acc  = np.mean((y_pred >= 0.5).astype(int) == y_true)
    actual_acc  = y_true.mean()
    avg_pred    = y_pred.mean()
    calibration = abs(actual_acc - avg_pred)  # lower = better calibrated

    # ── Interpret ──
    auc_label  = "🟢 Great"  if auc  > 0.80 else ("🟡 Acceptable" if auc  > 0.70 else "🔴 Needs work")
    rmse_label = "🟢 Great"  if rmse < 0.35 else ("🟡 Acceptable" if rmse < 0.50 else "🔴 Needs work")

    print(f"\n  Total interactions  : {len(df):,}")
    print(f"  Unique students     : {df['student_id'].nunique()}")
    print(f"  Unique concepts     : {df['concept_id'].nunique()}")
    print(f"\n  {'Metric':<28} {'Value':>8}   {'Interpretation'}")
    print(f"  {'-'*60}")
    print(f"  {'AUC-ROC':<28} {auc:>8.4f}   {auc_label}")
    print(f"  {'RMSE':<28} {rmse:>8.4f}   {rmse_label}")
    print(f"  {'Binary Accuracy':<28} {binary_acc:>8.2%}")
    print(f"  {'Actual Answer Accuracy':<28} {actual_acc:>8.2%}")
    print(f"  {'Avg Predicted P(correct)':<28} {avg_pred:>8.2%}")
    print(f"  {'Calibration Error':<28} {calibration:>8.4f}   (lower = better)")

    # ── Per difficulty ──
    print_section("Breakdown by Difficulty")
    print(f"  {'Difficulty':<10} {'Count':>6} {'Actual Acc':>12} {'Pred Acc':>10} {'AUC':>8}")
    print(f"  {'-'*50}")
    for diff in ["Easy", "Medium", "Hard"]:
        sub = df[df["difficulty"] == diff]
        if len(sub) < 5:
            continue
        sub_actual = sub["correct"].mean()
        sub_pred   = sub["p_correct_pred"].mean()
        try:
            sub_auc = roc_auc_score(sub["correct"], sub["p_correct_pred"]) \
                      if sub["correct"].nunique() > 1 else float("nan")
        except Exception:
            sub_auc = float("nan")
        auc_str = f"{sub_auc:.4f}" if not np.isnan(sub_auc) else "  N/A"
        print(f"  {diff:<10} {len(sub):>6} {sub_actual:>12.2%} {sub_pred:>10.2%} {auc_str:>8}")

    # ── Per subject ──
    print_section("Breakdown by Subject")
    print(f"  {'Subject':<42} {'Count':>6} {'Accuracy':>10} {'AUC':>8}")
    print(f"  {'-'*70}")
    for subj, sub in df.groupby("subject"):
        if len(sub) < 5:
            continue
        try:
            s_auc = roc_auc_score(sub["correct"], sub["p_correct_pred"]) \
                    if sub["correct"].nunique() > 1 else float("nan")
        except Exception:
            s_auc = float("nan")
        auc_str = f"{s_auc:.4f}" if not np.isnan(s_auc) else "  N/A"
        print(f"  {subj:<42} {len(sub):>6} {sub['correct'].mean():>10.2%} {auc_str:>8}")

    return {"auc": auc, "rmse": rmse, "binary_accuracy": binary_acc,
            "actual_accuracy": actual_acc, "calibration_error": calibration}


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 2 — QUESTION SELECTION QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_question_selection(db_path: str, student_id: str = None) -> dict:
    """
    Checks whether the system selected questions at the right difficulty
    given the student's mastery level at the time of selection.

    Expected mapping:
      mastery < 0.40  →  Easy
      mastery 0.40-0.70 →  Medium
      mastery > 0.70  →  Hard
    """
    print_header("LEVEL 2 — Question Selection Quality")

    df = load_interactions(db_path, student_id)
    if df.empty or len(df) < 5:
        print("  ⚠️  Not enough data.")
        return {}

    def expected_difficulty(mastery: float) -> str:
        if mastery < 0.40:  return "Easy"
        elif mastery < 0.70: return "Medium"
        else:                return "Hard"

    df["expected_diff"] = df["bkt_before"].apply(expected_difficulty)
    df["correct_selection"] = df["difficulty"] == df["expected_diff"]

    match_rate = df["correct_selection"].mean()
    label = "🟢 Great" if match_rate > 0.75 else ("🟡 Acceptable" if match_rate > 0.55 else "🔴 Needs work")

    print(f"\n  Difficulty-Mastery match rate : {match_rate:.2%}  {label}")
    print(f"  Total selections checked      : {len(df):,}")

    print_section("Confusion Matrix (Expected vs Actual Difficulty)")
    pivot = df.groupby(["expected_diff", "difficulty"]).size().unstack(fill_value=0)
    # Ensure all columns exist
    for col in ["Easy", "Medium", "Hard"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Easy", "Medium", "Hard"]]
    print(f"\n  {'Expected \\ Got':<15} {'Easy':>8} {'Medium':>8} {'Hard':>8}")
    print(f"  {'-'*42}")
    for idx, row in pivot.iterrows():
        print(f"  {idx:<15} {row.get('Easy',0):>8} {row.get('Medium',0):>8} {row.get('Hard',0):>8}")

    print_section("Mastery Distribution at Time of Selection")
    bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    df["mastery_band"] = pd.cut(df["bkt_before"], bins=bins, labels=labels)
    band_counts = df.groupby("mastery_band", observed=True).size()
    for band, count in band_counts.items():
        bar = "█" * (count * 30 // max(band_counts))
        print(f"  {band} : {bar:<30} {count:>5}")

    return {"match_rate": match_rate}


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 3 — MASTERY CONVERGENCE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_mastery_convergence(db_path: str, student_id: str = None) -> dict:
    """
    Checks whether mastery scores move in the correct direction:
      - Students with high accuracy → mastery should increase
      - Students with low accuracy  → mastery should stay low or decrease
      - Mastery should stabilize (stop jumping around) after enough attempts
    """
    print_header("LEVEL 3 — Mastery Convergence Check")

    df = load_interactions(db_path, student_id)
    if df.empty:
        print("  ⚠️  No data found.")
        return {}

    converging   = 0
    diverging    = 0
    stable       = 0
    total_checked = 0
    issues        = []

    print(f"\n  {'Student':<20} {'Concept':<35} {'Start':>6} {'End':>6} {'Acc':>6} {'Status'}")
    print(f"  {'-'*85}")

    for (sid, cid), group in df.groupby(["student_id", "concept_id"]):
        group = group.sort_values("timestamp")
        if len(group) < 3:
            continue

        total_checked += 1
        scores    = group["bkt_after"].values
        corrects  = group["correct"].values
        acc       = corrects.mean()
        start     = scores[0]
        end       = scores[-1]
        delta     = end - start

        # Convergence check
        # Good: high acc → rising mastery OR low acc → falling/stable mastery
        is_good = (acc >= 0.6 and delta >= -0.05) or \
                  (acc < 0.4 and delta <= 0.10)

        # Stability check: last 3 scores shouldn't vary much
        if len(scores) >= 4:
            recent_std = np.std(scores[-3:])
            is_stable  = recent_std < 0.15
        else:
            is_stable = True

        if is_good and is_stable:
            status = "✅ Good"
            converging += 1
        elif is_good and not is_stable:
            status = "🟡 Unstable"
            stable += 1
        else:
            status = "🔴 Check"
            diverging += 1
            issues.append((sid, cid, start, end, acc))

        short_cid = cid.split("::")[-1].replace("_", " ")[:33]
        print(f"  {sid:<20} {short_cid:<35} {start:>6.3f} {end:>6.3f} "
              f"{acc:>6.0%} {status}")

    if total_checked == 0:
        print("  ⚠️  No concept with 3+ attempts found. Answer more questions.")
        return {}

    print_section("Summary")
    print(f"  Concepts checked    : {total_checked}")
    print(f"  ✅ Converging       : {converging}  ({converging/total_checked:.0%})")
    print(f"  🟡 Unstable         : {stable}  ({stable/total_checked:.0%})")
    print(f"  🔴 Needs attention  : {diverging}  ({diverging/total_checked:.0%})")

    if issues:
        print_section("Concepts to Investigate")
        for sid, cid, start, end, acc in issues:
            direction = "↑" if end > start else "↓"
            print(f"  {sid} | {cid} | {start:.3f} {direction} {end:.3f} | acc={acc:.0%}")
            if acc < 0.4 and end > start + 0.2:
                print(f"    → Mastery rising despite low accuracy — check p_guess value")
            if acc > 0.7 and end < start - 0.1:
                print(f"    → Mastery falling despite high accuracy — check p_slip value")

    return {
        "total_checked": total_checked,
        "converging": converging,
        "convergence_rate": converging / total_checked if total_checked else 0
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 4 — A/B TEST: ADAPTIVE vs RANDOM SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def run_ab_test(csv_path: str, n_students: int = 100,
                n_questions: int = 30, subject: str = "Computer Networks") -> dict:
    """
    Simulates n_students students each answering n_questions questions.
    Compares adaptive selection vs random selection.

    Key metric: How many questions needed to reach first mastery (score >= 0.80)?
    Lower = better = system is more efficient at building mastery.
    """
    print_header("LEVEL 4 — A/B Test: Adaptive vs Random")

    if not os.path.exists(csv_path):
        print(f"  ❌ CSV not found: {csv_path}")
        print(f"     Provide path with --csv flag")
        return {}

    # Import here to avoid circular imports
    from core.concept_dag import build_default_dag
    from core.knowledge_model import StudentKnowledgeModel
    from core.question_selector import QuestionSelector

    print(f"\n  Simulating {n_students} students × {n_questions} questions each")
    print(f"  Subject: {subject}")
    print(f"  Metric: Questions needed to achieve first concept mastery (score ≥ 0.80)\n")

    questions_df = pd.read_csv(csv_path)
    questions_df = questions_df.reset_index().rename(columns={"index": "question_index"})
    questions_df["question_index"] = questions_df["question_index"].astype(str)
    dag = build_default_dag()

    results = defaultdict(list)
    accuracy_results = defaultdict(list)

    for mode in ["adaptive", "random"]:
        epsilon = 0.20 if mode == "adaptive" else 1.0
        # epsilon=1.0 → always explore → effectively random selection

        for i in range(n_students):
            db = f"_eval_temp_{mode}_{i}.db"
            try:
                km       = StudentKnowledgeModel(db_path=db)
                selector = QuestionSelector(questions_df, dag, km, epsilon=epsilon)

                student_id = f"sim_{i:04d}"
                # Simulate varied student ability (beta distribution — most students mid-range)
                ability = random.betavariate(2, 3)

                questions_to_first_mastery = None
                correct_count = 0

                for q_num in range(n_questions):
                    q = selector.select_question(student_id, subject)
                    if not q:
                        break

                    # Simulate answer: ability + small learning effect + noise
                    diff_modifier = {"Easy": +0.15, "Medium": 0.0, "Hard": -0.15}
                    p_correct = ability + diff_modifier.get(q.get("difficulty","Medium"), 0) \
                                + q_num * 0.005 + random.gauss(0, 0.05)
                    p_correct = max(0.05, min(0.95, p_correct))
                    correct   = random.random() < p_correct
                    if correct:
                        correct_count += 1

                    km.update_skill(
                        student_id    = student_id,
                        question_id   = str(q_num),
                        concept_id    = q.get("concept_id", "unknown"),
                        subject       = subject,
                        topic         = q.get("topic", ""),
                        subtopic      = q.get("subtopic", ""),
                        difficulty    = q.get("difficulty", "Medium"),
                        correct       = correct,
                        time_taken_sec= random.uniform(30, 300)
                    )

                    # Check for first mastery
                    if questions_to_first_mastery is None:
                        summary = km.get_subject_summary(student_id, subject)
                        if summary["mastered_count"] >= 1:
                            questions_to_first_mastery = q_num + 1

                results[mode].append(questions_to_first_mastery or n_questions + 1)
                accuracy_results[mode].append(correct_count / n_questions)

            finally:
                if os.path.exists(db):
                    os.remove(db)

        # Progress indicator
        print(f"  ✅ {mode.capitalize()} simulation complete")

    # ── Results ──
    for mode in ["adaptive", "random"]:
        r = results[mode]
        a = accuracy_results[mode]
        mastered_pct = sum(1 for x in r if x <= n_questions) / n_students
        print(f"\n  {'─'*50}")
        print(f"  Mode: {mode.upper()}")
        print(f"  {'─'*50}")
        print(f"  Avg questions to first mastery : {np.mean(r):.1f}")
        print(f"  Median                         : {np.median(r):.1f}")
        print(f"  % who achieved mastery         : {mastered_pct:.0%}")
        print(f"  Avg answer accuracy            : {np.mean(a):.2%}")

    adapt_avg  = np.mean(results["adaptive"])
    random_avg = np.mean(results["random"])
    improvement = (random_avg - adapt_avg) / random_avg * 100

    print(f"\n  {'═'*50}")
    if improvement > 5:
        print(f"  ✅ Adaptive is {improvement:.1f}% more efficient than random!")
    elif improvement > 0:
        print(f"  🟡 Adaptive is marginally better ({improvement:.1f}%)")
    else:
        print(f"  🔴 Random matched or beat adaptive ({improvement:.1f}%)")
        print(f"     Consider tuning epsilon or BKT parameters")
    print(f"  {'═'*50}")

    return {
        "adaptive_avg":  adapt_avg,
        "random_avg":    random_avg,
        "improvement_pct": improvement
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 5 — QUICK DATABASE HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_database_health(db_path: str):
    """Quick sanity check on what's in the SQLite database."""
    print_header("DATABASE HEALTH CHECK")

    if not os.path.exists(db_path):
        print(f"  ❌ Database not found: {db_path}")
        print(f"     Run demo.py first to generate data.")
        return

    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]

    print(f"\n  Database : {db_path}")
    print(f"  Size     : {os.path.getsize(db_path) / 1024:.1f} KB")
    print(f"\n  {'Table':<25} {'Rows':>8}  {'Columns'}")
    print(f"  {'-'*60}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for table in table_names:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            print(f"  {table:<25} {row_count:>8}  {', '.join(cols[:5])}"
                  f"{'...' if len(cols) > 5 else ''}")

    # Show sample interaction if exists
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sample = conn.execute(
            "SELECT * FROM interaction_log ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if sample:
            print_section("Most Recent Interaction")
            for key in sample.keys():
                print(f"  {key:<20} : {sample[key]}")

    # Show students
    with sqlite3.connect(db_path) as conn:
        students = conn.execute(
            "SELECT student_id, COUNT(*) as attempts, "
            "SUM(correct) as correct FROM interaction_log "
            "GROUP BY student_id ORDER BY attempts DESC LIMIT 10"
        ).fetchall()
        if students:
            print_section("Top Students by Activity")
            print(f"  {'Student ID':<25} {'Attempts':>10} {'Correct':>8} {'Accuracy':>10}")
            print(f"  {'-'*55}")
            for s in students:
                acc = s[2] / s[1] if s[1] > 0 else 0
                print(f"  {s[0]:<25} {s[1]:>10} {s[2]:>8} {acc:>10.2%}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Run all evaluations
# ══════════════════════════════════════════════════════════════════════════════

def run_all(db_path: str, student_id: str = None,
            csv_path: str = None, run_ab: bool = False):

    print("\n" + "╔" + "═"*58 + "╗")
    print("║     Adaptive Platform — Full Evaluation Report         ║")
    print("╚" + "═"*58 + "╝")

    check_database_health(db_path)

    results = {}

    results["bkt"]        = evaluate_bkt_accuracy(db_path, student_id)
    results["selection"]  = evaluate_question_selection(db_path, student_id)
    results["convergence"]= evaluate_mastery_convergence(db_path, student_id)

    if run_ab and csv_path:
        results["ab_test"] = run_ab_test(csv_path)
    elif run_ab and not csv_path:
        print("\n  ⚠️  A/B test skipped — provide --csv path to questions CSV")

    # ── Final scorecard ──
    print_header("OVERALL SCORECARD")
    print()

    if results.get("bkt"):
        auc = results["bkt"].get("auc", 0)
        auc_grade = "A" if auc > 0.80 else ("B" if auc > 0.70 else ("C" if auc > 0.60 else "D"))
        print(f"  BKT Prediction (AUC)        : {auc:.4f}  [{auc_grade}]")

    if results.get("selection"):
        mr = results["selection"].get("match_rate", 0)
        mr_grade = "A" if mr > 0.75 else ("B" if mr > 0.60 else ("C" if mr > 0.45 else "D"))
        print(f"  Question Selection Quality  : {mr:.2%}   [{mr_grade}]")

    if results.get("convergence"):
        cr = results["convergence"].get("convergence_rate", 0)
        cr_grade = "A" if cr > 0.80 else ("B" if cr > 0.65 else ("C" if cr > 0.50 else "D"))
        print(f"  Mastery Convergence Rate    : {cr:.2%}   [{cr_grade}]")

    if results.get("ab_test"):
        imp = results["ab_test"].get("improvement_pct", 0)
        ab_grade = "A" if imp > 15 else ("B" if imp > 5 else ("C" if imp > 0 else "D"))
        print(f"  Adaptive vs Random Gain     : {imp:+.1f}%  [{ab_grade}]")

    print()
    print("  Grade guide: A = Excellent, B = Good, C = Needs tuning, D = Review")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Adaptive Test Prep Platform")
    parser.add_argument("--db",      default="adaptive_platform.db",
                        help="Path to SQLite database (default: adaptive_platform.db)")
    parser.add_argument("--csv",     default=None,
                        help="Path to questions CSV (required for A/B test)")
    parser.add_argument("--student", default=None,
                        help="Evaluate specific student only")
    parser.add_argument("--ab",      action="store_true",
                        help="Run A/B test simulation (slower, needs --csv)")
    parser.add_argument("--level",   type=int, default=0,
                        help="Run specific level only (1-4). 0 = run all")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    if args.level == 1:
        evaluate_bkt_accuracy(args.db, args.student)
    elif args.level == 2:
        evaluate_question_selection(args.db, args.student)
    elif args.level == 3:
        evaluate_mastery_convergence(args.db, args.student)
    elif args.level == 4:
        if args.csv:
            run_ab_test(args.csv)
        else:
            print("Level 4 requires --csv path to questions CSV")
    else:
        run_all(args.db, args.student, args.csv, args.ab)