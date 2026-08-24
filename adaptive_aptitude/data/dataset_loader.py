"""
data/dataset_loader.py
----------------------
Utilities for:
1. Loading your questions-enriched.csv into the platform
2. Bootstrapping BKT parameters from ASSISTments / EdNet interaction logs
   (so your model starts with realistic priors instead of defaults)

ASSISTments SkillBuilder 2009-2010:
  Download: https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010
  Relevant columns: user_id, skill_name, correct, ms_first_response, problem_id

EdNet KT3:
  Download: https://www.kaggle.com/datasets/anhtu96/ednet-kt34
  Relevant columns: user_id, item_id, correct, elapsed_time, bundle_id
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional
from collections import defaultdict


# ── Load your question dataset ─────────────────────────────────────────────

def load_question_dataset(csv_path: str) -> pd.DataFrame:
    """Load the enriched questions CSV and add a concept_id column."""
    df = pd.read_csv(csv_path)
    df = df.reset_index().rename(columns={"index": "question_index"})
    df["question_index"] = df["question_index"].astype(str)

    # Validate required columns
    required = ["subject","topic","subtopic","question",
                "option_a","option_b","option_c","option_d",
                "correct_answer","difficulty","time_expected_minutes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"✅ Loaded {len(df)} questions across {df['subject'].nunique()} subjects")
    print(df.groupby("subject").size().to_string())
    return df


# ── ASSISTments loader ─────────────────────────────────────────────────────

def load_assistments(csv_path: str) -> pd.DataFrame:
    """
    Load ASSISTments SkillBuilder 2009-2010 dataset.

    Key columns we use:
      user_id          → student identifier
      skill_name       → concept label (maps to your topic/subtopic)
      correct          → 1 = correct, 0 = incorrect
      ms_first_response→ response time in milliseconds
      problem_id       → question identifier
      assistment_id    → session context
    """
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    # Standardize column names (the dataset has some variants)
    col_map = {
        "user_id": "student_id",
        "skill_name": "skill_name",
        "correct": "correct",
        "ms_first_response": "response_time_ms",
        "problem_id": "problem_id",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Keep only needed columns
    keep = [c for c in ["student_id","skill_name","correct","response_time_ms","problem_id"]
            if c in df.columns]
    df = df[keep].dropna(subset=["student_id","skill_name","correct"])
    df["correct"] = df["correct"].astype(int)

    print(f"✅ ASSISTments: {len(df)} interactions, "
          f"{df['student_id'].nunique()} students, "
          f"{df['skill_name'].nunique()} skills")
    return df


# ── Estimate BKT parameters from interaction data ─────────────────────────

def estimate_bkt_params_from_data(interactions_df: pd.DataFrame,
                                   skill_col: str = "skill_name",
                                   n_em_iters: int = 20) -> Dict[str, dict]:
    """
    Estimate BKT parameters per skill using simplified EM-like fitting.
    
    For each skill, we compute:
      p_init    ← fraction of students who got first attempt correct (proxy)
      p_transit ← improvement rate across consecutive attempts
      p_slip    ← error rate among students with high running accuracy
      p_guess   ← correct rate among students with low running accuracy

    Returns: {skill_name: {p_init, p_transit, p_slip, p_guess}}
    """
    results = {}

    for skill, group in interactions_df.groupby(skill_col):
        if len(group) < 30:
            continue  # skip skills with too little data

        # Group by student
        student_groups = group.groupby("student_id")["correct"].apply(list)

        first_attempts = [seq[0] for seq in student_groups if len(seq) >= 1]
        p_init = np.mean(first_attempts) if first_attempts else 0.3

        # Estimate transition: P(correct improves with practice)
        improvements = []
        for seq in student_groups:
            if len(seq) >= 4:
                early  = np.mean(seq[:len(seq)//2])
                late   = np.mean(seq[len(seq)//2:])
                improvements.append(max(0, late - early))
        p_transit = np.mean(improvements) if improvements else 0.09

        # Estimate slip/guess from high/low performers
        all_accs = [np.mean(seq) for seq in student_groups if len(seq) >= 3]
        if all_accs:
            high_perf = [a for a in all_accs if a >= 0.8]
            low_perf  = [a for a in all_accs if a <= 0.2]
            p_slip  = np.mean([1 - a for a in high_perf]) if high_perf else 0.10
            p_guess = np.mean(low_perf) if low_perf else 0.20
        else:
            p_slip, p_guess = 0.10, 0.20

        results[skill] = {
            "p_init":    round(float(np.clip(p_init,    0.05, 0.95)), 3),
            "p_transit": round(float(np.clip(p_transit, 0.02, 0.40)), 3),
            "p_slip":    round(float(np.clip(p_slip,    0.02, 0.30)), 3),
            "p_guess":   round(float(np.clip(p_guess,   0.05, 0.40)), 3),
        }

    print(f"✅ Estimated BKT params for {len(results)} skills")
    return results


def save_bkt_params(params: dict, path: str = "bkt_params.json"):
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"💾 Saved BKT params to {path}")


# ── EdNet loader ───────────────────────────────────────────────────────────

def load_ednet_kt3(feather_path: str) -> pd.DataFrame:
    """
    Load EdNet KT3 merged feather file.
    
    Columns used:
      user_id       → student identifier
      item_id       → question identifier
      correct       → 1/0
      elapsed_time  → ms taken to answer
      bundle_id     → groups questions by concept
    """
    df = pd.read_feather(feather_path)
    print(f"✅ EdNet KT3: {len(df)} interactions, "
          f"{df['user_id'].nunique()} students")
    return df


# ── Simulation: generate synthetic student data for testing ───────────────

def simulate_student_interactions(n_students: int = 50,
                                   n_interactions_per_student: int = 30,
                                   subjects: Optional[list] = None) -> pd.DataFrame:
    """
    Generate synthetic interaction logs for testing the platform.
    Useful before you have real student data.
    """
    if subjects is None:
        subjects = ["Computer Networks", "Operating System", "Digital Logic"]

    topics_by_subject = {
        "Computer Networks":  ["OSI Model", "Data Link Layer", "Network Layer", "Transport Layer"],
        "Operating System":   ["Process Management", "CPU Scheduling", "Memory Management", "Deadlocks"],
        "Digital Logic":      ["Number Systems", "Boolean Algebra", "Combinational Circuits", "Sequential Circuits"],
        "Mathematics":        ["Propositional Logic", "Graph Theory", "Combinatorics", "Probability"],
        "Theory of Computation": ["Regular Languages", "Context-Free Languages", "Turing Machines", "Decidability"],
    }

    records = []
    for sid in range(n_students):
        student_id = f"student_{sid:04d}"
        ability    = np.random.beta(2, 3)  # most students are below average

        subject    = np.random.choice(subjects)
        topics     = topics_by_subject.get(subject, ["General"])

        for i in range(n_interactions_per_student):
            topic   = np.random.choice(topics)
            # Simulate learning: ability improves over time
            p_correct = min(0.95, ability + i * 0.01 + np.random.normal(0, 0.05))
            correct   = int(np.random.random() < p_correct)

            records.append({
                "student_id":    student_id,
                "subject":       subject,
                "topic":         topic,
                "correct":       correct,
                "response_time_sec": np.random.exponential(60),
                "interaction_idx": i
            })

    df = pd.DataFrame(records)
    print(f"✅ Simulated {len(df)} interactions for {n_students} students")
    return df


# ── Quick dataset stats ────────────────────────────────────────────────────

def print_dataset_stats(df: pd.DataFrame, name: str = "Dataset"):
    print(f"\n{'='*50}")
    print(f" {name} Statistics")
    print(f"{'='*50}")
    print(f"  Total rows:    {len(df):,}")
    if "student_id" in df.columns:
        print(f"  Students:      {df['student_id'].nunique():,}")
    if "correct" in df.columns:
        print(f"  Accuracy:      {df['correct'].mean():.1%}")
    if "subject" in df.columns:
        print(f"\n  By subject:")
        for s, cnt in df.groupby("subject").size().items():
            print(f"    {s}: {cnt:,}")
    print()
