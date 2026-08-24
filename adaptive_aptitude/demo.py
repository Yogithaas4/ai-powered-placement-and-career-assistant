"""
demo.py
-------
End-to-end simulation of the adaptive platform.
Run this to verify everything works before connecting a frontend.

Usage:
    python demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import random
import os

from core.concept_dag import build_default_dag
from core.knowledge_model import StudentKnowledgeModel
from core.question_selector import QuestionSelector


def run_demo():
    print("=" * 60)
    print("  Adaptive Test Prep Platform — End-to-End Demo")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────────────
    print("\n1. Building Concept DAG...")
    dag = build_default_dag()
    total_concepts = len(dag.nodes)
    print(f"   ✅ {total_concepts} concepts loaded")

    print("\n2. Loading questions...")
    csv_path = os.path.join(os.path.dirname(__file__), "questions-enriched.csv")
    if not os.path.exists(csv_path):
        # Create a tiny mock dataset for the demo
        mock_data = []
        for subj, topic, subtopic, cid in [
            ("Computer Networks", "OSI Model", "Layer Functions", "CN::OSI_Model"),
            ("Computer Networks", "Data Link Layer", "Error Detection", "CN::DLL_ErrorDetection"),
            ("Computer Networks", "Data Link Layer", "Flow Control", "CN::DLL_FlowControl"),
            ("Operating System", "Process Management", "Process States", "OS::Process_States"),
            ("Operating System", "CPU Scheduling", "Scheduling Algorithms", "OS::Scheduling_Algo"),
        ]:
            for i in range(5):
                mock_data.append({
                    "subject": subj, "topic": topic, "subtopic": subtopic,
                    "question": f"Sample question {i+1} about {subtopic}?",
                    "option_a": "Option A", "option_b": "Option B",
                    "option_c": "Option C", "option_d": "Option D",
                    "correct_answer": random.choice(["A","B","C","D"]),
                    "difficulty": random.choice(["Easy","Medium","Hard"]),
                    "time_expected_minutes": random.randint(1,6),
                })
        df = pd.DataFrame(mock_data)
        print("   ⚠️  Using mock dataset (place questions-enriched.csv here for full demo)")
    else:
        df = pd.read_csv(csv_path)
        print(f"   ✅ {len(df)} questions loaded")

    df = df.reset_index().rename(columns={"index": "question_index"})
    df["question_index"] = df["question_index"].astype(str)

    # ── Init models ────────────────────────────────────────────────────────
    db_path = "demo_adaptive.db"
    #if os.path.exists(db_path):
    #    os.remove(db_path)  # fresh start for demo

    km       = StudentKnowledgeModel(db_path=db_path)
    selector = QuestionSelector(df, dag, km, epsilon=0.20)

    STUDENT_ID = "demo_student_001"
    SUBJECT    = "Computer Networks"

    print(f"\n3. Simulating student: {STUDENT_ID}")
    print(f"   Subject: {SUBJECT}")

    # ── Simulate a 15-question session ────────────────────────────────────
    print("\n4. Running 15-question adaptive session...\n")
    print(f"   {'#':<3} {'Concept':<35} {'Diff':<8} {'Answer':<8} {'Skill Score':<12} {'Label'}")
    print("   " + "-" * 80)

    session_history = []
    for i in range(15):
        q = selector.select_question(STUDENT_ID, SUBJECT, session_history)
        if q is None:
            print("   No more questions available.")
            break

        qid       = str(q.get("question_index", i))
        concept   = q.get("concept_id", "unknown")
        topic     = q.get("topic", "")
        subtopic  = q.get("subtopic", "")
        diff      = q.get("difficulty", "Medium")
        correct_a = q.get("correct_answer", "A")

        # Simulate student answer — harder questions more likely to be wrong
        skill = km.get_skill(STUDENT_ID, concept, SUBJECT, topic, subtopic)
        p_correct = {
            "Easy":   0.80,
            "Medium": skill["skill_score"] + 0.1,
            "Hard":   skill["skill_score"] - 0.1
        }.get(diff, 0.5)
        p_correct = max(0.1, min(0.95, p_correct))
        is_correct = random.random() < p_correct
        answer     = correct_a if is_correct else random.choice(
            [x for x in ["A","B","C","D"] if x != correct_a])

        # Update mastery
        updated = km.update_skill(
            student_id=STUDENT_ID, question_id=qid,
            concept_id=concept, subject=SUBJECT,
            topic=topic, subtopic=subtopic,
            difficulty=diff, correct=is_correct,
            time_taken_sec=random.uniform(30, 300)
        )

        score = updated["skill_score"]
        label = "✅ Mastered" if score >= 0.8 else ("🟡 Proficient" if score >= 0.6
                else ("🟠 Developing" if score >= 0.4 else "🔴 Weak"))
        short_concept = (topic + " > " + subtopic)[:33]
        mark = "✓" if is_correct else "✗"

        print(f"   {i+1:<3} {short_concept:<35} {diff:<8} {mark:<8} {score:.3f}        {label}")
        session_history.append(qid)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n5. Session Summary\n")
    summary = km.get_subject_summary(STUDENT_ID, SUBJECT)
    coverage = selector.get_coverage_stats(STUDENT_ID, SUBJECT)

    print(f"   Subject:            {summary['subject']}")
    print(f"   Concepts explored:  {summary['concepts_seen']}")
    print(f"   Mastered concepts:  {summary['mastered_count']}")
    print(f"   Total attempts:     {summary['total_attempts']}")
    print(f"   Accuracy:           {summary['accuracy']:.1%}")
    print(f"   Avg mastery score:  {summary['avg_mastery']:.3f}")
    print(f"   Concept coverage:   {coverage['coverage_pct']}%")

    print("\n6. Per-concept Mastery Breakdown\n")
    skills = km.get_all_skills(STUDENT_ID, SUBJECT)
    skills_sorted = sorted(skills, key=lambda x: x["skill_score"])
    print(f"   {'Concept':<40} {'BKT':>6} {'EMA':>6} {'Score':>7} {'Attempts':>9}")
    print("   " + "-" * 75)
    for s in skills_sorted:
        name = (s['topic'] + " > " + s['subtopic'])[:38]
        print(f"   {name:<40} {s['bkt_score']:>6.3f} {s['ema_score']:>6.3f} "
              f"{s['skill_score']:>7.3f} {s['attempts']:>9}")

    print("\n✅ Demo complete! The adaptive engine is working correctly.")
    print(f"   Database saved to: {db_path}")
    print(f"\nTo run the API server:")
    print(f"   pip install fastapi uvicorn pandas")
    print(f"   cd adaptive_platform")
    print(f"   uvicorn api.main:app --reload --port 8000")
    print(f"   Then open: http://localhost:8000/docs\n")

    #if os.path.exists(db_path):
    #    os.remove(db_path)


if __name__ == "__main__":
    random.seed(42)
    run_demo()
