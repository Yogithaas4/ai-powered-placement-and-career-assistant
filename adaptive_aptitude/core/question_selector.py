"""
question_selector.py
--------------------
Selects the next best question for a student using a combination of:

1. Concept DAG constraints  — don't test advanced concepts before prerequisites
2. BKT/EMA mastery scores   — target weak concepts (exploit)
3. Exploration              — occasionally probe new/unseen concepts (explore)
4. Difficulty progression   — match question difficulty to current mastery
5. Recency filter           — avoid repeating recently seen questions

Selection Strategy:
  - epsilon-greedy style: with probability epsilon → explore (pick a new concept)
                          with probability 1-epsilon → exploit (drill weakest concept)
  - Within selected concept: pick question whose difficulty matches mastery level
"""

import random
import math
from typing import List, Dict, Optional, Tuple
import pandas as pd


# ── Difficulty mapping ─────────────────────────────────────────────────────

DIFFICULTY_MASTERY_BANDS = {
    # mastery range → appropriate difficulty
    (0.00, 0.40): "Easy",
    (0.40, 0.70): "Medium",
    (0.70, 1.00): "Hard",
}

def mastery_to_difficulty(mastery: float) -> str:
    for (low, high), diff in DIFFICULTY_MASTERY_BANDS.items():
        if low <= mastery < high:
            return diff
    return "Medium"


# ── Concept scoring ────────────────────────────────────────────────────────

def score_concept_for_student(concept_id: str, mastery: float,
                               attempts: int, is_unlocked: bool) -> float:
    """
    Score how urgently a concept needs practice.
    Higher score = more likely to be selected.

    Factors:
    - Low mastery → higher priority (weak concepts need drilling)
    - Moderate attempts → slight boost (engagement signal)
    - Unlocked by prerequisites → eligible
    """
    if not is_unlocked:
        return -1.0  # not eligible

    # Urgency: inverse of mastery — weakest concepts score highest
    urgency = 1.0 - mastery

    # Bonus for concepts seen but not mastered (need reinforcement)
    reinforcement = 0.2 if 0 < attempts < 5 else 0.0

    # Bonus for never-seen concepts (encourage breadth)
    novelty = 0.15 if attempts == 0 else 0.0

    return urgency + reinforcement + novelty


# ── Main Question Selector ─────────────────────────────────────────────────

class QuestionSelector:

    def __init__(self,
                 questions_df: pd.DataFrame,
                 concept_dag,
                 knowledge_model,
                 epsilon: float = 0.20):
        """
        Args:
            questions_df   : Full question dataset as DataFrame
            concept_dag    : ConceptDAG instance
            knowledge_model: StudentKnowledgeModel instance
            epsilon        : Exploration probability (0.0–1.0)
        """
        self.questions  = questions_df
        self.dag        = concept_dag
        self.km         = knowledge_model
        self.epsilon    = epsilon

        # Build concept_id column from topic+subtopic if not present
        if "concept_id" not in self.questions.columns:
            self.questions = self.questions.copy()
            self.questions["concept_id"] = self._infer_concept_ids()

    def _infer_concept_ids(self) -> pd.Series:
        """Map (subject, topic, subtopic) → concept_id using the DAG."""
        lookup = {}
        for cid, node in self.dag.nodes.items():
            key = (node.subject, node.topic, node.subtopic)
            lookup[key] = cid

        def map_row(row):
            key = (row["subject"], row["topic"], row["subtopic"])
            return lookup.get(key, f"UNKNOWN::{row['topic']}::{row['subtopic']}")

        return self.questions.apply(map_row, axis=1)

    # ── Main selection entry point ─────────────────────────────────────────

    def select_question(self, student_id: str, subject: str,
                         session_history: Optional[List[str]] = None) -> Optional[dict]:
        """
        Select the single best next question for this student in this subject.

        Returns a dict with question data, or None if no suitable question found.
        """
        session_history = session_history or []
        recently_seen   = self.km.get_recently_seen_questions(student_id, n=50)
        all_seen        = recently_seen | set(session_history)

        # Get student mastery for all concepts in this subject
        mastery_dict = self.km.get_mastery_dict(student_id, subject)
        skills       = {s["concept_id"]: s for s in self.km.get_all_skills(student_id, subject)}

        # Get concepts whose prerequisites are satisfied
        unlocked = set(self.dag.get_unlocked_concepts(subject, mastery_dict, mastery_threshold=0.6))

        # All concepts in subject
        all_concepts = self.dag.get_concepts_by_subject(subject)

        # ── Decide: Explore or Exploit ─────────────────────────────────────
        if random.random() < self.epsilon:
            target_concept = self._explore(all_concepts, mastery_dict, unlocked)
        else:
            target_concept = self._exploit(all_concepts, mastery_dict, skills, unlocked)

        if target_concept is None:
            # Fallback: pick any unlocked concept
            target_concept = random.choice(list(unlocked)) if unlocked else (
                random.choice(all_concepts) if all_concepts else None)

        if target_concept is None:
            return None

        # ── Select question for chosen concept ────────────────────────────
        mastery   = mastery_dict.get(target_concept, 0.3)
        target_diff = mastery_to_difficulty(mastery)

        question = self._pick_question(target_concept, subject, target_diff, all_seen)

        if question is None:
            # Relax difficulty constraint
            question = self._pick_question(target_concept, subject, difficulty=None,
                                           exclude_ids=all_seen)

        if question is None:
            # Last resort: any question in subject not yet seen
            subject_qs = self.questions[
                (self.questions["subject"] == subject) &
                (~self.questions.index.astype(str).isin(all_seen))
            ]
            if not subject_qs.empty:
                question = subject_qs.sample(1).iloc[0].to_dict()

        return question

    # ── Explore: pick a concept not yet studied or rarely studied ──────────

    def _explore(self, concepts: List[str], mastery: Dict[str, float],
                  unlocked: set) -> Optional[str]:
        """
        Exploration: prefer unseen or rarely attempted concepts among unlocked ones.
        """
        unseen  = [c for c in concepts if c not in mastery and c in unlocked]
        if unseen:
            return random.choice(unseen)

        # Pick lowest-mastery unlocked concept as fallback exploration
        unlocked_list = [(c, mastery.get(c, 0.0)) for c in concepts if c in unlocked]
        if not unlocked_list:
            return None
        unlocked_list.sort(key=lambda x: x[1])
        # Pick randomly from bottom 30%
        bottom_n = max(1, len(unlocked_list) // 3)
        return random.choice(unlocked_list[:bottom_n])[0]

    # ── Exploit: pick concept most in need of drilling ─────────────────────

    def _exploit(self, concepts: List[str], mastery: Dict[str, float],
                  skills: dict, unlocked: set) -> Optional[str]:
        """
        Exploitation: pick the unlocked concept with lowest mastery (needs most work).
        """
        scored = []
        for cid in concepts:
            m        = mastery.get(cid, 0.3)
            attempts = skills.get(cid, {}).get("attempts", 0)
            score    = score_concept_for_student(cid, m, attempts, cid in unlocked)
            if score >= 0:
                scored.append((cid, score))

        if not scored:
            return None

        # Softmax sampling — weighted by score (not pure greedy)
        scores = [s for _, s in scored]
        total  = sum(math.exp(s * 3) for s in scores)  # temperature = 3
        probs  = [math.exp(s * 3) / total for s in scores]

        concepts_list = [c for c, _ in scored]
        return random.choices(concepts_list, weights=probs, k=1)[0]

    # ── Pick an actual question row ────────────────────────────────────────

    def _pick_question(self, concept_id: str, subject: str,
                        difficulty: Optional[str],
                        exclude_ids: set) -> Optional[dict]:
        """
        Find a question for the given concept and difficulty that hasn't been seen.
        """
        mask = (self.questions["concept_id"] == concept_id) & \
               (self.questions["subject"] == subject)

        if difficulty:
            mask = mask & (self.questions["difficulty"] == difficulty)

        candidates = self.questions[mask]

        # Exclude already-seen questions
        if exclude_ids:
            candidates = candidates[~candidates.index.astype(str).isin(exclude_ids)]

        if candidates.empty:
            return None

        return candidates.sample(1).iloc[0].to_dict()

    # ── Batch selection (for testing / simulation) ─────────────────────────

    def select_session_questions(self, student_id: str, subject: str,
                                  n: int = 10) -> List[dict]:
        """Select n questions for a full practice session."""
        questions = []
        session_ids = []
        for _ in range(n):
            q = self.select_question(student_id, subject, session_ids)
            if q:
                qid = str(q.get("name", q.get("index", id(q))))
                session_ids.append(qid)
                questions.append(q)
        return questions

    # ── Stats ──────────────────────────────────────────────────────────────

    def get_coverage_stats(self, student_id: str, subject: str) -> dict:
        """How much of the subject's concept graph has been explored."""
        all_concepts = self.dag.get_concepts_by_subject(subject)
        mastery      = self.km.get_mastery_dict(student_id, subject)
        seen         = [c for c in all_concepts if c in mastery]
        mastered     = [c for c in seen if mastery[c] >= 0.8]

        return {
            "total_concepts":    len(all_concepts),
            "concepts_seen":     len(seen),
            "concepts_mastered": len(mastered),
            "coverage_pct":      round(len(seen) / len(all_concepts) * 100, 1) if all_concepts else 0,
            "mastery_pct":       round(len(mastered) / len(all_concepts) * 100, 1) if all_concepts else 0,
        }
