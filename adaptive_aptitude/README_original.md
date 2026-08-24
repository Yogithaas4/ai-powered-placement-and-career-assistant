# Adaptive Test Prep Platform

## Architecture

```
questions-enriched.csv
        │
        ▼
  ConceptDAG (concept_dag.py)
  - Nodes: topic + subtopic per subject
  - Edges: prerequisite relationships
        │
        ▼
  StudentKnowledgeModel (knowledge_model.py)
  - BKT update per concept after each answer
  - EMA update for fast smoothing
  - Combined skill_score = 0.6×BKT + 0.4×EMA
  - Stored in SQLite: student_skill + interaction_log
        │
        ▼
  QuestionSelector (question_selector.py)
  - epsilon-greedy exploration/exploitation
  - DAG-aware: only unlocks concepts whose prerequisites are mastered
  - Difficulty matched to current mastery band
  - Recency filter: avoids repeating recent questions
        │
        ▼
  FastAPI Backend (api/main.py)
  - POST /session/start
  - GET  /question/next
  - POST /question/answer
  - GET  /student/{id}/summary
  - GET  /student/{id}/skills
  - GET  /student/{id}/history
  - GET  /dag/{subject}
```

## Database Schema

```sql
student_skill (
    student_id, concept_id,
    subject, topic, subtopic,
    bkt_score,     -- P(learned) from BKT
    ema_score,     -- smoothed accuracy from EMA
    skill_score,   -- combined: 0.6*BKT + 0.4*EMA
    attempts, correct_count, last_updated
)

interaction_log (
    log_id, student_id, question_id, concept_id,
    subject, topic, subtopic, difficulty,
    correct, time_taken_sec,
    bkt_before, bkt_after, ema_before, ema_after,
    timestamp
)
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the demo simulation
python demo.py

# Start the API server
cp /path/to/questions-enriched.csv .
uvicorn api.main:app --reload --port 8000

# API docs at: http://localhost:8000/docs
```

## Using ASSISTments Data for BKT Bootstrapping

```python
from data.dataset_loader import load_assistments, estimate_bkt_params_from_data, save_bkt_params

df = load_assistments("skill_builder_data_corrected.csv")
params = estimate_bkt_params_from_data(df)
save_bkt_params(params, "bkt_params.json")
# Then load these params into SUBJECT_BKT_PARAMS in knowledge_model.py
```

## BKT Math (quick reference)

After each answer:
```
P(correct | known)   = P(known) × (1 - p_slip)  +  (1-P(known)) × p_guess
P(known | correct)   = P(known) × (1 - p_slip)  /  P(correct)
P(known_next)        = P(known | obs)  +  (1 - P(known | obs)) × p_transit
```

Mastery thresholds:
- skill_score ≥ 0.80 → Mastered ✅
- skill_score ≥ 0.60 → Proficient 🟡
- skill_score ≥ 0.40 → Developing 🟠
- skill_score <  0.40 → Needs Practice 🔴

## Exploration vs Exploitation

epsilon = 0.20 means:
- 80% of the time: drill the weakest unlocked concept (exploit)
- 20% of the time: probe an unseen/new concept (explore)

Adjust epsilon in QuestionSelector.__init__() to control this tradeoff.
