"""
experiments/
------------
Phase 1 model-comparison harness for the research paper.

This package is intentionally SEPARATE from `core/` and `api/`. Those still
run the SQLite + hand-authored-CN-DAG prototype that the live demo/API use.
`experiments/` is the offline research harness: it reads the real Postgres
`questions_resolved` view and the real 2,539-concept DAG built in Phase 0,
runs every selection algorithm against identical simulated students, and
logs results into `experiment_runs` / `experiment_metrics`.

Nothing in `core/` or `api/` is modified by this package. When an algorithm
here is finalized, wiring it into the live API is a separate, later step.
"""
