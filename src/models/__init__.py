"""
src/models/
-----------
Three recommendation model implementations for benchmarking.

Models
------
confit_v2            ConFit v2 — Hypothetical Resume Embedding + Runner-Up Hard-Negative Mining
                     Paper: https://arxiv.org/abs/2502.12361

colbert_matcher      ColBERT  — Contextualized Late Interaction over BERT (MaxSim)
                     Paper: https://arxiv.org/abs/2004.12832

cross_encoder_matcher  Cross-Encoder — Full joint attention (bi-encoder recall + CE rerank)

Shared Interface
----------------
Each engine exposes:
    engine.recommend(resume_text, top_k=20) -> List[Dict]
    engine.export(recommendations, resume_name, output_path) -> str

Each module also exposes:
    evaluate(recommendations, relevant_job_titles, k_values) -> Dict

Usage
-----
    from models.confit_v2 import ConFitV2Engine
    from models.colbert_matcher import ColBERTEngine
    from models.cross_encoder_matcher import CrossEncoderEngine
    from models.evaluate_models import run_comparison
"""
