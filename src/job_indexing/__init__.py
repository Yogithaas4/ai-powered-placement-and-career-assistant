"""Job CSV loading, embedding, and Chroma job index persistence."""

from .job_ingestion import JobIngestionEngine, quick_ingest

__all__ = ["JobIngestionEngine", "quick_ingest"]
