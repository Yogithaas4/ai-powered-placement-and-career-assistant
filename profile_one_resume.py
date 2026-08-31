# profile_one_resume.py — repo root
import time
from resume_processing.step1_parser import parse_file
from resume_processing.step2_segmentation import segment_resume
from resume_processing.step3_ner import extract_all_entities
from resume_processing.step4_embeddings import build_embeddings
from models.confit_v2_fixed import ConFitV2Engine
from models.colbert_matcher_fixed import ColBERTEngine
from models.cross_encoder_matcher_fixed import CrossEncoderEngine

FILE_PATH = r"C:\Users\Student1\Downloads\archive\data\data\ACCOUNTANT\12802330.pdf"

t0 = time.time()
raw_text = parse_file(FILE_PATH)
t1 = time.time()
print(f"parse_file: {t1-t0:.2f}s")

sections = segment_resume(raw_text)
t2 = time.time()
print(f"segment_resume: {t2-t1:.2f}s")

entities = extract_all_entities(raw_text, sections)
t3 = time.time()
print(f"extract_all_entities (NER): {t3-t2:.2f}s")

embeddings = build_embeddings(entities, sections)
t4 = time.time()
print(f"build_embeddings: {t4-t3:.2f}s")

preprocessed = {
    "filename": "12802330.pdf", "file_type": "pdf", "raw_text": raw_text,
    "sections": sections, "entities": entities, "embeddings": embeddings,
}

print("\n--- Loading engines (one-time cost, not per-resume) ---")
t5 = time.time()
confit = ConFitV2Engine(hre_mode="rule")
colbert = ColBERTEngine()
cross_encoder = CrossEncoderEngine()
t6 = time.time()
print(f"Engine loading: {t6-t5:.2f}s")

print("\n--- Per-matcher recommend() timing ---")
t7 = time.time()
confit.recommend(preprocessed, top_k=15, stage1_n_results=60)
t8 = time.time()
print(f"ConFit v2 recommend: {t8-t7:.2f}s")

colbert.recommend(preprocessed, top_k=15, stage1_n_results=60)
t9 = time.time()
print(f"ColBERT recommend: {t9-t8:.2f}s")

cross_encoder.recommend(preprocessed, top_k=15, stage1_n_results=60)
t10 = time.time()
print(f"CrossEncoder recommend: {t10-t9:.2f}s")

print(f"\nTOTAL per-resume (excluding engine loading): {(t4-t0) + (t10-t7):.2f}s")