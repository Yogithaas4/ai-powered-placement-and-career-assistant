import json
from models.confit_v2_fixed import ConFitV2Engine

with open('data/processed/resume_full/Frontend_Developer_Resume.json', 'r', encoding='utf-8') as f:
    preprocessed = json.load(f)

print('[+] Loaded:', preprocessed['filename'])

engine = ConFitV2Engine(hre_mode='local', local_model='mistral')
recs = engine.recommend(preprocessed, top_k=15, stage1_n_results=15)

print()
print('=' * 60)
print(f'  TOP {len(recs)} RECOMMENDATIONS')
print('=' * 60)
for r in recs:
    line = "  {rank:2}. [{score:.4f}] {title} @ {company}".format(
        rank=r['rank'], score=r['score'], title=r['title'], company=r['company']
    )
    print(line)