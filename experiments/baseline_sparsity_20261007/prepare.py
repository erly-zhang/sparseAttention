import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'representative_patterns_fast_retry03_20261001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
m=json.loads((ORIGINAL/'inputs/manifest.json').read_text());c=json.loads((ORIGINAL/'config.json').read_text())
assert len(m['evaluation'])==200
jobs=[]
for method in ['flexprefill','minference']:
    bins=[{'method':method,'keys':[],'load':0} for _ in range(40)]
    for item in sorted(m['evaluation'],key=lambda x:-x['input_tokens']**2):
        b=min((x for x in bins if len(x['keys'])<5),key=lambda x:x['load'])
        b['keys'].append(item['key']);b['load']+=item['input_tokens']**2
    jobs+=bins
for i,p in enumerate(jobs):p['worker']=i
save(HERE/'work-plan.json',jobs)
save(HERE/'config.json',dict(model=c['model'],baselines=c['baselines'],samples_per_method=200,samples_per_task=50,
    layers=list(range(32)),heads=32,input_scope='full untruncated original inputs',prefill_only=True,decode_replayed=False,
    definition='1 - unique valid causal token pairs / (N*(N+1)/2); equal means over heads, layers, samples',
    workers=80,max_concurrent=80,gpu='h100-80',cpus=8,memory_gb=128,walltime_hours=4,save_progress_seconds=60,
    automatic_pdf_update_after_independent_audit=True))
paths=list(HERE.glob('*.py'))+list(HERE.glob('*.sbatch'))+[HERE/'config.json',HERE/'work-plan.json',ORIGINAL/'config.json',ORIGINAL/'source-lock.json',ORIGINAL/'inputs/manifest.json']
save(HERE/'source-lock.json',{str(p):sha(p) for p in paths})
save(HERE/'approval.json',dict(scope='baseline sparsity replay and update existing English PDF',user_message='FlexPrefill MInference 重新记录系数度，然后将结果补到之前的pdf中',
    source_lock_sha256=sha(HERE/'source-lock.json'),config_sha256=sha(HERE/'config.json'),authorized=True))
print('PREPARED',len(jobs),'workers; 400 full-prefill observations')
