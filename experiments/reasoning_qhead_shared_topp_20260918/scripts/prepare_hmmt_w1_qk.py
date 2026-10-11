"""Freeze the two authorized K1024 raw-QK W1 experiments and gated jobs."""
import ast
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E=Path(__file__).resolve().parents[1]
base=E/'results/hmmt_compact_interval500_k1024_k2048_k4096_20261007'
root=E/'results/hmmt_w1_raw_qk_k1024_full_compact500_20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def edit(p,old,new):
    s=p.read_text();assert old in s,(p,old);p.write_text(s.replace(old,new))

resume=root.exists() and '--resume-incomplete' in sys.argv
assert not root.exists() or (resume and not (root/'manifest.json').exists() and not (root/'jobs.json').exists()), 'Never overwrite a sealed/submitted run'
if not resume:root.mkdir()
def copy_folder(folder):
    # Independent snapshot files need no timestamps/xattrs; avoid slow serial
    # copy2 metadata round trips on the shared filesystem.
    files=[Path(s) for s in subprocess.check_output(['rg','--files','--hidden','--no-ignore',str(base/folder)],text=True).splitlines()]
    files=[p for p in files if '__pycache__' not in p.parts and p.suffix!='.pyc']
    for parent in sorted({(root/folder/p.relative_to(base/folder)).parent for p in files}):
        parent.mkdir(parents=True,exist_ok=True)
    def copy(p):
        with p.open('rb') as src,(root/folder/p.relative_to(base/folder)).open('xb') as dst:
            shutil.copyfileobj(src,dst)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy,files))
if not resume:
    for folder in ('source','inputs','data'):copy_folder(folder)
for name in ('model_assets_sha256.json','prompt_protection_stats.json'):
    shutil.copy2(base/name,root/name)
(root/'logs').mkdir(exist_ok=True);(root/'reference_probability').mkdir(exist_ok=True)
# Keep the established harness, including pair/input alignment telemetry.
for rel in ('ra_harness/kvcompress/engine/selector_grid_cache.py',
            'ra_harness/kvcompress/engine/full_w1_optimized_cache.py',
            'scripts/run_w1_qk.py','tests/test_w1_qk_grid.py','tests/test_full_w1_qk.py'):
    shutil.copy2(E/rel,root/'source'/rel)
cfg=read(base/'config.json')
cfg.update(created=datetime.datetime.now().astimezone().isoformat(),
           experiment='hmmt_w1_raw_qk_k1024_full_compact500_v1',
           scope_authorization='User requested K1024 W1 raw-QK full KV and interval500 compact on2026-10-09; inherit prompt+recent64 protection.',
           max_concurrent_gpus=2,score_mode='qk',cells=[],
           caveats=[
               'Accuracy is the mean of two sampled outputs per question, not pass@2.',
               'K1024 includes prompt and recent64; attention softmax is retained.',
               'Raw BF16 QK can produce different masks from BF16-scaled FP32 probabilities due to rounding/underflow ties.',
               'Raw-QK ties use stable earlier chronological positions.',
               'All KV heads share positions, not K/V values; one representative Q head per layer/event.',
               'Full KV uses the optimized stable-sort implementation; historical full-KV timing also differs in implementation.',
               'Compact500 attends all surviving KV between events, up to K+500; full KV gathers K positions before every decode attention.'])
for storage in ('compact','full'):
    method='grid_compact500_w1_topk_protected' if storage=='compact' else 'grid_full_w1_topk_protected'
    cell=dict(name=f'hmmt_k1024_w1_raw_qk_{storage}_protected',task='hmmt',count=60,K=1024,
              shards=4,method=method,storage=storage,scorer='w1',selector='topk',protection='protected',
              window=1,compaction_interval=500 if storage=='compact' else None,score_mode='qk')
    cfg['cells'].append(cell)
    old=(base/'analysis/completed_cells_20261009_1654' if storage=='compact' else
         E/'results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_full_w1_topk_protected/formal')
    for artifact in ('summary.json','grades.jsonl'):
        source=old/f'hmmt_k1024_grid_compact500_w1_topk_protected_{artifact}' if storage=='compact' else old/artifact
        shutil.copy2(source,root/'reference_probability'/f"{cell['name']}_{artifact}")
write(root/'config.json',cfg)
write(root/'smoke_work.json',[dict(cell=i,shard=0,num_shards=1) for i in range(2)])
write(root/'formal_work.json',[dict(cell=i,shard=s,num_shards=4) for s in range(4) for i in range(2)])
gen=root/'source/ra_harness/kvcompress/harness/generation_utils.py'
edit(gen,"        layers=current_cache.layers;first=layers[0]",
     "        layers=current_cache.layers;first=layers[0]\n        metrics.update(w1_score_mode=first.score_mode, w1_implementation=first.implementation, w1_rank_backend=first.rank_backend)")
script=root/'source/scripts/selector_grid.py'
edit(script,"str(root/'source/ra_harness/kvcompress/harness/eval_hf.py')","str(root/'source/scripts/run_w1_qk.py')")
edit(script,"        assert r['compaction_interval']==cell['compaction_interval']==r['eviction_interval']==500\n        assert r['compaction_phase']=='after_attention'\n        assert r['cache_semantics']=='persistent_eviction'\n        for row in r['final_cache_seqlens']:assert all(n<=max(P,cell['K']+500) for n in row)",
     "        assert r['w1_score_mode']=='qk' and r['w1_rank_backend']=='stable_sort'\n        assert r['w1_implementation']==('shared_w1_raw_qk_v1' if cell['storage']=='compact' else 'full_w1_fixed_ranges_batched_qk_v1')\n        if cell['storage']=='compact':\n            assert r['compaction_interval']==cell['compaction_interval']==r['eviction_interval']==500\n            assert r['compaction_phase']=='after_attention' and r['cache_semantics']=='persistent_eviction'\n            for row in r['final_cache_seqlens']:assert all(n<=max(P,cell['K']+500) for n in row)\n        else:\n            assert r['eviction_interval']==1 and r['compaction_interval'] is None\n            assert r['cache_semantics']=='full_history_temporary_mask'")
a=script.read_text();start=a.index("        for cell in cfg['cells']:\n            name=cell['name'];candidate_path=grades(cell)")
end=a.index("        write(root/'paired_topk_comparisons.json'",start)
a=a[:start]+"""        for cell in cfg['cells']:
            name=cell['name'];candidate_path=grades(cell)
            reference=root/'reference_probability'/f'{name}_grades.jsonl'
            comparisons.append(compare(name+'_probability',name,reference,candidate_path,'raw_QK_vs_probability_matched_K_storage_protection'))
"""+a[end:]
script.write_text(a.replace('Frozen12-cell HMMT interval500 grid','Frozen2-cell HMMT raw-QK W1 K1024 grid'))
checker=root/'source/scripts/check_budgets.py'
edit(checker,"('test_selector_grid','test_hmmt_budgets','test_hmmt_interval500')","('test_full_w1_qk','test_w1_qk_grid')")
edit(checker,"Legacy64 regression plus interval500 at K1024/2048/4096 x Random/W1 x protection/none; two500 events, bounded capacity, current-query score, GQA numeric oracle, chronological gather, protection, batch shrink, private RNG/replay",
     "Raw QK: W1 full/compact500, real K1024 and GQA geometry, no selector softmax, signed stable ties, protection, short history, absolute positions, two compactions, independent FP32 attention reference and batch shrink; unchanged probability default")
for stage in ('correctness','smoke','gate','formal','collect'):
    p=root/'source/scripts'/f'{stage}.sbatch'
    t=p.read_text().replace('hmmt-c500-','hmmt-qk-')
    if stage=='smoke':t=t.replace('--array=0-11%4','--array=0-1%1').replace('--time=2-00:00:00','--time=04:00:00')
    if stage=='formal':t=t.replace('--array=0-47%4','--array=0-7%2').replace('--time=2-00:00:00','--time=16:00:00')
    p.write_text(t)
write(root/'provenance.json',dict(harness_baseline_root=str(base),
      input_reference_sha256=sha(root/'inputs/hmmt.jsonl'),data_sha256=sha(root/'data/hmmt/test.jsonl'),
      implementation_sources={r:sha(E/r) for r in ('ra_harness/kvcompress/engine/selector_grid_cache.py','ra_harness/kvcompress/engine/full_w1_optimized_cache.py','scripts/run_w1_qk.py')},
      score='Current representative post-RoPE Q dot corresponding GQA K in BF16, no division, softmax, or mean in selection; stable ties by original chronological position.'))
(root/'README.md').write_text('''# HMMT W1 raw-QK selection: K1024

Two independent groups: full KV (optimized W1) and physical KV compaction every500 decode forwards.
Each uses the same canonical60 inputs, two sampled rollouts/question (120 scored answers),
Qwen3-4B, BF16, batch2, seed42, representative seed0, temperature0.6, generation top-p0.95,
total context+generation length cap32768. K includes the protected prompt and recent64.

Each layer/event selects one representative Q head with the existing private deterministic RNG.
It ranks unprotected positions using current post-RoPE QK logits, with stable chronological ties.
All heads share selected positions while preserving their own K/V values. Attention normalization,
RoPE absolute positions, GQA mapping, padding and generation sampling remain unchanged.
Raw BF16 scores can differ in rank from the previous scaled BF16 / FP32-softmax scores under ties.

Full KV retains all historical KV and gathers at most1024 positions before every decode attention.
Compact500 attends all surviving KV plus the new token; after attention at steps divisible by500,
when cache length exceeds1024 it keeps1024 positions and permanently removes the remainder.
Maximum compact attention length is1524. It captures the current query at each compaction event.

GPU correctness precedes complete generation on the same three inputs0,1,2 for both groups.
A CPU gate validates inputs, actual pair counts, score mode and all six outputs/group before
eight formal shards may start. Formal inference repeats all60 inputs in fresh directories.
At most two additional scheduler-assigned A10080GB GPUs; existing experiments are untouched.
The collector checks60 unique inputs/120 answers, hashes/token lengths, EOS/length limits,
analytic pair accounting, bounded grading, timing/memory/sparsity and paired old-probability accuracy.
Source/config/data/model assets are immutable and checked by all jobs.
''')
for p in (root/'source').rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(root/'static_validation.json',dict(status='passed',cells=2,formal_shards=8,smoke_ids=[0,1,2],
      inherited_generation_settings=True,python_ast='passed',source_snapshot='frozen',
      input_reference_sha256=sha(root/'inputs/hmmt.jsonl')))
files=[root/p for p in ('config.json','smoke_work.json','formal_work.json','model_assets_sha256.json','inputs/hmmt.jsonl','data/hmmt/test.jsonl','provenance.json')]
files += [p for folder in ('source','reference_probability') for p in (root/folder).rglob('*') if p.is_file()]
write(root/'manifest.json',{str(p.relative_to(root)):sha(p) for p in sorted(files)})
print(json.dumps(dict(root=str(root),cells=2,smoke=2,formal=8,static_validation='passed')),flush=True)
