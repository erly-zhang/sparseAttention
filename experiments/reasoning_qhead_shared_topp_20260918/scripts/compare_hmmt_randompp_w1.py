"""Source-backed full-HMMT comparison and separately matched optimized-W1 pilot."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1]).resolve();results_root=root.parent
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
refs={r['idx']:r for r in rows(root/'inputs/hmmt.jsonl')}
sources={}
def metrics(files):
    rr=[]
    for f in sorted(files):
        sources[str(f)]=sha(f);rr+=rows(f)
    assert len(rr)==len({r['idx'] for r in rr})
    for r in rr:
        ref=refs[r['idx']]
        assert (r['input_tokens'],r['input_ids_sha256'])==(ref['input_tokens'],ref['input_ids_sha256'])
    return rr

def record(name,storage,k,mm,grades,ids):
    assert len(mm)==len(ids) and {r['idx'] for r in mm}==set(ids)
    gg=[g for g in grades if g['idx'] in ids]
    assert len(gg)==2*len(ids) and len({(g['idx'],g['run']) for g in gg})==len(gg)
    seconds=sum(m['decode_seconds'] for m in mm);calls=sum(m['decode_forward_calls'] for m in mm)
    selected=sum(m['selected_pairs'] for m in mm);causal=sum(m['causal_pairs'] for m in mm)
    lengths=[t for m in mm for t in m['generated_tokens_per_rollout']]
    return dict(method=name,storage=storage,K=k,questions=len(mm),answers=len(gg),
        correct=sum(g['correct'] for g in gg),accuracy=sum(g['correct'] for g in gg)/len(gg),
        mean_decode_seconds=seconds/len(mm),aggregate_ms_per_forward=1000*seconds/calls,
        mean_output_tokens=statistics.mean(lengths),median_output_tokens=statistics.median(lengths),
        cap_hits=sum(not e for m in mm for e in m['eos_reached']),
        peak_allocated_gib=max(m['peak_memory_allocated'] for m in mm)/2**30,
        selected_pairs=selected,causal_pairs=causal,global_pair_sparsity=1-selected/causal)

assert read(root/'summary.json')['status']=='completed'
full=[];pilot=[];all_ids=set(range(60));pilot_ids={0,1,2}
pp_grades=rows(root/'grades.jsonl')
for cell in read(root/'config.json')['cells']:
    d=root/'cells'/cell['name']
    mm=metrics(list((d/'smoke').glob('*/cache_metrics.jsonl'))+list((d/'formal').glob('shard_*/cache_metrics.jsonl')))
    gg=[g for g in pp_grades if g['cell']==cell['name']]
    full.append(record('official_random_pp','compact_native64',cell['K'],mm,gg,all_ids))
    pilot.append(record('official_random_pp','compact_native64',cell['K'],[m for m in mm if m['idx'] in pilot_ids],gg,pilot_ids))

dense=results_root/'hmmt_dense_and_timing_diagnosis_20261008/formal'
assert read(dense/'summary.json')['questions']==60
dense_mm=metrics(dense.glob('shard_*/cache_metrics.jsonl'));dense_grades=rows(dense/'grades.jsonl')
full.append(record('Dense','full_dense',None,dense_mm,dense_grades,all_ids))
pilot.append(record('Dense','full_dense',None,[m for m in dense_mm if m['idx'] in pilot_ids],dense_grades,pilot_ids))

historical=results_root/'hmmt_k1024_k2048_full_compact_k4096_full_20261001'
for k in [1024,2048]:
    for storage in ['full','compact']:
        d=historical/'cells'/f'hmmt_k{k}_grid_{storage}_w1_topk_protected'/'formal'
        assert read(d/'summary.json')['questions']==60
        mm=metrics(d.glob('shard_*/cache_metrics.jsonl'));gg=rows(d/'grades.jsonl')
        full.append(record('W1 original','full' if storage=='full' else 'compact64',k,mm,gg,all_ids))

opt=results_root/'full_kv_w1_implementation_optimization_20261008'
assert read(opt/'paired_summary.json')['generated_token_ids']=='passed/exact'
opt_grades=read(opt/'grades.json')
for implementation in ['baseline','optimized']:
    mm=metrics((opt/'pairs').glob(f'input_*/{implementation}/cache_metrics.jsonl'))
    gg=[g for g in opt_grades if g['implementation']==implementation]
    pilot.append(record('W1 '+implementation,'full',2048,mm,gg,pilot_ids))

proof=dict(status='completed_comparison',input_alignment='passed/full_within_each_scope',
    full60=full,matched_first3=pilot,source_sha256=sources,
    caveats=['Optimized W1 is a three-input pilot; it must not be shown as a full60 result.',
             'W1 baseline/optimized are paired on the same GPU for each input; Dense and random_pp use other A100 nodes.',
             'K includes prompt+recent64 in these HMMT runs. Equal K does not equalize physical storage, head sharing, trigger phase or realized pair sparsity.',
             'Whole-generation timing mixes output-length changes and execution cost. All rows use two sampled answers/input, not pass@2.',
             'Dense uses standard DynamicCache and FlashAttention2; it is not a preallocated Dense implementation.'])
(root/'analysis/comparison.json').write_text(json.dumps(proof,indent=2)+'\n')
for name,rr in [('full60',full),('matched_first3',pilot)]:
    with (root/'analysis'/f'comparison_{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
text=['HMMT official random_pp, Dense and shared W1 comparison','',*proof['caveats'],'',
      'Full60 questions (120 sampled answers per row):',*map(json.dumps,full),'',
      'Matched inputs0,1,2 only (6 sampled answers per row):',*map(json.dumps,pilot)]
(root/'analysis/comparison_en.txt').write_text('\n'.join(text)+'\n')
print(json.dumps(proof),flush=True)
