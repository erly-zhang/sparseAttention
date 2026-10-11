"""Validate complete random-mask outputs, exact pair budgets, inputs and grade MATH."""
import argparse,hashlib,json,statistics,sys
from fractions import Fraction
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--phase',choices=['smoke','formal'],required=True);a=p.parse_args();root=a.root.resolve()
for rel,digest in json.loads((root/'manifest.json').read_text()).items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
cfg=json.loads((root/'config.json').read_text());expected={r['idx']:r for r in map(json.loads,(root/'inputs.jsonl').read_text().splitlines())}
assert set(expected)==set(range(500))
ids=set(cfg['smoke_sample_ids']) if a.phase=='smoke' else set(expected)
assert a.phase=='formal'
plan=json.loads((root/'repair_plan.json').read_text());old=Path(plan['old_root'])
dirs=[old/f'random/shard_{i}' for i in range(8)]+[root/f'oom_replay/worker_{i}' for i in range(2)]+[root/f'repair/worker_{i}' for i in range(4)]
(root/'random').mkdir(exist_ok=True)
assert json.loads((root/'smoke_regression.json').read_text())['status']=='passed'
provenance={}
for d in dirs:
    if not d.is_relative_to(old):
        assert json.loads((d/'validation.json').read_text())['status']=='passed'
    for pth in d.rglob('*.jsonl'):
        provenance[str(pth)]=hashlib.sha256(pth.read_bytes()).hexdigest()
(root/'collection_input_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
metrics={};comps={};hardware=[]
g=cfg['model_geometry'];factor=g['num_attention_heads']*g['num_hidden_layers'];budget=cfg['reasoning_budget']
for shard,d in enumerate(dirs):
    if not d.is_relative_to(old):assert json.loads((d/'exit.json').read_text())['exit_code']==0,str(d)
    hardware.append(json.loads((d/'hardware.json').read_text()))
    for r in map(json.loads,(d/'cache_metrics.jsonl').read_text().splitlines()):
        idx=r['idx'];assert idx in ids and idx not in metrics
        assert idx in (set(plan['completed_ids']) if d.is_relative_to(old) else set(plan['missing_ids']))
        assert r['selection_strategy']=='fixed_reasoning_budget_recent_plus_gpu_uniform_topk' and r['random_keep_ratio'] is None
        assert r['reasoning_budget']==budget==1024 and r['prompt_outside_budget']
        assert r['recent_window']==64 and r['recent_inside_budget'] and r['gather_backend']=='triton_fused_reusable'
        assert r['query_window'] is None and r['selection_p'] is None
        assert r['mask_seed']==42 and r['representative_seed']==0
        assert r['cache_semantics']=='full_history_temporary_mask'
        assert r['input_tokens']==expected[idx]['input_tokens'] and r['input_ids_sha256']==expected[idx]['input_ids_sha256']
        P=r['input_tokens'];generated=r['generated_tokens_per_rollout'];assert len(generated)==len(r['eos_reached'])==2
        for count,eos in zip(generated,r['eos_reached']):
            assert 1<=count<=cfg['max_tokens']-P
            if not eos:assert count==cfg['max_tokens']-P
        prefill=2*factor*P*(P+1)//2
        causal=factor*sum((P+t-1)*(P+t)//2 for t in generated)
        decode_selected=factor*sum((t-1)*P+sum(min(j,budget) for j in range(1,t)) for t in generated)
        assert r['prefill_selected_pairs']==r['prefill_causal_pairs']==prefill
        assert r['causal_pairs']==causal and r['decode_causal_pairs']==causal-prefill
        assert r['decode_selected_pairs']==decode_selected and r['selected_pairs']==prefill+decode_selected
        assert r['decode_query_tokens']==sum(generated)-2
        assert r['decode_forward_calls']==max(generated)-1
        assert r['selection_events']==r['decode_forward_calls']*g['num_hidden_layers']
        assert len(r['final_cache_seqlens'])==len(r['final_cumulative_length'])==g['num_hidden_layers']
        for lengths,cumulative in zip(r['final_cache_seqlens'],r['final_cumulative_length']):
            assert cumulative==P+max(generated)-1 and all(n==cumulative for n in lengths)
        assert r['prefill_seconds']>0 and r['decode_seconds']>0
        metrics[idx]=r
    for run in (0,1):
        files=list((d/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1
        for r in map(json.loads,files[0].read_text().splitlines()):
            idx=r['idx'];key=(run,idx);assert idx in ids and key not in comps
            assert idx in (set(plan['completed_ids']) if d.is_relative_to(old) else set(plan['missing_ids']))
            assert r['completion'].startswith(expected[idx]['prompt_text']);comps[key]=r['completion']
assert set(metrics)==ids and set(comps)=={(r,i) for r in (0,1) for i in ids}
sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
from Utils.parser import extract_answer
from Utils.grader import check_is_correct
import numpy as np

def stats(xs):return dict(mean=statistics.mean(xs),median=statistics.median(xs),p90=float(np.percentile(xs,90)),total=sum(xs))
grades=[];runs=[]
for run in (0,1):
    rr=[]
    for idx in sorted(ids):
        error=None
        try:prediction=extract_answer(comps[run,idx],'math');correct=int(bool(check_is_correct(prediction,expected[idx]['gold'])))
        except Exception as e:prediction='';correct=0;error=f'{type(e).__name__}: {e}'
        rr.append(dict(run=run,idx=idx,prediction=prediction,correct=correct,grading_error=error,eos_reached=metrics[idx]['eos_reached'][run],generated_tokens=metrics[idx]['generated_tokens_per_rollout'][run]))
    grades.extend(rr);runs.append(dict(run=run,correct=sum(x['correct'] for x in rr),count=len(rr),accuracy=statistics.mean(x['correct'] for x in rr)))
vals=list(metrics.values());sel=sum(x['selected_pairs'] for x in vals);causal=sum(x['causal_pairs'] for x in vals);ds=sum(x['decode_selected_pairs'] for x in vals);dc=sum(x['decode_causal_pairs'] for x in vals);dt=sum(x['decode_seconds'] for x in vals);tokens=sum(x['decode_query_tokens'] for x in vals)
summary=dict(status='passed' if a.phase=='smoke' else 'completed',phase=a.phase,method=cfg['method'],reasoning_budget=budget,prompt_outside_budget=True,random_keep_ratio=None,recent_window=cfg['recent_window'],recent_inside_budget=True,gather_backend=cfg['gather_backend'],query_window=None,selection_p=None,
    questions=len(ids),rollouts=len(grades),input_alignment='passed/full' if a.phase=='formal' else 'passed/3 smoke inputs',
    exact_random_pair_budget='passed',full_history_continuity='passed/final lengths and correctness tests',source_manifest='passed',
    runs=runs,correct=sum(x['correct'] for x in grades),accuracy=statistics.mean(x['correct'] for x in grades),eos_rate=statistics.mean(x['eos_reached'] for x in grades),
    length_limited_rollouts=sum(not x['eos_reached'] for x in grades),generated_tokens=stats([x['generated_tokens'] for x in grades]),
    prefill_seconds_per_question_batch=stats([x['prefill_seconds'] for x in vals]),decode_seconds_per_question_batch=stats([x['decode_seconds'] for x in vals]),batch_wall_seconds=stats([x['batch_wall_seconds'] for x in vals]),
    decode_tokens_per_second=tokens/dt,selected_pairs=sel,causal_pairs=causal,global_pair_sparsity=1-sel/causal,
    decode_selected_pairs=ds,decode_causal_pairs=dc,decode_pair_sparsity=1-ds/dc,peak_memory_allocated=max(x['peak_memory_allocated'] for x in vals),hardware=hardware,
    timing_scope=cfg['timing'],comparison_caveat=cfg['comparison_caveat'])
out=root/'smoke' if a.phase=='smoke' else root/'random'
with (out/'grades.jsonl').open('x') as f:f.writelines(json.dumps(x)+'\n' for x in grades)
with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
if a.phase=='smoke':
    assert json.loads((root/'correctness.json').read_text())['status']=='passed'
    with (root/'smoke_gate.json').open('x') as f:json.dump(dict(status='passed',summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()),f,indent=2)
print(json.dumps(summary,indent=2))

repaired=[r for r in vals if r['idx'] in plan['missing_ids']]
historical=[r for r in vals if r['idx'] in plan['completed_ids']]
assert len(repaired)==71 and len(historical)==429
summary['implementation']='full_kv_memory_repair_v1; 429 historical + 71 repaired outputs'
summary['timing_warning']='Mixed historical/repaired timings; not a fresh 500-question repaired speed measurement.'
summary['stratified_measurements']={label:dict(questions=len(rr),decode_seconds_mean=statistics.mean(r['decode_seconds'] for r in rr),peak_memory_allocated=max(r['peak_memory_allocated'] for r in rr)) for label,rr in [('historical_429',historical),('repaired_71',repaired)]}
summary['all_repaired_generations_released_cache']=all(r['memory_allocated_after_generation']<=r['memory_allocated_before_generation']+256*1024**2 for r in repaired)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'question_metrics.jsonl').write_text(''.join(json.dumps(dict(r,source='repaired' if r['idx'] in plan['missing_ids'] else 'historical'))+'\n' for r in sorted(vals,key=lambda x:x['idx'])))
(root/'collection_status.json').write_text(json.dumps(dict(status='completed',questions=500,rollouts=1000,input_alignment='passed/full',correct=summary['correct'],accuracy=summary['accuracy']),indent=2)+'\n')
(out/'REPORT.md').write_text(f"# Full-KV reasoning1024 memory repair\n\n500/500 questions; 1000 rollouts; input_alignment=passed/full. Correct: {summary['correct']}/1000 ({summary['accuracy']:.2%}).\n\n429 historical completed questions plus 71 repaired questions; no duplicates or missing IDs. Exact pair-budget checks passed for all samples.\n\nRepaired subset peak allocated: {max(r['peak_memory_allocated'] for r in repaired)/2**30:.3f} GiB. Historical and repaired timing must be reported separately; this is not a fresh full500 repaired-speed benchmark.\n")
