"""Validate and grade the complete W1 recent three-question smoke; compare same W16/W64 inputs."""
import hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();out=root/'smoke';cfg=json.loads((root/'config.json').read_text());ids={0,1,2}
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
assert json.loads((root/'correctness.json').read_text())['status']=='passed'
assert json.loads((out/'exit.json').read_text())['exit_code']==0
expected={r['idx']:r for r in map(json.loads,(root/'inputs.jsonl').read_text().splitlines())}
g=cfg['model_geometry'];factor=g['num_hidden_layers']*g['num_attention_heads'];metrics={};comps={}
for r in map(json.loads,(out/'cache_metrics.jsonl').read_text().splitlines()):
 idx=r['idx'];assert idx in ids and idx not in metrics
 assert r['query_window']==1 and r['selection_p']==.95 and r['cache_semantics']=='full_history_temporary_mask'
 assert r['selection_strategy']=='w1_conditional_topp_union_recent' and r['recent_window']==64
 assert r['recent_inside_budget']==False and r['random_keep_ratio'] is None
 assert r['decode_selected_pairs']==r['decode_topp_core_selected_pairs']+r['decode_recent_added_pairs']
 P=r['input_tokens'];assert (P,r['input_ids_sha256'])==(expected[idx]['input_tokens'],expected[idx]['input_ids_sha256'])
 lengths=r['generated_tokens_per_rollout'];assert len(lengths)==len(r['eos_reached'])==2
 for count,eos in zip(lengths,r['eos_reached']):
  assert 1<=count<=cfg['max_tokens']-P
  if not eos:assert count==cfg['max_tokens']-P
 assert len(r['final_cache_seqlens'])==len(r['final_cumulative_length'])==g['num_hidden_layers']
 for cache_lens,cum in zip(r['final_cache_seqlens'],r['final_cumulative_length']):assert cum==P+max(lengths)-1 and all(x==cum for x in cache_lens)
 assert r['prefill_selected_pairs']==r['prefill_causal_pairs']==2*factor*P*(P+1)//2
 assert r['causal_pairs']==factor*sum((P+n-1)*(P+n)//2 for n in lengths)
 assert r['selected_pairs']==r['prefill_selected_pairs']+r['decode_selected_pairs']
 assert r['causal_pairs']==r['prefill_causal_pairs']+r['decode_causal_pairs']
 assert 0<r['decode_selected_pairs']<=r['decode_causal_pairs']
 assert r['decode_query_tokens']==sum(lengths)-2 and r['decode_forward_calls']==max(lengths)-1
 assert r['selection_events']==r['decode_forward_calls']*g['num_hidden_layers']
 metrics[idx]=r
for run in (0,1):
 files=list((out/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1
 for r in map(json.loads,files[0].read_text().splitlines()):
  idx=r['idx'];assert idx in ids and (run,idx) not in comps and r['completion'].startswith(expected[idx]['prompt_text']);comps[run,idx]=r['completion']
assert set(metrics)==ids and set(comps)=={(run,idx) for run in (0,1) for idx in ids}
sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
from Utils.parser import extract_answer
from Utils.grader import check_is_correct
grades=[]
for (run,idx),text in sorted(comps.items()):
 err=None
 try:pred=extract_answer(text,'math');correct=int(bool(check_is_correct(pred,expected[idx]['gold'])))
 except Exception as exc:pred='';correct=0;err=f'{type(exc).__name__}: {exc}'
 grades.append(dict(run=run,idx=idx,prediction=pred,correct=correct,grading_error=err,eos_reached=metrics[idx]['eos_reached'][run],generated_tokens=metrics[idx]['generated_tokens_per_rollout'][run]))
def stats(xs):return dict(mean=statistics.mean(xs),median=statistics.median(xs),p90=float(np.percentile(xs,90)),total=sum(xs))
vals=list(metrics.values());sel=sum(x['selected_pairs'] for x in vals);causal=sum(x['causal_pairs'] for x in vals);ds=sum(x['decode_selected_pairs'] for x in vals);dc=sum(x['decode_causal_pairs'] for x in vals);dt=sum(x['decode_seconds'] for x in vals)
summary=dict(status='passed',questions=3,rollouts=6,query_window=1,selection_p=.95,correct=sum(x['correct'] for x in grades),accuracy=statistics.mean(x['correct'] for x in grades),eos_count=sum(x['eos_reached'] for x in grades),length_limited_rollouts=sum(not x['eos_reached'] for x in grades),grading_errors=sum(x['grading_error'] is not None for x in grades),input_alignment='passed/3 canonical inputs',pair_count_crosscheck='passed',full_history_continuity='passed',generated_tokens=stats([x['generated_tokens'] for x in grades]),prefill_seconds=stats([x['prefill_seconds'] for x in vals]),decode_seconds=stats([x['decode_seconds'] for x in vals]),batch_wall_seconds=stats([x['batch_wall_seconds'] for x in vals]),decode_tokens_per_second=sum(x['decode_query_tokens'] for x in vals)/dt,selected_pairs=sel,causal_pairs=causal,global_pair_sparsity=1-sel/causal,decode_selected_pairs=ds,decode_causal_pairs=dc,decode_pair_sparsity=1-ds/dc,peak_memory_allocated=max(x['peak_memory_allocated'] for x in vals),hardware=json.loads((out/'hardware.json').read_text()),comparison={})
reference=Path(cfg['reference_root'])
for w in (16,64):
 rr=[]
 for p in (reference/f'w{w}').glob('shard_*/cache_metrics.jsonl'):
  for x in map(json.loads,p.read_text().splitlines()):
   if x['idx'] in ids:assert x['input_ids_sha256']==metrics[x['idx']]['input_ids_sha256'];rr.append(x)
 assert len(rr)==3
 summary['comparison'][str(w)]=dict(mean_decode_seconds=statistics.mean(x['decode_seconds'] for x in rr),decode_tokens_per_second=sum(x['decode_query_tokens'] for x in rr)/sum(x['decode_seconds'] for x in rr),global_pair_sparsity=1-sum(x['selected_pairs'] for x in rr)/sum(x['causal_pairs'] for x in rr),generated_lengths={x['idx']:x['generated_tokens_per_rollout'] for x in rr})
(out/'grades.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in grades));(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'smoke_gate.json').write_text(json.dumps(dict(status='passed',summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()))+'\n');print(json.dumps(summary,indent=2))

summary.update(method=cfg['method'],recent_window=64,recent_semantics=cfg['recent_semantics'],
    decode_topp_core_selected_pairs=sum(x['decode_topp_core_selected_pairs'] for x in vals),
    decode_recent_added_pairs=sum(x['decode_recent_added_pairs'] for x in vals))
for name,rel in [('W1','full_kv_w1_p095_smoke_20260924'),('Random40_recent64','full_kv_random_r040_recent64_gpu_smoke_20260924')]:
 path=reference.parent/rel/'smoke/cache_metrics.jsonl'
 rr=list(map(json.loads,path.read_text().splitlines()))
 assert {x['idx'] for x in rr}==ids and len(rr)==3
 for x in rr:assert x['input_ids_sha256']==metrics[x['idx']]['input_ids_sha256']
 summary['comparison'][name]=dict(mean_decode_seconds=statistics.mean(x['decode_seconds'] for x in rr),
    decode_tokens_per_second=sum(x['decode_query_tokens'] for x in rr)/sum(x['decode_seconds'] for x in rr),
    global_pair_sparsity=1-sum(x['selected_pairs'] for x in rr)/sum(x['causal_pairs'] for x in rr),
    generated_lengths={x['idx']:x['generated_tokens_per_rollout'] for x in rr})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'smoke_gate.json').write_text(json.dumps(dict(status='passed',summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()))+'\n')
report=['# W1 Top-p0.95 union recent64 smoke','',
 f"Completed 3 questions / 6 outputs. Correct {summary['correct']}/6, EOS {summary['eos_count']}/6. Input alignment and full-KV/pair checks passed.",'',
 '| Method (same 3 inputs) | Mean decode seconds/question batch | Decode tokens/s | Global pair sparsity |',
 '| --- | ---: | ---: | ---: |',
 f"| W1 Top-p0.95 + recent64 | {summary['decode_seconds']['mean']:.3f} | {summary['decode_tokens_per_second']:.3f} | {summary['global_pair_sparsity']:.2%} |"]
for name,row in summary['comparison'].items():
 report.append(f"| {name} | {row['mean_decode_seconds']:.3f} | {row['decode_tokens_per_second']:.3f} | {row['global_pair_sparsity']:.2%} |")
report+=['','Times cover two concurrent rollouts per question. Top-p on all non-prompt positions, then union with last64; no40% quota or permanent KV eviction. Current-query selector uses the historical representative RNG and scoring. Fused reusable gather supports unequal batch lengths. Timing includes selection/gather/attention and generation bookkeeping; excludes model loading, warmup, diagnostic serialization. A3-question smoke cannot establish full-dataset accuracy or isolate recency from implementation effects.','',
 f"Peak allocated memory: {summary['peak_memory_allocated']/2**30:.3f} GiB. Recent-added pairs: {summary['decode_recent_added_pairs']}; Top-p core decode pairs: {summary['decode_topp_core_selected_pairs']}."]
(out/'REPORT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(summary,indent=2))
