"""Validate aligned official random_pp outputs, then grade and aggregate actual pairs."""
import argparse,csv,datetime,hashlib,json,math,statistics,subprocess,sys
from pathlib import Path
import numpy as np


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def stats(xs):
 return dict(mean=statistics.mean(xs),median=statistics.median(xs),p90=float(np.percentile(xs,90)),
  p95=float(np.percentile(xs,95)),p99=float(np.percentile(xs,99)),min=min(xs),max=max(xs),total=sum(xs))


def retention_trace(prompt,generated):
 """Independent integer simulation of official post-attention eviction timing."""
 stored=prompt;pairs=prompt*(prompt+1)//2;events=0
 for _ in range(generated-1):
  stored+=1;pairs+=stored
  if stored>1024 and stored%64==0:stored=1024;events+=1
 return stored,pairs,events


def analyze(root,smoke=False):
 cfg=json.loads((root/'config.json').read_text());method=cfg['method'];assert method=='official_random_pp'
 raw=[json.loads(s) for s in (root/'inputs.jsonl').read_text().splitlines()]
 assert len(raw)==500 and {x['idx'] for x in raw}==set(range(500))
 expected={x['idx']:x for x in raw};wanted=set(range(3 if smoke else 500))
 dirs=[root/'smoke'] if smoke else [root/f'formal/shard_{i}' for i in range(8)]
 out=root/'smoke' if smoke else root/'formal';out.mkdir(exist_ok=True)
 g=cfg['model_geometry'];factor=g['num_hidden_layers']*g['num_attention_heads'];metrics={};comps={};hardware=[]
 for shard,d in enumerate(dirs):
  assert json.loads((d/'exit.json').read_text())['exit_code']==0,str(d)
  if not smoke:assert json.loads((d/'preflight.json').read_text())['status']=='passed'
  hardware.append(json.loads((d/'hardware.json').read_text()))
  local=set()
  for r in map(json.loads,(d/'cache_metrics.jsonl').read_text().splitlines()):
   idx=r['idx'];assert idx in wanted and idx not in metrics
   assert smoke or idx%8==shard
   assert r['query_window'] is None and r['selection_p'] is None and r['cache_semantics']=='persistent_eviction'
   assert r['selection_strategy']=='official_random_pp' and r['token_budget']==1024 and r['residual_length']==64
   P=r['input_tokens'];assert (P,r['input_ids_sha256'])==(expected[idx]['input_tokens'],expected[idx]['input_ids_sha256'])
   lengths=r['generated_tokens_per_rollout'];assert len(lengths)==len(r['eos_reached'])==2
   for count,eos in zip(lengths,r['eos_reached']):
    assert 1<=count<=cfg['max_tokens']-P
    if not eos:assert count==cfg['max_tokens']-P
   assert len(r['final_cache_seqlens'])==len(r['final_cumulative_length'])==g['num_hidden_layers']
   for lens,cum in zip(r['final_cache_seqlens'],r['final_cumulative_length']):
    expected_cache,_,evictions=retention_trace(P,max(lengths))
    assert cum==P+max(lengths)-1 and len(lens)>=1 and all(x==expected_cache for x in lens)
   assert r['prefill_selected_pairs']==r['prefill_causal_pairs']==factor*P*(P+1)
   assert r['causal_pairs']==factor*sum((P+n-1)*(P+n)//2 for n in lengths)
   assert r['selected_pairs']==r['prefill_selected_pairs']+r['decode_selected_pairs']
   assert r['causal_pairs']==r['prefill_causal_pairs']+r['decode_causal_pairs']
   assert 0<r['decode_selected_pairs']<=r['decode_causal_pairs']
   assert r['decode_query_tokens']==sum(lengths)-2 and r['decode_forward_calls']==max(lengths)-1
   assert r['selection_events']==evictions*g['num_hidden_layers']
   assert r['selected_pairs']==factor*sum(retention_trace(P,n)[1] for n in lengths)
   assert r['max_selected_tokens']<=1088
   for key in ('prefill_seconds','decode_seconds','batch_wall_seconds'):assert math.isfinite(r[key]) and r[key]>0
   # Every decode query protects all P prompt positions. Recent is an exact
   # protected lower bound, not a fixed upper budget in this Top-p method.
   lower=factor*sum(sum(P+1 for t in range(1,n)) for n in lengths)
   assert r['decode_selected_pairs']>=lower
   metrics[idx]=r;local.add(idx)
  assert local==(wanted if smoke else set(range(shard,500,8)))
  for run in (0,1):
   files=list((d/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1,str(d)
   for r in map(json.loads,files[0].read_text().splitlines()):
    idx=r['idx'];assert idx in local and (run,idx) not in comps
    assert r['completion'].startswith(expected[idx]['prompt_text']);comps[run,idx]=r['completion']
 assert set(metrics)==wanted and set(comps)=={(run,i) for run in (0,1) for i in wanted}
 sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
 from Utils.parser import extract_answer
 from Utils.grader import check_is_correct
 grades=[]
 for (run,idx),text in sorted(comps.items()):
  err=None
  try:pred=extract_answer(text,'math');correct=int(bool(check_is_correct(pred,expected[idx]['gold'])))
  except Exception as exc:pred='';correct=0;err=f'{type(exc).__name__}: {exc}'
  grades.append(dict(run=run,idx=idx,prediction=pred,correct=correct,grading_error=err,
   eos_reached=metrics[idx]['eos_reached'][run],generated_tokens=metrics[idx]['generated_tokens_per_rollout'][run]))
 (out/'grades.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in grades))
 assert not any(x['grading_error'] for x in grades),'grading errors require review; not reporting full accuracy'
 vals=[metrics[i] for i in sorted(wanted)];sel=sum(x['selected_pairs'] for x in vals);causal=sum(x['causal_pairs'] for x in vals)
 ds=sum(x['decode_selected_pairs'] for x in vals);dc=sum(x['decode_causal_pairs'] for x in vals)
 dt=sum(x['decode_seconds'] for x in vals);tokens=sum(x['decode_query_tokens'] for x in vals)
 metadata={x['idx']:x for x in map(json.loads,(root/'dataset_metadata.jsonl').read_text().splitlines())}
 summary=dict(status='completed',method=method,questions=len(wanted),rollouts=len(grades),
  input_alignment='passed/3 canonical inputs' if smoke else 'passed/full',cache_policy_validation='passed/official permanent eviction and absolute RoPE positions',
  pair_count_crosscheck='passed/exact official eviction recurrence',selection_p=None,query_window=None,token_budget=1024,residual_length=64,
  correct=sum(x['correct'] for x in grades),accuracy=statistics.mean(x['correct'] for x in grades),
  eos_count=sum(x['eos_reached'] for x in grades),eos_rate=statistics.mean(x['eos_reached'] for x in grades),
  length_limited_rollouts=sum(not x['eos_reached'] for x in grades),grading_errors=0,
  correct_and_eos_rate=statistics.mean(bool(x['correct'] and x['eos_reached']) for x in grades),
  generated_tokens=stats([x['generated_tokens'] for x in grades]),
  prefill_seconds=stats([x['prefill_seconds'] for x in vals]),decode_seconds=stats([x['decode_seconds'] for x in vals]),
  batch_wall_seconds=stats([x['batch_wall_seconds'] for x in vals]),
  eviction_events=sum(x['selection_events'] for x in vals),decode_tokens_per_second=tokens/dt,decode_query_tokens=tokens,decode_forward_calls=sum(x['decode_forward_calls'] for x in vals),
  selected_pairs=sel,causal_pairs=causal,global_pair_sparsity=1-sel/causal,
  decode_selected_pairs=ds,decode_causal_pairs=dc,decode_pair_sparsity=1-ds/dc,
  peak_memory_allocated=max(x['peak_memory_allocated'] for x in vals),peak_memory_distribution=stats([x['peak_memory_allocated'] for x in vals]),
  hardware=hardware,hardware_uniform=len({(h['gpu_name'],h['gpu_total_memory']) for h in hardware})==1,
  timing_scope=cfg['timing'],runs=[],per_level={},baseline_comparisons={})
 for run in (0,1):
  rows=[x for x in grades if x['run']==run]
  summary['runs'].append(dict(run=run,correct=sum(x['correct'] for x in rows),count=len(rows),accuracy=statistics.mean(x['correct'] for x in rows)))
 for level in sorted({str(metadata[i].get('level','unknown')) for i in wanted}):
  rows=[x for x in grades if str(metadata[x['idx']].get('level','unknown'))==level]
  summary['per_level'][level]=dict(rollouts=len(rows),correct=sum(x['correct'] for x in rows),accuracy=statistics.mean(x['correct'] for x in rows))
 current={(x['run'],x['idx']):x for x in grades}
 summary['questions_both_correct']=sum(all(current[r,i]['correct'] for r in (0,1)) for i in wanted)
 summary['questions_at_least_one_correct']=sum(any(current[r,i]['correct'] for r in (0,1)) for i in wanted)
 for d in sorted((root/'reference_baselines').iterdir()):
  baseline={(x['run'],x['idx']):x for x in map(json.loads,(d/'grades.jsonl').read_text().splitlines())}
  assert all(k in baseline for k in current)
  acc=statistics.mean(baseline[k]['correct'] for k in current)
  summary['baseline_comparisons'][d.name]=dict(accuracy=acc,accuracy_difference=summary['accuracy']-acc,
   new_only_correct=sum(x['correct'] and not baseline[k]['correct'] for k,x in current.items()),
   baseline_only_correct=sum(baseline[k]['correct'] and not x['correct'] for k,x in current.items()))
 (out/'question_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in vals))
 for name,rows in [('question_metrics',vals),('rollout_metrics',grades)]:
  keys=sorted(set().union(*(x.keys() for x in rows)))
  with (out/f'{name}.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
   for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 report=[f'# MATH500 {method}','',f"Completed {len(wanted)} questions / {len(grades)} answers; input alignment {summary['input_alignment']}. Correct {summary['correct']}/{len(grades)} ({summary['accuracy']:.2%}).",'',
 '| Metric | Value |','| --- | ---: |',f"| Prefill mean / median / P90 seconds | {summary['prefill_seconds']['mean']:.4f} / {summary['prefill_seconds']['median']:.4f} / {summary['prefill_seconds']['p90']:.4f} |",
 f"| Decode mean / median / P90 seconds | {summary['decode_seconds']['mean']:.3f} / {summary['decode_seconds']['median']:.3f} / {summary['decode_seconds']['p90']:.3f} |",
 f"| Total generation mean seconds | {summary['batch_wall_seconds']['mean']:.3f} |",f"| Aggregate decode tokens/s | {tokens/dt:.3f} |",
 f"| Overall / decode pair sparsity | {summary['global_pair_sparsity']:.2%} / {summary['decode_pair_sparsity']:.2%} |",
 f"| Peak allocated GPU memory GiB | {summary['peak_memory_allocated']/2**30:.3f} |",'',
 f"EOS {summary['eos_count']}/{len(grades)}, length-limit outputs {summary['length_limited_rollouts']}. Selected/causal pairs {sel}/{causal}.", '',
 'Times are per question batch with two concurrent rollouts. Prefill/decode are GPU-synchronized; loading, warmup and diagnostic serialization are excluded. Throughput sums active tokens over total decode time. Official random_pp permanently evicts KV per KV head at budget1024/residual64; pairs count attention BEFORE eviction and use complete causal history as the denominator. This is not a budget-matched comparison against dynamic Top-p or random40 masks. Upstream code is unchanged; private mask RNG isolation overhead is included. Different generation lengths, nodes and gather implementations affect time. Historical dense has a different timing scope and no saved runtime input hashes. Smoke quality is not a full-dataset estimate.','']
 for name,b in summary['baseline_comparisons'].items():report.append(f"Compared with {name}: accuracy difference {100*b['accuracy_difference']:+.2f} percentage points; new-only/baseline-only correct {b['new_only_correct']}/{b['baseline_only_correct']}.")
 (out/'REPORT.md').write_text('\n'.join(report)+'\n')
 return summary


def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--validate-smoke',action='store_true');a=p.parse_args();root=a.root.resolve()
 for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
 statefile=root/('smoke_gate.json' if a.validate_smoke else 'collection_status.json')
 def status(phase,**kw):statefile.write_text(json.dumps(dict(status=phase,phase=phase,timestamp=datetime.datetime.now().astimezone().isoformat(),**kw),indent=2)+'\n')
 try:
  status('validating')
  if not a.validate_smoke:
   proof={str(p.relative_to(root)):sha(p) for p in sorted((root/'formal').glob('shard_*/**/*')) if p.is_file() and (p.name.startswith('completions') or p.name in ('cache_metrics.jsonl','exit.json','hardware.json','preflight.json'))}
   f=root/'collector_input_hashes.json'
   if f.exists():assert json.loads(f.read_text())==proof,'raw outputs changed since collection'
   else:f.write_text(json.dumps(proof,indent=2)+'\n')
  s=analyze(root,a.validate_smoke)
  if a.validate_smoke:
   status('passed',questions=3,rollouts=6,correct=s['correct'],summary_sha256=sha(root/'smoke/summary.json'),input_alignment=s['input_alignment'])
  else:
   sub=json.loads((root/'submission.json').read_text())
   r=subprocess.run(['sacct','-j',sub['array_job_id'],'--parsable2','--format=JobIDRaw,State,ExitCode,Submit,Start,End,Elapsed,NodeList,AllocTRES'],capture_output=True,text=True)
   (root/'scheduler_accounting.txt').write_text(r.stdout);(root/'scheduler_accounting_error.txt').write_text(r.stderr)
   status('completed',questions=500,rollouts=1000,input_alignment='passed/full',accuracy=s['accuracy'])
 except Exception as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':main()
