"""Fail closed on missing records, input drift, cache-budget drift or ungradeable tests."""
import argparse,concurrent.futures,importlib.util,json,multiprocessing,re,statistics,sys
from pathlib import Path
from compact_multibench_common import verify,rows,sha

def gpqa_answer(text):
 # Require a complete final boxed answer. Letters in reasoning are never a fallback.
 matches=list(re.finditer(r'\\boxed\s*\{',text))
 if not matches:return ''
 start=matches[-1].end();depth=1;end=start
 while end<len(text) and depth:
  if text[end]=='{':depth+=1
  elif text[end]=='}':depth-=1
  end+=1
 if depth:return ''
 value=text[start:end-1].strip()
 for _ in range(3):
  wrapped=re.fullmatch(r'\\(?:text|mathrm|mathbf|textbf)\s*\{\s*([\s\S]*)\s*\}',value)
  if wrapped:value=wrapped[1].strip()
 value=value.strip(' $().')
 return value if re.fullmatch('[A-D]',value) else ''

def lcb_module(root):
 p=Path(root)/'source/ra_harness/kvcompress/eval/grader_lcb.py'
 spec=importlib.util.spec_from_file_location('compact_lcb_grader',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def grade_lcb(item):
 root,idx,run,text=item;m=lcb_module(root);data=Path(root)/'data/livecodebench'
 store=m.ProblemStore(str(data/'test.jsonl'),str(data/'livecodebench_tests'),cache_mb=0)
 problem=store.get(idx)
 assert problem is not None and len(problem[1])==len(problem[2])>0,('ungradeable',idx)
 result=m.judge_completion(text,problem,timeout=20,mem_mb=2048,py=sys.executable,max_tests=0)
 assert result in (0,1),('ungradeable result',idx)
 return dict(idx=idx,run=run,correct=int(result),code_extracted=bool(m.extract_code(text)),tests_available=len(problem[1]),all_tests_required=True)

def check_metric(r,expected,cfg):
 assert r['selection_strategy']=='persistent_shared_uniform_topk_recent' and r['random_keep_ratio'] is None
 assert r['reasoning_budget']==1024 and r['prompt_outside_budget']
 assert r['recent_window']==64 and r['recent_inside_budget'] and r['gather_backend']=='triton_fused_bounded_scratch'
 assert r['query_window'] is None and r['selection_p'] is None
 assert r['mask_seed']==42 and r['representative_seed']==0
 assert r['cache_semantics']=='persistent_eviction' and r['compaction_interval']==64 and r['compaction_phase']=='after_attention'
 assert r['input_tokens']==expected['input_tokens'] and r['input_ids_sha256']==expected['input_ids_sha256']
 g=cfg['model_geometry'];factor=g['num_attention_heads']*g['num_hidden_layers'];P=r['input_tokens'];generated=r['generated_tokens_per_rollout'];budget=1024
 assert len(generated)==len(r['eos_reached'])==2
 for count,eos in zip(generated,r['eos_reached']):
  assert 1<=count<=cfg['max_tokens']-P
  if not eos:assert count==cfg['max_tokens']-P
 prefill=2*factor*P*(P+1)//2
 causal=factor*sum((P+t-1)*(P+t)//2 for t in generated)
 decode_selected=factor*sum(sum(P+(j if j<=budget+64 else budget+1+(j-budget-65)%64) for j in range(1,t)) for t in generated)
 assert r['prefill_selected_pairs']==r['prefill_causal_pairs']==prefill
 assert r['causal_pairs']==causal and r['decode_causal_pairs']==causal-prefill
 assert r['decode_selected_pairs']==decode_selected and r['selected_pairs']==prefill+decode_selected
 assert r['decode_query_tokens']==sum(generated)-2 and r['decode_forward_calls']==max(generated)-1
 steps=max(generated)-1;events=0 if steps<budget+64 else 1+(steps-budget-64)//64
 assert r['selection_events']==events*g['num_hidden_layers']
 assert r['max_capacity']==P+budget+64 and r['max_selected_tokens']==P+min(steps,budget+64)
 cached=steps if steps<budget+64 else budget+(steps-budget-64)%64
 assert len(r['final_cache_seqlens'])==len(r['final_cumulative_length'])==g['num_hidden_layers']
 for lengths,cumulative in zip(r['final_cache_seqlens'],r['final_cumulative_length']):
  assert cumulative==P+steps and all(n==P+cached for n in lengths)
 assert r['prefill_seconds']>0 and r['decode_seconds']>0

def check(root,task,phase):
 root=Path(root).resolve();cfg=verify(root,tests=task=='livecodebench');spec=next(t for t in cfg['tasks'] if t['name']==task)
 expected={r['idx']:r for r in rows(root/'inputs'/f'{task}.jsonl')};assert set(expected)==set(range(spec['count']))
 ids=set(cfg['smoke_sample_ids']) if phase=='smoke' else set(expected)
 n=1 if phase=='smoke' else spec['shards'];out=root/task/phase
 metrics={};comps={};hardware=[];source_hashes={}
 for shard in range(n):
  d=out/f'shard_{shard}';assert json.loads((d/'exit.json').read_text())['exit_code']==0,str(d)
  hardware.append(json.loads((d/'hardware.json').read_text()))
  source_hashes[str((d/'cache_metrics.jsonl').relative_to(root))]=sha(d/'cache_metrics.jsonl')
  for r in rows(d/'cache_metrics.jsonl'):
   idx=r['idx'];assert idx in ids and idx not in metrics
   if phase=='formal':assert idx%n==shard
   check_metric(r,expected[idx],cfg);metrics[idx]=r
  for run in (0,1):
   files=list((d/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1
   source_hashes[str(files[0].relative_to(root))]=sha(files[0])
   for r in rows(files[0]):
    idx=r['idx'];key=run,idx;assert idx in ids and key not in comps
    if phase=='formal':assert idx%n==shard
    prompt=expected[idx]['prompt_text'];assert r['completion'].startswith(prompt)
    # Never grade starter code or boxed values embedded in the input prompt.
    comps[key]=r['completion'][len(prompt):]
 assert set(metrics)==ids and set(comps)=={(r,i) for r in (0,1) for i in ids}
 print(json.dumps(dict(task=task,phase=phase,technical_validation='passed',questions=len(ids),grading='starting')),flush=True)
 sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
 from Utils.parser import extract_answer
 from Utils.grader import check_is_correct
 import numpy as np
 def stats(xs):return dict(mean=statistics.mean(xs),median=statistics.median(xs),p90=float(np.percentile(xs,90)),total=sum(xs))
 grades=[]
 if task=='livecodebench':
  items=[(str(root),idx,run,comps[run,idx]) for idx in sorted(ids) for run in (0,1)]
  # Processes, not threads: the candidate executor uses POSIX preexec_fn.
  with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
   for i,r in enumerate(pool.map(grade_lcb,items,chunksize=1)):
    grades.append(r)
    if (i+1)%20==0:print(json.dumps(dict(task=task,graded=i+1,total=len(items))),flush=True)
 else:
  for run in (0,1):
   for idx in sorted(ids):
    prediction=gpqa_answer(comps[run,idx]) if task=='gpqa' else extract_answer(comps[run,idx])
    correct=(prediction==expected[idx]['gold']) if task=='gpqa' else bool(check_is_correct(prediction,expected[idx]['gold']))
    grades.append(dict(idx=idx,run=run,prediction=prediction,correct=int(correct)))
 for x in grades:
  x.update(eos_reached=metrics[x['idx']]['eos_reached'][x['run']],generated_tokens=metrics[x['idx']]['generated_tokens_per_rollout'][x['run']])
 runs=[]
 for run in (0,1):
  rr=[r for r in grades if r['run']==run];runs.append(dict(run=run,correct=sum(x['correct'] for x in rr),count=len(rr),accuracy=statistics.mean(x['correct'] for x in rr)))
 vals=list(metrics.values());sel=sum(x['selected_pairs'] for x in vals);causal=sum(x['causal_pairs'] for x in vals);ds=sum(x['decode_selected_pairs'] for x in vals);dc=sum(x['decode_causal_pairs'] for x in vals);dt=sum(x['decode_seconds'] for x in vals);tokens=sum(x['decode_query_tokens'] for x in vals)
 summary=dict(status='passed' if phase=='smoke' else 'completed',phase=phase,task=task,dataset_version=cfg['versions'][task],method=cfg['method'],questions=len(ids),rollouts=len(grades),input_alignment='passed/full' if phase=='formal' else 'passed/3 smoke inputs',exact_pair_budget='passed',source_manifest='passed',runs=runs,correct=sum(x['correct'] for x in grades),accuracy=statistics.mean(x['correct'] for x in grades),score_definition='mean correctness over two independent sampled rollouts; LCB pass@1 estimate, not pass@2',eos_rate=statistics.mean(x['eos_reached'] for x in grades),length_limited_rollouts=sum(not x['eos_reached'] for x in grades),generated_tokens=stats([x['generated_tokens'] for x in grades]),prefill_seconds_per_question_batch=stats([x['prefill_seconds'] for x in vals]),decode_seconds_per_question_batch=stats([x['decode_seconds'] for x in vals]),batch_wall_seconds=stats([x['batch_wall_seconds'] for x in vals]),decode_tokens_per_second=tokens/dt,selected_pairs=sel,causal_pairs=causal,global_pair_sparsity=1-sel/causal,decode_selected_pairs=ds,decode_causal_pairs=dc,decode_pair_sparsity=1-ds/dc,peak_memory_allocated=max(x['peak_memory_allocated'] for x in vals),hardware=hardware,timing_scope=cfg['timing'],comparison_caveat=cfg['comparison_caveat'],raw_files_sha256=source_hashes)
 with (out/'grades.jsonl').open('x') as f:f.writelines(json.dumps(x)+'\n' for x in grades)
 with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
 print(json.dumps({k:summary[k] for k in ('task','phase','status','questions','rollouts','correct','accuracy','global_pair_sparsity')}),flush=True)
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('task');p.add_argument('phase',choices=['smoke','formal']);a=p.parse_args();check(a.root,a.task,a.phase)
