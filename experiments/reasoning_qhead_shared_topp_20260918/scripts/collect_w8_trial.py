import datetime,json,statistics,sys
from pathlib import Path
from rep_compact_math_common import sha,rows
from check_w8_metrics import check_metric
root=Path(sys.argv[1]).resolve();cfg=json.loads((root/'config.json').read_text())
def dump(name,obj):
 (root/name).write_text(json.dumps(obj,indent=2)+'\n')
try:
 dump('collection_status.json',dict(status='validating',time=datetime.datetime.now().astimezone().isoformat()))
 for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
 assert json.loads((root/'correctness/status.json').read_text())['phase']=='completed'
 sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
 from Utils.parser import extract_answer
 from Utils.grader import check_is_correct
 import numpy as np
 expected={r['idx']:r for r in rows(root/'inputs.jsonl')};assert set(expected)==set(range(20))
 results={};allgrades={};allmetrics={}
 def stats(v):return dict(mean=statistics.mean(v),median=statistics.median(v),p90=float(np.percentile(v,90)),total=sum(v))
 for name,w in [('w1',1),('w8',8)]:
  metrics={};comps={};hardware=[];hashes={};ccfg={**cfg,'query_window':w}
  for shard in range(4):
   parent=root/f'shard_{shard}';assert json.loads((parent/'status.json').read_text())['phase']=='completed'
   hardware.append(json.loads((parent/'hardware.json').read_text()));d=parent/name
   assert json.loads((d/'exit.json').read_text())['exit_code']==0
   hashes[str((d/'cache_metrics.jsonl').relative_to(root))]=sha(d/'cache_metrics.jsonl')
   for r in rows(d/'cache_metrics.jsonl'):
    idx=r['idx'];assert idx in expected and idx%4==shard and idx not in metrics
    check_metric(r,expected[idx],ccfg)
    assert r['implementation']==(cfg['baseline_method'] if w==1 else cfg['method'])
    assert r['representative_role']==('current_query_probability_topk' if w==1 else 'recent_queries_probability_mean_topk')
    metrics[idx]=r
   for run in (0,1):
    files=list((d/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1
    hashes[str(files[0].relative_to(root))]=sha(files[0])
    for r in rows(files[0]):
     idx=r['idx'];key=run,idx;assert idx in expected and idx%4==shard and key not in comps
     prompt=expected[idx]['prompt_text'];assert r['completion'].startswith(prompt);comps[key]=r['completion'][len(prompt):]
  assert set(metrics)==set(expected) and set(comps)=={(r,i) for r in (0,1) for i in expected}
  dump('collection_status.json',dict(status='grading',method=name,technical_validation='passed/20',time=datetime.datetime.now().astimezone().isoformat()))
  grades=[]
  for run in (0,1):
   for idx in sorted(expected):
    pred=extract_answer(comps[run,idx]);correct=int(bool(check_is_correct(pred,expected[idx]['gold'])))
    grades.append(dict(run=run,idx=idx,prediction=pred,correct=correct,generated_tokens=metrics[idx]['generated_tokens_per_rollout'][run],eos_reached=metrics[idx]['eos_reached'][run]))
  (root/f'{name}_grades.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in grades));allgrades[name]={(g['run'],g['idx']):g for g in grades};allmetrics[name]=metrics
  vals=list(metrics.values());sel=sum(r['selected_pairs'] for r in vals);causal=sum(r['causal_pairs'] for r in vals)
  results[name]=dict(questions=20,rollouts=40,correct=sum(g['correct'] for g in grades),accuracy=statistics.mean(g['correct'] for g in grades),runs=[dict(run=run,correct=sum(g['correct'] for g in grades if g['run']==run),total=20) for run in (0,1)],input_alignment='passed/full20',exact_pair_budget='passed',decode_seconds_per_question_batch=stats([r['decode_seconds'] for r in vals]),prefill_seconds_per_question_batch=stats([r['prefill_seconds'] for r in vals]),total_seconds_per_question_batch=stats([r['batch_wall_seconds'] for r in vals]),generated_tokens=stats([g['generated_tokens'] for g in grades]),decode_tokens_per_second=sum(r['decode_query_tokens'] for r in vals)/sum(r['decode_seconds'] for r in vals),eos_rate=statistics.mean(g['eos_reached'] for g in grades),length_limited_rollouts=sum(not g['eos_reached'] for g in grades),peak_memory_allocated=max(r['peak_memory_allocated'] for r in vals),selected_pairs=sel,causal_pairs=causal,global_pair_sparsity=1-sel/causal,hardware=hardware,raw_files_sha256=hashes)
 a,b=results['w1'],results['w8'];changes=[]
 for idx in range(20):
  changes.append(dict(idx=idx,w1_correct=sum(allgrades['w1'][run,idx]['correct'] for run in (0,1)),w8_correct=sum(allgrades['w8'][run,idx]['correct'] for run in (0,1)),w1_decode_seconds=allmetrics['w1'][idx]['decode_seconds'],w8_decode_seconds=allmetrics['w8'][idx]['decode_seconds'],w1_generated_tokens=allmetrics['w1'][idx]['generated_tokens_per_rollout'],w8_generated_tokens=allmetrics['w8'][idx]['generated_tokens_per_rollout']))
 comparison=dict(accuracy_change_percentage_points=100*(b['accuracy']-a['accuracy']),decode_mean_change_fraction=b['decode_seconds_per_question_batch']['mean']/a['decode_seconds_per_question_batch']['mean']-1,decode_throughput_change_fraction=b['decode_tokens_per_second']/a['decode_tokens_per_second']-1,generated_mean_change_fraction=b['generated_tokens']['mean']/a['generated_tokens']['mean']-1,wrong_to_right=sum(not allgrades['w1'][k]['correct'] and allgrades['w8'][k]['correct'] for k in allgrades['w1']),right_to_wrong=sum(allgrades['w1'][k]['correct'] and not allgrades['w8'][k]['correct'] for k in allgrades['w1']),per_question=changes)
 result=dict(status='completed',scope=cfg['scope'],input_alignment='passed/full20',results=results,comparison=comparison,caveat=cfg['comparison_caveat'],timing_scope=cfg['timing'],score_definition='40 rollout correctness mean for each method; two rollouts per question, not pass@2')
 dump('summary.json',result)
 lines=['# Optimized W1 vs W8: first20 MATH500 questions','',cfg['scope'],'','| Method | Correct /40 | Accuracy | Mean decode s / question batch | Decode tokens/s | Mean generated tokens / rollout | Mean prefill s | Mean total s | Length-limit hits |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,s in results.items():lines.append(f"| {name} | {s['correct']} | {s['accuracy']:.1%} | {s['decode_seconds_per_question_batch']['mean']:.3f} | {s['decode_tokens_per_second']:.3f} | {s['generated_tokens']['mean']:.1f} | {s['prefill_seconds_per_question_batch']['mean']:.4f} | {s['total_seconds_per_question_batch']['mean']:.3f} | {s['length_limited_rollouts']} |")
 lines+=['',f"W8 minus W1: accuracy {comparison['accuracy_change_percentage_points']:+.1f} percentage points; mean decode wall time {comparison['decode_mean_change_fraction']:+.2%}; aggregate decode throughput {comparison['decode_throughput_change_fraction']:+.2%}; output length {comparison['generated_mean_change_fraction']:+.2%}.",'',cfg['comparison_caveat'],'','All 20 inputs aligned by token count and input_ids SHA256, no duplicate/missing records, complete outputs and exact pair/budget formulas checked. Accuracy pools two sampled rollouts per question; this is not a full MATH500 score. Wall time is for a question batch of two rollouts. Throughput uses actual decode query tokens, including the reduced batch after early EOS.','', 'W8 scores each of the latest8 post-RoPE queries against all currently surviving keys, applies independent FP32 softmax, then averages probabilities. No historical-query causal mask in the importance estimator (matches historical window semantics); actual generation attention remains causal. One representative Q head per layer/event, correct GQA mapping, same private representative stream; prompt outside reasoning1024; recent64 inside; stable TopK960; after-attention compaction at reasoning1088 every64 steps.','', 'Timing differences include both query-window execution costs and changed trajectories/output lengths. The 20-question, single-seed comparison cannot establish a reliable full-dataset accuracy improvement or isolate selector overhead from end-to-end timing alone.']
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n');dump('collection_status.json',dict(status='completed',report=str(root/'REPORT.md'),time=datetime.datetime.now().astimezone().isoformat()))
 print(json.dumps(result,indent=2),flush=True)
except Exception as exc:
 dump('collection_status.json',dict(status='failed',error=repr(exc),time=datetime.datetime.now().astimezone().isoformat()));raise
