"""Compare only identical three input IDs; never mix full-dataset and smoke rates."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve()
E=Path(__file__).resolve().parents[1]
assert json.loads((root/'smoke_gate.json').read_text())['status']=='passed'
newmetrics={r['idx']:r for r in map(json.loads,(root/'smoke/cache_metrics.jsonl').read_text().splitlines())}
newgrades={(r['idx'],r['run']):r for r in map(json.loads,(root/'smoke/grades.jsonl').read_text().splitlines())}
ids=set(newmetrics);assert ids=={0,1,2}
for r in newmetrics.values():
 assert r['max_selected_tokens']==r['input_tokens']+min(1088,max(r['generated_tokens_per_rollout'])-1)
paths={
 'compact_prompt_plus_reasoning1024_recent64':root/'smoke',
 'full_history_reasoning1024':E/'results/full_kv_random_reason1024_recent64_smoke_20260925/smoke',
 'random40_recent64':E/'results/full_kv_random_r040_recent64_gpu_20260924/random',
 'official_random_pp':E/'results/official_random_pp_k1024_r64_20260924/formal'}
rows=[];sources=[]
for name,path in paths.items():
 if name=='compact_prompt_plus_reasoning1024_recent64':metrics=newmetrics;grades=newgrades;files=[path/'cache_metrics.jsonl',path/'grades.jsonl']
 elif name=='full_history_reasoning1024':
  files=[path/'cache_metrics.jsonl',path/'grades.jsonl']
  metrics={r['idx']:r for r in map(json.loads,files[0].read_text().splitlines())}
  grades={(r['idx'],r['run']):r for r in map(json.loads,files[1].read_text().splitlines())}
 else:
  metrics={};files=list(path.glob('shard_*/cache_metrics.jsonl'))
  for f in files:
   for r in map(json.loads,f.read_text().splitlines()):
    if r['idx'] in ids:
     assert r['idx'] not in metrics;metrics[r['idx']]=r
  grades={}
  f=path/'rollout_metrics.csv';files.append(f)
  for r in csv.DictReader(f.open()):
   if int(r['idx']) in ids:grades[int(r['idx']),int(r['run'])]=dict(correct=int(r['correct']))
 assert set(metrics)==ids and set(grades)=={(i,r) for i in ids for r in (0,1)}
 for i in ids:
  assert metrics[i]['input_ids_sha256']==newmetrics[i]['input_ids_sha256'] and metrics[i]['input_tokens']==newmetrics[i]['input_tokens']
 vals=list(metrics.values());dt=sum(r['decode_seconds'] for r in vals);tokens=sum(r['decode_query_tokens'] for r in vals)
 row=dict(method=name,questions=3,rollouts=6,correct=sum(r['correct'] for r in grades.values()),
          eos_count=sum(sum(r['eos_reached']) for r in vals),mean_generated_tokens=statistics.mean(n for r in vals for n in r['generated_tokens_per_rollout']),
          mean_prefill_seconds=statistics.mean(r['prefill_seconds'] for r in vals),mean_decode_seconds=dt/3,mean_total_seconds=statistics.mean(r['batch_wall_seconds'] for r in vals),
          decode_tokens_per_second=tokens/dt,selected_pairs=sum(r['selected_pairs'] for r in vals),causal_pairs=sum(r['causal_pairs'] for r in vals),
          peak_memory_gib=max(r['peak_memory_allocated'] for r in vals)/2**30,
          lengths_by_input={i:metrics[i]['generated_tokens_per_rollout'] for i in sorted(ids)})
 row['global_pair_sparsity']=1-row['selected_pairs']/row['causal_pairs'];rows.append(row)
 sources.extend(dict(method=name,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files)
result=dict(scope='same MATH500 input IDs0,1,2; two complete rollouts each; not full500 accuracy',input_alignment='passed/identical token lengths and SHA256 across methods',rows=rows,
            caveats=['Same A10080GB class, different nodes/run times.','Different attention budgets, cache retention and generated lengths; total latency is not an isolated kernel comparison.','Prompt is outside the new1024 reasoning budget but inside official random_pp total1024 budget.','Compacted method evicts permanently every64 reasoning tokens, after attention; shared positions across KV heads. Full-history method retains all KV.'])
(root/'smoke/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'smoke/comparison_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
lines=['# Compacted prompt + reasoning1024 + recent64: completed three-input trial','',result['scope'],'',result['input_alignment'],'',
       '| Method | Correct/6 | EOS/6 | Mean generated tokens | Decode seconds/question batch | Decode tokens/s | Global pair sparsity | Peak allocated GiB |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:lines.append(f"| {r['method']} | {r['correct']}/6 | {r['eos_count']}/6 | {r['mean_generated_tokens']:.1f} | {r['mean_decode_seconds']:.3f} | {r['decode_tokens_per_second']:.3f} | {r['global_pair_sparsity']:.2%} | {r['peak_memory_gib']:.3f} |")
lines+=['','Timing excludes loading, warmup and metric serialization; each question has two concurrent rollouts.','']+result['caveats']
(root/'smoke/REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
