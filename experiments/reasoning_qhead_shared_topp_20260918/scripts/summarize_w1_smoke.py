"""Post-run same-input comparison; inference remains byte-identical to W16/W64."""
from pathlib import Path
import hashlib,json,statistics,sys
root=Path(sys.argv[1]).resolve();s=json.loads((root/'smoke/summary.json').read_text());cfg=json.loads((root/'config.json').read_text());ref=Path(cfg['reference_root']);ids={0,1,2}
methods={};questions=[];sources={}
for w in (1,16,64):
 paths=[root/'smoke/cache_metrics.jsonl'] if w==1 else sorted((ref/f'w{w}').glob('shard_*/cache_metrics.jsonl'))
 rr=[]
 for p in paths:
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  rr.extend(r for r in map(json.loads,p.read_text().splitlines()) if r['idx'] in ids)
 assert len(rr)==3 and {r['idx'] for r in rr}==ids
 gp=root/'smoke/grades.jsonl' if w==1 else ref/f'w{w}/grades.jsonl'
 grades=[r for r in map(json.loads,gp.read_text().splitlines()) if r['idx'] in ids];assert len(grades)==6
 dt=sum(r['decode_seconds'] for r in rr);calls=sum(r['decode_forward_calls'] for r in rr);tokens=sum(r['decode_query_tokens'] for r in rr)
 methods[str(w)]=dict(window=w,questions=3,rollouts=6,correct=sum(r['correct'] for r in grades),eos_count=sum(r['eos_reached'] for r in grades),total_decode_seconds=dt,mean_decode_seconds=dt/3,decode_tokens_per_second=tokens/dt,decode_forward_steps_per_second=calls/dt,mean_active_rows_per_decode_step=tokens/calls,mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in grades),global_pair_sparsity=1-sum(r['selected_pairs'] for r in rr)/sum(r['causal_pairs'] for r in rr),peak_memory_allocated=max(r['peak_memory_allocated'] for r in rr))
 for r in sorted(rr,key=lambda x:x['idx']):
  questions.append(dict(window=w,idx=r['idx'],lengths=r['generated_tokens_per_rollout'],decode_seconds=r['decode_seconds'],decode_tokens_per_second=r['decode_query_tokens']/r['decode_seconds'],decode_forward_steps_per_second=r['decode_forward_calls']/r['decode_seconds'],input_ids_sha256=r['input_ids_sha256'],correct_count=sum(x['correct'] for x in grades if x['idx']==r['idx'])))
for idx in ids:assert len({r['input_ids_sha256'] for r in questions if r['idx']==idx})==1
result=dict(status='completed',scope='same three complete input problems; not full500 accuracy',methods=methods,questions=questions,sources_sha256=sources)
(root/'smoke/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# W1 current-query smoke: completed','',f"11 correctness tests passed. Three canonical inputs, two complete generations each. W1 correct {s['correct']}/6; EOS {s['eos_count']}/6; length-limit exits {s['length_limited_rollouts']}. Input hashes, full-cache lengths and pair-counter crosschecks passed.",'','| Window | Correct / 6 | Mean decode per question (s) | Decode tokens/s | Forward steps/s | Mean active rows/step | Overall pair sparsity |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for m in methods.values():lines.append(f"| W{m['window']} | {m['correct']}/6 | {m['mean_decode_seconds']:.2f} | {m['decode_tokens_per_second']:.2f} | {m['decode_forward_steps_per_second']:.2f} | {m['mean_active_rows_per_decode_step']:.3f} | {m['global_pair_sparsity']:.2%} |")
lines+=['','| Window | Input | Generated lengths | Decode (s) | Correct / 2 |','| --- | ---: | --- | ---: | ---: |']
for r in questions:lines.append(f"| W{r['window']} | {r['idx']} | {r['lengths'][0]}, {r['lengths'][1]} | {r['decode_seconds']:.2f} | {r['correct_count']}/2 |")
lines+=['','All inference source files are identical to the frozen W16/W64 implementation. Only query_window changes. W1 overwrites its sole query slot before each sparse decode selection, uses the current representative Q head over all currently cached legal keys, protects the prompt and applies conditional non-prompt Top-p0.95. Full KV remains available.','',
'The actual historical W16/W64 selector rescores its query buffer against the current complete cache; it does not independently enforce each old query\'s original causal boundary. This corrects an earlier explanatory formula. W1 has only a current query, for which all currently cached keys are legal. Historical code and results were preserved.','',
'Decode wall time includes selection, sorting, KV gather, attention, model computation, sampling and bookkeeping; loading/warmup excluded. The two active rollouts shrink to one when one answer reaches EOS first. Forward steps/s and tokens/s measure different things; neither isolates selector/kernel latency when generated lengths and attention masks differ. Nodes differ but all are A10080GB. Three questions cannot establish full MATH500 accuracy or a stable speed advantage.','',
'No automatic full500 submission is enabled. Sources: summary.json, cache_metrics.jsonl, grades.jsonl, comparison.json, ../correctness.log and ../inference_source_identity.json.']
(root/'smoke/REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(methods,indent=2))
