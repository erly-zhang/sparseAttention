import json,statistics,sys
from pathlib import Path
from rep_compact_math_common import sha,rows
from check_rep_compact_math import check_metric
root=Path(sys.argv[1]).resolve();cfg=json.loads((root/'config.json').read_text())
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
from Utils.parser import extract_answer
from Utils.grader import check_is_correct
expected={r['idx']:r for r in rows(root/'inputs.jsonl')};passes={};reference_outputs=None;reference_metrics=None
fields=('input_tokens','input_ids_sha256','generated_tokens_per_rollout','eos_reached','selected_pairs','causal_pairs','decode_selected_pairs','decode_causal_pairs','prefill_selected_pairs','prefill_causal_pairs','selection_events','final_cache_seqlens','final_cumulative_length')
for name in cfg['pair_order']:
 out=root/name;assert json.loads((out/'exit.json').read_text())['exit_code']==0
 rr=rows(out/'cache_metrics.jsonl');assert len(rr)==3 and {r['idx'] for r in rr}=={0,1,2}
 metric={r['idx']:r for r in rr};outputs={};grades=[]
 for r in rr:
  check_metric(r,expected[r['idx']],cfg)
  want=cfg['baseline_method'] if name.startswith('baseline') else cfg['method'];assert r['implementation']==want
 for run in (0,1):
  cc=rows(out/f'run_{run}/completions_shard0.jsonl');assert len(cc)==3 and {x['idx'] for x in cc}=={0,1,2}
  for row in cc:
   idx=row['idx'];prefix=expected[idx]['prompt_text'];assert row['completion'].startswith(prefix)
   outputs[run,idx]=row['completion'];pred=extract_answer(row['completion'][len(prefix):])
   grades.append(dict(run=run,idx=idx,prediction=pred,correct=int(bool(check_is_correct(pred,expected[idx]['gold'])))))
 if reference_outputs is None:reference_outputs=outputs;reference_metrics=metric
 else:
  assert reference_outputs==outputs,('output mismatch',name)
  for idx in metric:
   for field in fields:assert metric[idx][field]==reference_metrics[idx][field],(name,idx,field)
 passes[name]=dict(correct=sum(g['correct'] for g in grades),rollouts=6,decode_mean_seconds=statistics.mean(x['decode_seconds'] for x in rr),prefill_mean_seconds=statistics.mean(x['prefill_seconds'] for x in rr),total_mean_seconds=statistics.mean(x['batch_wall_seconds'] for x in rr),decode_tokens_per_second=sum(x['decode_query_tokens'] for x in rr)/sum(x['decode_seconds'] for x in rr),peak_memory_allocated=max(x['peak_memory_allocated'] for x in rr),generated_mean_tokens=statistics.mean(t for x in rr for t in x['generated_tokens_per_rollout']),metrics=rr,grades=grades)
summary={}
for mode in ('baseline','optimized'):
 ss=[v for name,v in passes.items() if name.startswith(mode)];mm=[r for x in ss for r in x['metrics']]
 summary[mode]=dict(decode_mean_seconds=statistics.mean(r['decode_seconds'] for r in mm),prefill_mean_seconds=statistics.mean(r['prefill_seconds'] for r in mm),total_mean_seconds=statistics.mean(r['batch_wall_seconds'] for r in mm),decode_tokens_per_second=sum(r['decode_query_tokens'] for r in mm)/sum(r['decode_seconds'] for r in mm),peak_memory_allocated=max(r['peak_memory_allocated'] for r in mm),repeat_decode_means=[x['decode_mean_seconds'] for x in ss])
improvement=1-summary['optimized']['decode_mean_seconds']/summary['baseline']['decode_mean_seconds']
result=dict(status='passed',input_alignment='passed/all3',full_generation_outputs='exactly identical for all6 outputs across all4 passes',metrics_and_pair_counts='exactly identical',unique_questions=3,unique_rollouts=6,repetitions_per_method=2,passes=passes,summary=summary,decode_time_reduction=improvement,hardware=json.loads((root/'hardware.json').read_text()),microbenchmark=json.loads((root/'microbenchmark.json').read_text()),comparison_scope=cfg['comparison_caveat'])
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# W1 execution optimization: controlled same-GPU comparison','', 'MATH500 inputs0,1,2; six unique sampled outputs; two repeated runs per method in baseline/optimized/optimized/baseline order. All six complete outputs, input hashes, generated lengths, EOS, exact attention pair counts, selection event counts and final cache/position lengths match across all four passes.','',
 '| Method | Mean decode seconds / two-rollout question | Decode tokens/s | Peak allocated GiB | Repeat decode means |', '|---|---:|---:|---:|---|']
for mode,s in summary.items():lines.append(f"| {mode} | {s['decode_mean_seconds']:.3f} | {s['decode_tokens_per_second']:.3f} | {s['peak_memory_allocated']/1024**3:.3f} | {', '.join(f'{x:.3f}' for x in s['repeat_decode_means'])} |")
lines+=['',f'Decode time reduction on this controlled three-input trial: {improvement:.2%}.', '', 'Optimizations: Q copy only on compaction events; protected prompt copied to spare buffer once; reasoning-only fused K/V gather; full-capacity buffer swap instead of copying gathered KV back. Original representative RNG, BF16 QK, FP32 softmax, stable tie rule, TopK960, chronological ordering, prompt/recent protection, GQA, RoPE and 64-step compression cadence remain unchanged.', '', 'Two full-capacity buffers require 64 more scratch positions per layer than the old compressed-length scratch. No accuracy improvement is claimed. These are three-input repeated timings, not a new full MATH500 benchmark; microbench numbers exclude projections/MLP and must not be presented as model-level acceleration. Model loading, warmup and diagnostic serialization are outside recorded decode timing.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('passes','microbenchmark')},indent=2),flush=True)
