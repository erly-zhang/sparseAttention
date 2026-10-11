"""Post-process saved outputs only; no work added to timed inference."""
import argparse,csv,json,statistics
from fractions import Fraction
from pathlib import Path
import numpy as np

def stats(values):
 x=list(values)
 return dict(n=len(x),mean=statistics.mean(x),median=statistics.median(x),p90=float(np.percentile(x,90)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),minimum=min(x),maximum=max(x),total=sum(x)) if x else None

def write_csv(path,rows):
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def summarize(root,out,phase):
 cfg=json.loads((root/'config.json').read_text());base=root/('smoke' if phase=='smoke' else 'random')
 summary=json.loads((base/'summary.json').read_text())
 grades=[json.loads(x) for x in (base/'grades.jsonl').read_text().splitlines()];grade={(x['run'],x['idx']):x for x in grades}
 paths=[base/'cache_metrics.jsonl'] if phase=='smoke' else sorted(base.glob('shard_*/cache_metrics.jsonl'))
 raw=[json.loads(x) for p in paths for x in p.read_text().splitlines()]
 assert len(raw)==summary['questions'] and len(grades)==2*len(raw)
 metadata={x['idx']:x for x in map(json.loads,(root/'dataset_metadata.jsonl').read_text().splitlines())} if (root/'dataset_metadata.jsonl').exists() else {}
 g=cfg['model_geometry'];factor=g['num_hidden_layers']*g['num_attention_heads'];ratio=Fraction(str(cfg['random_keep_ratio']))
 # Precompute discrete-budget cumulative sums once; use actual observed lengths.
 max_r=max(max(r['generated_tokens_per_rollout'])-1 for r in raw)
 keep=[0];recent=[0]
 for r in range(1,max_r+1):
  k=(ratio.numerator*r+ratio.denominator-1)//ratio.denominator
  keep.append(keep[-1]+k);recent.append(recent[-1]+min(cfg['recent_window'],k))
 rows=[]
 for r in sorted(raw,key=lambda x:x['idx']):
  idx=r['idx'];P=r['input_tokens'];counts=r['generated_tokens_per_rollout'];rs=[t-1 for t in counts]
  prompt=factor*sum(P*x for x in rs);protected=factor*sum(recent[x] for x in rs)
  random=factor*sum(keep[x]-recent[x] for x in rs);reasoning=factor*sum(x*(x+1)//2 for x in rs)
  assert prompt+protected+random==r['decode_selected_pairs']
  assert prompt+reasoning==r['decode_causal_pairs']
  correct=sum(grade[run,idx]['correct'] for run in (0,1));dt=r['decode_seconds'];tokens=r['decode_query_tokens']
  row=dict(idx=idx,level=metadata.get(idx,{}).get('level'),input_tokens=P,input_ids_sha256=r['input_ids_sha256'],correct_count=correct,accuracy=correct/2,run0_correct=grade[0,idx]['correct'],run1_correct=grade[1,idx]['correct'],run0_generated_tokens=counts[0],run1_generated_tokens=counts[1],run0_eos=r['eos_reached'][0],run1_eos=r['eos_reached'][1],prefill_seconds=r['prefill_seconds'],decode_seconds=dt,batch_wall_seconds=r['batch_wall_seconds'],decode_forward_calls=r['decode_forward_calls'],decode_query_tokens=tokens,decode_tokens_per_second=tokens/dt,decode_forward_steps_per_second=r['decode_forward_calls']/dt,aggregate_ms_per_decode_token=1000*dt/tokens,prefill_selected_pairs=r['prefill_selected_pairs'],prefill_causal_pairs=r['prefill_causal_pairs'],decode_selected_pairs=r['decode_selected_pairs'],decode_causal_pairs=r['decode_causal_pairs'],selected_pairs=r['selected_pairs'],causal_pairs=r['causal_pairs'],global_pair_sparsity=1-r['selected_pairs']/r['causal_pairs'],decode_pair_sparsity=1-r['decode_selected_pairs']/r['decode_causal_pairs'],derived_prompt_decode_pairs=prompt,derived_recent_decode_pairs=protected,derived_random_decode_pairs=random,derived_reasoning_causal_pairs=reasoning,reasoning_only_sparsity=1-(protected+random)/reasoning,peak_memory_allocated_bytes=r['peak_memory_allocated'],max_full_cache_tokens=r['max_visible_tokens'],max_selected_tokens=r['max_selected_tokens'],max_cache_capacity=r['max_capacity'],selection_events=r['selection_events'])
  rows.append(row)
 baselines={}
 for name in ('w16','dense'):
  d=root/'reference_baselines'/name
  if not (d/'summary.json').exists():continue
  b=json.loads((d/'summary.json').read_text());bg=[]
  for path in sorted(d.glob('grades*.jsonl')):bg.extend(json.loads(x) for x in path.read_text().splitlines())
  bi={(r['run'],r['idx']):r for r in bg};assert set(grade)<=set(bi)
  acc=b['accuracy'] if name=='w16' else b['pooled']['flag_acc']
  baselines[name]=dict(accuracy=acc,accuracy_difference_percentage_points=100*(summary['accuracy']-acc),new_only_correct=sum(grade[k]['correct'] and not bi[k]['correct'] for k in grade),baseline_only_correct=sum(bi[k]['correct'] and not grade[k]['correct'] for k in grade),both_correct=sum(grade[k]['correct'] and bi[k]['correct'] for k in grade),both_wrong=sum(not grade[k]['correct'] and not bi[k]['correct'] for k in grade),summary=b)
 s=dict(summary)
 s.update(correct_and_eos_count=sum(r['correct'] and r['eos_reached'] for r in grades),correct_and_eos_rate=sum(r['correct'] and r['eos_reached'] for r in grades)/len(grades),grading_error_count=sum(r['grading_error'] is not None for r in grades),questions_both_correct=sum(r['correct_count']==2 for r in rows),questions_at_least_one_correct=sum(r['correct_count']>0 for r in rows),questions_both_wrong=sum(r['correct_count']==0 for r in rows),distribution_stats={name:stats(r[name] for r in rows) for name in ('input_tokens','prefill_seconds','decode_seconds','batch_wall_seconds','decode_tokens_per_second','aggregate_ms_per_decode_token','peak_memory_allocated_bytes','max_full_cache_tokens','max_selected_tokens','selection_events')},generated_tokens_distribution=stats(r['generated_tokens'] for r in grades),decode_query_tokens=sum(r['decode_query_tokens'] for r in rows),decode_forward_calls=sum(r['decode_forward_calls'] for r in rows),inverse_aggregate_throughput_ms=1000*sum(r['decode_seconds'] for r in rows)/sum(r['decode_query_tokens'] for r in rows),selection_events=sum(r['selection_events'] for r in rows),pair_breakdown={name:sum(r[name] for r in rows) for name in ('derived_prompt_decode_pairs','derived_recent_decode_pairs','derived_random_decode_pairs','derived_reasoning_causal_pairs')},baselines=baselines)
 s['reasoning_only_sparsity']=1-(s['pair_breakdown']['derived_recent_decode_pairs']+s['pair_breakdown']['derived_random_decode_pairs'])/s['pair_breakdown']['derived_reasoning_causal_pairs']
 s['per_level']={str(level):dict(questions=len(sub),correct=sum(r['correct_count'] for r in sub),rollouts=2*len(sub),accuracy=sum(r['correct_count'] for r in sub)/(2*len(sub)),mean_decode_seconds=statistics.mean(r['decode_seconds'] for r in sub),pair_sparsity=1-sum(r['selected_pairs'] for r in sub)/sum(r['causal_pairs'] for r in sub)) for level in sorted({r['level'] for r in rows},key=str) if (sub:=[r for r in rows if r['level']==level])}
 s['measurement_notes']=['Latency is per question with two concurrent rollouts; throughput sums tokens over active rows.','Percentile distributions use actual question batches; global sparsity uses summed pairs, not mean percentages.','Prompt/recent/random pair breakdown is derived from exact budget and observed lengths, and checked against measured selected/causal totals.','Full persistent KV is retained; pair sparsity does not imply equivalent KV-memory reduction.','Mean inverse aggregate throughput is not individual-token latency; no intrusive per-step profiler is enabled.','Dense legacy timing includes the whole generation batch and lacks synchronized phase timing; input hash validation is unavailable for historical dense, while decoded prompt-prefix alignment was previously audited.','GPU nodes, masks and output lengths differ; observed end-to-end timing does not isolate sampler/kernel speedup.']
 out.mkdir(parents=True,exist_ok=True)
 (out/'question_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));write_csv(out/'question_metrics.csv',rows);write_csv(out/'rollout_metrics.csv',grades)
 (out/'comprehensive_summary.json').write_text(json.dumps(s,indent=2)+'\n')
 lines=['# MATH500 random40 + recent64: full-history shared masks','',f"Status: {s['status']}; {s['questions']} questions, {s['rollouts']} completions; input alignment {s['input_alignment']}.",'',f"Correct: {s['correct']}/{s['rollouts']} ({s['accuracy']:.2%}); EOS {s['eos_rate']:.2%}; length-limit exits {s['length_limited_rollouts']}; grading errors {s['grading_error_count']}.",'','| Metric | Result |','| --- | ---: |']
 for name,label in [('prefill_seconds','Prefill seconds'),('decode_seconds','Decode seconds / question batch'),('batch_wall_seconds','Total generation seconds / question batch')]:
  q=s['distribution_stats'][name];lines.append(f"| {label}: mean / median / P90 / P95 / P99 | {q['mean']:.4f} / {q['median']:.4f} / {q['p90']:.4f} / {q['p95']:.4f} / {q['p99']:.4f} |")
 lines += [f"| Aggregate decode tokens/s | {s['decode_tokens_per_second']:.4f} |",f"| Overall / decode pair sparsity | {s['global_pair_sparsity']:.2%} / {s['decode_pair_sparsity']:.2%} |",f"| Peak allocated memory GiB | {s['peak_memory_allocated']/2**30:.3f} |",'',f"Raw overall selected/causal pairs: {s['selected_pairs']} / {s['causal_pairs']}. Decode selected/causal pairs: {s['decode_selected_pairs']} / {s['decode_causal_pairs']}.",'']
 for name,b in baselines.items():lines.append(f"Compared with {name}: accuracy {b['accuracy']:.2%}; new minus baseline {b['accuracy_difference_percentage_points']:+.2f} percentage points; new-only/baseline-only correct {b['new_only_correct']}/{b['baseline_only_correct']}.")
 lines+=['','See comprehensive_summary.json for per-level results, per-run accuracy, paired outcomes, termination, hardware and memory distributions; question_metrics.csv and rollout_metrics.csv contain every question/answer.','']+[f'- {note}' for note in s['measurement_notes']]
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return s

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--phase',choices=['smoke','formal'],default='formal');p.add_argument('--out',type=Path,required=True);a=p.parse_args();s=summarize(a.root,a.out,a.phase);print(json.dumps({k:s[k] for k in ('questions','rollouts','accuracy','decode_tokens_per_second','global_pair_sparsity')},indent=2))
