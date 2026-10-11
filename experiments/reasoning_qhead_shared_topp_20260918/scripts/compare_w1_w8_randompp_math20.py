"""Compare saved methods on exactly IDs0..19; no new model runs."""
import hashlib,json,statistics,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1]
w=E/'results/w8_optimized_math20_20260927';rp=E/'results/official_random_pp_k1024_r64_20260924';out=w/'random_pp_comparison';out.mkdir(exist_ok=True)
def rows(p):
 with p.open() as f:return [json.loads(l) for l in f if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
wc=json.loads((w/'config.json').read_text());rc=json.loads((rp/'config.json').read_text())
matched=['model','model_geometry','model_file_hashes','dataset_sha256','seed','batch_size','rollouts','max_tokens','warmup_new_tokens','generation_temperature','generation_top_p','generation_top_k_applied','packages','timing']
for k in matched:assert wc[k]==rc[k],k
for f in ['Utils/grader.py','Utils/parser.py']:
 assert sha(w/'source/ra_harness/kvcompress/harness'/f)==sha(rp/'source/ra_harness/kvcompress/harness'/f),f
assert json.loads((rp/'formal/summary.json').read_text())['status']=='completed'
assert json.loads((w/'collection_status.json').read_text())['status']=='completed'
ref={r['idx']:r for r in rows(w/'inputs.jsonl')};ids=set(range(20));assert set(ref)==ids
sys.path[:0]=[str(w/'source/ra_harness/kvcompress/harness')]
from Utils.parser import extract_answer
from Utils.grader import check_is_correct
metrics={};grades={};hardware={};provenance={};results={}
for mode,root,pat,gfile in [('random_pp',rp,'formal/shard_*',rp/'formal/grades.jsonl'),('w1',w,'shard_*',w/'w1_grades.jsonl'),('w8',w,'shard_*',w/'w8_grades.jsonl')]:
 mm={};cc={};hw=[]
 for d in sorted(root.glob(pat)):
  base=d if mode=='random_pp' else d/mode
  assert json.loads((base/'exit.json').read_text())['exit_code']==0
  hw.append(json.loads((d/'hardware.json').read_text()))
  p=base/'cache_metrics.jsonl';provenance[str(p)]=sha(p)
  for r in rows(p):
   i=r['idx']
   if i not in ids:continue
   assert i not in mm
   for k in ['input_tokens','input_ids_sha256']:assert r[k]==ref[i][k],(mode,i,k)
   assert len(r['generated_tokens_per_rollout'])==len(r['eos_reached'])==2
   assert r['decode_query_tokens']==sum(r['generated_tokens_per_rollout'])-2
   assert r['selected_pairs']==r['prefill_selected_pairs']+r['decode_selected_pairs']
   assert r['causal_pairs']==r['prefill_causal_pairs']+r['decode_causal_pairs']
   for count,eos in zip(r['generated_tokens_per_rollout'],r['eos_reached']):
    assert 1<=count<=wc['max_tokens']-r['input_tokens']
    if not eos:assert count==wc['max_tokens']-r['input_tokens']
   mm[i]=r
  for run in (0,1):
   files=list((base/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1
   provenance[str(files[0])]=sha(files[0])
   for r in rows(files[0]):
    i=r['idx']
    if i not in ids:continue
    key=(run,i);assert key not in cc;prompt=ref[i]['prompt_text'];assert r['completion'].startswith(prompt)
    cc[key]=r['completion'][len(prompt):]
 assert set(mm)==ids and set(cc)=={(run,i) for run in (0,1) for i in ids}
 gg={}
 for r in rows(gfile):
  if r['idx'] not in ids:continue
  key=r['run'],r['idx'];assert key not in gg
  assert r.get('grading_error') is None
  assert r['generated_tokens']==mm[r['idx']]['generated_tokens_per_rollout'][r['run']]
  assert r['eos_reached']==mm[r['idx']]['eos_reached'][r['run']]
  # Regrade all120 outputs using the same pinned parser/grader and prompt strip.
  pred=extract_answer(cc[key]);correct=int(bool(check_is_correct(pred,ref[r['idx']]['gold'])))
  assert correct==r['correct'],(mode,key,'grade mismatch')
  gg[key]=r
 assert set(gg)==set(cc)
 provenance[str(gfile)]=sha(gfile);vv=list(mm.values());gl=list(gg.values())
 for h in hw:assert (h.get('gpu_name') or h.get('gpu'))=='NVIDIA A100-SXM4-80GB'
 results[mode]=dict(questions=20,rollouts=40,correct=sum(g['correct'] for g in gl),accuracy=statistics.mean(g['correct'] for g in gl),per_run_correct=[sum(g['correct'] for (run,i),g in gg.items() if run==j) for j in (0,1)],mean_decode_seconds=statistics.mean(r['decode_seconds'] for r in vv),median_decode_seconds=statistics.median(r['decode_seconds'] for r in vv),mean_prefill_seconds=statistics.mean(r['prefill_seconds'] for r in vv),mean_total_seconds=statistics.mean(r['batch_wall_seconds'] for r in vv),decode_tokens_per_second=sum(r['decode_query_tokens'] for r in vv)/sum(r['decode_seconds'] for r in vv),mean_generated_tokens=statistics.mean(g['generated_tokens'] for g in gl),length_limited_rollouts=sum(not g['eos_reached'] for g in gl),selected_pairs=sum(r['selected_pairs'] for r in vv),causal_pairs=sum(r['causal_pairs'] for r in vv),global_pair_sparsity=1-sum(r['selected_pairs'] for r in vv)/sum(r['causal_pairs'] for r in vv),peak_memory_gib=max(r['peak_memory_allocated'] for r in vv)/1024**3)
 metrics[mode]=mm;grades[mode]=gg;hardware[mode]=hw
baseline=results['random_pp'];comparisons={}
for mode in ('w1','w8'):
 s=results[mode];comparisons[mode]=dict(accuracy_change_percentage_points=100*(s['accuracy']-baseline['accuracy']),decode_time_change=s['mean_decode_seconds']/baseline['mean_decode_seconds']-1,throughput_change=s['decode_tokens_per_second']/baseline['decode_tokens_per_second']-1,output_length_change=s['mean_generated_tokens']/baseline['mean_generated_tokens']-1,only_method_correct=[list(k) for k,g in grades[mode].items() if g['correct'] and not grades['random_pp'][k]['correct']],only_random_pp_correct=[list(k) for k,g in grades[mode].items() if not g['correct'] and grades['random_pp'][k]['correct']])
summary=dict(status='completed',validation='Share with caveats: all20 inputs aligned, no missing/duplicates, all120 outputs regraded identically',scope='MATH500 IDs0..19, two rollouts each, existing results only',matched_config_fields=matched,prompt_tokens=dict(min=min(r['input_tokens'] for r in ref.values()),max=max(r['input_tokens'] for r in ref.values()),mean=statistics.mean(r['input_tokens'] for r in ref.values())),results=results,versus_random_pp=comparisons,hardware=hardware,timing_caveat='Same A10080 model and timing scope; historical random_pp (17 allocated CPUs) vs newer paired W1/W8 (8 allocated CPUs), different dates/nodes. Not a three-way same-GPU paired timing or equal-budget algorithm ablation.',budget_caveat='Random_pp total retained1024 includes prompt and recent64; W1/W8 retain prompt plus1024 reasoning including recent64. Random_pp selects independently per KV head, W1/W8 share positions across all heads within each layer/rollout.',source_sha256=provenance)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Optimized W1 / W8 / official random_pp: matched first20 MATH500 inputs','',summary['scope']+'. All input token counts and SHA256 match; all120 complete outputs regraded with the same pinned parser/grader, agreeing with saved grades. Accuracy is mean correctness over40 outputs per method, not pass@2.','', '| Method | Correct/40 | Accuracy | Mean decode s/question batch | Decode tokens/s | Mean output tokens | Length-limit hits/40 | Pair sparsity | Peak GiB |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,s in results.items():lines.append(f"| {name} | {s['correct']} | {s['accuracy']:.1%} | {s['mean_decode_seconds']:.2f} | {s['decode_tokens_per_second']:.2f} | {s['mean_generated_tokens']:.1f} | {s['length_limited_rollouts']} | {s['global_pair_sparsity']:.2%} | {s['peak_memory_gib']:.2f} |")
lines+=['',summary['timing_caveat'],'',summary['budget_caveat'],'','Mean decode time is for a batch of two sampled rollouts. It excludes loading/warmup/metric serialization; includes selector, gather, attention, generation sampling and bookkeeping. Pair sparsity is1 minus summed selected pairs divided by summed causal pairs; longer outputs can raise this sparsity without lowering the per-step retained budget. The first20 questions and single seed do not establish full-dataset superiority.','']
for name,s in comparisons.items():lines.append(f"{name} versus random_pp: accuracy {s['accuracy_change_percentage_points']:+.1f}pp; mean decode {s['decode_time_change']:+.2%}; throughput {s['throughput_change']:+.2%}; output length {s['output_length_change']:+.2%}.")
lines+=['','Sources: paired W1/W8 `../summary.json` and raw shard outputs; random_pp `../../official_random_pp_k1024_r64_20260924/formal/` raw outputs and grades. Full file hashes and hardware in this comparison summary.json.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('hardware','source_sha256')},indent=2),flush=True)
