"""Grade only complete aligned W16/W64 runs and aggregate decode/pair measurements."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root.resolve()
for rel,digest in json.loads((root/'manifest.json').read_text()).items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
expected={r['idx']:r for r in map(json.loads,(root/'inputs.jsonl').read_text().splitlines())};assert set(expected)==set(range(500))
missing=[]
for w in (16,64):
    for s in range(8):
        p=root/f'w{w}/shard_{s}/exit.json'
        if not p.exists() or json.loads(p.read_text()).get('exit_code')!=0:missing.append(str(p))
if missing:
    (root/'comparison_status.json').write_text(json.dumps(dict(status='incomplete_or_failed',missing_successful_workers=missing),indent=2)+'\n')
    raise SystemExit('Incomplete workers: formal accuracy not reported; see comparison_status.json')
sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
from Utils.parser import extract_answer
from Utils.grader import check_is_correct
import numpy as np
config=json.loads((root/'config.json').read_text());geometry=config['model_geometry'];factor=geometry['num_attention_heads']*geometry['num_hidden_layers']

def stats(values):
    return dict(mean=statistics.mean(values),median=statistics.median(values),p90=float(np.percentile(values,90)),total=sum(values))

summary=dict(status='completed',input_alignment='passed/full',expected_questions_per_window=500,rollouts_per_question=2,methods={},timing_scope=config['timing'],comparison_caveat=config['comparison_caveat'])
all_grades={};all_metrics={}
for w in (16,64):
    metrics={};completions={};hardware=[]
    for shard in range(8):
        d=root/f'w{w}/shard_{shard}';hardware.append(json.loads((d/'hardware.json').read_text()))
        for r in map(json.loads,(d/'cache_metrics.jsonl').read_text().splitlines()):
            idx=r['idx'];assert idx in expected and idx%8==shard and idx not in metrics
            assert r['query_window']==w and r['selection_p']==.95 and r['cache_semantics']=='full_history_temporary_mask'
            assert r['input_tokens']==expected[idx]['input_tokens'] and r['input_ids_sha256']==expected[idx]['input_ids_sha256']
            assert len(r['generated_tokens_per_rollout'])==len(r['eos_reached'])==2
            for lengths,cumulative in zip(r['final_cache_seqlens'],r['final_cumulative_length']):assert all(n==cumulative for n in lengths)
            assert len(r['final_cache_seqlens'])==geometry['num_hidden_layers']
            assert r['prefill_selected_pairs']==r['prefill_causal_pairs']==2*factor*r['input_tokens']*(r['input_tokens']+1)//2
            assert r['selected_pairs']==r['prefill_selected_pairs']+r['decode_selected_pairs']
            assert r['causal_pairs']==r['prefill_causal_pairs']+r['decode_causal_pairs']
            assert r['decode_query_tokens']==sum(r['generated_tokens_per_rollout'])-2
            assert r['selection_events']==r['decode_forward_calls']*geometry['num_hidden_layers']
            assert r['causal_pairs']==factor*sum((r['input_tokens']+g-1)*(r['input_tokens']+g)//2 for g in r['generated_tokens_per_rollout'])
            assert 0<=r['decode_selected_pairs']<=r['decode_causal_pairs']
            assert r['prefill_seconds']>=0 and r['decode_seconds']>=0
            metrics[idx]=r
        for run in (0,1):
            for r in map(json.loads,(d/f'run_{run}/completions.jsonl').read_text().splitlines()):
                idx=r['idx'];key=(run,idx);assert idx in expected and idx%8==shard and key not in completions
                assert r['completion'].startswith(expected[idx]['prompt_text'])
                completions[key]=r['completion']
    assert set(metrics)==set(expected)
    assert set(completions)=={(r,i) for r in (0,1) for i in expected}
    grades=[];runs=[]
    for run in (0,1):
        run_rows=[]
        for idx in range(500):
            grading_error=None
            try:
                prediction=extract_answer(completions[(run,idx)],'math')
                correct=int(bool(check_is_correct(prediction,expected[idx]['gold'])))
            except Exception as exc:
                prediction='';correct=0;grading_error=f'{type(exc).__name__}: {exc}'
            row=dict(run=run,idx=idx,prediction=prediction,correct=correct,grading_error=grading_error,eos_reached=metrics[idx]['eos_reached'][run],generated_tokens=metrics[idx]['generated_tokens_per_rollout'][run])
            grades.append(row);run_rows.append(row)
        runs.append(dict(run=run,correct=sum(r['correct'] for r in run_rows),count=500,accuracy=sum(r['correct'] for r in run_rows)/500,eos_rate=sum(r['eos_reached'] for r in run_rows)/500))
    with (root/f'w{w}/grades.jsonl').open('x') as stream:
        stream.writelines(json.dumps(row)+'\n' for row in grades)
    vals=list(metrics.values());sel=sum(r['selected_pairs'] for r in vals);causal=sum(r['causal_pairs'] for r in vals)
    ds=sum(r['decode_selected_pairs'] for r in vals);dc=sum(r['decode_causal_pairs'] for r in vals)
    dt=sum(r['decode_seconds'] for r in vals);tokens=sum(r['decode_query_tokens'] for r in vals)
    info=dict(questions=500,rollouts=1000,runs=runs,correct=sum(r['correct'] for r in grades),accuracy=sum(r['correct'] for r in grades)/1000,
              correct_and_eos=sum(r['correct'] and r['eos_reached'] for r in grades)/1000,eos_rate=sum(r['eos_reached'] for r in grades)/1000,
              length_limited_rollouts=sum(not r['eos_reached'] for r in grades),generated_tokens=stats([r['generated_tokens'] for r in grades]),
              prefill_seconds_per_question_batch=stats([r['prefill_seconds'] for r in vals]),decode_seconds_per_question_batch=stats([r['decode_seconds'] for r in vals]),
              batch_wall_seconds=stats([r['batch_wall_seconds'] for r in vals]),decode_query_tokens=tokens,decode_tokens_per_second=tokens/dt if dt else None,
              decode_milliseconds_per_token=1000*dt/tokens if tokens else None,
              selected_pairs=sel,causal_pairs=causal,global_pair_sparsity=1-sel/causal,
              decode_selected_pairs=ds,decode_causal_pairs=dc,decode_pair_sparsity=1-ds/dc if dc else 0,
              max_full_cache_tokens=max(r['max_visible_tokens'] for r in vals),max_selected_tokens=max(r['max_selected_tokens'] for r in vals),
              peak_memory_allocated=max(r['peak_memory_allocated'] for r in vals),hardware=hardware)
    summary['methods'][str(w)]=info;all_grades[w]={(r['run'],r['idx']):r for r in grades};all_metrics[w]=metrics
for idx in range(500):assert all_metrics[16][idx]['input_ids_sha256']==all_metrics[64][idx]['input_ids_sha256']
summary['paired_outcomes']=dict(w16_only_correct=sum(all_grades[16][k]['correct'] and not all_grades[64][k]['correct'] for k in all_grades[16]),w64_only_correct=sum(all_grades[64][k]['correct'] and not all_grades[16][k]['correct'] for k in all_grades[16]))
summary['hardware_uniform']=len({(h['gpu_name'],h['gpu_total_memory']) for m in summary['methods'].values() for h in m['hardware']})==1
with (root/'comparison_summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
lines=['# MATH500 full-history shared-mask W16 / W64 comparison','', 'Top-p=0.95; 500 questions per window, two rollouts per question. Input alignment: passed/full.','', '| Query window | Correct / 1000 | Accuracy | Mean decode batch (s) | Decode tokens/s | Decode pair sparsity | Overall pair sparsity |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for w in (16,64):
    m=summary['methods'][str(w)];lines.append(f"| {w} | {m['correct']}/1000 | {m['accuracy']:.2%} | {m['decode_seconds_per_question_batch']['mean']:.3f} | {m['decode_tokens_per_second']:.3f} | {m['decode_pair_sparsity']:.2%} | {m['global_pair_sparsity']:.2%} |")
lines+=['','Decode batch latency covers two concurrently generated rollouts, with finished rows removed. It is not per-rollout latency. Loading, two-token warmup and metric serialization are excluded. Formal per-step diagnostics are off. Output lengths may differ; latency and throughput must be interpreted together. Full KV memory is retained.','',f"GPU model/memory uniform across workers: {summary['hardware_uniform']}. Raw totals, median/P90 times, termination counts, per-run accuracy and hardware records are in comparison_summary.json.",'']
(root/'REPORT.md').write_text('\n'.join(lines));(root/'comparison_status.json').write_text(json.dumps(dict(status='completed',input_alignment='passed/full'),indent=2)+'\n')
print(json.dumps({w:{k:v for k,v in m.items() if k in ('accuracy','decode_tokens_per_second','decode_pair_sparsity','global_pair_sparsity')} for w,m in summary['methods'].items()},indent=2))
