"""Smoke gate and automatic full MATH500 with a frozen matched random baseline."""
import datetime,json,os,subprocess,sys
from pathlib import Path
from rep_compact_math_common import verify,sha,rows

def launch(root,phase,cfg):
 with (root/f'{phase}_submission.lock').open('x') as f:f.write(str(os.getpid()))
 n=1 if phase=='smoke' else 8
 state=dict(phase=phase,status='submitting',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 def save():
  (root/f'{phase}_submission.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
 def submit(script,extra):
  cmd=['sbatch','--parsable',*extra,f'--output={root}/logs/%x-%A_%a.out',f'--error={root}/logs/%x-%A_%a.err',str(root/'source/scripts'/script),str(root),phase]
  r=subprocess.run(cmd,text=True,capture_output=True,check=True);job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
  return job,cmd
 try:
  state['array_job_id'],state['array_command']=submit('rep_compact_math_gpu.sbatch',[f'--array=0-{n-1}%{min(4,n)}',f'--job-name=rep-w1-{phase}'])
  state['status']='gpu_array_submitted';save()
  state['collector_job_id'],state['collector_command']=submit('rep_compact_math_cpu.sbatch',[f"--dependency=afterany:{state['array_job_id']}",f'--job-name=rep-w1-{phase}-grade'])
  state['status']='submitted';save()
 except Exception as e:
  state.update(status='submission_failed',error=str(e));save();raise

def compare(root,phase):
 current=root/'math'/phase;baseline=root/'reference_random'/phase
 new=json.loads((current/'summary.json').read_text());old=json.loads((baseline/'summary.json').read_text())
 gnew={(r['run'],r['idx']):r for r in rows(current/'grades.jsonl')};gold={(r['run'],r['idx']):r for r in rows(baseline/'grades.jsonl')};assert set(gnew)==set(gold)
 paired=dict(new_only_correct=sum(gnew[k]['correct'] and not gold[k]['correct'] for k in gnew),random_only_correct=sum(gold[k]['correct'] and not gnew[k]['correct'] for k in gnew))
 result=dict(phase=phase,representative=new,random_baseline=old,paired=paired,accuracy_difference=new['accuracy']-old['accuracy'],pair_budget_formula='Identical at matched prompt and generated lengths; observed totals may differ because output lengths differ.',scope='Single seed, two rollouts per question; different masks and A100 nodes; end-to-end latency is not isolated selector speed.')
 (current/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# W1 representative TopK compact: MATH500','',f'Phase: {phase}. Prompt outside reasoning1024; recent64 inside; after-attention compaction1088→1024 every64 reasoning tokens.','',
 '| Method | Correct / rollouts | Accuracy | Mean generated tokens | EOS | Prefill s/batch | Decode s/batch | Total s/batch | Global sparsity | Peak GiB |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,s in [('W1 representative TopK',new),('Uniform random TopK (completed baseline)',old)]:
  lines.append(f"| {name} | {s['correct']}/{s['rollouts']} | {s['accuracy']:.2%} | {s['generated_tokens']['mean']:.1f} | {s['eos_rate']:.2%} | {s['prefill_seconds_per_question_batch']['mean']:.3f} | {s['decode_seconds_per_question_batch']['mean']:.2f} | {s['batch_wall_seconds']['mean']:.2f} | {s['global_pair_sparsity']:.2%} | {s['peak_memory_allocated']/1024**3:.2f} |")
 lines+=['',f"Paired outcomes: representative-only correct {paired['new_only_correct']}; random-only correct {paired['random_only_correct']}.",
 '', 'Representative selection uses one privately sampled Q head per layer/compaction; GQA maps it to its KV head. The current post-RoPE query scores all surviving keys, then selects TopK960 among older reasoning only. Complete prompt and recent64 are protected. Tied probabilities prefer earlier original positions. The representative seed is 0 and generation seed42; selection never changes the global generation RNG.',
 '', 'All inputs, record coverage, exact attention pair budgets and final cache/cumulative lengths passed. Prefill/decode/total timings are synchronized per two-rollout question batch and exclude loading/warmup. Sources, model and data hashes are verified. Candidate sets contain only surviving KV; evicted positions cannot return.',
 '', 'Identical cache budget/cadence does not imply equal observed global sparsity: generated lengths and EOS differ. No W16/W64 trials or additional seeds are included. Smoke scores are not full MATH500 accuracy. Full results remain single-seed observations; compare output length and tokens/sec before attributing end-to-end changes to selector cost.']
 (root/f'{phase.upper()}_REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 root=Path(sys.argv[1]).resolve();phase=sys.argv[2];cfg=verify(root);assert cfg['full_run_authorized'] and cfg['query_window']==1
 if phase=='submit':
  t=json.loads((root/'pipeline_tests.json').read_text());assert t['status']=='passed' and t['manifest_sha256']==sha(root/'manifest.json')
  launch(root,'smoke',cfg);return
 assert phase in ('smoke','formal')
 status=dict(phase=phase,status='validating_and_grading',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p=root/f'{phase}_collection_status.json'
 def save():p.write_text(json.dumps(status,indent=2)+'\n')
 save()
 try:
  with (root/'logs'/f'grade_{phase}_math.log').open('x') as f:
   subprocess.run([sys.executable,'-u',str(root/'source/scripts/check_rep_compact_math.py'),str(root),'math',phase],stdout=f,stderr=subprocess.STDOUT,check=True)
  compare(root,phase)
  status.update(status='passed' if phase=='smoke' else 'completed',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
  if phase=='smoke':
   assert json.loads((root/'correctness.json').read_text())['status']=='passed'
   gate=dict(status='passed',summaries={'math/smoke/summary.json':sha(root/'math/smoke/summary.json')})
   with (root/'smoke_gate.json').open('x') as f:json.dump(gate,f,indent=2)
   launch(root,'formal',cfg)
 except Exception as e:
  status.update(status='failed',error=str(e));save();raise

if __name__=='__main__':main()
