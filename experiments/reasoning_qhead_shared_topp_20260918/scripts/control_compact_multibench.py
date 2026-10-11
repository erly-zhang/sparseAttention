"""Submit once, validate every smoke before formal, collect per-benchmark results."""
import datetime,json,os,subprocess,sys,traceback
from pathlib import Path
from compact_multibench_common import verify,sha

def submit(root,script,phase,extra):
 cmd=['sbatch','--parsable',*extra,f'--output={root}/logs/%x-%A_%a.out',f'--error={root}/logs/%x-%A_%a.err',str(root/'source/scripts'/script),str(root),phase]
 r=subprocess.run(cmd,text=True,capture_output=True,check=True);job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
 return job,cmd

def launch(root,phase,cfg):
 with (root/f'{phase}_submission.lock').open('x') as f:f.write(str(os.getpid()))
 n=len(cfg['tasks']) if phase=='smoke' else len(json.loads((root/'formal_work.json').read_text()))
 state=dict(phase=phase,status='submitting',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 def save():
  (root/f'{phase}_submission.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
 try:
  state['array_job_id'],state['array_command']=submit(root,'compact_multibench_gpu.sbatch',phase,[f'--array=0-{n-1}%{min(8,n)}',f'--job-name=compact-bench-{phase}'])
  state['status']='gpu_array_submitted';save()
  state['collector_job_id'],state['collector_command']=submit(root,'compact_multibench_cpu.sbatch',phase,[f"--dependency=afterany:{state['array_job_id']}",f'--job-name=compact-bench-{phase}-grade'])
  state['status']='submitted';save()
 except Exception as e:
  state.update(status='submission_failed',error=str(e));save();raise

def main():
 root=Path(sys.argv[1]).resolve();phase=sys.argv[2];cfg=verify(root)
 assert cfg['full_run_authorized']
 if phase=='submit':
  test=json.loads((root/'pipeline_tests.json').read_text());assert test['status']=='passed' and test['manifest_sha256']==sha(root/'manifest.json')
  launch(root,'smoke',cfg);return
 assert phase in ('smoke','formal')
 status=dict(phase=phase,status='validating_and_grading',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks={})
 statusfile=root/f'{phase}_collection_status.json'
 def save():statusfile.write_text(json.dumps(status,indent=2)+'\n')
 save();summaries={};errors={}
 for task in cfg['tasks']:
  name=task['name']
  try:
   with (root/'logs'/f'grade_{phase}_{name}.log').open('x') as f:
    r=subprocess.run([sys.executable,'-u',str(root/'source/scripts/check_compact_multibench.py'),str(root),name,phase],stdout=f,stderr=subprocess.STDOUT)
   assert r.returncode==0,f'grading process exited {r.returncode}; inspect logs/grade_{phase}_{name}.log'
   summaries[name]=json.loads((root/name/phase/'summary.json').read_text());status['tasks'][name]='passed';save()
  except Exception as e:
   errors[name]=str(e);status['tasks'][name]=dict(status='failed',error=str(e));save()
 if errors:
  status.update(status='failed',errors=errors);save();raise RuntimeError(errors)
 lines=['# Compressed KV multi-benchmark experiment','',f'Phase: {phase}. Two sampled rollouts per question; scores are mean per-rollout correctness (LCB pass@1 estimate).','',
 'Qwen3-4B, BF16, A100 80GB, temperature 0.6, generation top-p 0.95, seed42, total length cap32768. Entire prompt protected outside reasoning1024; recent64 inside budget; post-attention compaction1088→1024 every64 steps.','',
 '| Dataset | Questions | Correct / rollouts | Accuracy | Mean output tokens | Mean prefill s / batch | Mean decode s / batch | Global pair sparsity | Peak GiB |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,s in summaries.items():
  lines.append(f"| {s['dataset_version']} | {s['questions']} | {s['correct']}/{s['rollouts']} | {s['accuracy']:.2%} | {s['generated_tokens']['mean']:.1f} | {s['prefill_seconds_per_question_batch']['mean']:.3f} | {s['decode_seconds_per_question_batch']['mean']:.2f} | {s['global_pair_sparsity']:.2%} | {s['peak_memory_allocated']/1024**3:.2f} |")
 lines+=['', 'All inputs verified by token count and input_ids SHA256; all exact pair budgets and record coverage passed. Timings include the two-rollout batch; loading/warmup are excluded. No dense or random_pp comparison on these datasets is implied.', '', 'GPQA uses official Diamond data with deterministic choice shuffling; LCB is release_v6 medium only, not the entire release. LCB requires all available public/private tests, 20s per test, 2048MiB process limit. This is the local project grader; comparison to external leaderboard scores requires matching grader and prompt settings.', '', 'A smoke gate certifies technical correctness and grading completion, not that every model answer is correct.']
 (root/f'{phase.upper()}_REPORT.md').write_text('\n'.join(lines)+'\n')
 status.update(status='passed' if phase=='smoke' else 'completed',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
 if phase=='smoke':
  gate=dict(status='passed',summaries={f'{name}/smoke/summary.json':sha(root/name/'smoke/summary.json') for name in summaries})
  with (root/'smoke_gate.json').open('x') as f:json.dump(gate,f,indent=2)
  launch(root,'formal',cfg)

if __name__=='__main__':main()
