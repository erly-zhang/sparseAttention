"""Submit the already validated full experiment exactly once; preserve both job IDs."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
assert json.loads((root/'config.json').read_text())['full_run_authorized']
assert json.loads((root/'report_validation.json').read_text())['status']=='passed'
with (root/'submission.lock').open('x') as f:f.write(str(os.getpid()))
state=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),status='submitting',root=str(root))
def save():
 (root/'submission.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
def submit(script,extra):
 cmd=['sbatch','--parsable',*extra,f'--output={root}/logs/%x-%A_%a.out',f'--error={root}/logs/%x-%A_%a.err',str(root/'source/scripts'/script),str(root)]
 r=subprocess.run(cmd,text=True,capture_output=True,check=True);job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout;return job,cmd
try:
 cfg=json.loads((root/'config.json').read_text());name='mask-w1-rec64' if cfg['method']=='qhead_shared_mask_w1_recent_gpu' else 'mask-w1'
 state['array_job_id'],state['array_command']=submit('w1_full_array.sbatch',[f'--job-name={name}']);state['status']='array_submitted';save()
 state['collector_job_id'],state['collector_command']=submit('w1_full_collect.sbatch',[f"--dependency=afterany:{state['array_job_id']}"]);state['status']='submitted';save()
except Exception as exc:
 state.update(status='submission_failed',error=f'{type(exc).__name__}: {exc}');save();raise
