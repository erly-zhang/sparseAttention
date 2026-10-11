"""User-authorized submission controller: wait for technical smoke success, fail closed."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
control=Path(sys.argv[1]).resolve();cfg=json.loads((control/'config.json').read_text())
smoke=Path(cfg['smoke']);root=Path(cfg['formal_root']);scripts=control/'scripts'
with (control/'controller.lock').open('x') as f:f.write(str(os.getpid()))
env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def status(phase,**kw):
 row=dict(phase=phase,timestamp=datetime.datetime.now().astimezone().isoformat(),**kw)
 (control/'status.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
try:
 status('waiting_for_smoke',smoke=str(smoke),formal_root=str(root))
 deadline=time.monotonic()+3*3600
 while True:
  state=json.loads((smoke/'pipeline_status.json').read_text())
  if state['phase']=='completed':break
  if state['phase']=='failed':raise RuntimeError(f'Smoke failed: {state}')
  if time.monotonic()>deadline:raise TimeoutError('Smoke not complete within three hours; no full run submitted')
  time.sleep(30)
 status('preparing_frozen_full_run')
 subprocess.run([sys.executable,str(scripts/'prepare_w1_full.py'),str(root),'--smoke',str(smoke)],env=env,check=True)
 status('validating_report_on_passed_smoke')
 subprocess.run([sys.executable,str(root/'source/scripts/collect_w1_full.py'),str(root),'--validate-smoke'],env=env,check=True)
 status('submitting_full_run')
 subprocess.run([sys.executable,str(root/'source/scripts/submit_w1_full.py'),str(root)],env=env,check=True)
 sub=json.loads((root/'submission.json').read_text());assert sub['status']=='submitted'
 status('submitted',array_job_id=sub['array_job_id'],collector_job_id=sub['collector_job_id'],formal_root=str(root))
except Exception as exc:status('failed_no_further_submission',error=f'{type(exc).__name__}: {exc}');raise
