"""Correctness -> complete same3 smoke -> exact-input grading -> authorized full500 submission."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();py=sys.executable
with (root/'pipeline.lock').open('x') as f:f.write(str(os.getpid()))
env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
for k in ('QHEAD_LOG','KEEPLOG','KEEPLOG_POS','FORCE_KEEP_RANGE','LOG_UNION','EVICT_TIMING'):env.pop(k,None)
def status(phase,**kw):
 row=dict(phase=phase,timestamp=datetime.datetime.now().astimezone().isoformat(),**kw)
 (root/'pipeline_status.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
try:
 status('correctness_running')
 with (root/'correctness.log').open('x') as log:
  for pattern in ('test*shared*mask*.py','test_official_random_pp.py'):
   subprocess.run([py,'-m','unittest','discover','-s',str(root/'source/tests'),'-p',pattern,'-v'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 (root/'correctness.json').write_text(json.dumps(dict(status='passed',log_sha256=hashlib.sha256((root/'correctness.log').read_bytes()).hexdigest()))+'\n')
 status('smoke_running',sample_ids=[0,1,2])
 subprocess.run(['bash',str(root/'source/scripts/official_random_pp_smoke.sh'),str(root)],env=env,check=True)
 status('validating_smoke')
 with (root/'smoke_validation.log').open('x') as log:
  subprocess.run([py,str(root/'source/scripts/collect_official_random_pp.py'),str(root),'--validate-smoke'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 status('submitting_full500')
 subprocess.run([py,str(root/'source/scripts/submit_official_random_pp.py'),str(root)],env=env,check=True)
 sub=json.loads((root/'submission.json').read_text());assert sub['status']=='submitted'
 status('full_submitted',array_job_id=sub['array_job_id'],collector_job_id=sub['collector_job_id'])
except Exception as exc:status('failed_no_further_submission',error=f'{type(exc).__name__}: {exc}');raise
