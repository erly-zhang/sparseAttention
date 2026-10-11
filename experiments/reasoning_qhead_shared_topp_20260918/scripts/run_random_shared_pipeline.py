"""Run correctness + three complete generations, then submit the authorized MATH500 run."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();py='/project/yl258/jz648/miniconda3/envs/random-attention/bin/python'
with (root/'pipeline.lock').open('x') as f:f.write(str(os.getpid()))
env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
env.pop('QHEAD_LOG',None)
def status(phase,**kwargs):
    row=dict(phase=phase,timestamp=datetime.datetime.now().astimezone().isoformat(),**kwargs)
    (root/'pipeline_status.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
def submit(script,extra=()):
    result=subprocess.run(['sbatch','--parsable',*extra,f'--output={root}/logs/%x-%A_%a.out',f'--error={root}/logs/%x-%A_%a.err',str(root/'source/scripts'/script),str(root)],env=env,text=True,capture_output=True,check=True)
    job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout;return job
try:
    status('correctness_running')
    with (root/'correctness.log').open('x') as log:
        subprocess.run([py,'-m','unittest','discover','-s',str(root/'source/tests'),'-p','test*shared*mask*.py','-v'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/'correctness.json').write_text(json.dumps(dict(status='passed',log_sha256=hashlib.sha256((root/'correctness.log').read_bytes()).hexdigest()))+'\n')
    status('smoke_running',sample_ids=[0,1,2],random_keep_ratio=.4,max_tokens=32768)
    subprocess.run(['bash',str(root/'source/scripts/random_shared_worker.sh'),str(root),'smoke'],env=env,check=True)
    status('smoke_validating')
    with (root/'smoke_validation.log').open('x') as log:
        subprocess.run([py,str(root/'source/scripts/check_random_shared.py'),str(root),'--phase','smoke'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('smoke_passed_submitting')
    array=submit('random_shared_array.sbatch')
    (root/'submission.json').write_text(json.dumps(dict(array_job_id=array),indent=2)+'\n')
    status('formal_submitted',array_job_id=array)
    collector=submit('random_shared_collect.sbatch',[f'--dependency=afterany:{array}'])
    (root/'submission.json').write_text(json.dumps(dict(array_job_id=array,collector_job_id=collector),indent=2)+'\n')
    status('formal_submitted',array_job_id=array,collector_job_id=collector)
except Exception as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}')
    raise
