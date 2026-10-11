"""Correctness, three full generations, validation. No automatic formal run."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();py='/project/yl258/jz648/miniconda3/envs/random-attention/bin/python'
with (root/'pipeline.lock').open('x') as f:f.write(str(os.getpid()))
env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4');env.pop('QHEAD_LOG',None)
def status(phase,**extra):
    row=dict(phase=phase,timestamp=datetime.datetime.now().astimezone().isoformat(),auto_full_submission=False,**extra)
    (root/'pipeline_status.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
try:
    import torch
    assert torch.cuda.is_available(), 'GPU correctness and inference require a Slurm GPU allocation'
    status('correctness_running')
    with (root/'correctness.log').open('x') as log:
        subprocess.run([py,'-m','unittest','discover','-s',str(root/'source/tests'),'-p','test*shared*mask*.py','-v'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/'correctness.json').write_text(json.dumps(dict(status='passed',log_sha256=hashlib.sha256((root/'correctness.log').read_bytes()).hexdigest()))+'\n')
    status('smoke_running',sample_ids=[0,1,2],reasoning_budget=1024,prompt_outside_budget=True,recent_window=64)
    subprocess.run(['bash',str(root/'source/scripts/random_fixed_recent_smoke.sh'),str(root)],env=env,check=True)
    status('validating')
    with (root/'smoke_validation.log').open('x') as log:
        subprocess.run([py,str(root/'source/scripts/check_random_fixed_recent_smoke.py'),str(root),'--phase','smoke'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('completed',summary=str(root/'smoke/summary.json'))
except Exception as e:
    status('failed',error=f'{type(e).__name__}: {e}');raise
