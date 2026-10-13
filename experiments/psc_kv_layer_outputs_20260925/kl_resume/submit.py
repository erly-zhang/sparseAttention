"""Submit only after user approves a revised cumulative cap; lock prevents duplicate jobs."""
import argparse,hashlib,json,subprocess,os,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
p=argparse.ArgumentParser();p.add_argument('--approved-total-seconds',required=True,type=int);a=p.parse_args()
assert a.approved_total_seconds>=787+6360, '106-minute job does not fit this cumulative cap'
active=subprocess.check_output(['squeue','-h','-u','jzhang114','-o','%j'],text=True)
assert not any(x.startswith(('psc-kl-resume','psc-kv-layer5')) for x in active.splitlines()), 'experiment already active'
assert not (R/'run_kl_resume').exists(), 'existing output directory; do not duplicate'
(P/'.submission-lock').mkdir()
files=list(P.glob('*.py'))+[P/'run.sbatch',R/'compare.py',R/'adapters.py',R/'summarize.py',R/'experiment_config.json',R/'environment.sh']
(P/'source_sha256.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
with tarfile.open(P/'source_snapshot.tar.gz','w:gz') as tar:
    for f in files:tar.add(f,arcname=str(f.relative_to(R)))
env=dict(os.environ,PSC_APPROVED_TOTAL_SECONDS=str(a.approved_total_seconds))
result=subprocess.run(['sbatch','--parsable',str(P/'run.sbatch')],env=env,text=True,capture_output=True)
(P/'submission.json').write_text(json.dumps({'approved_total_seconds':a.approved_total_seconds,'prior_seconds':787,'wall_seconds':6360,'stdout':result.stdout,'stderr':result.stderr,'returncode':result.returncode},indent=2))
print(result.stdout,result.stderr);result.check_returncode()
