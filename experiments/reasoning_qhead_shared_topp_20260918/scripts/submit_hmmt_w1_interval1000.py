"""Submit isolated correctness -> smoke -> gate -> formal -> collection."""
import hashlib
import argparse
import json
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--budget',type=int,choices=(1024,2048),default=1024)
parser.add_argument('--after',help='Scheduler job that must succeed before this budget starts')
options=parser.parse_args()
root=Path(__file__).resolve().parents[1]/f'results/hmmt_w1_raw_qk_k{options.budget}_hybrid4096_compact1000_20261011'
assert not (root/'jobs.json').exists(), 'Do not submit twice'
for rel,h in json.loads((root/'manifest.json').read_text()).items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
for stage in ('correctness','smoke','gate','formal','collect'):
    subprocess.run(['bash','-n',str(root/'source/scripts'/f'{stage}.sbatch')],check=True)
jobs={};previous=options.after
for stage in ('correctness','smoke','gate','formal','collect'):
    args=['sbatch','--parsable','--output',str(root/'logs'/f'{stage}-%A_%a.out')]
    if previous:args+=['--dependency',f'afterok:{previous}']
    args += [str(root/'source/scripts'/f'{stage}.sbatch'),str(root)]
    out=subprocess.check_output(args,text=True).strip();jid=out.split(';')[0]
    assert jid.isdigit(),out
    jobs[stage]=dict(job_id=jid,dependency=previous,command=args)
    (root/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    previous=jid
    print(json.dumps(dict(stage=stage,job_id=jid,dependency=jobs[stage]['dependency'])),flush=True)
