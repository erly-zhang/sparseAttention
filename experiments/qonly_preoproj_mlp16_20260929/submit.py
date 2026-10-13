"""One actual submission attempt. Never retry ambiguous scheduler outcomes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from pilot import HERE, check
from protocol import sha, save

parser=argparse.ArgumentParser()
parser.add_argument('--compute-node-exception',action='store_true')
args=parser.parse_args()
assert Path.cwd().resolve()==HERE
host=os.uname().nodename.split('.')[0]
assert host.startswith('login') or (host=='gl002' and args.compute_node_exception)
assert not (HERE/'failure.json').exists()
assert not (HERE/'submitted-job.txt').exists()
check()
assert json.loads((HERE/'cpu-validation.json').read_text())==dict(status='passed',tests=11,
    source_manifest_sha256=sha(HERE/'SHA256SUMS'))
save(HERE/'submit-intent.json',dict(time=time.time(),host=host,
    script_sha256=sha(HERE/'run.sbatch'),manifest_sha256=sha(HERE/'inputs/manifest.json'),
    compute_node_exception=args.compute_node_exception,
    budget='User approved fresh48 sequential MLP training/test,1H10080GB,8CPU,128GB,max4h'))
with (HERE/'test-only.log').open('x') as handle:
    test=subprocess.run(['sbatch','--test-only','run.sbatch'],stdout=handle,stderr=subprocess.STDOUT,timeout=90)
assert test.returncode==0,'Test-only failed; preserve and stop'
with (HERE/'submitted-job.txt').open('x') as out, (HERE/'submit-stderr.log').open('x') as err:
    result=subprocess.run(['sbatch','--parsable','run.sbatch'],stdout=out,stderr=err,timeout=90)
assert result.returncode==0,'Submission failed or ambiguous; no retry'
job=(HERE/'submitted-job.txt').read_text().strip()
assert job.split(';')[0].isdigit()
print('ACTUAL SUBMITTED JOB',job,flush=True)
