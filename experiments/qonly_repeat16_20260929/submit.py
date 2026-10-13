"""One attempt only; ambiguous Slurm outcomes require reconciliation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from evaluate import HERE, setup, sha, save

parser = argparse.ArgumentParser()
parser.add_argument('--compute-node-exception',action='store_true')
args = parser.parse_args()
assert Path.cwd().resolve() == HERE
host = os.uname().nodename.split('.')[0]
assert host.startswith('login') or args.compute_node_exception, 'Login node or explicit user exception required'
assert not (HERE/'failure.json').exists()
assert not (HERE/'submitted-job.txt').exists()
q, manifest, _ = setup()
validation = json.loads((HERE/'cpu-validation.json').read_text())
assert validation == dict(status='passed',tests=10,source_manifest_sha256=sha(HERE/'SHA256SUMS'))
with (HERE/'submit-intent.json').open('x') as handle:
    json.dump(dict(time=time.time(),host=host,script_sha256=sha(HERE/'run.sbatch'),
        manifest_sha256=sha(HERE/'inputs/manifest.json'),
        compute_node_exception=args.compute_node_exception,
        budget='User approved150 generations,1H10080GB,8CPU,128GB,max2h'),handle)
    handle.flush()
    os.fsync(handle.fileno())
with (HERE/'test-only.log').open('x') as handle:
    test = subprocess.run(['sbatch','--test-only','run.sbatch'],stdout=handle,stderr=subprocess.STDOUT,timeout=90)
assert test.returncode == 0, 'Test-only failed; preserve artifacts and stop'
with (HERE/'submitted-job.txt').open('x') as out, (HERE/'submit-stderr.log').open('x') as err:
    result = subprocess.run(['sbatch','--parsable','run.sbatch'],stdout=out,stderr=err,timeout=90)
assert result.returncode == 0, 'Submission failed or ambiguous; do not retry'
job = (HERE/'submitted-job.txt').read_text().strip()
assert job.split(';')[0].isdigit()
print('ACTUAL SUBMITTED JOB',job,flush=True)
