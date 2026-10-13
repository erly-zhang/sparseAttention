"""One fail-closed submission; explicit v020 exception applies to this pilot only."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from smoke import HERE, check, save, sha

parser=argparse.ArgumentParser()
parser.add_argument('--compute-node-exception',action='store_true')
args=parser.parse_args()
assert Path.cwd().resolve()==HERE
host=socket.gethostname().split('.')[0]
assert host.startswith('login') or (host=='v020' and args.compute_node_exception), host
assert not (HERE/'submitted-job.txt').exists() and not (HERE/'failure.json').exists()
check()
assert json.loads((HERE/'cpu-validation.json').read_text())==dict(
    passed=True,tests=11,source_manifest_sha256=sha(HERE/'SHA256SUMS'))
save(HERE/'submit-intent.json',dict(host=host,compute_exception=args.compute_node_exception,
    sbatch_sha256=sha(HERE/'run.sbatch'),input_manifest_sha256=sha(HERE/'inputs/manifest.json')))
with (HERE/'test-only.log').open('x') as f:
    result=subprocess.run(['sbatch','--test-only','run.sbatch'],stdout=f,stderr=subprocess.STDOUT,timeout=90)
assert result.returncode==0, 'Test-only failed; no submission'
with (HERE/'submitted-job.txt').open('x') as out, (HERE/'submit-stderr.log').open('x') as err:
    result=subprocess.run(['sbatch','--parsable','run.sbatch'],stdout=out,stderr=err,timeout=90)
assert result.returncode==0, 'Submission failed or ambiguous; inspect, never blindly retry'
job=(HERE/'submitted-job.txt').read_text().strip().split(';')[0]
assert job.isdigit()
print('ACTUAL SUBMITTED JOB',job,flush=True)
