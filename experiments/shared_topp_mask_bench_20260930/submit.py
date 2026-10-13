"""Exactly one four-element array submission after frozen validation and test-only."""
import json
import os
from pathlib import Path
import socket
import subprocess
from protocol import RUN, sha, save

assert Path(__file__).resolve().parent == RUN
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
host=socket.gethostname().split('.')[0]
assert os.environ.get('AUTHORIZED_SUBMIT_HOST')==host, 'Set only after live host and one-time user authorization check'
assert not (RUN/'submitted-job.txt').exists() and not (RUN/'submit-intent.json').exists()
for line in (RUN/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split()
    assert sha(RUN/name)==expected
validation=json.loads((RUN/'cpu-validation.json').read_text())
assert validation['status']=='passed' and validation['tests']==25
assert validation['source_manifest_sha256']==sha(RUN/'SHA256SUMS')
assert validation['input_manifest_sha256']==sha(RUN/'inputs/manifest.json')
assert validation['log_sha256']==sha(RUN/'cpu-tests.log')
script=RUN/'run.sbatch'
for option in ('--array=0-3%4','--gres=gpu:h100-80:1','--cpus-per-task=8','--mem=128G','--time=24:00:00','--no-requeue'):
    assert '#SBATCH '+option in script.read_text()
test=subprocess.run(['sbatch','--test-only',str(script)],cwd=RUN,capture_output=True,text=True)
with (RUN/'test-only.log').open('x') as f:
    f.write(test.stdout+test.stderr)
print(test.stdout+test.stderr,flush=True)
assert test.returncode==0
save(RUN/'submit-intent.json',dict(source_manifest_sha256=sha(RUN/'SHA256SUMS'),
    input_manifest_sha256=sha(RUN/'inputs/manifest.json'),script_sha256=sha(script),
    submit_host=host,
    authorization='2026-09-30 user approved NEW 4 H10080GB, each 8CPU 128GB max24h; one authenticated compute-node submission; no retry',
    test_only_sha256=sha(RUN/'test-only.log')))
# Failure after this point is ambiguous: preserve intent; NEVER automatically retry.
result=subprocess.run(['sbatch','--parsable',str(script)],cwd=RUN,capture_output=True,text=True)
with (RUN/'submit-stdout.log').open('x') as f:
    f.write(result.stdout)
with (RUN/'submit-stderr.log').open('x') as f:
    f.write(result.stderr)
assert result.returncode==0,(result.stdout,result.stderr)
job=result.stdout.strip().split(';')[0]
assert job.isdigit()
with (RUN/'submitted-job.txt').open('x') as f:
    f.write(result.stdout.strip()+'\n')
    f.flush()
    os.fsync(f.fileno())
print('ACTUAL SUBMITTED ARRAY',job,flush=True)
