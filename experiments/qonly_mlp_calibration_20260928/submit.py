"""Explicitly authorized one-shot independent pilot submission."""
import json
import os
import subprocess
import time
from pilot import HERE, check
from protocol import save, sha

assert HERE == __import__('pathlib').Path.cwd().resolve()
assert not (HERE / 'failure.json').exists()
assert not (HERE / 'run-start.json').exists()
assert not (HERE / 'submitted-job.txt').exists()
assert 'ALL16 CPU TESTS PASS' in (HERE / 'cpu-tests.log').read_text()
assert (HERE / 'test-only.log').is_file()
check()
save(HERE / 'submit-intent.json', dict(time=time.time(), host=os.uname().nodename,
    manifest_sha256=sha(HERE / 'inputs/manifest.json'), script_sha256=sha(HERE / 'run.sbatch'),
    authorization='User approved48 synthetic pilot,H10080GB x1,8CPU,128GB,2h; explicitly allowed compute-node sbatch'))
with (HERE / 'submitted-job.txt').open('x') as out, (HERE / 'submit-stderr.log').open('x') as err:
    result = subprocess.run(['sbatch', '--parsable', 'run.sbatch'], stdout=out, stderr=err, timeout=90)
assert result.returncode == 0, 'Preserve failed intent and stderr; no automatic retry'
job = (HERE / 'submitted-job.txt').read_text().strip()
assert job.split(';')[0].isdigit(), 'Ambiguous submission: reconcile before any further action'
print('ACTUAL SUBMITTED JOB', job, flush=True)
