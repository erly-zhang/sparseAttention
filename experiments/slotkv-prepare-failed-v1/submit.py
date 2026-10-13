"""Single-attempt Slurm submission; uncertain outcomes require manual reconciliation."""
import json
import os
from pathlib import Path
import subprocess
import time
from evaluate import setup, save, sha, HERE

assert Path.cwd().resolve() == HERE
assert not (HERE / 'failure.json').exists()
assert not (HERE / 'submitted-job.txt').exists()
assert (HERE / 'test-only.log').is_file()
setup()
# Exclusive intent prevents retries even when sbatch times out after accepting a job.
with (HERE / 'submit-intent.json').open('x') as handle:
    json.dump(dict(time=time.time(), host=os.uname().nodename,
                   script_sha256=sha(HERE / 'run.sbatch'),
                   manifest_sha256=sha(HERE / 'inputs/manifest.json'),
                   authorization='User approved compute-node submission and H100/8CPU/128GB/6h, 500 generations'), handle)
    handle.flush()
    os.fsync(handle.fileno())
with (HERE / 'submitted-job.txt').open('x') as out, (HERE / 'submit-stderr.log').open('x') as err:
    result = subprocess.run(['sbatch', '--parsable', 'run.sbatch'], stdout=out, stderr=err, timeout=90)
assert result.returncode == 0, 'Submission failed: inspect stderr, do not retry automatically'
job = (HERE / 'submitted-job.txt').read_text().strip()
assert job.split(';')[0].isdigit(), 'Uncertain job ID: reconcile Slurm before any action'
print('ACTUAL SUBMITTED JOB', job, flush=True)
