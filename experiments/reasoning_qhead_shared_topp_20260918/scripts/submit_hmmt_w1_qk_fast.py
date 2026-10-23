"""Submit immutable K1024 W1 jobs; no formal inference before both gates pass."""
import hashlib
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012')
parser.add_argument('--validation-dependency', help='Wait for an earlier smoke/gate before allocating validation GPU')
options = parser.parse_args()
root = options.root.resolve()
assert not (root/'jobs.json').exists(), 'Never submit twice'
for rel, digest in json.loads((root/'manifest.json').read_text()).items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest() == digest, rel
jobs = {}
previous = options.validation_dependency
for stage in ('correctness', 'smoke', 'gate', 'formal', 'collect'):
    script = root/'source/scripts'/f'{stage}.sbatch'
    subprocess.run(['bash', '-n', str(script)], check=True)
    args = ['sbatch', '--parsable', '--output', str(root/'logs'/f'{stage}-%A_%a.out')]
    if stage == 'correctness':
        # Available A100 MIG20GB for validation; formal uses full A10080GB.
        args += ['--gres=gpu:a100_20g:1']
    if stage == 'gate':
        args += ['--time=01:00:00']
    if stage == 'collect':
        args += ['--time=03:00:00']
    if previous:
        args += ['--dependency', f'afterok:{previous}']
    args += [str(script), str(root)]
    result = subprocess.check_output(args, text=True).strip()
    jid = result.split(';')[0]
    assert jid.isdigit(), result
    jobs[stage] = dict(job_id=jid, dependency=previous, command=args)
    (root/'jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    previous = jid
    print(json.dumps(dict(stage=stage, job_id=jid)), flush=True)
