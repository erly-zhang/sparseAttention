"""Submit exactly four gated jobs; compute submission explicitly authorized."""
import fcntl
import json
import socket
import subprocess
import time
from pathlib import Path
from .gpu_preflight import allocation, OUT
from .launch import checked_proof, read, save
from experiments.learned_mask_nnonly_20260914.model import VARIANTS
from experiments.learned_mask_nnonly_20260914.pipeline import sha


def main():
    access = allocation()
    smoke = read(OUT/'smoke_passed.json')
    assert smoke['status'] == 'passed' and smoke['count_per_variant'] == 3
    proof_path = Path(smoke['correctness_path'])
    assert sha(proof_path) == smoke['correctness_sha256']
    checked_proof(proof_path)
    lock = (OUT/'submission.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for variant in VARIANTS:
        folder = OUT/'smoke'/variant
        assert read(folder/'exit.json')['returncode'] == 0
        assert read(folder/'verified_complete.json')['count'] == 3
    queue = subprocess.check_output(['squeue', '-u', 'jz648', '-h', '-o', '%j'], text=True)
    assert not any('nnkv-'+v in queue.splitlines() for v in VARIANTS)
    submissions = OUT/'submissions'
    submissions.mkdir(exist_ok=False)
    script = Path(__file__).parent/'formal.sbatch'
    for variant in VARIANTS:
        assert not (OUT/'formal'/variant).exists()
        test = ['sbatch', '--test-only', '--job-name=nnkv-'+variant, str(script), variant, str(proof_path)]
        result = subprocess.run(test, text=True, capture_output=True)
        save(submissions/f'{variant}_test_only.json', dict(command=test, returncode=result.returncode,
                                                       stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, 'Site/Slurm rejected test-only; no bypass permitted'
    for variant in VARIANTS:
        command = ['sbatch', '--parsable', '--job-name=nnkv-'+variant, str(script), variant, str(proof_path)]
        # Intent marker survives an uncertain return. Never automatically resubmit.
        save(submissions/f'{variant}_intent.json', dict(command=command, timestamp=time.time(), allocation=access))
        result = subprocess.run(command, capture_output=True, text=True)
        save(submissions/f'{variant}.json', dict(command=command, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr, host=socket.gethostname(), timestamp=time.time()))
        assert result.returncode == 0 and result.stdout.strip().split(';')[0].isdigit()
        print('SUBMITTED', variant, result.stdout.strip(), flush=True)


if __name__ == '__main__':
    main()
