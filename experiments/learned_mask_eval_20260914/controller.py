"""One-shot development-allocation gates, then independent formal submission."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .gpu_preflight import OUT
from experiments.learned_mask_nnonly_20260914.pipeline import ROOT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = OUT/f'controller_{time.time_ns()}'
    directory.mkdir()
    env = dict(os.environ, PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
               PYTHONPATH=str(ROOT)+':'+str(ROOT/'FlexPrefill'), TOKENIZERS_PARALLELISM='false')
    status = dict(pid=os.getpid(), start=time.time(), returncode=None)
    def stage(name, command):
        status.update(stage=name, command=command, updated=time.time())
        (directory/'status.json').write_text(json.dumps(status, indent=2))
        print('STAGE_START', name, directory, flush=True)
        with (directory/f'{name}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('STAGE_PASSED', name, flush=True)
    try:
        old = set(OUT.glob('preflight_*/correctness_passed.json'))
        stage('gpu_preflight', [sys.executable, '-u', '-m', 'experiments.learned_mask_eval_20260914.gpu_preflight'])
        new = set(OUT.glob('preflight_*/correctness_passed.json')) - old
        assert len(new) == 1
        proof = next(iter(new))
        stage('smoke', [sys.executable, '-u', '-m', 'experiments.learned_mask_eval_20260914.launch',
                        'smoke', '--proof', str(proof)])
        stage('submit', [sys.executable, '-u', '-m', 'experiments.learned_mask_eval_20260914.submit'])
        status.update(stage='formal_submitted', returncode=0)
    except BaseException as exc:
        status.update(returncode=1, error=repr(exc))
        raise
    finally:
        status['end'] = time.time()
        (directory/'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
