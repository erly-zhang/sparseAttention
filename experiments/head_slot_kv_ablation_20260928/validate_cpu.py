"""Persist test output; never run tensor tests with CUDA visible."""
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
here = Path(__file__).resolve().parent
for name in ('test_assembly.py', 'test_ablation.py'):
    result = subprocess.run([sys.executable, '-u', str(here / name)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with (here / (name + '.log')).open('x') as handle:
        handle.write(result.stdout)
    print(result.stdout, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
