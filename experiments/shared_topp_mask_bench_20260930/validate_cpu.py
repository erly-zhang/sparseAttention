"""Record source-bound CPU validation without submitting or using a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from protocol import RUN, sha, save

assert Path(__file__).resolve().parent == RUN
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
modules=['test_attention','test_protocol','test_runtime','test_reuse']
count=unittest.TestLoader().loadTestsFromNames(modules).countTestCases()
assert count==25
result=subprocess.run([sys.executable,'-m','unittest','-v',*modules],
    cwd=RUN,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
with (RUN/'cpu-tests.log').open('x') as f:
    f.write(result.stdout)
print(result.stdout,flush=True)
assert result.returncode == 0
from evaluate import check
_,m,_,_,config=check(retokenize=True)
save(RUN/'cpu-validation.json',dict(status='passed',source_manifest_sha256=sha(RUN/'SHA256SUMS'),
    input_manifest_sha256=sha(RUN/'inputs/manifest.json'),log_sha256=sha(RUN/'cpu-tests.log'),
    tests=count,config=config))
print('CPU VALIDATION PASS',count,flush=True)
