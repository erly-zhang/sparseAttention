"""Record source-bound CPU validation without submitting or using a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
from protocol import RUN, sha, save

assert Path(__file__).resolve().parent == RUN
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
result=subprocess.run([sys.executable,'-m','unittest','-v','test_attention','test_protocol','test_runtime'],
    cwd=RUN,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
with (RUN/'cpu-tests.log').open('x') as f:
    f.write(result.stdout)
print(result.stdout,flush=True)
assert result.returncode == 0
from evaluate import check
_,m,_,_,config=check(retokenize=True)
save(RUN/'cpu-validation.json',dict(status='passed',source_manifest_sha256=sha(RUN/'SHA256SUMS'),
    input_manifest_sha256=sha(RUN/'inputs/manifest.json'),log_sha256=sha(RUN/'cpu-tests.log'),
    tests=15,config=config))
print('CPU VALIDATION PASS15',flush=True)
