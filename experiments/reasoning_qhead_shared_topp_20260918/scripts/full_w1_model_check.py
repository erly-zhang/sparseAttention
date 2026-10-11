import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
import torch

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'source/ra_harness'))
spec=importlib.util.spec_from_file_location('test_full_w1_optimized_model',root/'source/tests/test_full_w1_optimized_model.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with (root/'model_correctness.log').open('x') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
proof=dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',tests=result.testsRun,
           skipped=result.skipped,gpu=torch.cuda.get_device_name(),node=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID'),
           scope='Tiny random-weight Qwen3 model; exact sampled token trajectory, RoPE/GQA and real runner/factory. Not HMMT accuracy.')
(root/'model_correctness.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
assert proof['status']=='passed'
