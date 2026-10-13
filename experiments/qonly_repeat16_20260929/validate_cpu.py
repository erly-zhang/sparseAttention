"""Submission gate: never treat skipped tensor checks as success."""
from pathlib import Path
import unittest
import torch
from evaluate import save, sha, HERE

assert not torch.cuda.is_available(), 'Hide CUDA for CPU validation'
torch.set_num_threads(1)
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py'))
assert result.wasSuccessful() and not result.skipped and result.testsRun == 10
save(HERE/'cpu-validation.json',dict(status='passed',tests=result.testsRun,
    source_manifest_sha256=sha(HERE/'SHA256SUMS')))
print('ALL10 CPU TESTS PASS',flush=True)
