"""Fail closed on failed or skipped tensor tests; no GPU work."""
import unittest
import torch
from pilot import HERE, preflight
from protocol import sha, save

assert not torch.cuda.is_available(), 'Hide CUDA for CPU validation'
preflight()
torch.set_num_threads(1)
result=unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py'))
assert result.wasSuccessful() and not result.skipped and result.testsRun==11
save(HERE/'cpu-validation.json',dict(status='passed',tests=11,
    source_manifest_sha256=sha(HERE/'SHA256SUMS')))
print('ALL11 CPU TESTS PASS',flush=True)
