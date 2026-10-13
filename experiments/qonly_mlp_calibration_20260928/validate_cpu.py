"""Fail closed if any required CPU test is skipped or fails."""
import os
from pathlib import Path
import sys
import unittest
import torch

assert not torch.cuda.is_available(), 'CPU validation requires hidden CUDA devices'
torch.set_num_threads(1)
here = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(here), pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful() and not result.skipped and result.testsRun == 13
from pilot import ABLATION, ROUTING
os.environ['ROUTING_SOURCE'] = str(ROUTING)
# The frozen route and native-slot tests are required in addition to new tests.
sys.path.insert(0, str(ABLATION))
from pilot import load
load('assembly', ABLATION / 'assembly.py')
load('adapter', ABLATION / 'adapter.py')
load('test_assembly', ABLATION / 'test_assembly.py')
original_protocol = sys.modules['protocol']
try:
    load('protocol', ABLATION / 'protocol.py')
    checks = load('frozen_ablation_tests', ABLATION / 'test_ablation.py')
    frozen = unittest.defaultTestLoader.loadTestsFromTestCase(checks.TensorTests)
    result = unittest.TextTestRunner(verbosity=2).run(frozen)
    assert result.wasSuccessful() and not result.skipped and result.testsRun == 3
finally:
    sys.modules['protocol'] = original_protocol
print('ALL16 CPU TESTS PASS', flush=True)
