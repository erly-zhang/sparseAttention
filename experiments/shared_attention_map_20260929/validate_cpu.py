import unittest
from pathlib import Path
from smoke import sha, save

here = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(here), pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful() and not result.skipped and result.testsRun == 11
save(here/'cpu-validation.json',dict(passed=True,tests=result.testsRun,
    source_manifest_sha256=sha(here/'SHA256SUMS')))
