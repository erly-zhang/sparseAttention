"""Freeze a fresh source bundle; never overwrite an existing archive."""
import hashlib
from pathlib import Path
import tarfile
import sys

here = Path(__file__).resolve().parent
archive = Path(sys.argv[1])
assert not archive.exists()
files = sorted(p for p in here.iterdir() if p.suffix in ('.py', '.sh', '.sbatch') or p.name == 'README.md')
manifest = here / 'SHA256SUMS'
assert not manifest.exists(), 'Preserve existing source freeze; version repairs explicitly'
with manifest.open('x') as handle:
    for path in files:
        handle.write(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n')
with tarfile.open(archive, 'x:gz') as tar:
    for path in files + [manifest]:
        tar.add(path, arcname='qmlpstage20260928/' + path.name, recursive=False)
print('ARCHIVE', archive, hashlib.sha256(archive.read_bytes()).hexdigest())
