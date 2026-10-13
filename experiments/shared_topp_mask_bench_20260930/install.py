"""Install a verified new code package without replacing frozen data or history."""
from pathlib import Path
import shutil
from protocol import RUN, sha

stage=Path(__file__).resolve().parent
assert stage!=RUN and (RUN/'inputs/manifest.json').exists()
assert not (RUN/'submitted-job.txt').exists() and not (RUN/'submit-intent.json').exists()
entries=[]
for line in (stage/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split()
    assert Path(name).name==name and sha(stage/name)==expected
    assert not (RUN/name).exists(),name
    entries.append(name)
assert not (RUN/'SHA256SUMS').exists()
for name in entries:
    with (stage/name).open('rb') as src,(RUN/name).open('xb') as dst:
        shutil.copyfileobj(src,dst)
with (stage/'SHA256SUMS').open('rb') as src,(RUN/'SHA256SUMS').open('xb') as dst:
    shutil.copyfileobj(src,dst)
print('INSTALLED',len(entries),'source files; no submission',flush=True)
