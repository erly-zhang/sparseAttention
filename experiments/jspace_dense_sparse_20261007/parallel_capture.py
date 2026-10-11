"""Stage sample2 independently; publish only a completed sample directory."""
import json
import shutil
import capture

root=capture.HERE
staged=root/'staged_sample_2'
staged.mkdir(exist_ok=True)
shutil.copyfile(root/'smoke.json',staged/'smoke.json')
capture.HERE=staged
model,tok=capture.initialize()
capture.capture_study(model,tok,4096,[2])
source=staged/'captures/sample_2'
target=root/'captures/sample_2'
assert json.loads((source/'complete.json').read_text())['status']=='completed'
if target.exists():
    # The original worker remains authoritative if it has started this sample.
    print('STAGED_SAMPLE_RETAINED: original worker already owns sample2',flush=True)
else:
    source.rename(target)
    print('SAMPLE2_PUBLISHED_ATOMICALLY',flush=True)
