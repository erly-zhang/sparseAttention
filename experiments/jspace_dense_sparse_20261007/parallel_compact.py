"""Capture the two compact sample1 arms on the second L40, atomically."""
import json
import os
from pathlib import Path
import torch
import capture

root=capture.HERE
staged=root/'staged_compact_1'
staged.mkdir(exist_ok=True)
model,tok=capture.initialize()
capture.wrap_evict_attn_forward('/scratch/yl258/jz648/Qwen3-4B')
ref=next(r for r in capture.references() if r['idx']==1)
ids=capture.canonical_prompt_ids(tok,ref)
directory=root/'captures/sample_1'
tr=json.loads((directory/'dense_trajectory.json').read_text())
assert tr['input_ids_sha256']==capture.ids_sha(ids)
continuation=tr['continuation_ids']
points=capture.checkpoints(len(ids),len(continuation),2048)
dense=torch.load(directory/'dense.pt',map_location='cpu',weights_only=True)
for arm in ['compact_random','compact_w1']:
    target=directory/f'{arm}.pt'
    if target.exists():
        continue
    record=capture.replay(model,ids,continuation,arm,2048,1,points)
    before=record['steps']<=2048-len(ids)
    torch.testing.assert_close(record['residuals'][before],dense['residuals'][before],rtol=0,atol=0)
    record.update(before_sparse_residuals='bitwise_identical',paired_dense_control='dense',
                  prompt_tokens=len(ids),continuation_tokens=len(continuation),gpu=torch.cuda.get_device_name(),
                  cross_gpu_prebudget_alignment='bitwise_identical')
    source=staged/f'{arm}.pt'
    torch.save(record,source)
    stats=staged/f'{arm}_stats.json'
    capture.dump(stats,record['stats'])
    try:
        os.link(source,target)
    except FileExistsError:
        print(f'STAGED_ARM_RETAINED: original worker owns {arm}',flush=True)
    else:
        os.link(stats,directory/stats.name)
        print(f'ARM_PUBLISHED_ATOMICALLY: sample1/{arm}',flush=True)
