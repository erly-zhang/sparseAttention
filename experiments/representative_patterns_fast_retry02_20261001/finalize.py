"""The last completed worker performs CPU postprocessing, once, without waiting.

Called only after releasing the checkpoint and emptying its CUDA allocator.
If any peer is incomplete the worker simply exits; no GPU is held waiting.
"""
import json
import os
from protocol import RUN, sha, save


def finalize_if_ready(stage, config):
    counts={'calibration':config['calibration_job']['gpus'],
            'formal':config['formal']['gpus']}
    if stage not in counts:
        raise ValueError(stage)
    folders=[RUN/'outputs'/f'{stage}-{i}' for i in range(counts[stage])]
    if not all((folder/'complete.json').exists() for folder in folders):
        return False
    # Validate completion statuses before taking the exclusive claim.
    for folder in folders:
        if (folder/'failure.json').exists():
            return False
        done=json.loads((folder/'complete.json').read_text())
        if done['status']!='complete':
            return False
    claim=RUN/'outputs'/(stage+'-postprocess-claim.json')
    try:
        with claim.open('x') as file:
            json.dump(dict(stage=stage,job=os.environ.get('SLURM_JOB_ID'),
                source_lock_sha256=sha(RUN/'source-lock.json')),file)
            file.write('\n')
    except FileExistsError:
        return False
    print('CPU POSTPROCESS',stage,'claimed',flush=True)
    if stage=='calibration':
        from aggregate import aggregate
        aggregate(require_hidden=False)
    else:
        from audit import audit
        audit(require_hidden=False)
    save(RUN/'outputs'/(stage+'-postprocess-complete.json'),dict(status='complete',stage=stage,
        job=os.environ.get('SLURM_JOB_ID'),source_lock_sha256=sha(RUN/'source-lock.json')))
    return True
