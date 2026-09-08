"""Continue matched-count Top-p on GPU4-7 without disturbing fixed jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import schedule as shared


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    root=shared.ROOT
    if args.detach:
        logpath=root/f'scheduler_topp_{int(time.time())}.log'
        with logpath.open('w') as log:
            child=subprocess.Popen([shared.PY,str(Path(__file__).resolve())],cwd=shared.WORK,
                env={**os.environ,'PYTHONPATH':str(shared.WORK)},stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':child.pid,'log':str(logpath)}))
        return
    lock=(root/'topp_scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((shared.HERE/'correctness.json').read_text())['passed']
    rules=json.loads((root/'policies.json').read_text())
    assert rules['top_p_mode']=='equal-count replacement; final mass may be below .99'
    assert json.loads((root/'data_manifest.json').read_text())['calibration_evaluation_disjoint']
    # Do not modify an inference source file already used by the fixed experiment.
    hashes={}
    for name in ('online.py','residual_selector.py','policy.py','schedule.py'):
        source=shared.HERE/name
        frozen=root/'formal_source'/name
        assert source.read_bytes()==frozen.read_bytes(),f'Inference source changed: {name}'
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (root/'topp_source_hashes.json').write_text(json.dumps(hashes,indent=2))
    method=shared.METHODS[1]
    shared.status(event='topp_scheduler_started',pid=os.getpid(),gpus=[4,5,6,7],
                  policy='same-QK count replacement; target .99, final mass not guaranteed')
    path=shared.job(4,method,0,3,'smoke')
    summary,records,_=shared.validate(path,3)
    reference=shared.rows(shared.REFERENCE)
    fixed_smokes=list((root/'attempts/smoke').glob(
        'probmean_shared_residual_compact10240_0_3/attempt*/probmean_shared_residual_compact10240/kv_retrieval/summary.json'))
    assert fixed_smokes
    fixed_path=fixed_smokes[-1].parent
    _,fixed_records,_=shared.validate(fixed_path,3)
    assert [shared.identity(x) for x in records]==[shared.identity(x) for x in fixed_records]
    for record in records:
        assert record['max_probability_sum_error']<1e-4
        npz=Path(record['selection_stats_file'])
        metadata=json.loads(npz.with_suffix('.json').read_text())
        seen=set()
        with np.load(npz) as arrays:
            for item in metadata['mapping']:
                arr=arrays[item['array']]
                assert np.isfinite(arr).all()
                assert np.array_equal(arr[:,1],arr[:,11])
                assert (arr[:,2]>=.99-1e-4).all()
                ends=np.minimum((np.arange(len(arr))+1)*128,record['input_tokens'])
                assert (arr[:,0]>=np.minimum(ends,1024)).all()
                assert (arr[:,1]<=ends).all()
                for head in item['member_heads']:
                    identity=(item['layer'],head)
                    assert identity not in seen
                    seen.add(identity)
        assert len(seen)==1024
    (root/'topp_smoke_passed.json').write_text(json.dumps({'passed':True,'path':str(path),
        'count':3,'input_alignment':summary['input_alignment'],
        'same_qk_original_final_count_preserved':True},indent=2))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(shared.job,4+i,method,a,b,'formal') for i,(a,b) in enumerate(shared.SHARDS)]
        paths=[f.result() for f in futures]
    from merge_aligned import merge, completed_paths
    merge(method,paths)
    # The fixed scheduler might still be preparing its report; keep separate artifacts.
    from experiments.probmean_compact_20260906.summarize import aggregate
    aggregate(root/'merged'/method/'kv_retrieval')
    shared.status(event='topp_complete',count=497,method=method)
    fixed_paths=completed_paths(shared.METHODS[0])
    if fixed_paths is not None:
        merge(shared.METHODS[0],fixed_paths)
        from report import summarize
        summarize(root)
        shared.status(event='both_aligned_complete',count=994)


if __name__=='__main__':
    try:main()
    except Exception as error:
        shared.status(event='topp_scheduler_failed',error=repr(error))
        raise
