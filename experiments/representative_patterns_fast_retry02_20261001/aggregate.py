"""CPU aggregation of three completed, disjoint, parallel calibration samples."""
import json
import os
import shutil
import numpy as np
from protocol import RUN, sha, save, require_approval
from run import preflight, fixed_random_heads
from selection import vote_representatives
from audit import verify_completion


def aggregate(require_hidden=True):
    require_approval('full_suite')
    if require_hidden and os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('aggregation requires GPUs hidden')
    c,m=preflight()
    samples=[];sources={}
    for index,item in enumerate(m['calibration']):
        folder=RUN/'outputs'/f'calibration-{index}'
        done=verify_completion(folder,c,1)
        record=json.loads((folder/(item['key']+'.json')).read_text())
        assert record['input']==item and set(record['layers'])==set(map(str,c['layers']))
        parity=RUN/'outputs'/('gpu-parity-'+done['job']+'.json')
        proof=json.loads(parity.read_text())
        assert proof['status']=='passed' and proof['config_sha256']==sha(RUN/'config.json')
        assert proof['source_lock_sha256']==sha(RUN/'source-lock.json')
        for stat in record['layers'].values():
            matrix=np.asarray(stat['similarity'])
            assert matrix.shape==(32,32) and np.isfinite(matrix).all()
            assert np.allclose(matrix.mean(-1),stat['outgoing_scores'],rtol=0,atol=1e-10)
            assert stat['head']==int(matrix.mean(-1).argmax())
        samples.append(record['layers'])
        sources[str(folder/'complete.json')]=sha(folder/'complete.json')
        sources[str(parity)]=sha(parity)
    coverage=vote_representatives(samples,c['layers'])
    random=fixed_random_heads(c)
    out=RUN/'outputs/calibration'
    out.mkdir(exist_ok=False)
    for index,item in enumerate(m['calibration']):
        src=RUN/'outputs'/f'calibration-{index}'/(item['key']+'.json')
        with (out/src.name).open('xb') as dst,src.open('rb') as reader:
            shutil.copyfileobj(reader,dst)
    provenance=dict(config_sha256=sha(RUN/'config.json'),source_lock_sha256=sha(RUN/'source-lock.json'),
        input_manifest_sha256=sha(RUN/'inputs/manifest.json'))
    save(out/'representative-heads.json',dict(representative_heads=coverage,random_heads=random,
        random_seed=c['calibration']['random_head_seed'],calibration_keys=[r['key'] for r in m['calibration']],
        selection='mean outgoing directional coverage, majority vote, voter-score tie break, lowest head',
        calibration_source_hashes=sources,**provenance))
    save(out/'complete.json',dict(status='complete',records=3,frozen_base_unchanged=True,
        representative_heads={'coverage':coverage,'random':random},**provenance,
        job=os.environ.get('SLURM_JOB_ID'),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print('CALIBRATION AGGREGATION PASS',json.dumps({'coverage':coverage,'random':random}),flush=True)


if __name__=='__main__':
    aggregate()
