"""Independent CPU artifact audit; scheduler completion is a separate mandatory gate."""
from collections import Counter
import json
import os
import numpy as np
from evaluate import check
from protocol import RUN, TASKS, METHODS, sha, digest, save, validate_pair, repetition


def kl(p,q):
    p=np.asarray(p,dtype=np.float64)
    q=np.asarray(q,dtype=np.float64)
    assert p.shape==q.shape==(128256,) and np.isfinite(p).all() and np.isfinite(q).all()
    p=p-p.max()
    q=q-q.max()
    lp=p-np.log(np.exp(p).sum())
    lq=q-np.log(np.exp(q).sum())
    value=float((np.exp(lp)*(lp-lq)).sum())
    assert value>=-1e-10 and np.isfinite(value)
    return max(0,value)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    _,m,tok,scorers,config=check(retokenize=True)
    mh=sha(RUN/'inputs/manifest.json')
    ch=digest(config)
    rows=[]
    examples=[]
    completions={}
    for shard in range(4):
        out=RUN/f'worker-{shard}'
        assert not (out/'failure.json').exists()
        done=json.loads((out/'complete.json').read_text())
        assert done['records']==100 and done['frozen_base_unchanged']
        assert done['manifest_sha256']==mh and done['config_sha256']==ch
        for name,value in done['files'].items():
            assert sha(out/name)==value,name
        smoke=json.loads((out/'smoke-complete.json').read_text())
        assert smoke['status']=='passed' and smoke['projected_seconds']<smoke['limit_seconds']==84600
        completions[str(shard)]=sha(out/'complete.json')
        for item in (r for r in m['records'] if r['shard']==shard):
            pair=[]
            logits={}
            for method in METHODS:
                path=out/(item['key']+'-'+method+'.json')
                assert path.name in done['files']
                row=json.loads(path.read_text())
                assert row['prediction']==tok.decode(row['generated_ids'],skip_special_tokens=True)
                assert row['score']==float(scorers[item['task']](row['prediction'],item['answers'],'llama'))
                assert row['repetition']==repetition(row['generated_ids'])
                assert row['source_manifest_sha256']==sha(RUN/'SHA256SUMS')
                assert row['array_job']==(RUN/'submitted-job.txt').read_text().strip().split(';')[0]
                assert sha(out/row['logits_file'])==row['logits_sha256']
                logits[method]=np.load(out/row['logits_file'],allow_pickle=False)
                n=item['input_tokens']
                for stat in row['prefill'].values():
                    assert stat['stitched_causal_pairs']==(n*(n+1)//2 if method=='shared_map' else None)
                    assert stat['query_block_size']==256
                pair.append(row)
            validate_pair(pair,item,mh,ch)
            for row in pair:
                row['independent_finalprompt_vocabulary_kl']=kl(logits['dense_full'],logits[row['method']])
            rows.extend(pair)
            examples.append(dict(key=item['key'],task=item['task'],input_tokens=item['input_tokens'],
                answers=item['answers'],methods={r['method']:dict(prediction=r['prediction'],score=r['score'],
                    stop_reason=r['stop_reason'],generated_ids=r['generated_ids'],repetition=r['repetition']) for r in pair}))
    assert len(rows)==len({(r['key'],r['method']) for r in rows})==400
    assert Counter((r['task'],r['method']) for r in rows)==Counter({(t,m):50 for t in TASKS for m in METHODS})
    summary={}
    for task in TASKS:
        summary[task]={}
        for method in METHODS:
            subset=[r for r in rows if r['task']==task and r['method']==method]
            summary[task][method]=dict(count=50,mean_official_score=float(np.mean([r['score'] for r in subset])),
                caps=sum(r['stop_reason']=='max_new_tokens' for r in subset),
                repetitive=sum(r['repetition']['flagged'] for r in subset),
                mean_finalprompt_vocabulary_kl=float(np.mean([r['independent_finalprompt_vocabulary_kl'] for r in subset])))
    report=dict(status='PASS',records=400,manifest_sha256=mh,source_manifest_sha256=sha(RUN/'SHA256SUMS'),
        completion_hashes=completions,summary=summary,examples=examples,auditor_sha256=sha(__file__),
        limitations=['Slurm COMPLETED/0:0 for all four workers must be checked separately',
            'Finite/cache/pair-count assertions grounded in frozen runtime, not a GPU replay',
            'Intact length-eligible 50-input subsets, not the full InfiniteBench',
            'Final-prompt vocabulary KL, not layerwise KL; no speedup claim'])
    save(RUN/'independent-audit.json',report)
    print(json.dumps(summary,indent=2),flush=True)
    print('INDEPENDENT ARTIFACT AUDIT PASS400',flush=True)


if __name__=='__main__':
    audit()
