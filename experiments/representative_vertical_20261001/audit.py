"""CPU artifact audit of the approved run; no GPU replay or new generations."""
import argparse
from collections import Counter
import json
import os
import numpy as np
from protocol import RUN,ORIGIN,TASKS,sha,save,official,repetition
from run import preflight,load_heads


def kl(p,q):
    p,q=np.asarray(p,dtype=np.float64),np.asarray(q,dtype=np.float64)
    assert p.shape==q.shape==(128256,) and np.isfinite(p).all() and np.isfinite(q).all()
    p-=p.max();q-=q.max()
    lp=p-np.log(np.exp(p).sum());lq=q-np.log(np.exp(q).sum())
    value=float((np.exp(lp)*(lp-lq)).sum())
    assert np.isfinite(value) and value>=-1e-10
    return max(0.,value)


def audit(profile):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('audit requires GPUs hidden')
    from transformers import AutoTokenizer
    c,m=preflight()
    heads=load_heads(c,m)
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    _,_,scorers=official(ORIGIN/'official')
    old={x['key']:x['methods']['dense_full'] for x in json.loads((ORIGIN/'independent-audit.json').read_text())['examples']}
    folders=[RUN/'outputs/pilot'] if profile=='pilot' else [RUN/'outputs'/f'formal-{i}' for i in range(4)]
    rows=[];completion_hashes={}
    for folder in folders:
        assert not (folder/'failure.json').exists()
        done=json.loads((folder/'complete.json').read_text())
        assert done['status']=='complete' and done['frozen_base_unchanged']
        assert done['records']==(32 if profile=='pilot' else 200)
        assert done['source_lock_sha256']==sha(RUN/'source-lock.json')
        assert done['input_manifest_sha256']==sha(RUN/'inputs/manifest.json')
        assert done['config_sha256']==sha(RUN/'config.json')
        assert done['representative_heads']==heads
        for name,expected in done['files'].items():
            assert sha(folder/name)==expected
        completion_hashes[folder.name]=sha(folder/'complete.json')
        worker=int(folder.name.split('-')[-1]) if profile=='formal' else None
        items=[r for r in m['evaluation'] if r['key'] in m['pilot_keys']] if profile=='pilot' else [
            r for r in m['evaluation'] if r['shard']==worker]
        for item in items:
            pair=[];logits={}
            for method in c['methods']:
                path=folder/(item['key']+'-'+method+'.json')
                assert path.name in done['files']
                row=json.loads(path.read_text())
                for field,value in item.items():
                    assert row[field]==value,field
                for field in ('source_lock_sha256','config_sha256','input_manifest_sha256','representative_heads','node','gpu','job'):
                    assert row[field]==done[field],field
                assert row['method']==method and row['finite_all_steps'] and row['full_kv_cache'] and row['dense_decode']
                ids=row['generated_ids']
                assert 1<=len(ids)<=item['max_new_tokens']
                assert not any(i in row['eos_ids'] for i in ids[:-1])
                assert row['stop_reason']==('eos' if ids[-1] in row['eos_ids'] else 'max_new_tokens')
                assert row['stop_reason']=='eos' or len(ids)==item['max_new_tokens']
                assert row['prediction']==tok.decode(ids,skip_special_tokens=True)
                assert row['score']==float(scorers[item['task']](row['prediction'],item['answers'],'llama'))
                assert row['repetition']==repetition(ids)
                assert row['decode_calls']=={str(i):len(ids)-1 for i in c['layers']}
                assert set(row['prefill'])==set(map(str,c['layers']))
                n=item['input_tokens']
                for layer,stat in row['prefill'].items():
                    assert stat['mode']==method
                    if method=='dense_full':
                        assert stat['full_maps_computed']==32 and stat['all_keys_per_block']==n
                        assert stat['max_row_sum_error']<=2e-5
                    else:
                        assert stat['representative_head']==heads[layer]
                        assert stat['shared_with_heads']==list(range(32))
                        assert stat['own_qk_renormalized'] and stat['probabilities_shared'] is False
                        assert stat['target_token_budget']==c['selection']['target_token_budget']
                        assert stat['causal_pairs']==n*(n+1)//2
                        assert n<=stat['base_selected_pairs']<=stat['mask_selected_pairs']<=stat['causal_pairs']
                        assert stat['mask_density']==stat['mask_selected_pairs']/stat['causal_pairs']
                        assert stat['vertical_added_pairs']==stat['mask_selected_pairs']-stat['base_selected_pairs']
                        assert stat['member_coverage_queries']==min(n,128)
                        for field in ('member_mean_retained_mass','member_min_retained_mass'):
                            assert len(stat[field])==32 and all(0<=v<=1+2e-5 for v in stat[field])
                        if method.endswith('_vertical'):
                            assert stat['slash_added'] is False and stat['vertical_source_head']==heads[layer]
                            assert stat['vertical_gamma']==c['selection']['vertical_gamma']
                            if not stat['vertical_budget_capped']:
                                assert stat['vertical_retained_mass']>=stat['vertical_gamma']-2e-5
                        else:
                            assert stat['vertical_added_pairs']==0
                if method=='dense_full':
                    assert all(row[k]==old[item['key']][k] for k in ('generated_ids','prediction','score','stop_reason'))
                assert sha(folder/row['logits_file'])==row['logits_sha256']
                logits[method]=np.load(folder/row['logits_file'],allow_pickle=False)
                pair.append(row)
            for row in pair:
                row['finalprompt_vocabulary_kl']=kl(logits['dense_full'],logits[row['method']])
            rows.extend(pair)
    count=2 if profile=='pilot' else 50
    assert len(rows)==len({(r['key'],r['method']) for r in rows})==count*4*4
    assert Counter((r['task'],r['method']) for r in rows)==Counter({(t,method):count for t in TASKS for method in c['methods']})
    summary={}
    for task in TASKS:
        summary[task]={}
        for method in c['methods']:
            subset=[r for r in rows if r['task']==task and r['method']==method]
            stats=[s for r in subset for s in r['prefill'].values()]
            summary[task][method]=dict(count=count,mean_official_score=float(np.mean([r['score'] for r in subset])),
                capped=sum(r['stop_reason']=='max_new_tokens' for r in subset),
                repetitive=sum(r['repetition']['flagged'] for r in subset),
                mean_finalprompt_vocabulary_kl=float(np.mean([r['finalprompt_vocabulary_kl'] for r in subset])),
                mean_mask_density=float(np.mean([s['mask_density'] for s in stats])) if method!='dense_full' else None,
                mean_vertical_added_pairs=float(np.mean([s['vertical_added_pairs'] for s in stats])) if method!='dense_full' else None)
    destination=folders[0] if profile=='pilot' else RUN/'outputs'
    save(destination/'independent-audit.json',dict(status='PASS',profile=profile,records=len(rows),
        summary=summary,completion_hashes=completion_hashes,source_lock_sha256=sha(RUN/'source-lock.json'),
        input_manifest_sha256=sha(RUN/'inputs/manifest.json'),auditor_sha256=sha(__file__),
        limitations=['Scheduler COMPLETED/0:0 must be checked separately',
            'Artifact validation, no GPU replay', 'Last-query per-head mask coverage is not whole-sequence coverage',
            'Final-prompt vocabulary KL is not layerwise KL', 'Pilot is a diagnostic subset; no speedup claim']))
    print(json.dumps(summary,indent=2))
    print('INDEPENDENT AUDIT PASS',len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=('pilot','formal'),default='pilot')
    audit(p.parse_args().profile)
