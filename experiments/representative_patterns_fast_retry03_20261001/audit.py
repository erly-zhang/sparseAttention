"""Independent CPU verification of all 200 paired examples and head strategies."""
from collections import Counter
import json
import os
import numpy as np
from protocol import RUN,ORIGIN,TASKS,sha,save,official,repetition
from run import preflight,load_heads


def kl(p,q):
    p,q=np.array(p,dtype=np.float64,copy=True),np.array(q,dtype=np.float64,copy=True)
    assert p.shape==q.shape==(128256,) and np.isfinite(p).all() and np.isfinite(q).all()
    p-=p.max();q-=q.max()
    lp=p-np.log(np.exp(p).sum());lq=q-np.log(np.exp(q).sum())
    value=float((np.exp(lp)*(lp-lq)).sum())
    assert np.isfinite(value) and value>=-1e-10
    return max(0.,value)


def verify_completion(folder,c,records):
    assert not (folder/'failure.json').exists()
    done=json.loads((folder/'complete.json').read_text())
    assert done['status']=='complete' and done['frozen_base_unchanged'] and done['records']==records
    for field,path in [('source_lock_sha256','source-lock.json'),('config_sha256','config.json'),
                       ('input_manifest_sha256','inputs/manifest.json')]:
        assert done[field]==sha(RUN/path)
    for name,expected in done['files'].items():
        assert sha(folder/name)==expected
    return done


def audit(require_hidden=True):
    if require_hidden and os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('audit requires GPUs hidden')
    from transformers import AutoTokenizer
    c,m=preflight()
    heads=load_heads(c,m)
    calibration=RUN/'outputs/calibration'
    done=verify_completion(calibration,c,3)
    assert done['representative_heads']==heads
    from selection import vote_representatives
    samples=[json.loads((calibration/(x['key']+'.json')).read_text()) for x in m['calibration']]
    for item,record in zip(m['calibration'],samples):
        assert record['input']==item and set(record['layers'])==set(map(str,c['layers']))
        for stat in record['layers'].values():
            similarity=np.asarray(stat['similarity'])
            assert similarity.shape==(32,32) and np.isfinite(similarity).all()
            assert np.allclose(similarity.mean(-1),stat['outgoing_scores'],rtol=0,atol=1e-10)
            assert stat['head']==int(similarity.mean(-1).argmax())
    assert vote_representatives([r['layers'] for r in samples],c['layers'])==heads['coverage']
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    _,_,scorers=official(ORIGIN/'official')
    history={x['key']:x['methods']['dense_full'] for x in json.loads((ORIGIN/'independent-audit.json').read_text())['examples']}
    rows=[];completion_hashes={};historical_dense_matches=0
    index={}
    plan=json.loads((RUN/'work-plan.json').read_text())
    by_key={item['key']:item for item in m['evaluation']}
    for assignment in plan:
        worker=assignment['worker'];method=assignment['method']
        folder=RUN/'outputs'/f'formal-{worker}'
        done=verify_completion(folder,c,len(assignment['keys']))
        assert done['representative_heads']==heads and done['worker']==worker
        assert done['worker_method']==method and done['work_shard']==assignment['shard']
        completion_hashes[folder.name]=sha(folder/'complete.json')
        for key in assignment['keys']:
            path=folder/(key+'-'+method+'.json')
            assert path.name in done['files'] and (key,method) not in index
            index[key,method]=(folder,json.loads(path.read_text()),done)
    for item in m['evaluation']:
        pair=[];logits={}
        for method in c['methods']:
            folder,row,done=index[item['key'],method]
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
            assert row['applied_layers']==list(range(32))
            assert row['decode_calls']=={str(i):len(ids)-1 for i in range(32)}
            assert set(row['prefill'])==set(map(str,range(32)))
            spec=c['scenarios'].get(method)
            assert row['head_strategy']==(spec['head_strategy'] if spec else None)
            selected=heads[spec['head_strategy']] if spec else None
            assert row['selected_heads']==selected
            expected_method=spec['selector_method'] if spec else method
            assert row['mask_method']==expected_method
            n=item['input_tokens']
            for layer,stat in row['prefill'].items():
                assert stat['mode']==expected_method
                if method=='dense_full':
                    assert stat['full_maps_computed']==32 and stat['all_keys_per_block']==n
                    assert stat['max_row_sum_error']<=2e-5
                elif method in c['baselines']:
                    assert stat['baseline_configuration']==c['baselines'][method]
                    assert stat['own_qk_renormalized'] and not stat['probabilities_shared']
                else:
                    assert stat['representative_head']==selected[layer]
                    assert stat['shared_with_heads']==list(range(32))
                    assert stat['own_qk_renormalized'] and stat['probabilities_shared'] is False
                    assert stat['target_token_budget']==c['selection']['target_token_budget']
                    assert stat['causal_pairs']==n*(n+1)//2
                    assert n<=stat['base_selected_pairs']<=stat['mask_selected_pairs']<=stat['candidate_causal_pairs']<=stat['causal_pairs']
                    assert stat['mask_density']==stat['mask_selected_pairs']/stat['causal_pairs']
                    assert stat['candidate_density']==stat['candidate_causal_pairs']/stat['causal_pairs']
                    assert stat['vertical_added_pairs']+stat['slash_added_pairs']==stat['mask_selected_pairs']-stat['base_selected_pairs']
                    assert stat['exact_query_mask']
                    assert stat['member_coverage_queries']==min(n,128)
                    for field in ('member_mean_retained_mass','member_min_retained_mass'):
                        assert len(stat[field])==32 and all(0<=v<=1+2e-5 for v in stat[field])
                    for kind in ('vertical','slash'):
                        if kind in expected_method:
                            assert stat[kind+'_source_head']==selected[layer]
                            assert stat[kind+'_gamma']==c['selection'][kind+'_gamma']
                            if not stat[kind+'_budget_capped']:
                                assert stat[kind+'_retained_mass']>=stat[kind+'_gamma']-2e-5
                        else:
                            assert stat[kind+'_added_pairs']==0
                    assert bool(stat.get('slash_added',False))==expected_method.endswith('_slash')
            if method=='dense_full':
                historical_dense_matches+=all(row[k]==history[item['key']][k] for k in ('generated_ids','prediction','score','stop_reason'))
            assert sha(folder/row['logits_file'])==row['logits_sha256']
            logits[method]=np.load(folder/row['logits_file'],allow_pickle=False)
            pair.append(row)
        for row in pair:
            row['finalprompt_vocabulary_kl']=kl(logits['dense_full'],logits[row['method']])
        rows.extend(pair)
    assert len(rows)==len({(r['key'],r['method']) for r in rows})==200*len(c['methods'])
    assert Counter((r['task'],r['method']) for r in rows)==Counter({(t,method):50 for t in TASKS for method in c['methods']})
    summary={}
    for task in TASKS:
        summary[task]={}
        for method in c['methods']:
            subset=[r for r in rows if r['task']==task and r['method']==method]
            stats=[s for r in subset for s in r['prefill'].values()]
            sparse=method in c['scenarios']
            summary[task][method]=dict(count=50,mean_official_score=float(np.mean([r['score'] for r in subset])),
                capped=sum(r['stop_reason']=='max_new_tokens' for r in subset),
                repetitive=sum(r['repetition']['flagged'] for r in subset),
                mean_finalprompt_vocabulary_kl=float(np.mean([r['finalprompt_vocabulary_kl'] for r in subset])),
                mean_mask_density=float(np.mean([s['mask_density'] for s in stats])) if sparse else None,
                mean_candidate_density=float(np.mean([s['candidate_density'] for s in stats])) if sparse else None,
                mean_vertical_added_pairs=float(np.mean([s['vertical_added_pairs'] for s in stats])) if sparse else None,
                mean_slash_added_pairs=float(np.mean([s['slash_added_pairs'] for s in stats])) if sparse else None)
    deltas={t:{method:summary[t][method]['mean_official_score']-summary[t]['dense_full']['mean_official_score']
        for method in c['methods'] if method!='dense_full'} for t in TASKS}
    save(RUN/'outputs/independent-audit.json',dict(status='PASS',records=len(rows),summary=summary,
        score_deltas_from_dense=deltas,representative_heads=heads,historical_dense_exact_matches=historical_dense_matches,
        completion_hashes=completion_hashes,source_lock_sha256=sha(RUN/'source-lock.json'),
        input_manifest_sha256=sha(RUN/'inputs/manifest.json'),auditor_sha256=sha(__file__),
        limitations=['Scheduler state must be checked separately','CPU artifact validation, no GPU replay',
        'Coverage diagnostics only cover last 128 queries','Final-prompt vocabulary KL is not layerwise KL',
        'Kernel candidate density may exceed effective mask density; timings include diagnostics',
        'Single fixed random seed; no across-seed variance estimate',
        'Old Dense used 16 explicit-attention layers; new Dense uses all 32']))
    print(json.dumps(summary,indent=2))
    print('INDEPENDENT AUDIT PASS',len(rows))


if __name__=='__main__':
    audit()
