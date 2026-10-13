"""Source-bound progress and paired quality/density summaries; CPU only."""
import collections
from support import HERE,read,save,sha


def summarize():
    p=read(HERE/'prepared.json');digest=sha(HERE/'lock.json')
    rows=[];completed=[]
    for worker in p['workers']:
        directory=HERE/'outputs'/('worker-%03d'%worker['index'])
        for key in worker['keys']:
            path=directory/(key+'-'+worker['method']+'.json')
            if not path.exists():continue
            r=read(path)
            assert r['status']=='complete' and r['lock_sha256']==digest
            assert r['key']==key and r['method']==worker['method'] and r['pattern_scope']=='per_head'
            assert sha(directory/r['logits_file'])==r['logits_sha256']
            assert len(r['prefill'])==32 and all(len(s['per_head'])==32 for s in r['prefill'].values())
            for s in r['prefill'].values():
                assert s['total_causal_pairs']==32*r['input_tokens']*(r['input_tokens']+1)//2
                assert s['mask_selected_pairs']==sum(x['mask_selected_pairs'] for x in s['per_head'])
                assert all(x['pattern_source_head']==h for h,x in enumerate(s['per_head']))
            rows.append(r)
        if (directory/'complete.json').exists():
            done=read(directory/'complete.json');assert done['worker']==worker and done['lock_sha256']==digest
            assert all((directory/(key+'-'+worker['method']+'.json')).exists() for key in worker['keys'])
            completed.append(worker['index'])
    table=[]
    for method in read(HERE/'config.json')['methods']:
        for task in read(HERE/'config.json')['tasks']:
            group=[r for r in rows if r['method']==method and r['task']==task]
            historical=[r for r in p['references'] if r['method']==method.replace('_own_patterns','') and r['task']==task and r['key'] in {x['key'] for x in group}]
            assert len(group)==len(historical)
            n=len(group)
            score=100*sum(r['score'] for r in group)/n if n else None
            old=100*sum(r['score'] for r in historical)/n if n else None
            density=sum(sum(s['mask_density'] for s in r['prefill'].values())/32 for r in group)/n if n else None
            candidate=sum(sum(s['candidate_density'] for s in r['prefill'].values())/32 for r in group)/n if n else None
            table.append(dict(method=method,task=task,samples=n,complete=n==50,score=score,
                paired_shared_score=old,score_delta=score-old if n else None,
                sparsity_percent=100*(1-density) if n else None,
                candidate_density_percent=100*candidate if n else None,
                mean_prefill_seconds=sum(r['prefill_seconds'] for r in group)/n if n else None))
    baseline_table=[]
    for method in read(HERE/'config.json')['comparison_methods']:
        for task in read(HERE/'config.json')['tasks']:
            group=[r for r in p['references'] if r['method']==method and r['task']==task]
            assert len(group)==50
            baseline_table.append(dict(method=method,task=task,samples=50,score=100*sum(r['score'] for r in group)/50,source='frozen historical paired inputs'))
    status=dict(status='complete' if len(rows)==800 else 'running_or_queued',samples_completed=len(rows),
        samples_total=800,workers_completed=len(completed),workers_total=160,completed_workers=completed,
        lock_sha256=digest,table=table,baseline_table=baseline_table,
        density_definition='Actual causal retained pairs, equal mean over query heads, layers and samples; candidate density is a scheduling superset')
    save(HERE/'results-summary.json',status)
    return status

if __name__=='__main__':
    print(summarize()['samples_completed'])
