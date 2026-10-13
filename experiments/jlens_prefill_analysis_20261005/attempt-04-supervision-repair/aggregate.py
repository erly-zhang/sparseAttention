"""Pair and audit new observations; historical scores remain historical labels."""
from collections import defaultdict
import fcntl
from pathlib import Path
import numpy as np

from common import HERE,config,read,save,sha


def interval(values,seed=42):
    values=np.asarray(values,dtype=float)
    if not len(values): return None
    rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(values,size=(2000,len(values)),replace=True),axis=1)
    return dict(mean=float(values.mean()),bootstrap_95_low=float(np.quantile(means,.025)),
                bootstrap_95_high=float(np.quantile(means,.975)),n=len(values))


def correlate(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if len(x)<3 or x.std()==0 or y.std()==0: return None
    return float(np.corrcoef(x,y)[0,1])


def audit_and_aggregate():
    c=config()
    lens_hash=sha(HERE/'outputs/lens.pt')
    manifest=read(HERE/'input-manifest.json')
    original={item['key']:item for item in manifest['evaluation']}
    summaries=defaultdict(list)
    seen=set()
    bitwise,first_matches=0,0
    layer_sums=defaultdict(list)
    for method in c['replay']['methods']:
        for shard in range(c['replay']['workers_per_method']):
            directory=HERE/'outputs'/('replay-'+method+'-'+str(shard))
            complete=read(directory/'complete.json')
            for name,expected in complete['files'].items():
                if sha(directory/name)!=expected: raise ValueError('Worker result changed')
            records=[directory/name for name in complete['files'] if name not in ['failure.json','snapshot-error.json']]
            if len(records)!=complete['records']: raise ValueError('Worker record count mismatch')
            for path in records:
                row=read(path)
                key=method,row['key']
                if key in seen: raise ValueError('Duplicate result')
                seen.add(key)
                item=original[row['key']]
                if row['positions']!=item['analysis_positions']: raise ValueError('Changed positions')
                if row['artifact_sha256']!=sha(path.parent/row['artifact_file']): raise ValueError('Artifact changed')
                if row['lens_sha256']!=lens_hash: raise ValueError('Mixed lenses')
                if row['config_sha256']!=sha(HERE/'config.json'): raise ValueError('Mixed configurations')
                if set(map(int,row['layer_metrics']))!=set(c['replay']['layers']): raise ValueError('Missing layers')
                bitwise += row['history_replay_match']['bitwise_equal']
                first_matches += row['history_replay_match']['first_token_equal']
                groups={g:[i for i,p in enumerate(row['positions']) if p in ps]
                        for g,ps in row['position_groups'].items()}
                sample=dict(key=row['key'],score=row['historical_score'],
                            history_first_token_equal=row['history_replay_match']['first_token_equal'])
                for group,indices in groups.items():
                    if not indices: continue
                    numbers=[]
                    for layer,metrics in row['layer_metrics'].items():
                        arr=np.asarray(metrics['js_nats'],dtype=float)[indices]
                        if not np.isfinite(arr).all(): raise ValueError('Nonfinite metric')
                        value=float(arr.mean())
                        layer_sums[(row['task'],method,group,int(layer))].append(value)
                        numbers.append(value)
                    sample[group+'_mean_js_nats']=float(np.mean(numbers))
                summaries[(row['task'],method)].append(sample)
    expected={(m,key) for m in c['replay']['methods'] for key in original}
    if seen!=expected: raise ValueError('Incomplete paired observations')
    tables={}
    for (task,method),rows in sorted(summaries.items()):
        result={}
        for group in ['prefix','tail','uniform','kv_evidence']:
            field=group+'_mean_js_nats'
            selected=[r for r in rows if field in r]
            if selected:
                result[group]=interval([r[field] for r in selected])
                dense_scores={r['key']:r['score'] for r in summaries[(task,'dense_full')]}
                result[group]['pearson_with_historical_score_loss']=correlate(
                    [r[field] for r in selected],[dense_scores[r['key']]-r['score'] for r in selected])
                dense_matches={r['key']:r['history_first_token_equal'] for r in summaries[(task,'dense_full')]}
                matched=[r for r in selected if r['history_first_token_equal'] and dense_matches[r['key']]]
                result[group]['replay_first_token_matched_count']=len(matched)
                result[group]['matched_subset_js']=interval([r[field] for r in matched])
        tables.setdefault(task,{})[method]=result
    paired={}
    for task in sorted({t for t,m in summaries}):
        for strategy in ['coverage','random']:
            left={r['key']:r for r in summaries[(task,strategy+'_token_vertical')]}
            right={r['key']:r for r in summaries[(task,strategy+'_token_vertical_slash')]}
            paired.setdefault(task,{})[strategy]=interval([
                right[key]['tail_mean_js_nats']-left[key]['tail_mean_js_nats'] for key in sorted(left)])
    save(HERE/'outputs/summary.json',dict(groups=tables,slash_minus_vertical_tail_js=paired,
        layer_means=[dict(task=t,method=m,group=g,layer=l,summary=interval(v))
                     for (t,m,g,l),v in sorted(layer_sums.items())],sample_summaries=[dict(task=t,method=m,**r)
                     for (t,m),rows in sorted(summaries.items()) for r in rows],
        confidence_intervals='2000 paired-sample bootstrap draws, seed 42; descriptive, no multiple-testing claims'))
    save(HERE/'outputs/independent-audit.json',dict(status='PASS',records=len(seen),
        history_bitwise_matches=bitwise,history_first_token_matches=first_matches,
        lens_sha256=lens_hash,config_sha256=sha(HERE/'config.json'),
        source_lock_sha256=sha(HERE/'source-lock.json'),input_manifest_sha256=sha(HERE/'input-manifest.json'),
        limitations=['Observational readouts, no causal intervention',
          'Sampled token positions and KV only; complete forward contexts',
          'Historical scores are labels, no newly scored generation',
          'Readout drift is not the sparsely decomposed J-space component',
          'Short-corpus lens transfer to long-context is not assumed proven',
          'Pattern comparisons are not budget matched',
          'Replay differences are reported explicitly',
          'Uniform positions cannot guarantee coverage of every relevant evidence span']))


def finalize_if_ready():
    c=config()
    if any(not (HERE/'outputs'/('replay-'+method+'-'+str(shard))/'complete.json').exists()
           for method in c['replay']['methods'] for shard in range(c['replay']['workers_per_method'])):
        return False
    with (HERE/'outputs/aggregate.lock').open('a') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return False
        if not (HERE/'outputs/independent-audit.json').exists(): audit_and_aggregate()
    return True


if __name__=='__main__':
    if not finalize_if_ready(): raise SystemExit('Not all replay workers have completed')
