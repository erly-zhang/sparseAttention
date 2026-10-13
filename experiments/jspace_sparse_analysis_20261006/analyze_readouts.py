"""Independently aggregate saved position metrics, equal weighting of prompts."""
from collections import defaultdict
import numpy as np
from common import HERE,SOURCE,METHODS,read,save

def main():
    manifest=read(SOURCE/'input-manifest.json');groups=defaultdict(list);scores=[]
    original=read(SOURCE/'outputs/summary.json')
    expected={(r['task'],r['method'],r['group'],r['layer']):r['summary']['mean'] for r in original['layer_means']}
    for method in METHODS:
        for item in manifest['evaluation']:
            v=read(SOURCE/'outputs'/('replay-'+method+'-'+str(item['analysis_shard']))/(item['key']+'.json'))
            lookup={p:i for i,p in enumerate(v['positions'])};scores.append(dict(task=v['task'],method=method,key=v['key'],historical_score=v['historical_score'],history_first_token_equal=v['history_replay_match']['first_token_equal']))
            for layer,metrics in v['layer_metrics'].items():
                arrays=dict(js=np.array(metrics['js_nats']),top10_recall=np.array(metrics['top_k_overlap']),
                    top1_preserved=np.array(metrics['dense_top1_rank_in_sparse'])==1,
                    residual_relative_l2=np.array(metrics['residual']['relative_l2']),
                    transported_relative_l2=np.array(metrics['transported_residual']['relative_l2']))
                for group,ps in v['position_groups'].items():
                    if not ps:continue
                    idx=[lookup[p] for p in ps]
                    for metric,vals in arrays.items():
                        value=float(vals[idx].mean());key=(v['task'],method,group,int(layer),metric)
                        groups[key].append((v['key'],value,v['history_replay_match']['first_token_equal']))
        print('AGGREGATED',method,flush=True)
    rng=np.random.default_rng(42);rows=[];maxdiff=0
    for (task,method,group,layer,metric),values in sorted(groups.items()):
        a=np.array([v for key,v,match in values]);mean=float(a.mean())
        if not np.isfinite(a).all():raise ValueError('Nonfinite metric')
        if metric=='js':maxdiff=max(maxdiff,abs(mean-expected[(task,method,group,layer)]))
        boot=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
        rows.append(dict(task=task,method=method,group=group,layer=layer,metric=metric,mean=mean,n=len(a),bootstrap_95_low=float(np.quantile(boot,.025)),bootstrap_95_high=float(np.quantile(boot,.975))))
    if maxdiff>1e-10:raise ValueError('Source summary disagreement')
    save(HERE/'outputs/readout-summary.json',dict(layer_means=rows,historical_scores=scores,independent_js_reconciliation_max_abs=maxdiff))
    from visualize import main as visualize
    visualize()
if __name__=='__main__':main()
