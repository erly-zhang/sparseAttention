import numpy as np
from collections import defaultdict
from common import HERE,TASKS,METHODS,read,save,verify,sha

def main():
    m=verify();groups=defaultdict(list);seen=set()
    for i in range(128):
        d=HERE/'outputs'/('worker-'+str(i));complete=read(d/'complete.json')
        if complete['records']!=50:raise ValueError('Wrong sample count')
        for path in sorted(d.glob('*.json')):
            if path.name in ['complete.json','progress.json','gpu-check.json']:continue
            row=read(path);identity=(row['layer'],row['key'])
            if identity in seen:raise ValueError('Duplicate')
            seen.add(identity)
            for method,ks in row['methods'].items():
                for k,metrics in ks.items():
                    for metric,gs in metrics.items():
                        for g,v in gs.items():
                            if not np.isfinite(v):raise ValueError('Nonfinite output')
                            groups[(row['task'],method,int(k),row['layer'],g,metric)].append(v)
    if len(seen)!=6400:raise ValueError('Incomplete sample/layer pairs')
    if read(HERE/'source-verification.json')['status']!='PASS':raise ValueError('Source bytes unverified')
    rng=np.random.default_rng(42);res=[]
    for (task,method,k,l,g,metric),vs in sorted(groups.items()):
        values=np.array(vs);boot=values[rng.integers(0,len(vs),size=(2000,len(vs)))].mean(1)
        res.append(dict(task=task,method=method,k=k,layer=l,group=g,metric=metric,mean=float(values.mean()),n=len(vs),bootstrap_95_low=float(np.quantile(boot,.025)),bootstrap_95_high=float(np.quantile(boot,.975))))
    save(HERE/'outputs/jspace-summary.json',dict(layer_means=res,scope='Full-context prefill; all saved positions; sample-weighted summaries',sparsity=[16,25],random_control=m['random_control']))
    save(HERE/'outputs/audit.json',dict(status='PASS',workers=128,sample_layer_pairs=6400,method_sample_layer_pairs=44800,k_values=[16,25],source_verification='PASS',lens_sha256=m['lens_sha256'],lock_sha256=sha(HERE/'lock.json')))
    from visualize import main as visualize
    visualize()
    from report import main as report
    report()
    print('ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
