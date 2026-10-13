"""Independently reconcile published scores and all recorded branch decisions.

Read-only with respect to experiment inputs, implementations and old results.
"""
from collections import defaultdict, Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE.parents[1]/'experiments'
SCORES=EXPERIMENTS/'representative_patterns_fast_retry03_20261001'
USAGE=EXPERIMENTS/'flexprefill_head_usage_retry01_20261002'
TASKS=['kv_retrieval','longbook_choice_eng','longbook_qa_eng','code_debug']
METHODS=['dense_full','coverage_token_vertical','coverage_token_vertical_slash',
         'random_token_vertical','random_token_vertical_slash','flexprefill','minference']
EXPECTED=[['62.00','78.00','30.40','18.00'],['4.00','74.00','23.16','12.00'],
          ['4.00','78.00','26.46','20.00'],['0.00','68.00','25.80','8.00'],
          ['0.00','68.00','22.35','8.00'],['42.00','78.00','32.08','18.00'],
          ['10.00','74.00','28.12','20.00']]


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate():
    score_audit=read(SCORES/'outputs/independent-audit.json')
    usage_audit=read(USAGE/'outputs/independent-audit.json')
    assert score_audit['status']==usage_audit['status']=='PASS'
    assert usage_audit['source_lock_sha256']==sha(USAGE/'source-lock.json')
    assert score_audit['source_lock_sha256']==sha(SCORES/'source-lock.json')
    scores=defaultdict(list)
    score_keys=set()
    for folder in sorted((SCORES/'outputs').glob('formal-*')):
        if not folder.is_dir():continue
        completion=read(folder/'complete.json')
        for filename,digest in completion['files'].items():
            f=folder/filename
            if not filename.endswith('.json'):continue
            assert sha(f)==digest, str(f)
            r=read(f)
            if 'prediction' not in r:continue
            k=(r['method'],r['key'])
            assert k not in score_keys
            score_keys.add(k)
            assert math.isfinite(r['score']) and 0<=r['score']<=1
            scores[(r['task'],r['method'])].append(r['score'])
    assert len(score_keys)==1400
    table=[]
    for method,expected in zip(METHODS,EXPECTED):
        values=[]
        for task in TASKS:
            sample=scores[(task,method)]
            assert len(sample)==50
            mean=sum(sample)/len(sample)
            assert abs(mean-score_audit['summary'][task][method]['mean_official_score'])<1e-12
            values.append(format(mean*100,'.2f'))
        assert values==expected,(method,values,expected)
        table.append(dict(method=method,scores=values,samples_per_task=50))
    records=[]
    finite=[]
    counts=Counter()
    actual_calls=[0,0]
    nonfinite=[0,0]
    mismatches=[0,0]
    layer_counts=defaultdict(list)
    task_layer_counts=defaultdict(list)
    changes=0
    for folder in sorted((USAGE/'outputs').glob('worker-*')):
        completion=read(folder/'complete.json')
        for filename,digest in completion['files'].items():
            if not filename.endswith('.json'):continue
            f=folder/filename
            assert sha(f)==digest, str(f)
            r=read(f)
            if 'layers' not in r:continue
            records.append(r)
            for p,layer_field in enumerate(('layers','repeat_layers')):
                for layer,v in r[layer_field].items():
                    qa=v['query_aware_heads'];vs=v['vertical_slash_heads']
                    assert sorted(qa+vs)==list(range(32)) and not set(qa)&set(vs)
                    assert len(qa)==v['query_aware_count']==v['actual_query_aware_branch_calls']
                    assert len(vs)==v['vertical_slash_count'] and v['tau']==.1
                    actual_calls[p]+=len(qa)
                    counts[(p,'qa')]+=len(qa);counts[(p,'vs')]+=len(vs)
                    assert len(v['sqrt_js_divergence'])==32
                    for h,x in enumerate(v['sqrt_js_divergence']):
                        if x is None or not math.isfinite(x):nonfinite[p]+=1;continue
                        mismatches[p]+=((x<.1)!=(h in qa))
                        if p==0:finite.append(x)
                    if p==0:
                        layer_counts[layer].append(len(qa))
                        task_layer_counts[(r['task'],layer)].append(len(qa))
            for layer in r['layers']:
                changes+=len(set(r['layers'][layer]['query_aware_heads'])^
                             set(r['repeat_layers'][layer]['query_aware_heads']))
    assert len(records)==len({r['key'] for r in records})==200
    assert Counter(r['task'] for r in records)==Counter({t:50 for t in TASKS})
    assert nonfinite==[0,0] and mismatches==[0,0]
    assert counts[(0,'qa')]==counts[(1,'qa')]==usage_audit['total_query_aware_decisions']==1050
    assert counts[(0,'vs')]==counts[(1,'vs')]==usage_audit['total_vertical_slash_decisions']==203750
    assert changes==usage_audit['stability']['changed_decisions']==0
    for layer,values in layer_counts.items():
        s=usage_audit['all']['layers'][layer]
        assert sum(values)==s['query_aware_total']
        assert sum(values)/len(values)==s['query_aware_mean']
        assert (min(values),max(values))==(s['query_aware_min'],s['query_aware_max'])
    for (task,layer),values in task_layer_counts.items():
        assert sum(values)==usage_audit['by_task'][task]['layers'][layer]['query_aware_total']
    vals=sorted(finite)
    quantiles={str(q):vals[round(q*(len(vals)-1))] for q in (0,.01,.05,.25,.5,.75,.95,.99,1)}
    result=dict(status='PASS',score_units='mean official task score x100',score_records=1400,
        scores=table,head_samples=200,head_decisions_per_pass=204800,passes=2,
        query_aware_decisions_per_pass=1050,vertical_slash_decisions_per_pass=203750,
        query_aware_percent=1050/204800*100,vertical_slash_percent=203750/204800*100,
        finite_js_values_per_pass=[204800,204800],nonfinite_js_values_per_pass=nonfinite,
        threshold_mismatches_per_pass=mismatches,actual_query_aware_calls=actual_calls,
        changed_head_decisions_between_passes=changes,sqrt_js_quantiles=quantiles,
        by_layer=usage_audit['all']['layers'],
        counterfactual_threshold_counts={str(t):sum(x<t for x in finite) for t in (.05,.1,.15,.2,.3)},
        counterfactual_scope='Reclassify saved JS only; no changed-threshold forward run or benchmark scores.',
        head_statistics_scope=usage_audit['statistical_scope'],
        historical_logits_comparison=usage_audit['historical_logits_comparison'],
        paper_crosscheck=dict(url='https://proceedings.iclr.cc/paper_files/paper/2025/file/03645743ea35690f30d795d6bac149a5-Paper-Conference.pdf',
            section='Appendix H and Figure 8',tau=.1,context_tokens=131072,
            paper_example_query_aware_head_counts=[13,2],total_heads_per_example=1024,
            observation='Most heads use Vertical-Slash; Query-Aware heads mainly in the first layer.'),
        sources={str(SCORES/'outputs/independent-audit.json'):sha(SCORES/'outputs/independent-audit.json'),
                 str(USAGE/'outputs/independent-audit.json'):sha(USAGE/'outputs/independent-audit.json')})
    (HERE/'report-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS: all 1400 scores reconcile; 409600 recorded decisions finite and correctly classified; repeat agreement 100%.')
    return result


if __name__=='__main__':validate()
