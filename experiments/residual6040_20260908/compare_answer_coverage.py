"""Matched-three-input coverage, own trajectories; raw counts and provenance."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/local/results/residual6040_20260908')
OLD=Path('/home/ubuntu/work/experiments/outputs/llama_kv_probmean_compact_20260906/merged')
B='probmean_shared6144_member2q4096_allmembers'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def add(dst,values):
    for k,v in values.items():dst[k]=dst.get(k,0)+int(v)
def finish(x):
    result=dict(x)
    for e in ('value','kv'):
        for unit in ('slots','pairs'):
            result[e+'_'+unit+'_coverage']=x[e+'_kept_'+unit]/x[e+'_legal_'+unit]
    return result


def main():
    profile=next(ROOT.glob('attempts/profile/'+B+'/**/profiling_summary.json')).parent
    br=rows(profile/'online_metrics.jsonl');assert len(br)==3
    reference={x['input_ids_sha256']:x for x in br}
    ranges={};results={};provenance=[]
    btotal={};blayers={};bsamples={};bshadow={}
    for row in br:
        path=Path(row['profiling_file']);meta=read(path.with_suffix('.json'))
        sha=row['input_ids_sha256'];n=row['input_tokens'];ranges[sha]=meta['evidence_ranges'];sample={};seen=set()
        cols={name:i for i,name in enumerate(meta['fields'])}
        with np.load(path) as arrays:
            for item in meta['mapping']:
                layer,h=item['layer'],item['head'];assert (layer,h) not in seen;seen.add((layer,h))
                a=arrays[item['array']];assert len(a)==(n+127)//128
                values={};shadow={}
                for label,prefix in [('value','value'),('kv','kv_evidence')]:
                    for unit in ('slots','pairs'):
                        legal=int(a[:,cols[prefix+'_legal_'+unit]].sum(dtype=np.float64))
                        keep=int(a[:,cols[prefix+'_final_'+unit]].sum(dtype=np.float64))
                        old=int(a[:,cols[prefix+'_shared10240_'+unit]].sum(dtype=np.float64))
                        assert 0<=keep<=legal and 0<=old<=legal
                        values.update({label+'_legal_'+unit:legal,label+'_kept_'+unit:keep})
                        shadow.update({label+'_legal_'+unit:legal,label+'_kept_'+unit:old})
                add(btotal,values);add(sample,values);add(blayers.setdefault(str(layer),{}),values);add(bshadow,shadow)
        assert len(seen)==1024;bsamples[sha]=finish(sample);provenance.append(str(path))
    results['B_member2q']={'overall':finish(btotal),'per_layer':{k:finish(v) for k,v in blayers.items()},'per_sample':bsamples}
    for label,method in [('Shared10240','probmean_shared_compact10240'),('Perhead10240','probmean_perhead_compact10240')]:
        directory=OLD/method/'kv_retrieval';source=rows(directory/'online_metrics.jsonl')
        selected=[r for r in source if r['input_ids_sha256'] in reference];assert len(selected)==3
        total={};layers={};samples={}
        for row in selected:
            sha=row['input_ids_sha256'];assert row['input_tokens']==reference[sha]['input_tokens']
            n=row['input_tokens'];path=Path(row['selection_stats_file']);meta=read(path.with_suffix('.json'))
            assert meta['input_ids_sha256']==sha and meta['ranges']==ranges[sha]
            provenance.append(str(path));sample={};seen=set()
            with np.load(path) as arrays:
                for item in meta['mapping']:
                    layer=item['layer'];members=item['member_heads'];weight=len(members)
                    for h in members:assert (layer,h) not in seen;seen.add((layer,h))
                    a=arrays[item['array']];assert len(a)==(n+127)//128
                    start=np.arange(len(a))*128;end=np.minimum(start+128,n);m=end-start
                    assert np.array_equal(a[:,1],np.minimum(end,10240))
                    values={}
                    for e,(lo,hi),legal_col,kept_col in [('value',ranges[sha][0],5,7),('kv',ranges[sha][1],8,10)]:
                        legal=a[:,legal_col].astype(np.int64);kept=a[:,kept_col].astype(np.int64)
                        assert np.array_equal(legal,np.maximum(0,np.minimum(end,hi)-lo))
                        # Sink/local are forced in all compared methods. Only local
                        # keys have fewer than m visible queries; all are retained.
                        lower=np.maximum(start,lo);upper=np.minimum(end,hi)
                        span=np.maximum(0,upper-lower)
                        correction=span*(lower+upper-1-2*start)//2
                        correction=np.where(span>0,correction,0)
                        kept_pairs=kept*m-correction;legal_pairs=legal*m-correction
                        assert ((kept_pairs>=0)&(kept_pairs<=legal_pairs)).all()
                        for unit,kk,ll in [('slots',kept,legal),('pairs',kept_pairs,legal_pairs)]:
                            values[e+'_kept_'+unit]=int(kk.sum())*weight
                            values[e+'_legal_'+unit]=int(ll.sum())*weight
                    add(total,values);add(sample,values);add(layers.setdefault(str(layer),{}),values)
            assert len(seen)==1024;samples[sha]=finish(sample)
        results[label]={'overall':finish(total),'per_layer':{k:finish(v) for k,v in layers.items()},'per_sample':samples}
    for method,data in results.items():
        for e in ('value','kv'):
            for unit in ('slots','pairs'):
                assert data['overall'][e+'_legal_'+unit]==btotal[e+'_legal_'+unit]
    # Cross-check the pair reconstruction against B's directly recorded pairs.
    for row in br:
        p=Path(row['profiling_file']);meta=read(p.with_suffix('.json'));n=row['input_tokens'];cols={s:i for i,s in enumerate(meta['fields'])}
        with np.load(p) as arrays:
            for a in arrays.values():
                start=np.arange(len(a))*128;end=np.minimum(start+128,n);m=end-start
                for prefix,(lo,hi) in zip(('value','kv_evidence'),meta['evidence_ranges']):
                    lower=np.maximum(start,lo);upper=np.minimum(end,hi);span=np.maximum(0,upper-lower)
                    correction=np.where(span>0,span*(lower+upper-1-2*start)//2,0)
                    derived=a[:,cols[prefix+'_final_slots']]*m-correction
                    assert np.array_equal(derived,a[:,cols[prefix+'_final_pairs']])
    output={'input_count':3,'input_identities':[{'sha256':sha,'tokens':r['input_tokens'],'ranges':ranges[sha]} for sha,r in reference.items()],
        'aggregation':'pooled raw numerator/denominator across 3 samples, 32 layers, 32 Q heads and all real query tiles',
        'own_trajectory_results':results,'B_same_QK_shadow_Shared10240':finish(bshadow),
        'qa':{'same_input_hashes_lengths_ranges':True,'no_duplicate_missing_layer_heads':True,
            'equal_evidence_denominators':True,'historical_pair_reconstruction_matches_direct_B_pairs':True},
        'caveats':['Coverage is not 497-sample aggregate. Formal accuracy has a different scope.',
            'Token-slot coverage counts each evidence key once per head and tile that can see it.',
            'Causal-pair coverage counts each legally visible query-evidence-key interaction.',
            'KV span includes queried key, intervening separators, and value; it is not an all-or-nothing complete-span rate.',
            'Own trajectories can differ after sparse attention. B shadow Shared10240 fixes B Q/K but does not re-run Shared end to end.'],
        'sources':provenance}
    dest=ROOT/'answer_coverage_comparison';dest.mkdir(exist_ok=True)
    (dest/'coverage.json').write_text(json.dumps(output,indent=2))
    import csv
    with (dest/'per_layer.csv').open('w') as f:
        names=['method','layer','value_slots_coverage','value_pairs_coverage','kv_slots_coverage','kv_pairs_coverage']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for name,data in results.items():
            for layer,v in data['per_layer'].items():writer.writerow({'method':name,'layer':layer,**{k:v[k] for k in names[2:]}})
    print(json.dumps({k:v['overall'] for k,v in results.items()},indent=2))
    print('B_same_QK_shadow_Shared10240',finish(bshadow))
    print('output',dest)


if __name__=='__main__':main()
