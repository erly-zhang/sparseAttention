"""CPU-only aggregation outside measured generation; raw tile arrays are retained."""
import json
from pathlib import Path
import numpy as np

BINS=np.array([0,128,256,512,1024,2048,4096,8192,10240,16384,32768,65536,131073])

def new_bucket():
    return {'rows':0,'target_sum':0.,'final_sum':0.,'target_min':float('inf'),'final_min':float('inf'),
            'target_max':0.,'final_max':0.,'target_hist':np.zeros(len(BINS)-1,dtype=np.int64),
            'final_hist':np.zeros(len(BINS)-1,dtype=np.int64),'target_mass_sum':0.,'final_mass_sum':0.,
            'value_legal':0.,'value_target':0.,'value_final':0.,'pair_legal':0.,'pair_target':0.,'pair_final':0.}

def add(bucket,array,weight=1):
    bucket['rows']+=len(array)*weight
    for idx,name in [(0,'target'),(1,'final')]:
        bucket[name+'_sum']+=float(array[:,idx].sum(dtype=np.float64))*weight
        bucket[name+'_min']=min(bucket[name+'_min'],float(array[:,idx].min()))
        bucket[name+'_max']=max(bucket[name+'_max'],float(array[:,idx].max()))
        bucket[name+'_hist']+=np.histogram(array[:,idx],BINS)[0]*weight
    bucket['target_mass_sum']+=float(array[:,2].sum(dtype=np.float64))*weight
    bucket['final_mass_sum']+=float(array[:,3].sum(dtype=np.float64))*weight
    for i,name in enumerate(['value_legal','value_target','value_final','pair_legal','pair_target','pair_final'],5):
        bucket[name]+=float(array[:,i].sum(dtype=np.float64))*weight

def finish(b):
    out={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items()}
    out['target_count_mean']=b['target_sum']/b['rows']; out['final_count_mean']=b['final_sum']/b['rows']
    out['target_mass_mean']=b['target_mass_sum']/b['rows']; out['final_mass_mean']=b['final_mass_sum']/b['rows']
    for evidence in ('value','pair'):
        for stage in ('target','final'):
            den=b[evidence+'_legal']
            out[evidence+'_'+stage+'_coverage']=b[evidence+'_'+stage]/den if den else None
    return out

def aggregate(directory):
    rr=[json.loads(s) for s in (directory/'online_metrics.jsonl').read_text().splitlines()]
    overall=new_bucket(); by_layer={}; by_head={}
    for r in rr:
        path=Path(r['selection_stats_file'])
        metadata=json.loads(path.with_suffix('.json').read_text())
        assert metadata['input_ids_sha256']==r['input_ids_sha256']
        with np.load(path) as data:
            for item in metadata['mapping']:
                arr=data[item['array']]
                layer=str(item['layer']); members=item['member_heads']
                add(overall,arr,len(members))
                add(by_layer.setdefault(layer,new_bucket()),arr,len(members))
                for head in members:
                    add(by_head.setdefault(f'{layer}:{head}',new_bucket()),arr)
    out={'sample_count':len(rr),'weighting':'each Q head and query tile once; shared rows expanded using member mapping',
         'coverage_definition':'evidence-token slots visible to at least one query in tile; not exact pair coverage',
         'count_histogram_edges_left_closed':BINS.tolist(),'overall':finish(overall),
         'per_layer':{k:finish(v) for k,v in by_layer.items()},'per_layer_head':{k:finish(v) for k,v in by_head.items()},
         'per_sample_head_tile_arrays':[r['selection_stats_file'] for r in rr]}
    (directory/'selection_summary.json').write_text(json.dumps(out,indent=2))
    return out

def comparison(root):
    historical=Path('/home/ubuntu/work/diagnostics_review_20260906/verified_historical.json')
    old=json.loads(historical.read_text())['comparisons'] if historical.exists() else []
    selected=[]
    for row in old:
        path=row['path']
        if row['count']==497 and any(t in path for t in ('shard_token_compact/merged/','/llama31/shareprefill_ae3_token_block_auto/','/llama31/flexprefill/','/llama31/minference/')):
            summary=json.loads(Path(path).read_text())
            selected.append({'path':path,'kind':'historical','input_match':row['exact_full_input_match'],
                             'score':row['score'],'sparsity':row['pair_sparsity_recomputed'],
                             'runtime':summary['runtime'],'hardware':summary['hardware'],'timing_scope':summary['timing_scope']})
    for path in sorted((root/'merged').glob('*/kv_retrieval/summary.json')):
        s=json.loads(path.read_text()); selection=aggregate(path.parent)
        selected.append({'path':str(path),'kind':'new','input_match':s['input_alignment'],
                         'score':s['official_lm_eval_results'],'runtime':s['runtime'],
                         'hardware':s['hardware'],'selection':selection['overall']})
    (root/'comparison.json').write_text(json.dumps({'results':selected,'caveats':[
        'Historical tile_sum_plus_tail1024 differs in normalization order, logit scale and tail term.',
        'Do not compare speed across different hardware or timing/instrumentation scopes.',
        'TopK and Top-p have different measured sparsity; no matched-sparsity claim.',
        'Coverage counts evidence slots, not causal necessity or task accuracy.']},indent=2))

if __name__=='__main__':
    comparison(Path('/home/ubuntu/work/experiments/outputs/llama_kv_probmean_compact_20260906'))
