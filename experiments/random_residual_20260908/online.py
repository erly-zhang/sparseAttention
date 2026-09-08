"""Full-generation canonical runner; no calibrated residual policy dependency."""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'probmean_compact_20260906'))
import run as base
from experiments.benchmark_shareprefill_ae3 import input_ids_sha256
sys.path.insert(0,str(HERE))
from random_selector import RandomSelector,FIELDS
from random_policy import prepare,random_final

METHOD='probmean_shared9216_random1024_allmembers_seed42'
base.runner.METHODS=tuple(base.runner.METHODS)+(METHOD,)
base.runner.SHAREPREFILL_METHODS=tuple(base.runner.SHAREPREFILL_METHODS)+(METHOD,)


def install(model,method,**kwargs):
    assert method==METHOD
    original=base.sparse.RepresentativeTokenFirstBlockSelector
    class NewSelector(RandomSelector):
        def __init__(self,config,**unused):super().__init__(config,seed=42)
    base.sparse.RepresentativeTokenFirstBlockSelector=NewSelector
    try:result,patch,metadata=base.old_install(model,'shareprefill_ae3_token_compact',**kwargs)
    finally:base.sparse.RepresentativeTokenFirstBlockSelector=original
    groups=patch.selector.layers
    for layer in groups.values():
        assert len(layer)==3
        assert sorted(h for g in layer for h in g['members'])==list(range(32))
        assert all(int(g['representative']) in g['members'] for g in layer)
        assert len({int(g['representative']) for g in layer})==3
    tokenizer=AutoTokenizer.from_pretrained(kwargs['model_name'])
    base.state['value_resolver']=base.build_kv_retrieval_range_resolver(tokenizer,value_only=True)
    base.state['pair_resolver']=base.build_kv_retrieval_range_resolver(tokenizer)
    metadata={'implementation':METHOD,'num_groups_per_layer':3,'model':kwargs['model_name'],
        'group_config_path':str(kwargs['group_config_path']),
        'online_config':{'score':'mean_of_per_query_causal_softmax','query_tile':128,
            'softmax_again':False,'global_tail':False,'final_budget':10240,'member_core_budget':9216,
            'member_residual':'B-min(9216,B)','sink_tokens':128,'local':'current real query tile',
            'representatives_per_layer':3,'random_members_per_layer':29,'offline_residual_calibration':False,
            'seed':42,'seed_fields':['experiment_seed','input_ids_sha256','layer','head','query_tile'],
            'seed_hash':'SHA256 first64 bits','rng':'Triton Philox randint, unbiased range rejection, first unique draws',
            'sampling':'uniform without replacement over all legal keys outside shared core',
            'block_projection':False,'global_model_rng_changed_by_sampler':False}}
    device=next(model.parameters()).device
    # Warmup active random paths as the runner's 4096-token warmup is dense.
    n=11009;ends=(torch.arange((n+127)//128,device=device)+1).mul(128).clamp(max=n)
    legal=torch.arange(n,device=device)[None,:]<ends[:,None]
    p=legal.float()/ends[:,None]
    bundle=prepare(p)
    random_final(bundle,'warmup',42,0,0)
    for length in (4096,4097):
        q=torch.randn(1,length,32,128,device=device,dtype=torch.bfloat16)
        k=torch.randn(1,length,8,128,device=device,dtype=torch.bfloat16)
        base.probability_mean(q,k,0,0)
    del q,k,p,bundle
    torch.cuda.synchronize()
    return result,patch,metadata


base.runner.install_benchmark_method=install
OriginalRecorder=base.Recorder.__bases__[0]


class Recorder(OriginalRecorder):
    def install(self):
        super().install();inner=self.model.generate;recorder=self
        def generate(model,*args,**kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            sel=recorder.patch.selector
            sel.input_hash=input_ids_sha256(ids)
            sel.ranges=(base.state['value_resolver'](ids)[0],base.state['pair_resolver'](ids)[0])
            sel.records=[];sel.collect=True;stamp={}
            def before(_m,_a,named):
                inp=named.get('input_ids')
                if 'start' not in stamp and inp is not None and inp.shape[1]>1:
                    torch.cuda.synchronize();stamp['start']=time.perf_counter()
            def after(_m,_a,named,out):
                if 'start' in stamp and 'end' not in stamp:
                    torch.cuda.synchronize();stamp['end']=time.perf_counter()
            handles=[model.register_forward_pre_hook(before,with_kwargs=True),model.register_forward_hook(after,with_kwargs=True)]
            try:result=inner(*args,**kwargs)
            finally:
                for h in handles:h.remove()
                sel.collect=False
            row=recorder.rows[-1]
            assert row['input_ids_sha256']==sel.input_hash
            row['prefill_cuda_event_sec']=row['prefill_latency_sec']
            row['prefill_latency_sec']=stamp['end']-stamp['start']
            row['decode_latency_sec']=row['total_latency_sec']-row['prefill_latency_sec']
            row['timing_scope']='GPU-synchronized first forward including scoring, random sampling, deduplication, indices, attention and GPU statistics'
            arrays={};mapping=[];seen=set();reps=0
            for layer,rep,members,tensor in sel.records:
                assert len(members)==1;h=members[0]
                assert (layer,h) not in seen;seen.add((layer,h))
                key=f'layer{layer}_head{h}'
                a=tensor.detach().cpu().numpy().astype(np.float32)
                arrays[key]=a;mapping.append({'array':key,'layer':layer,'source_head':rep,'member_heads':[h]})
                ends=np.minimum((np.arange(len(a))+1)*128,row['input_tokens']);b=np.minimum(ends,10240)
                assert np.isfinite(a).all() and np.abs(a[:,4]-1).max()<1e-4
                assert np.array_equal(a[:,1],b)
                core=b if h==rep else np.minimum(9216,b)
                assert np.array_equal(a[:,11],core)
                assert np.array_equal(a[:,12],b-core)
                assert (a[:,13]<=a[:,12]).all()
                assert (a[:,14]==int(h==rep)).all()
                reps+=h==rep
            assert len(seen)==1024 and reps==96
            directory=recorder.output_path.parent/'selection_stats';directory.mkdir(exist_ok=True)
            stem=f"{len(recorder.rows)-1:04d}_{sel.input_hash[:12]}"
            path=directory/f'{stem}.npz'
            np.savez_compressed(path,**arrays)
            (directory/f'{stem}.json').write_text(json.dumps({'input_ids_sha256':sel.input_hash,
                'input_tokens':row['input_tokens'],'seed':42,'ranges':sel.ranges,'mapping':mapping,'fields':FIELDS,
                'mass_definition':'own-group representative probability, not member probability',
                'evidence_definition':'original-position token slots visible to at least one tile query'},indent=2))
            row['selection_stats_file']=str(path)
            row['max_probability_sum_error']=max(float(np.abs(a[:,4]-1).max()) for a in arrays.values())
            recorder.output_path.write_text(''.join(json.dumps(r)+'\n' for r in recorder.rows))
            sel.records=[]
            return result
        self.model.generate=types.MethodType(generate,self.model)


base.runner.GenerationMetricsRecorder=Recorder
if __name__=='__main__':base.runner.main()
