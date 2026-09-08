"""Isolated canonical full-generation runner, with an optional profiling-only mode."""
import json
import os
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
sys.path.insert(0,str(HERE))
from selector6040 import Selector,METHODS
from experiments.benchmark_shareprefill_ae3 import input_ids_sha256

base.runner.METHODS=tuple(base.runner.METHODS)+tuple(METHODS)
base.runner.SHAREPREFILL_METHODS=tuple(base.runner.SHAREPREFILL_METHODS)+tuple(METHODS)
PROFILE=os.environ.get('RESIDUAL6040_PROFILE')=='1'
VALIDATE=os.environ.get('RESIDUAL6040_VALIDATE')=='1'


def install(model,method,**kwargs):
    mode=METHODS[method];original=base.sparse.RepresentativeTokenFirstBlockSelector
    class NewSelector(Selector):
        def __init__(self,config,**unused):super().__init__(config,mode)
    base.sparse.RepresentativeTokenFirstBlockSelector=NewSelector
    try:result,patch,metadata=base.old_install(model,'shareprefill_ae3_token_compact',**kwargs)
    finally:base.sparse.RepresentativeTokenFirstBlockSelector=original
    sel=patch.selector
    for groups in sel.layers.values():
        assert len(groups)==3 and len({g['representative'] for g in groups})==3
        assert sorted(h for g in groups for h in g['members'])==list(range(32))
        assert all(g['representative'] in g['members'] for g in groups)
    device=next(model.parameters()).device
    # Actual sparse paths and tensor strides, outside formal timing and RNG state.
    with torch.random.fork_rng(devices=[device]):
        for n in (10496,10497):
            q=torch.randn(1,32,n,128,device=device,dtype=torch.bfloat16).transpose(1,2)
            k=torch.randn(1,8,n,128,device=device,dtype=torch.bfloat16).transpose(1,2)
            v=torch.randn_like(k);token=sel.current_layer.set(0)
            index=sel(q,k,v,128,.95,1,None)
            out=base.sparse.triton_token_compacted_prefill_attention(q,k,v,index,token_chunk_size=32)
            assert torch.isfinite(out).all()
            sel.current_layer.reset(token)
        del q,k,v,index,out
    torch.cuda.synchronize();sel.stats=type(sel.stats)()
    sel.profile=PROFILE
    sel.validate=VALIDATE
    if PROFILE:
        tokenizer=AutoTokenizer.from_pretrained(kwargs['model_name'])
        base.state['value_resolver']=base.build_kv_retrieval_range_resolver(tokenizer,value_only=True)
        base.state['pair_resolver']=base.build_kv_retrieval_range_resolver(tokenizer)
        original_kernel=base.sparse.triton_token_compacted_prefill_attention
        def measured(*args,**kw):
            with sel.phase('attention'):return original_kernel(*args,**kw)
        base.sparse.triton_token_compacted_prefill_attention=measured
    metadata={'implementation':method,'model':kwargs['model_name'],'num_groups_per_layer':3,
        'group_config_path':str(kwargs['group_config_path']),
        'online_config':{'query_tile':128,'score':'mean_of_per_real_query_causal_softmax',
            'member_score':'two_real_query_probability_mean' if mode=='member2q' else 'uniform_without_replacement',
            'sampling_positions':'a+floor((u+0.5)*m/min(2,m))',
            'final_budget':10240,'member_core_budget':6144,'member_residual_budget':4096,
            'representatives_per_layer':3,'all_nonrepresentatives':29,'offline_residual_calibration':False,
            'sink':128,'local':'current real query tile','global_tail':False,'softmax_again':False,
            'block_projection':False,'seed':42,'random_stream':'SHA256(seed,input hash,layer,head,tile)',
            'member2q_tie_break':'descending float32 probability then ascending original key position'},
        'index_implementation':'shared short core; member short residual; sorted fixed-stride CSR; original compact kernel unchanged',
        'profile_only':PROFILE,'timing_scope':'synchronized wall-clock first forward; all selector/index/attention work and required pair counts; no extra evidence diagnostics in formal runs',
        'diagnostics_scope':'separate same-3-input full-generation run only; member full128 probability is trajectory-specific'}
    return result,patch,metadata


base.runner.install_benchmark_method=install
OriginalRecorder=base.Recorder.__bases__[0]


class Recorder(OriginalRecorder):
    def install(self):
        super().install();inner=self.model.generate;recorder=self
        def generate(model,*args,**kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None);sel=recorder.patch.selector
            # Hash is canonical identity preparation. Deriving all sampling streams
            # from it, sampling and indices are within the timed selector.
            sel.input_hash=input_ids_sha256(ids);sel.diagnostics=[];sel.events=[]
            sel.validation=[];finite=[]
            if PROFILE:sel.ranges=(base.state['value_resolver'](ids)[0],base.state['pair_resolver'](ids)[0])
            stamp={}
            def before(_m,_a,named):
                x=named.get('input_ids')
                if 'start' not in stamp and x is not None and x.shape[1]>1:
                    torch.cuda.synchronize();stamp['start']=time.perf_counter()
            def after(_m,_a,named,out):
                if 'start' in stamp and 'end' not in stamp:
                    torch.cuda.synchronize();stamp['end']=time.perf_counter()
                if VALIDATE and hasattr(out,'logits'):finite.append(torch.isfinite(out.logits).all())
            hooks=[model.register_forward_pre_hook(before,with_kwargs=True),model.register_forward_hook(after,with_kwargs=True)]
            try:result=inner(*args,**kwargs)
            finally:
                for h in hooks:h.remove()
            row=recorder.rows[-1]
            assert row['input_ids_sha256']==sel.input_hash
            assert result.dtype in (torch.int32,torch.int64) and result.shape[1]>ids.shape[1]
            if VALIDATE:
                assert len(sel.validation)==1024 and torch.stack(sel.validation).all()
                assert finite and torch.stack(finite).all()
                row['smoke_indices_and_finite_logits_passed']=True
            row['prefill_cuda_event_sec']=row['prefill_latency_sec']
            row['prefill_latency_sec']=stamp['end']-stamp['start']
            row['decode_latency_sec']=row['total_latency_sec']-row['prefill_latency_sec']
            row['profile_only']=PROFILE
            row['timing_scope']='synchronized full prefill including selection, sampling, indices, required counts and attention'+('; includes separate profiling diagnostics, not formal timing' if PROFILE else '')
            if PROFILE:
                artifacts=recorder.output_path.parent/'profiling';artifacts.mkdir(exist_ok=True)
                stem=f"{len(recorder.rows)-1:04d}_{sel.input_hash[:12]}";arrays={};mapping=[]
                for layer,h,rep,tensor,fields in sel.diagnostics:
                    name=f'layer{layer}_head{h}';arrays[name]=tensor.cpu().numpy()
                    mapping.append({'array':name,'layer':layer,'head':h,'representative':rep})
                assert len(mapping)==1024
                for a in arrays.values():
                    assert np.isfinite(np.delete(a,[11,12],axis=1)).all()
                    assert np.abs(a[:,4]-1).max()<1e-4
                    ends=np.minimum((np.arange(len(a))+1)*128,row['input_tokens'])
                    assert np.array_equal(a[:,2],np.minimum(10240,ends))
                np.savez_compressed(artifacts/f'{stem}.npz',**arrays)
                phase={}
                for name,a,b in sel.events:phase[name]=phase.get(name,0.)+a.elapsed_time(b)/1000
                (artifacts/f'{stem}.json').write_text(json.dumps({'input_ids_sha256':sel.input_hash,'input_tokens':row['input_tokens'],
                    'fields':fields,'mapping':mapping,'phase_gpu_stream_sec':phase,'evidence_ranges':sel.ranges,
                    'normalization':'member128 and member2q kept distinct; mass is member trajectory, not shared proxy',
                    'source_scope':'three-input independent profiling only'},indent=2))
                row['profiling_file']=str(artifacts/f'{stem}.npz');row['phase_gpu_stream_sec']=phase
            recorder.output_path.write_text(''.join(json.dumps(r)+'\n' for r in recorder.rows))
            sel.diagnostics=[];sel.events=[];sel.validation=[]
            return result
        self.model.generate=types.MethodType(generate,self.model)


base.runner.GenerationMetricsRecorder=Recorder
if __name__=='__main__':base.runner.main()
