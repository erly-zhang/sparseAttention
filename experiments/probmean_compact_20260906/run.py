"""Launch new methods through the unchanged canonical InfiniteBench runner."""
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
import experiments.token_compacted_sparse as sparse
import experiments.run_dense_multimodel_infinitebench as runner
from experiments.run_kv_oracle_infinitebench import build_kv_retrieval_range_resolver
from selector import ProbabilitySelector, probability_mean, choose

METHODS = {
    'probmean_shared_compact10240': (False,10240,False),
    'probmean_perhead_compact10240': (True,10240,False),
    'probmean_shared_compact_topp99': (False,None,False),
    'probmean_perhead_compact_topp99': (True,None,False),
    'logitmean_notail_shared_compact10240': (False,10240,True),
}
runner.METHODS = tuple(runner.METHODS) + tuple(METHODS)
runner.SHAREPREFILL_METHODS = tuple(runner.SHAREPREFILL_METHODS) + tuple(METHODS)
old_install = runner.install_benchmark_method
state = {}


def install(model, method, **kwargs):
    independent,budget,control=METHODS[method]
    # Swap only the constructor within this isolated process. The existing patch
    # retains the model's RoPE, per-Q-head GQA mapping and sparse/decode dispatch.
    original_class=sparse.RepresentativeTokenFirstBlockSelector
    # Installer also calls isinstance, so use a subclass instead of a function.
    class NewSelector(ProbabilitySelector):
        def __init__(self, config, **unused):
            super().__init__(config,per_head=independent,budget=budget,control=control)
    sparse.RepresentativeTokenFirstBlockSelector=NewSelector
    try:
        result,patch,metadata=old_install(model,'shareprefill_ae3_token_compact',**kwargs)
    finally:
        sparse.RepresentativeTokenFirstBlockSelector=original_class
    state['patch']=patch
    tokenizer=AutoTokenizer.from_pretrained(kwargs['model_name'])
    state['value_resolver']=build_kv_retrieval_range_resolver(tokenizer,value_only=True)
    state['pair_resolver']=build_kv_retrieval_range_resolver(tokenizer)
    metadata={
        'implementation':method,'model':kwargs['model_name'],'num_groups_per_layer':32 if independent else 3,
        'group_config_path':str(kwargs['group_config_path']),
        'online_config':{'query_tile':128,'score':'causal_logit_visible_query_mean' if control else 'mean_of_per_query_causal_softmax',
                         'probability_mean_denominator':'real_queries_in_tile', 'global_tail':False,
                         'per_head':independent,'target_topk':budget,'target_top_p':None if budget else .99,
                         'target_min_tokens':None if budget else 1024,'final_budget':budget,
                         'sink_tokens':128,'local':'current_query_tile','block_projection':False,
                         'protection_policy':'reserve_before_filling' if budget else 'union_without_truncation',
                         'target_definition':'unprotected_topk_prefix' if budget else 'shortest_mass_prefix_then_floor',
                         'control_causal_denominator':'number_of_queries_i_in_tile_with_i>=j' if control else None},
        'statistics':'GPU reductions within prefill; CPU transfer, serialization and validation outside timed generate',
    }
    metadata['online_config']['selection_rank']='raw_logits' if control else 'probability_mean'
    # Precompile length/alignment specializations outside measured model calls.
    device=next(model.parameters()).device
    for n in (4096,4097):
        q=torch.randn(1,32,n,128,device=device,dtype=torch.bfloat16).transpose(1,2)
        k=torch.randn(1,8,n,128,device=device,dtype=torch.bfloat16).transpose(1,2)
        for h in (0,1,4,16):
            p=probability_mean(q,k,h,h//4)
            choose(p,budget=budget)
    del q,k,p
    torch.cuda.synchronize()
    return result,patch,metadata
runner.install_benchmark_method=install


class Recorder(runner.GenerationMetricsRecorder):
    def install(self):
        super().install()
        inner=self.model.generate
        recorder=self
        def generate(model_self,*args,**kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            sel=recorder.patch.selector
            sel.ranges=(state['value_resolver'](ids)[0],state['pair_resolver'](ids)[0])
            sel.records=[]; sel.collect=True
            stamp={}
            def before(_module,_args,named):
                inp=named.get('input_ids')
                if 'start' not in stamp and inp is not None and inp.shape[1]>1:
                    torch.cuda.synchronize(); stamp['start']=time.perf_counter()
            def after(_module,_args,named,out):
                if 'start' in stamp and 'end' not in stamp:
                    torch.cuda.synchronize(); stamp['end']=time.perf_counter()
            handles=[model_self.register_forward_pre_hook(before,with_kwargs=True),model_self.register_forward_hook(after,with_kwargs=True)]
            try: output=inner(*args,**kwargs)
            finally:
                for h in handles: h.remove()
                sel.collect=False
            row=recorder.rows[-1]
            row['prefill_cuda_event_sec']=row['prefill_latency_sec']
            row['prefill_latency_sec']=stamp['end']-stamp['start']
            row['decode_latency_sec']=row['total_latency_sec']-row['prefill_latency_sec']
            row['timing_scope']='synchronized wall-clock first model forward; selector+index+attention+required stats'
            artifacts=recorder.output_path.parent/'selection_stats'; artifacts.mkdir(exist_ok=True)
            arrays={}; mapping=[]
            for layer,head,members,tensor in sel.records:
                key=f'layer{layer}_source{head}'
                arrays[key]=tensor.detach().cpu().numpy()
                mapping.append({'array':key,'layer':layer,'source_head':head,'member_heads':members})
            assert len(mapping)==32*(32 if sel.per_head else 3),len(mapping)
            for array in arrays.values():
                if not np.isfinite(array).all(): raise RuntimeError('Non-finite selector statistics')
                ends=np.minimum((np.arange(len(array))+1)*128,row['input_tokens'])
                if sel.budget is not None:
                    if not np.array_equal(array[:,1],np.minimum(ends,sel.budget)):
                        raise RuntimeError('Fixed final budget mismatch')
                else:
                    if not (array[:,2]>=.99-1e-4).all(): raise RuntimeError('Top-p target mass below threshold')
                    if not (array[:,0]>=np.minimum(ends,1024)).all(): raise RuntimeError('Top-p floor mismatch')
                    if not (array[:,1]>=array[:,0]).all(): raise RuntimeError('Protection truncated Top-p target')
                if not (array[:,1]<=ends).all(): raise RuntimeError('Too many legal final keys')
            max_mass_error=max(float(np.abs(a[:,4]-1).max()) for a in arrays.values())
            if max_mass_error>1e-4: raise RuntimeError(f'P normalization failed: {max_mass_error}')
            stem=f"{len(recorder.rows)-1:04d}_{row['input_ids_sha256'][:12]}"
            np.savez_compressed(artifacts/f'{stem}.npz',**arrays)
            (artifacts/f'{stem}.json').write_text(json.dumps({'input_ids_sha256':row['input_ids_sha256'],
                'input_tokens':row['input_tokens'],'ranges':sel.ranges,'mapping':mapping,
                'fields':['target_count','final_count','target_mass','final_mass','p_sum',
                          'value_legal','value_target','value_final','pair_legal','pair_target','pair_final'],
                'interpretation':'one row per query tile; replicate member_heads only for head-weighted aggregates'},indent=2))
            row['selection_stats_file']=str(artifacts/f'{stem}.npz')
            row['max_probability_sum_error']=max_mass_error
            recorder.output_path.write_text(''.join(json.dumps(r)+'\n' for r in recorder.rows))
            sel.records=[]
            return output
        self.model.generate=types.MethodType(generate,self.model)
runner.GenerationMetricsRecorder=Recorder

# Keep per-sample scores and generated strings without duplicating long prompts.
import lm_eval
original_evaluate=lm_eval.simple_evaluate
def evaluate(*args,**kwargs):
    kwargs['log_samples']=True
    result=original_evaluate(*args,**kwargs)
    for task,samples in result.get('samples',{}).items():
        result['samples'][task]=[{k:v for k,v in s.items() if k in (
            'doc_id','target','resps','filtered_resps','doc_hash','prompt_hash','target_hash',
            'get_score_one_kv_retrieval')} for s in samples]
    return result
lm_eval.simple_evaluate=evaluate

if __name__=='__main__':
    runner.main()
