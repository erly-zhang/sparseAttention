"""Isolated full-generation runner with calibrated proxy residuals."""
import hashlib
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
from residual_selector import ResidualSelector

METHODS={'probmean_shared_residual_compact10240':'fixed',
         'probmean_shared_residual_topp99_matched':'topp'}
base.runner.METHODS=tuple(base.runner.METHODS)+tuple(METHODS)
base.runner.SHAREPREFILL_METHODS=tuple(base.runner.SHAREPREFILL_METHODS)+tuple(METHODS)
RULE_PATH=Path(os.environ['RESIDUAL_POLICY'])
RULE=json.loads(RULE_PATH.read_text())


def install(model,method,**kwargs):
    kind=METHODS[method]
    assert hashlib.sha256(Path(kwargs['group_config_path']).read_bytes()).hexdigest()==RULE['group_sha256']
    old_class=base.sparse.RepresentativeTokenFirstBlockSelector
    class NewSelector(ResidualSelector):
        def __init__(self,config,**unused):
            super().__init__(config,kind=kind,rules=RULE['policies'][kind]['layers'])
    base.sparse.RepresentativeTokenFirstBlockSelector=NewSelector
    try:
        result,patch,metadata=base.old_install(model,'shareprefill_ae3_token_compact',**kwargs)
    finally:
        base.sparse.RepresentativeTokenFirstBlockSelector=old_class
    tokenizer=AutoTokenizer.from_pretrained(kwargs['model_name'])
    base.state['value_resolver']=base.build_kv_retrieval_range_resolver(tokenizer,value_only=True)
    base.state['pair_resolver']=base.build_kv_retrieval_range_resolver(tokenizer)
    config={'score':'mean_of_per_query_causal_softmax','query_tile':128,'global_tail':False,
        'representative_distributions_per_layer':3,'per_member_online_qk':False,
        'fixed_budget':10240 if kind=='fixed' else None,'target_top_p':.99 if kind=='topp' else None,
        'target_minimum':1024 if kind=='topp' else None,'sink_tokens':128,'local':'current query tile',
        'residual_budget':1024,'residual_mode':'equal-count replacement',
        'top_p_final_mass_guaranteed':False if kind=='topp' else None,
        'final_count_rule':'same as original shared mask on the same current Q/K',
        'block_projection':False,'policy_sha256':hashlib.sha256(RULE_PATH.read_bytes()).hexdigest(),
        'policy_file':str(RULE_PATH),'enabled_member_layer_pairs':RULE['policies'][kind]['enabled']}
    metadata={'implementation':method,'num_groups_per_layer':3,'model':kwargs['model_name'],
              'group_config_path':str(kwargs['group_config_path']),'online_config':config}
    # Compile probability kernels for full and partial query tiles outside timing.
    device=next(model.parameters()).device
    for n in (4096,4097):
        q=torch.randn(1,n,32,128,device=device,dtype=torch.bfloat16)
        k=torch.randn(1,n,8,128,device=device,dtype=torch.bfloat16)
        for h in (0,1,4,16): base.probability_mean(q,k,h,h//4)
    del q,k
    torch.cuda.synchronize()
    return result,patch,metadata


base.runner.install_benchmark_method=install
OriginalRecorder=base.Recorder.__bases__[0]


class Recorder(OriginalRecorder):
    def install(self):
        super().install(); inner=self.model.generate; recorder=self
        def generate(model,*args,**kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            sel=recorder.patch.selector
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
            try: result=inner(*args,**kwargs)
            finally:
                for h in handles:h.remove()
                sel.collect=False
            row=recorder.rows[-1]
            row['prefill_cuda_event_sec']=row['prefill_latency_sec']
            row['prefill_latency_sec']=stamp['end']-stamp['start']
            row['decode_latency_sec']=row['total_latency_sec']-row['prefill_latency_sec']
            row['timing_scope']='GPU-synchronized first model forward including representative scoring, residual selection, index, attention and GPU statistics'
            arrays={};mapping=[];seen=set()
            for serial,(layer,head,members,tensor) in enumerate(sel.records):
                key=f'layer{layer}_pattern{serial}'
                array=tensor.detach().cpu().numpy().astype(np.float32)
                arrays[key]=array
                mapping.append({'array':key,'layer':layer,'source_head':head,'member_heads':members})
                for member in members:
                    assert (layer,member) not in seen
                    seen.add((layer,member))
                ends=np.minimum((np.arange(len(array))+1)*128,row['input_tokens'])
                assert np.isfinite(array).all()
                assert np.array_equal(array[:,1],array[:,11]),'Same-QK final key count changed'
                assert (array[:,1]<=ends).all()
                assert (np.abs(array[:,4]-1)<1e-4).all()
                if sel.kind=='fixed':
                    assert np.array_equal(array[:,1],np.minimum(ends,10240))
                else:
                    assert (array[:,2]>=.99-1e-4).all()
                    assert (array[:,0]>=np.minimum(ends,1024)).all()
                    # Final mass can be <.99: this is deliberately a matched-count replacement.
            assert len(seen)==1024
            path=recorder.output_path.parent/'selection_stats';path.mkdir(exist_ok=True)
            stem=f"{len(recorder.rows)-1:04d}_{row['input_ids_sha256'][:12]}"
            np.savez_compressed(path/f'{stem}.npz',**arrays)
            (path/f'{stem}.json').write_text(json.dumps({'input_ids_sha256':row['input_ids_sha256'],
                'input_tokens':row['input_tokens'],'ranges':sel.ranges,'mapping':mapping,
                'fields':['target_count','final_count','target_mass','final_mass','p_sum','value_legal',
                          'value_target','value_final','pair_legal','pair_target','pair_final',
                          'original_shared_count','original_shared_mass','novel_count','dropped_count','quarter'],
                'mass_definition':'current own-group representative P; not the member probability',
                'timing':'CPU transfer and serialization excluded from generate timing'},indent=2))
            row['selection_stats_file']=str(path/f'{stem}.npz')
            row['max_probability_sum_error']=max(float(np.abs(a[:,4]-1).max()) for a in arrays.values())
            recorder.output_path.write_text(''.join(json.dumps(r)+'\n' for r in recorder.rows))
            sel.records=[]
            return result
        self.model.generate=types.MethodType(generate,self.model)


base.runner.GenerationMetricsRecorder=Recorder
if __name__=='__main__':base.runner.main()
