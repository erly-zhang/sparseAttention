"""Temporary read-only hooks on final official kernel selection tensors."""
import importlib
import torch
from counts import flex_pairs, minference_pairs


class SparsityObserver:
    def __init__(self,method,progress=None):
        self.method=method;self.progress=progress;self.records={};self.restore=[]
        self.layer=None;self.pending=None

    def patch(self,module,name,value):
        self.restore.append((module,name,getattr(module,name)))
        setattr(module,name,value)

    def __enter__(self):
        baselines=importlib.import_module('baselines')
        old=baselines.baseline_attention
        def baseline(q,k,v,method,layer,options):
            assert method==self.method and layer not in self.records
            self.layer=layer;self.heads=[]
            result=old(q,k,v,method,layer,options)
            n=q.shape[1]
            if method=='flexprefill':
                if self.pending is None:
                    assert n<=max(2*options[method]['block_size'],options[method]['min_budget'])
                    self.heads=[{'head':h,'selected_pairs':n*(n+1)//2,'dense_fallback':True} for h in range(q.shape[2])]
                else:
                    assert len(self.pending)==1 and len(self.pending[0])==q.shape[2]
                    for h,ids in enumerate(self.pending[0]):
                        total,invalid=flex_pairs(ids,n,options[method]['block_size'])
                        total,invalid=int(total),int(invalid)
                        assert invalid==0
                        self.heads.append({'head':h,'selected_pairs':total,'selected_blocks':ids.numel()})
                self.pending=None
            assert [h['head'] for h in self.heads]==list(range(q.shape[2]))
            denominator=n*(n+1)//2
            assert all(0<h['selected_pairs']<=denominator for h in self.heads)
            self.records[layer]={'heads':self.heads,'causal_pairs_per_head':denominator,
                'mask_density':sum(h['selected_pairs'] for h in self.heads)/(denominator*len(self.heads))}
            if self.progress:self.progress(self.records)
            self.layer=None
            return result
        self.patch(baselines,'baseline_attention',baseline)
        if self.method=='flexprefill':
            flex=importlib.import_module('flex_prefill.ops.flex_prefill_attention')
            old_index=flex.get_active_blocks
            def active(*args,**kwargs):
                assert self.layer is not None and self.pending is None
                result=old_index(*args,**kwargs);self.pending=result
                return result
            self.patch(flex,'get_active_blocks',active)
        else:
            mf=importlib.import_module('minference.modules.minference_forward')
            ops=importlib.import_module('minference.ops.pit_sparse_flash_attention_v2')
            assert ops.convert_vertical_slash_indexes_opt is None, 'unexpected optimized converter; add verified counter before use'
            self.converter='minference.cuda.convert_vertical_slash_indexes'
            self.kernel='sparse_attn_func' if ops.sparse_attn_func is not None else 'triton_mixed_sparse_attention'
            old_convert=ops.convert_vertical_slash_indexes
            def convert(*args,**kwargs):
                assert self.layer is not None and self.pending is None
                result=old_convert(*args,**kwargs);self.pending=result
                return result
            self.patch(ops,'convert_vertical_slash_indexes',convert)
            old_kernel=mf.minference_prefill_kernel
            def kernel(q,k,v,head,layer,config):
                assert layer==self.layer and self.pending is None
                assert config['best_pattern'][layer][str(head)][0]=='vertical_and_slash'
                result=old_kernel(q,k,v,head,layer,config)
                assert self.pending is not None, 'expected VS converter not called'
                total,invalid,blocks,columns=minference_pairs(self.pending,q.shape[2])
                total,invalid,blocks,columns=map(int,(total,invalid,blocks,columns))
                assert invalid==0,(layer,head,'converter support invariants',invalid)
                self.heads.append({'head':head,'selected_pairs':total,'block_pairs':blocks,'column_pairs':columns})
                self.pending=None
                return result
            self.patch(mf,'minference_prefill_kernel',kernel)
        return self

    def __exit__(self,*exc):
        for module,name,old in reversed(self.restore):setattr(module,name,old)
        self.restore=[];self.pending=None
