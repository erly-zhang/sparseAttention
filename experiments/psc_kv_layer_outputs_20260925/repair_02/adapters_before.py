"""PSC Transformers 4.44 prefill-only adapters; original algorithm kernels."""
import sys
import types
from pathlib import Path
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
ROOT=Path(__file__).resolve().parents[2]
FLEX=Path('/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/FlexPrefill')
sys.path[:0]=[str(ROOT),str(FLEX),str(ROOT/'experiments/residual6040_20260908'),str(ROOT/'experiments/prefill_token_head_routing_20260924')]
from routing import routed_attention, repeat_project

METHODS=('dense','routing_c','shared6144_member2q4096','flexprefill','minference')


def install(model, method, groups=None):
    if method not in METHODS: raise ValueError(method)
    state={'method':method,'pairs':{},'selector':None}
    if method=='dense': return model,state
    if method in ('flexprefill','minference'):
        from experiments.benchmark_shareprefill_ae3 import install_benchmark_method
        model,patch,metadata=install_benchmark_method(model,method,
            model_name='meta-llama/Llama-3.1-8B-Instruct',flexprefill_root=FLEX,record_sparsity=False)
        state['metadata']=metadata
        return model,state
    if method=='shared6144_member2q4096':
        from selector6040 import Selector
        state['selector']=Selector(groups,'member2q')
    for layer in model.model.layers:
        attn=layer.self_attn
        def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                    output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None,**kwargs):
            b,n,_=hidden_states.shape
            if b!=1 or n<1 or (past_key_value is not None and past_key_value.get_seq_length(self.layer_idx)):
                raise ValueError('full one-sample prefill, empty cache only')
            h,hkv,d=self.num_heads,self.num_key_value_heads,self.head_dim
            q=self.q_proj(hidden_states).view(b,n,h,d).transpose(1,2)
            k=self.k_proj(hidden_states).view(b,n,hkv,d).transpose(1,2)
            v=self.v_proj(hidden_states).view(b,n,hkv,d).transpose(1,2)
            cos,sin=position_embeddings if position_embeddings is not None else self.rotary_emb(v,position_ids)
            q,k=apply_rotary_pos_emb(q,k,cos,sin)
            if past_key_value is not None:
                k,v=past_key_value.update(k,v,self.layer_idx,{'cos':cos,'sin':sin,'cache_position':cache_position})
            if method=='routing_c':
                merged,counts=routed_attention(q,k,v,position_ids,torch.ones_like(position_ids,dtype=torch.bool),
                    'q_kv',d**-0.5,backend='flash_attention_2')
                state['pairs'][self.layer_idx]=counts['eligible_pairs']
                output=repeat_project(merged,self.o_proj,h)
            else:
                from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
                selector=state['selector'];token=selector.current_layer.set(self.layer_idx)
                try:
                    qt,kt,vt=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
                    indices=selector(qt,kt,vt,128,.95,1024,None)
                    value=triton_token_compacted_prefill_attention(qt,kt,vt,indices,
                        token_chunk_size=32,softmax_scale=d**-0.5)
                    # Required selector counts retained as device tensors until diagnostics.
                    state['pairs'][self.layer_idx]=selector.stats.pending[-1][0][:,:,1].sum()
                    selector.stats.pending.clear()
                    output=self.o_proj(value.reshape(b,n,h*d))
                finally: selector.current_layer.reset(token)
            return output,None,past_key_value
        attn.forward=types.MethodType(forward,attn)
    return model,state
