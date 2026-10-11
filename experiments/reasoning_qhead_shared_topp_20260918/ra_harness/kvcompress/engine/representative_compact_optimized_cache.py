"""Execution-only W1 optimization; original selector and private RNG unchanged.

Save Q only when the next attention call will compact. Gather reasoning into a
second full-capacity buffer and swap buffers; the protected prompt is copied
once. Both buffers retain the exact original strides, values and RoPE layout.
"""
import torch
import triton
import triton.language as tl
from kvcompress.engine.representative_compact_recent_cache import RepresentativeCompactRecentLayer

@triton.jit(do_not_specialize=['P','KS','TS','IS'])
def _gather_reasoning(K,V,IDS,TK,TV,P,KS,TS,IS,N:tl.constexpr,HD:tl.constexpr,BLOCK:tl.constexpr):
    b=tl.program_id(1)
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<N*HD
    index=tl.load(IDS+b*IS+P+i//HD,mask=valid,other=0)
    src=b*KS+index*HD+i%HD
    dst=b*TS+P*HD+i
    kval=tl.load(K+src,mask=valid,other=0);vval=tl.load(V+src,mask=valid,other=0)
    tl.store(TK+dst,kval,mask=valid);tl.store(TV+dst,vval,mask=valid)

class RepresentativeCompactOptimizedLayer(RepresentativeCompactRecentLayer):
    def update_queries(self,query_states):
        # The model calls this immediately before attention. Preserve the exact
        # contiguous Q layout consumed by the baseline matmul, only on events.
        if self.cumulative_length and self._cached_tokens+query_states.shape[2]==self.capacity:
            super().update_queries(query_states)

    def lazy_initialization(self,key_states,value_states=None):
        super().lazy_initialization(key_states,value_states)
        # Capacity, not compressed length: this spare becomes the append buffer.
        self._temp_k=torch.empty_like(self.k_cache);self._temp_v=torch.empty_like(self.v_cache)
        self._spare_prompt_ready=False

    def compact(self):
        ids=self.select_indices();b,n=ids.shape;p=self.prompt_len
        if not self._spare_prompt_ready:
            self._temp_k[:b,:p].copy_(self.k_cache[:,:p]);self._temp_v[:b,:p].copy_(self.v_cache[:,:p])
            self._spare_prompt_ready=True
        if self.device.type=='cuda':
            hd=self.k_cache.shape[2]*self.k_cache.shape[3]
            _gather_reasoning[(triton.cdiv(self.reasoning_budget*hd,512),b)](
                self.k_cache,self.v_cache,ids,self._temp_k,self._temp_v,p,
                self.k_cache.stride(0),self._temp_k.stride(0),ids.stride(0),
                N=self.reasoning_budget,HD=hd,BLOCK=512)
        else:
            index=ids[:,p:,None,None].expand(-1,-1,*self.k_cache.shape[2:])
            self._temp_k[:b,p:n].copy_(torch.gather(self.k_cache,1,index))
            self._temp_v[:b,p:n].copy_(torch.gather(self.v_cache,1,index))
        self.k_cache,self._temp_k=self._temp_k,self.k_cache
        self.v_cache,self._temp_v=self._temp_v,self.v_cache
        if self.track_compact_positions:
            kept=self.compact_positions.gather(1,ids)
            self.compact_positions[:,:n].copy_(kept);self.compact_positions[:,n:].fill_(-1)
        self._last_selected_ids=ids
        self._cached_tokens=n;self.cache_seqlens.fill_(n);self.selection_events+=1

    def batch_select_indices(self,indices):
        if self.is_initialized:
            self._temp_k=self._temp_k[indices].contiguous();self._temp_v=self._temp_v[indices].contiguous()
        super().batch_select_indices(indices)
