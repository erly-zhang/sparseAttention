"""Window probability-mean selector with the optimized physical KV compactor.

Historical window semantics: each of the W recent post-RoPE queries scores
ALL currently surviving keys, independent softmax then arithmetic mean. There
is no historical-query causal mask in this importance estimator. Actual model
attention retains its exact causal mask. One representative is drawn per event.
Only the W decode queries immediately preceding each compaction are copied;
interval >= W ensures that none can be missing after a previous compaction.
"""
import math
import torch
from kvcompress.engine.representative_compact_optimized_cache import RepresentativeCompactOptimizedLayer

class RepresentativeWindowCompactOptimizedLayer(RepresentativeCompactOptimizedLayer):
    def __init__(self,*args,query_window=8,**kwargs):
        super().__init__(*args,query_window=1,**kwargs)
        if not isinstance(query_window,int) or not 1 <= query_window <= self.eviction_interval:
            raise ValueError('window must be positive and no larger than compaction interval')
        self.query_window=query_window;self._query_buf_len=query_window;self._q_valid=0

    def update_queries(self,query_states):
        if not self.cumulative_length:return
        if query_states.shape[2]!=1:raise ValueError('single-token decode required')
        remaining=self.capacity-self._cached_tokens
        if remaining>self.query_window:return
        if remaining<1:raise RuntimeError('query update after full cache')
        slot=self.query_window-remaining
        if slot==0:self._q_valid=0
        if self._q_valid!=slot:raise RuntimeError('missing queries in compression window')
        if not hasattr(self,'queries'):
            b,h,_,d=query_states.shape
            self.queries=query_states.new_empty((b,h,self.query_window,d))
        self.queries[:,:,slot:slot+1].copy_(query_states)
        self._q_valid=slot+1

    def select_indices(self):
        if self._q_valid!=self.query_window:raise RuntimeError('incomplete query window')
        p=self.prompt_len;n=self._cached_tokens;r=self.recent_window
        count=self.reasoning_budget-r;hq=self.queries.shape[1];hkv=self.k_cache.shape[2]
        if hq%hkv:raise ValueError('GQA requires Q-head count divisible by KV-head count')
        rep=int(torch.randint(hq,(1,),device=self.device,generator=self._rep_gen).item())
        kv=rep//(hq//hkv)
        logits=self.queries[:,rep]@self.k_cache[:,:n,kv].transpose(1,2)
        logits=logits/(math.sqrt(self.k_cache.shape[-1])*self.temperature)
        scores=torch.softmax(logits,dim=-1,dtype=torch.float32).mean(dim=1)
        if not bool(torch.isfinite(scores).all()):raise RuntimeError('non-finite representative scores')
        order=torch.argsort(scores[:,p:n-r],dim=-1,descending=True,stable=True)[:,:count]
        chosen=(order+p).sort(dim=-1).values;b=chosen.shape[0]
        prefix=torch.arange(p,device=self.device).expand(b,-1)
        recent=torch.arange(n-r,n,device=self.device).expand(b,-1)
        self._rep_head=rep;self._rep_kv=kv
        if self.track_compact_positions:self._last_scores=scores.clone()
        return torch.cat((prefix,chosen,recent),dim=1)
