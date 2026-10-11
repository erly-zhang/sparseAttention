"""W1 representative-probability TopK with the existing bounded KV compactor.

Only the selector differs from RandomCompactRecentLayer. One Q head is drawn
per layer/compaction from a private rep_seed/sample/layer stream, shared across
active rollouts (as in the historical representative selector). The current
post-RoPE query scores every surviving key through the correct GQA KV head.
Prompt and recent are excluded BEFORE ranking; ties prefer earlier positions.
"""
import math
import torch
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.random_compact_recent_cache import RandomCompactRecentLayer

class RepresentativeCompactRecentLayer(RandomCompactRecentLayer):
    def __init__(self,*args,query_window=1,**kwargs):
        if query_window!=1:raise ValueError('representative compact trial requires W1')
        super().__init__(*args,query_window=query_window,**kwargs)
        self.query_window=1;self._query_buf_len=1;self._q_valid=0
        if self.temperature<=0:raise ValueError('positive score temperature required')

    def update_queries(self,query_states):
        # Store only the current post-RoPE Q; do not retain the prefill tensor.
        last=query_states[:,:,-1:,:]
        if hasattr(self,'queries'):self.queries.copy_(last)
        else:self.queries=last.clone()
        self._q_valid=1

    def lazy_initialization(self,key_states,value_states=None):
        super().lazy_initialization(key_states,value_states)
        self._rep_gen=torch.Generator(device=self.device).manual_seed(
            _stable_hash(self.rep_seed,self.sample_id or 0,self.layer_idx))

    def select_indices(self):
        p=self.prompt_len;n=self._cached_tokens;r=self.recent_window
        count=self.reasoning_budget-r;hq=self.queries.shape[1];hkv=self.k_cache.shape[2]
        if hq%hkv:raise ValueError('GQA requires Q-head count divisible by KV-head count')
        rep=int(torch.randint(hq,(1,),device=self.device,generator=self._rep_gen).item())
        kv=rep//(hq//hkv)
        logits=self.queries[:,rep,:1]@self.k_cache[:,:n,kv].transpose(1,2)
        logits=logits/(math.sqrt(self.k_cache.shape[-1])*self.temperature)
        scores=torch.softmax(logits,dim=-1,dtype=torch.float32).squeeze(1)
        if not bool(torch.isfinite(scores).all()):raise RuntimeError('non-finite representative scores')
        # n-r is valid also when r=0. Stable sorting provides exact TopK with
        # original chronological position as deterministic tie-break.
        eligible=scores[:,p:n-r]
        order=torch.argsort(eligible,dim=-1,descending=True,stable=True)[:,:count]
        chosen=(order+p).sort(dim=-1).values
        b=chosen.shape[0]
        prefix=torch.arange(p,device=self.device).expand(b,-1)
        recent=torch.arange(n-r,n,device=self.device).expand(b,-1)
        self._rep_head=rep;self._rep_kv=kv
        if self.track_compact_positions:self._last_scores=scores.clone()
        return torch.cat((prefix,chosen,recent),dim=1)
