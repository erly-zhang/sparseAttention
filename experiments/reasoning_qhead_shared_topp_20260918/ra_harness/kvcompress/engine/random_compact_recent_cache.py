"""Prompt-exempt fixed reasoning budget with permanent, periodic shared compaction.

Attention first consumes all surviving KV plus the new key. At B+interval
reasoning slots, retain B slots (recent included). Never recover evicted KV.
"""
import torch
from kvcompress.engine.cache_utils import EvictLayer, _stable_hash
from kvcompress.engine.random_recent_shared_mask_cache import gpu_uniform_positions, fused_gather_kv


class RandomCompactRecentLayer(EvictLayer):
    records_attention_metrics = True
    uses_temporary_mask = False

    def __init__(self, *args, reasoning_budget, recent_window=64, mask_seed=42,
                 random_keep_ratio=None, track_positions=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(reasoning_budget, int) or reasoning_budget < 1:
            raise ValueError('positive reasoning_budget required')
        if not 0 <= recent_window <= reasoning_budget or self.eviction_interval < 1:
            raise ValueError('invalid recent_window or compaction interval')
        if self.token_budget != 0 or self.residual_length != 0 or random_keep_ratio is not None:
            raise ValueError('use reasoning_budget; legacy budget/residual/ratio are unused')
        self.reasoning_budget=reasoning_budget;self.recent_window=recent_window
        self.mask_seed=mask_seed;self.top_p=None;self.query_window=None
        self.prefill_pairs=0;self.max_selected_tokens=0;self._cached_tokens=0
        self._mask_generators={};self.track_compact_positions=track_positions

    def update_queries(self, query_states):
        pass  # Uniform selection consumes no Q vectors.

    def lazy_initialization(self, key_states, value_states=None):
        b,p,h,d=key_states.shape
        self.device=key_states.device;self.dtype=key_states.dtype;self.prompt_len=p
        self.capacity=p+self.reasoning_budget+self.eviction_interval
        shape=(b,self.capacity,h,d)
        self.k_cache=key_states.new_empty(shape);self.v_cache=key_states.new_empty(shape)
        # Bounded scratch for safe gather before overwriting the persistent buffers.
        self._temp_k=key_states.new_empty((b,p+self.reasoning_budget,h,d))
        self._temp_v=torch.empty_like(self._temp_k)
        self.cache_seqlens=torch.zeros(b,device=self.device,dtype=torch.int32)
        self._batch_ids=list(range(b))
        if self.track_compact_positions:
            self.compact_positions=torch.full((b,self.capacity),-1,device=self.device,dtype=torch.long)
        self.is_initialized=True

    def select_indices(self):
        p=self.prompt_len;n=self._cached_tokens;recent=self.recent_window
        older=n-p-recent;count=self.reasoning_budget-recent
        prefix=torch.arange(p,device=self.device);tail=torch.arange(n-recent,n,device=self.device)
        rows=[]
        for rollout in self._batch_ids:
            rng=self._mask_generators.get(rollout)
            if rng is None:
                rng=torch.Generator(device=self.device);self._mask_generators[rollout]=rng
            rng.manual_seed(_stable_hash(0x475055524543,self.mask_seed,self.sample_id or 0,
                                         self.cumulative_length-p,self.layer_idx,rollout))
            chosen=gpu_uniform_positions(older,count,rng,self.device)
            rows.append(torch.cat((prefix,p+chosen,tail)))
        return torch.stack(rows)

    def compact(self):
        ids=self.select_indices();b=ids.shape[0];n=ids.shape[1]
        if self.device.type=='cuda':
            fused_gather_kv(self.k_cache,self.v_cache,ids,self._temp_k[:b],self._temp_v[:b])
        else:  # CPU correctness oracle path; inference is CUDA-only.
            index=ids[:,:,None,None].expand(-1,-1,*self.k_cache.shape[2:])
            torch.gather(self.k_cache,1,index,out=self._temp_k[:b])
            torch.gather(self.v_cache,1,index,out=self._temp_v[:b])
        self.k_cache[:,:n].copy_(self._temp_k[:b]);self.v_cache[:,:n].copy_(self._temp_v[:b])
        if self.track_compact_positions:
            kept=self.compact_positions.gather(1,ids)
            self.compact_positions[:,:n].copy_(kept);self.compact_positions[:,n:].fill_(-1)
        self._last_selected_ids=ids
        self._cached_tokens=n;self.cache_seqlens.fill_(n);self.selection_events+=1

    def attention(self,query,key,value,scale):
        from flash_attn import flash_attn_with_kvcache
        first=self.cumulative_length==0;n=key.shape[1];b=key.shape[0];hq=query.shape[2]
        if not first and n!=1:raise ValueError('single-token decode required')
        if not self.is_initialized:self.lazy_initialization(key,value)
        assert self._cached_tokens+n<=self.capacity
        out=flash_attn_with_kvcache(query,self.k_cache,self.v_cache,k=key,v=value,
            cache_seqlens=self.cache_seqlens,causal=True,softmax_scale=scale,window_size=(-1,-1))
        triangular=n*(n+1)//2
        selected=b*hq*(n*self._cached_tokens+triangular)
        self.selected_pairs+=selected
        self.causal_pairs+=b*hq*(n*self.cumulative_length+triangular)
        if first:self.prefill_pairs+=selected
        if self.track_compact_positions:
            self.compact_positions[:,self._cached_tokens:self._cached_tokens+n]=torch.arange(
                self.cumulative_length,self.cumulative_length+n,device=self.device)
        self._cached_tokens+=n;self.cumulative_length+=n;self.cache_seqlens+=n
        self.max_selected_tokens=max(self.max_selected_tokens,self._cached_tokens)
        self.max_visible_tokens=max(self.max_visible_tokens,self.cumulative_length)
        if not first and self._cached_tokens==self.capacity:self.compact()
        return out

    def batch_select_indices(self,indices):
        super().batch_select_indices(indices)
        if self.track_compact_positions:self.compact_positions=self.compact_positions[indices].contiguous()
