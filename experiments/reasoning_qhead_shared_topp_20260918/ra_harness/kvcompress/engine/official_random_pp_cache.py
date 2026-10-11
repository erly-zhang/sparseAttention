"""Unmodified upstream random_pp algorithm under the common evaluation protocol.

Only adapters: independent mask RNG (generation stream remains matched) and
actual attention pair counters. Upstream trigger/selection/sort/compaction are
called verbatim. RNG context overhead is INCLUDED in measured decode time.
"""
import torch
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.official_random_attention.cache_utils import EvictLayer as UpstreamLayer


class OfficialRandomPPLayer(UpstreamLayer):
    records_attention_metrics = True
    uses_temporary_mask = False

    def __init__(self, layer_idx, token_budget, residual_length, eviction_mode,
                 rkv_lambda, smooth, verbose, sample_id=None, mask_seed=42,
                 top_p=.95, query_window=64, eviction_interval=1, rep_seed=0, **kwargs):
        if eviction_mode!='official_random_pp' or token_budget<=residual_length or residual_length<=0:
            raise ValueError('invalid official random_pp configuration')
        super().__init__(layer_idx,token_budget,residual_length,'random_pp',rkv_lambda,smooth,verbose,**kwargs)
        self.sample_id=sample_id;self.mask_seed=mask_seed;self.query_window=None
        self.selected_pairs=self.causal_pairs=self.prefill_pairs=self.selection_events=0
        self.max_visible_tokens=self.max_selected_tokens=0
        self._cached_tokens=0;self._mask_gen=None

    def _run_eviction(self,cur_len):
        if self._mask_gen is None:
            self._mask_gen=torch.Generator(device=self.device).manual_seed(
                _stable_hash(self.mask_seed,self.sample_id or 0,self.layer_idx))
        # Run the byte-identical upstream implementation with a dedicated RNG
        # stream. Restoring global state preserves the runner's generation RNG.
        cuda=self.device.type=='cuda'
        devices=[self.device.index if self.device.index is not None else torch.cuda.current_device()] if cuda else []
        with torch.random.fork_rng(devices=devices):
            if cuda:torch.cuda.set_rng_state(self._mask_gen.get_state(),self.device)
            else:torch.set_rng_state(self._mask_gen.get_state())
            super()._run_eviction(cur_len)
            self._mask_gen.set_state(torch.cuda.get_rng_state(self.device) if cuda else torch.get_rng_state())
        self._cached_tokens=self.token_budget
        self.selection_events+=1

    def attention(self,query,key,value,scale):
        from flash_attn import flash_attn_with_kvcache
        n=key.shape[1];b=key.shape[0];hq=query.shape[2]
        first=self.cumulative_length==0
        if not first and n!=1:raise ValueError('single-token decode required')
        kc,vc,lengths=self.prepare(key,value)
        # Count pairs actually consumed BEFORE the upstream post-attention
        # eviction. Every compact key is in the current query's causal past.
        triangular=n*(n+1)//2
        selected=b*hq*(n*self._cached_tokens+triangular)
        causal=b*hq*(n*self.cumulative_length+triangular)
        self.selected_pairs+=selected;self.causal_pairs+=causal
        if first:self.prefill_pairs+=selected
        self.max_selected_tokens=max(self.max_selected_tokens,self._cached_tokens+n)
        self.max_visible_tokens=max(self.max_visible_tokens,self.cumulative_length+n)
        self._cached_tokens+=n
        out=flash_attn_with_kvcache(query,kc,vc,k=key,v=value,cache_seqlens=lengths,
            causal=True,softmax_scale=scale,window_size=(-1,-1))
        # Original implementation remembers prompt for prefill>1. Formal inputs
        # are32..814tokens; do not change its treatment of single-token prefill.
        self.post_attention_update(n)
        return out
