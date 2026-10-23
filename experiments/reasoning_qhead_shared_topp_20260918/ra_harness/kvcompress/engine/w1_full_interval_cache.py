"""Opt-in Full-KV raw-QK W1 whose shared token mask is refreshed every 64 steps.

This deliberately changes the per-step Full-KV W1 algorithm, following the
official random_pp cadence. All KV is retained. At decode steps 1, 65, 129, ...
the current post-RoPE representative query ranks the complete history exactly
as the v2 Full-KV layer: total K including prompt + recent64, raw BF16 QK,
earlier-position ties, chronological IDs, same private representative stream.
Until the next refresh, each newly generated KV is appended to that attention
set, so attention covers K..K+63 positions (random_pp K1024 covers 1024..1087).
Nothing is evicted: every refresh can recover any earlier token.
"""
import torch
import triton

from kvcompress.engine.selector_grid_cache import SelectorGridLayer
from kvcompress.engine.full_w1_optimized_cache import _append_full
from kvcompress.engine.w1_interval1000_cache import _append_bounded
from kvcompress.engine.w1_qk_fast_cache import FastFullW1Layer

MODE = 'grid_full64_w1_topk_protected'
BASE_MODE = 'grid_full_w1_topk_protected'
INTERVAL = 64


class FullW1Interval64Layer(FastFullW1Layer):
    def __init__(self, *args, rank_backend='stable_sort', **kwargs):
        args, kwargs = list(args), dict(kwargs)
        actual = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        if actual != MODE or kwargs.get('eviction_interval', INTERVAL) != INTERVAL:
            raise ValueError('full64 requires its mode and eviction_interval=64')
        if len(args) > 3:
            args[3] = BASE_MODE
        else:
            kwargs['eviction_mode'] = BASE_MODE
        kwargs['eviction_interval'] = 1
        super().__init__(*args, rank_backend=rank_backend, **kwargs)
        self.MODE = actual
        self.mask_interval = self.eviction_interval = INTERVAL
        self.implementation = 'full_w1_raw_qk_interval64_v1'
        self.compaction_events = 0
        self._mask_step = None
        self._attend_count = 0
        self._attend_lengths = None

    def lazy_initialization(self, k, value_states=None):
        super().lazy_initialization(k, value_states)
        # Attention scratch holds one refresh (<=K) plus up to63 appended tokens.
        b, _, h, d = k.shape
        scratch = min(self.token_budget+self.mask_interval, self.capacity)
        self._temp_k = k.new_empty((b, scratch, h, d))
        self._temp_v = torch.empty_like(self._temp_k)
        self._temp_positions = torch.full((b, scratch), -1, device=k.device, dtype=torch.long)

    def attention(self, q, k, v, scale):
        from flash_attn import flash_attn_with_kvcache
        if not self.cumulative_length:
            # Dense prefill, identical to every other grid method.
            return SelectorGridLayer.attention(self, q, k, v, scale)
        b, new, hkv, d = k.shape
        if new != 1 or not k.is_contiguous() or not v.is_contiguous():
            raise ValueError('expected contiguous single-token decode KV')
        old = self.cumulative_length
        if old+1 > self.capacity or not all(n == old for n in self._lengths):
            raise RuntimeError('full cache capacity/length contract violated')
        cuda = self.device.type == 'cuda'
        # 1. Retain the new KV in the complete history.
        if cuda:
            _append_full[(triton.cdiv(hkv*d, 256), b)](
                k, v, self.k_cache, self.v_cache, self.slot_positions, self.cache_seqlens,
                old, hkv*d, k.stride(0), v.stride(0), self.k_cache.stride(0),
                self.slot_positions.stride(0), BLOCK=256)
        else:
            self.k_cache[:, old:old+1].copy_(k)
            self.v_cache[:, old:old+1].copy_(v)
            self.slot_positions[:, old] = old
            self.cache_seqlens.fill_(old+1)
        self.cumulative_length = old+1
        self._lengths = [old+1]*b
        self.max_visible_tokens = self.cumulative_length
        step = self.cumulative_length-self.prompt_len
        if self._mask_step is None or (step-1) % self.mask_interval == 0:
            # 2a. Refresh: identical selection to per-step Full-KV W1 at this step.
            ids, counts = self.select_indices()
            count = counts[0]
            self.gather(ids)
            self._temp_positions[:, :count].copy_(ids)
            if self._attend_lengths is None or self._attend_lengths.shape[0] != b:
                self._attend_lengths = torch.empty(b, device=self.device, dtype=torch.int32)
            self._attend_lengths.fill_(count)
            self._mask_step = step
        else:
            # 2b. Reuse the previous mask; append only the current token.
            count = self._attend_count
            if self._attend_lengths is None or self._attend_lengths.shape[0] != b:
                self._attend_lengths = torch.full((b,), count, device=self.device, dtype=torch.int32)
            if cuda:
                _append_bounded[(triton.cdiv(hkv*d, 256), b)](
                    k, v, self._temp_k, self._temp_v, self._temp_positions, self._attend_lengths,
                    count, old, hkv*d, k.stride(0), v.stride(0), self._temp_k.stride(0),
                    self._temp_positions.stride(0), BLOCK=256)
            else:
                self._temp_k[:, count:count+1].copy_(k)
                self._temp_v[:, count:count+1].copy_(v)
                self._temp_positions[:, count] = old
                self._attend_lengths.fill_(count+1)
            count += 1
        self._attend_count = count
        out = flash_attn_with_kvcache(q, self._temp_k, self._temp_v,
                                     cache_seqlens=self._attend_lengths,
                                     causal=True, softmax_scale=scale)
        hq = q.shape[2]
        self.selected_pairs += hq*b*count
        self.causal_pairs += hq*b*self.cumulative_length
        self.max_selected_tokens = max(self.max_selected_tokens, count)
        self.attention_token_sum += b*count
        self.attention_token_observations += b
        return out

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self._temp_positions = self._temp_positions[indices].contiguous()
        self._attend_lengths = None


def install(rank_backend='stable_sort'):
    """Process-local opt-in; all historical modes keep their factory."""
    import kvcompress.engine.selector_grid_cache as grid
    from kvcompress.engine.w1_qk_fast_cache import install as install_fast
    install_fast(rank_backend)
    original_decode, original_factory = grid.decode_mode, grid.SelectorGridLayer

    def decode(mode):
        if mode == MODE:
            return 'full', 'w1', 'topk', 'protected'
        return original_decode(mode)

    def factory(*args, **kwargs):
        mode = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        if mode == MODE:
            return FullW1Interval64Layer(*args, rank_backend=rank_backend, **kwargs)
        return original_factory(*args, **kwargs)

    grid.decode_mode, grid.SelectorGridLayer = decode, factory
