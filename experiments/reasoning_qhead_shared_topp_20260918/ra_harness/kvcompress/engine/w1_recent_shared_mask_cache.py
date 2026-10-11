"""Current-query Top-p union recent positions, full KV, reusable fused gather.

Top-p is computed over ALL non-prompt positions before union. Recency is not
subtracted from a fixed budget. Historical SharedMaskLayer scoring/sort/RNG are
reused verbatim; only the selected set is extended. Valid gather prefixes are
sorted and unique; padded slots are excluded by FlashAttention cache lengths.
"""
import json
import os
import torch
from kvcompress.engine.shared_mask_cache import SharedMaskLayer
from kvcompress.engine.random_recent_shared_mask_cache import RandomRecentSharedMaskLayer, fused_gather_kv


class W1RecentSharedMaskLayer(SharedMaskLayer):
    # Full-history storage optimizations do not depend on random selection.
    prepare = RandomRecentSharedMaskLayer.prepare
    _advance = RandomRecentSharedMaskLayer._advance

    def __init__(self, *args, recent_window=64, **kwargs):
        super().__init__(*args, **kwargs)
        if self.query_window != 1 or recent_window < 0:
            raise ValueError('W1 recent requires query_window=1 and recent_window>=0')
        self.recent_window = recent_window
        self._gather_capacity = 0
        self._gather_batch_capacity = 0
        self._temp_k = self._temp_v = None
        self.gather_allocations = 0
        self.topp_core_selected_pairs = 0
        self.recent_added_pairs = 0

    def select_indices(self):
        core = super().select_indices()
        n = self.cumulative_length
        start = max(self.prompt_len, n-self.recent_window)
        tail = torch.arange(start, n, device=self.device)
        # Core indices are already sorted. Removing overlap before concatenating
        # preserves order/uniqueness without a second probability sort.
        final = [torch.cat((ids[ids < start], tail)) for ids in core]
        self._last_topp_ids = core
        self._last_selected_ids = final
        if os.environ.get('QHEAD_DEBUG') == '1':
            for old, ids in zip(core, final):
                assert bool(torch.isin(old, ids).all())
                assert bool((ids[1:] > ids[:-1]).all())
                assert ids.numel() <= n
        if os.environ.get('QHEAD_LOG'):
            with open(os.environ['QHEAD_LOG'], 'a') as stream:
                for b, (old, ids) in enumerate(zip(core, final)):
                    stream.write(json.dumps(dict(implementation='w1_topp_union_recent',
                        layer=self.layer_idx, step=n, batch=b, sample_id=self.sample_id,
                        recent_window=self.recent_window, topp_core_tokens=old.numel(),
                        recent_added_tokens=ids.numel()-old.numel(), selected_tokens=ids.numel()))+'\n')
        return final

    def attention(self, query, key, value, scale):
        from flash_attn import flash_attn_with_kvcache
        n_new = key.shape[1]
        first = self.cumulative_length == 0
        if not first and n_new != 1:
            raise ValueError('single-token decode required')
        kc, vc, lengths = self.prepare(key, value)
        b, hq = key.shape[0], query.shape[2]
        if first:
            self.prompt_len = n_new
            out = flash_attn_with_kvcache(query, kc, vc, k=key, v=value,
                cache_seqlens=lengths, causal=True, softmax_scale=scale, window_size=(-1,-1))
            self._advance(n_new)
            pairs = b*hq*n_new*(n_new+1)//2
            self.selected_pairs += pairs
            self.causal_pairs += pairs
            self.prefill_pairs += pairs
            self.max_selected_tokens = n_new
            return out
        offset = self.cumulative_length
        kc[:, offset:offset+1].copy_(key)
        vc[:, offset:offset+1].copy_(value)
        self._advance(1)
        chosen = self.select_indices()
        counts = [ids.numel() for ids in chosen]
        extent = max(counts)
        # Batch rows can select different counts; pad only AFTER the valid
        # prefix, never sort padding into it. FlashAttention ignores the suffix.
        ids = torch.zeros((b, extent), device=self.device, dtype=torch.long)
        for row, positions in enumerate(chosen):
            ids[row, :positions.numel()].copy_(positions)
        if self._temp_k is None or extent > self._gather_capacity or b > self._gather_batch_capacity:
            self._gather_capacity = max(extent, 2*self._gather_capacity, 256)
            self._gather_batch_capacity = b
            shape = (b, self._gather_capacity, kc.shape[2], kc.shape[3])
            self._temp_k, self._temp_v = kc.new_empty(shape), vc.new_empty(shape)
            self.gather_allocations += 1
        tk, tv = self._temp_k[:b], self._temp_v[:b]
        fused_gather_kv(kc, vc, ids, tk, tv)
        selected_lengths = torch.tensor(counts, device=self.device, dtype=torch.int32)
        out = flash_attn_with_kvcache(query, tk, tv, cache_seqlens=selected_lengths,
            causal=True, softmax_scale=scale, window_size=(-1,-1))
        core_count = sum(ids.numel() for ids in self._last_topp_ids)
        self.topp_core_selected_pairs += hq*core_count
        self.recent_added_pairs += hq*(sum(counts)-core_count)
        self.selected_pairs += hq*sum(counts)
        self.causal_pairs += hq*b*self.cumulative_length
        self.max_selected_tokens = max(self.max_selected_tokens, extent)
        return out
