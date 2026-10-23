"""Opt-in v3: assemble chronological IDs and gather KV in one CUDA launch.

The v2 BF16 matmul, stable ranking, representative and residual RNG are reused.
The attention implementation and physical cache policies are unchanged.
"""
import torch
import triton
import triton.language as tl

from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.w1_interval1000_cache import HYBRID_MODE
from kvcompress.engine.w1_qk_fast_cache import (
    FastFullW1Layer, FastHybridW1Layer, exact_qk_topk, install as install_v2)


@triton.jit(do_not_specialize=['COUNT', 'START', 'END', 'EXTRA', 'CS', 'IS',
                             'KS', 'VS', 'TS'])
def _assemble_gather(CHOSEN, IDS, K, V, TK, TV, LENGTHS,
                     COUNT, START, END, EXTRA, CS, IS, KS, VS, TS,
                     HD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT*HD
    slot, inner = i//HD, i % HD
    middle = (slot >= START) & (slot < START+EXTRA) & valid
    selected = tl.load(CHOSEN+row*CS+slot-START, middle, other=0)+START
    position = tl.where(slot < START, slot,
                        tl.where(slot < START+EXTRA, selected,
                                 END+slot-START-EXTRA))
    tl.store(IDS+row*IS+slot, position, valid & (inner == 0))
    kval = tl.load(K+row*KS+position*HD+inner, valid, other=0)
    vval = tl.load(V+row*VS+position*HD+inner, valid, other=0)
    tl.store(TK+row*TS+i, kval, valid)
    tl.store(TV+row*TS+i, vval, valid)
    tl.store(LENGTHS+row, COUNT, tl.program_id(0) == 0)


class FusedQKSelection:
    """Same v2 selection, with the subsequent gather performed during assembly."""
    def select_indices(self):
        if self.device.type != 'cuda':
            return super().select_indices()
        n, b = self._lengths[0], len(self._lengths)
        if not all(length == n for length in self._lengths):
            raise ValueError('equal live lengths required')
        hq, hkv = self._num_query_heads, self.k_cache.shape[2]
        if hq % hkv or self._q_valid != 1:
            raise ValueError('invalid GQA geometry or missing current query')
        step = self.cumulative_length-self.prompt_len
        self._representative_rng.manual_seed(_stable_hash(
            0x47524550, self.rep_seed, self.sample_id or 0, self.layer_idx, step))
        rep = int(torch.randint(hq, (1,), generator=self._representative_rng))
        self._rep_head, self._rep_kv = rep, rep//(hq//hkv)
        count = min(self.token_budget, n)
        if self._ids is None or self._ids.shape != (b, count):
            self._ids = torch.empty((b, count), device=self.device, dtype=torch.long)
        if n > self.token_budget or self.trace:
            # Preserve the exact v2 matmul, including all protected keys.
            scores = (self.queries[:, rep, :1] @
                      self.k_cache[:, :n, self._rep_kv].transpose(1, 2)).squeeze(1)
            if self.trace:
                self._last_scores = scores.clone()
        if n <= self.token_budget:
            start, end, extra = n, n, 0
            chosen = self._ids  # Not read: the middle interval is empty.
        else:
            start = self.prompt_len if self.protection == 'protected' else 0
            end = max(self.prompt_len, n-64) if self.protection == 'protected' else n
            extra = count-start-(n-end)
            if extra < 0:
                raise ValueError('protection exceeds total K')
            if self.rank_backend != 'stable_sort' and (
                    self._key_buffer is None or self._key_buffer.shape != (b, self.capacity)):
                dtype = torch.int32 if self.rank_backend == 'packed_qk32_topk' else torch.int64
                self._key_buffer = torch.empty((b, self.capacity), device=self.device, dtype=dtype)
            chosen = exact_qk_topk(scores, extra, start, end, self._key_buffer, self.rank_backend)
        if self._selected_lengths is None or self._selected_lengths.shape[0] != b:
            self._selected_lengths = torch.empty(b, device=self.device, dtype=torch.int32)
        hd = self.k_cache.shape[2]*self.k_cache.shape[3]
        _assemble_gather[(triton.cdiv(count*hd, 512), b)](
            chosen, self._ids, self.k_cache, self.v_cache, self._temp_k, self._temp_v,
            self._selected_lengths, count, start, end, extra, chosen.stride(0),
            self._ids.stride(0), self.k_cache.stride(0), self.v_cache.stride(0),
            self._temp_k.stride(0), hd, BLOCK=512)
        self._selected_count = count
        self._gather_ready_ids = self._ids
        counts = [count]*b
        self._last_selected_ids = self._ids
        self.selection_events += 1
        self.selection_token_sum += b*count
        self.selection_token_observations += b
        if self.trace:
            self.event_trace.append(dict(step=step, representative=rep, counts=counts,
                positions=self.slot_positions.gather(1, self._ids).cpu().tolist()))
        return self._ids, counts

    def gather(self, ids):
        if ids is getattr(self, '_gather_ready_ids', None):
            self._gather_ready_ids = None
            return
        return super().gather(ids)

    def batch_select_indices(self, indices):
        self._gather_ready_ids = None
        return super().batch_select_indices(indices)


class FusedFullW1Layer(FusedQKSelection, FastFullW1Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.implementation = 'full_w1_raw_qk_fused_gather_v3'


class FusedHybridW1Layer(FusedQKSelection, FastHybridW1Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.implementation = 'hybrid_w1_raw_qk_fused_gather_v3'


def install(rank_backend='packed_qk_topk'):
    """Explicit process-local opt-in; no existing launcher is changed."""
    if rank_backend not in ('stable_sort', 'packed_qk_topk', 'packed_qk32_topk'):
        raise ValueError(rank_backend)
    import kvcompress.engine.selector_grid_cache as grid
    install_v2(rank_backend)
    original = grid.SelectorGridLayer

    def factory(*args, **kwargs):
        mode = args[3] if len(args)>3 else kwargs.get('eviction_mode')
        if mode in ('grid_full_w1_topk_protected', 'grid_full_w1_topk_none'):
            return FusedFullW1Layer(*args, rank_backend=rank_backend, **kwargs)
        if mode == HYBRID_MODE:
            return FusedHybridW1Layer(*args, rank_backend=rank_backend, **kwargs)
        return original(*args, **kwargs)

    grid.SelectorGridLayer = factory
