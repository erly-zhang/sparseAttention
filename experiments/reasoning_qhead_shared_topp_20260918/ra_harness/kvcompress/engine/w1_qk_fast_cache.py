"""Opt-in exact raw-BF16-QK selection optimizations for full and hybrid W1.

The historical layers stay intact. Packed keys encode the numerical BF16 order
and earlier-slot tie priority, so TopK selects exactly the stable-sort mask.
The matmul, representative stream, attention and hybrid random stream are intact.
"""
import torch
import triton
import triton.language as tl

from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer
from kvcompress.engine.w1_interval1000_cache import (
    HYBRID_MODE, W1Hybrid1000Layer, uniform_residual_ids, install as install_interval)
from kvcompress.engine.random_recent_shared_mask_cache import fused_gather_kv


@triton.jit(do_not_specialize=['N', 'START', 'SS', 'OS'])
def _pack_qk(SCORES, OUT, N, START, SS, OS, IB: tl.constexpr, PACK32: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(1)
    j = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    score = tl.load(SCORES+row*SS+START+j, j < N, other=0)
    bits = score.to(tl.uint16, bitcast=True).to(tl.int32)
    order = tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
    # CUDA's large BF16 stable sort is a CUB radix sort: preserve NaN sign
    # and payload ordering too. Use bits for zero to avoid FP32 subnormal FTZ.
    order = tl.where((bits & 0x7FFF) == 0, 0x8000, order)
    if PACK32:
        # BF16 order uses16 bits; <=65536 positions need <=16 more. Center
        # the rank to fit signed int32 without changing the numerical order.
        key = ((order-0x8000) << IB) | ((1 << IB)-1-j)
    else:
        key = (order.to(tl.int64) << 32) | (0xFFFFFFFF-j.to(tl.int64))
    tl.store(OUT+row*OS+j, key, j < N)


def packed_qk_keys(scores):
    """CPU-readable reference encoding, also used by CPU selection tests."""
    if scores.dtype != torch.bfloat16:
        raise ValueError('exact packed QK requires BF16 scores')
    bits = scores.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    order = torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
    order = torch.where((bits & 0x7FFF) == 0, 0x8000, order)
    if scores.device.type != 'cuda':
        # CPU uses a comparison sort, whose NaNs all tie above finite scores.
        order = torch.where((bits & 0x7FFF) > 0x7F80, 0xFFFF, order)
    j = torch.arange(scores.shape[1], device=scores.device, dtype=torch.int64)
    return (order << 32) | (0xFFFFFFFF-j)


def exact_qk_topk(scores, count, start=0, end=None, keys=None, backend='packed_qk_topk'):
    end = scores.shape[1] if end is None else end
    n = end-start
    if not 0 <= count <= n:
        raise ValueError('invalid TopK budget')
    if not count:
        return scores.new_empty((scores.shape[0], 0), dtype=torch.long)
    if backend == 'stable_sort':
        ids = scores[:, start:end].argsort(dim=1, descending=True, stable=True)[:, :count]
    elif backend in ('packed_qk_topk', 'packed_qk32_topk'):
        if scores.dtype != torch.bfloat16:
            raise ValueError('exact packed QK requires BF16 scores')
        if scores.device.type == 'cuda' and (n < 4096 or (backend == 'packed_qk32_topk' and n > 65536)):
            # Torch may use a comparison-based small sort. Preserve its NaN
            # semantics and avoid a packing launch where full sort is cheap.
            ids = scores[:, start:end].argsort(dim=1, descending=True, stable=True)[:, :count]
            return ids.sort(dim=1).values
        if scores.device.type == 'cuda':
            packed32 = backend == 'packed_qk32_topk'
            dtype = torch.int32 if packed32 else torch.int64
            if keys is None:
                keys = scores.new_empty((scores.shape[0], n), dtype=dtype)
            if keys.dtype != dtype:
                raise ValueError('key buffer dtype does not match backend')
            _pack_qk[(triton.cdiv(n, 256), scores.shape[0])](
                scores, keys, n, start, scores.stride(0), keys.stride(0),
                max(1, (n-1).bit_length()), packed32, BLOCK=256)
            ranking = keys[:, :n]
        else:
            ranking = packed_qk_keys(scores[:, start:end])
        ids = ranking.topk(count, dim=1, sorted=False).indices
    else:
        raise ValueError(backend)
    # Keep zero-based sorted middle slots; the assembly kernel adds start.
    return ids.sort(dim=1).values


@triton.jit(do_not_specialize=['COUNT', 'START', 'END', 'EXTRA', 'CS', 'OS'])
def _assemble_ids(CHOSEN, OUT, COUNT, START, END, EXTRA, CS, OS, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    j = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    mid = (j >= START) & (j < START+EXTRA) & (j < COUNT)
    selected = tl.load(CHOSEN+row*CS+j-START, mid, other=0)+START
    value = tl.where(j < START, j, tl.where(j < START+EXTRA, selected, END+j-START-EXTRA))
    tl.store(OUT+row*OS+j, value, j < COUNT)


@triton.jit(do_not_specialize=['N'])
def _pack_uniform(PRIMARY, SECONDARY, OUT, N, IB: tl.constexpr, SB: tl.constexpr,
                  BLOCK: tl.constexpr):
    j = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    primary = (tl.load(PRIMARY+j, j < N, other=0)*16777216).to(tl.int64)
    secondary = tl.load(SECONDARY+j, j < N, other=0)
    key = (primary << (SB+IB)) | (secondary << IB) | j.to(tl.int64)
    tl.store(OUT+j, key, j < N)


@triton.jit(do_not_specialize=['COUNT'])
def _exclude_core(CORE, PRIORITY, COUNT, BLOCK: tl.constexpr):
    j = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    slot = tl.load(CORE+j, j < COUNT, other=0)
    tl.store(PRIORITY+slot, -1, j < COUNT)


def fast_uniform_residual_ids(n, core, count, generator, workspace=None):
    """Same draws/keys as the old sampler; skip the redundant residual sort.

    The caller sorts the union. Core exclusion happens before global TopK.
    CPU and all-eligible branches retain the historical helper.
    """
    count = min(count, n-core.numel())
    if core.device.type != 'cuda' or count in (0, n-core.numel()):
        return uniform_residual_ids(n, core, count, generator)
    ib = max(1, (n-1).bit_length())
    sb = min(24, 63-24-ib)
    primary = torch.rand(n, device=core.device, generator=generator)
    secondary = torch.randint(2**sb, (n,), device=core.device, generator=generator, dtype=torch.long)
    priority = core.new_empty(n) if workspace is None else workspace[:n]
    _pack_uniform[(triton.cdiv(n, 256),)](primary, secondary, priority, n, ib, sb, BLOCK=256)
    _exclude_core[(triton.cdiv(core.numel(), 256),)](core, priority, core.numel(), BLOCK=256)
    return priority.topk(count, sorted=False).indices


class FastQKSelection:
    def select_indices(self):
        n = self._lengths[0]
        b = len(self._lengths)
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
        # No scoring is needed when every legal position will be selected.
        if n > self.token_budget or self.trace:
            scores = (self.queries[:, rep, :1] @ self.k_cache[:, :n, self._rep_kv].transpose(1, 2)).squeeze(1)
            if self.trace:
                self._last_scores = scores.clone()
        if n <= self.token_budget:
            self._ids.copy_(self._position_range[:n].expand(b, -1))
        else:
            start = self.prompt_len if self.protection == 'protected' else 0
            end = max(self.prompt_len, n-64) if self.protection == 'protected' else n
            extra = count-start-(n-end)
            if extra < 0:
                raise ValueError('protection exceeds total K')
            if self._key_buffer is None or self._key_buffer.shape != (b, self.capacity):
                dtype = torch.int32 if self.rank_backend == 'packed_qk32_topk' else torch.int64
                self._key_buffer = torch.empty((b, self.capacity), device=self.device, dtype=dtype)
            chosen = exact_qk_topk(scores, extra, start, end, self._key_buffer, self.rank_backend)
            if self.device.type == 'cuda':
                _assemble_ids[(triton.cdiv(count, 256), b)](
                    chosen, self._ids, count, start, end, extra, chosen.stride(0),
                    self._ids.stride(0), BLOCK=256)
            else:
                self._ids[:, :start].copy_(self._position_range[:start].expand(b, -1))
                self._ids[:, start:start+extra].copy_(chosen+start)
                self._ids[:, start+extra:].copy_(self._position_range[end:n].expand(b, -1))
        counts = [count]*b
        self._last_selected_ids = self._ids
        self.selection_events += 1
        self.selection_token_sum += b*count
        self.selection_token_observations += b
        if self.trace:
            self.event_trace.append(dict(step=step, representative=rep, counts=counts,
                positions=self.slot_positions.gather(1, self._ids).cpu().tolist()))
        return self._ids, counts


class FastFullW1Layer(FastQKSelection, FullW1OptimizedLayer):
    def __init__(self, *args, rank_backend='packed_qk_topk', **kwargs):
        if rank_backend not in ('stable_sort', 'packed_qk_topk', 'packed_qk32_topk'):
            raise ValueError(rank_backend)
        super().__init__(*args, rank_backend='stable_sort', score_mode='qk', **kwargs)
        self.rank_backend = rank_backend
        self.implementation = 'full_w1_raw_qk_fast_v2'


class FastHybridW1Layer(FastQKSelection, W1Hybrid1000Layer):
    def __init__(self, *args, rank_backend='packed_qk_topk', **kwargs):
        if rank_backend not in ('stable_sort', 'packed_qk_topk', 'packed_qk32_topk'):
            raise ValueError(rank_backend)
        super().__init__(*args, **kwargs)
        self.rank_backend = rank_backend
        self.implementation = 'hybrid_w1_raw_qk_fast_v2'
        self._random_priority_buffer = None

    def compact_candidate_pool(self, core_ids):
        n = self._lengths[0]
        if n <= self.candidate_budget:
            return
        if self._random_priority_buffer is None:
            self._random_priority_buffer = core_ids.new_empty(self.capacity)
        rows = []
        step = self.cumulative_length-self.prompt_len
        for row, rollout in enumerate(self._batch_ids):
            if rollout not in self._generators:
                self._generators[rollout] = torch.Generator(device=self.device)
            rng = self._generators[rollout]
            rng.manual_seed(_stable_hash(0x48595253, self.mask_seed, self.sample_id or 0,
                                       self.layer_idx, step, rollout))
            residual = fast_uniform_residual_ids(n, core_ids[row], self.extra_budget,
                                                rng, self._random_priority_buffer)
            rows.append(torch.cat((core_ids[row], residual)).sort().values)
        ids = torch.stack(rows)
        if self.device.type == 'cuda':
            fused_gather_kv(self.k_cache, self.v_cache, ids, self._compact_k, self._compact_v)
        else:
            ix = ids[:, :, None, None].expand(-1, -1, *self.k_cache.shape[2:])
            self._compact_k[:, :ids.shape[1]].copy_(self.k_cache.gather(1, ix))
            self._compact_v[:, :ids.shape[1]].copy_(self.v_cache.gather(1, ix))
        positions = self.slot_positions.gather(1, ids)
        self.k_cache, self._compact_k = self._compact_k, self.k_cache
        self.v_cache, self._compact_v = self._compact_v, self.v_cache
        self.slot_positions.fill_(-1)
        self.slot_positions[:, :ids.shape[1]].copy_(positions)
        self._lengths = [ids.shape[1]]*len(rows)
        self.cache_seqlens.fill_(ids.shape[1])
        self.compaction_events += 1
        self.compaction_core_token_sum += core_ids.numel()
        self.compaction_residual_token_sum += len(rows)*self.extra_budget
        self.compaction_token_observations += len(rows)
        if self.trace:
            self.last_compaction_ids = ids.clone()


def install(rank_backend='packed_qk_topk'):
    """Enable just full raw-QK and hybrid1000; leave historical files unchanged."""
    if rank_backend not in ('stable_sort', 'packed_qk_topk', 'packed_qk32_topk'):
        raise ValueError(rank_backend)
    import kvcompress.engine.selector_grid_cache as grid
    install_interval()
    original = grid.SelectorGridLayer

    def factory(*args, **kwargs):
        mode = args[3] if len(args)>3 else kwargs.get('eviction_mode')
        if mode in ('grid_full_w1_topk_protected', 'grid_full_w1_topk_none'):
            return FastFullW1Layer(*args, rank_backend=rank_backend, **kwargs)
        if mode == HYBRID_MODE:
            return FastHybridW1Layer(*args, rank_backend=rank_backend, **kwargs)
        return original(*args, **kwargs)

    grid.SelectorGridLayer = factory
