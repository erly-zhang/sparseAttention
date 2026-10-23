"""Opt-in hybrid W1 v6: one Triton launch per decode layer, launched without JIT dispatch.

Bitwise identical to v4/v5 for every hybrid variant. Per decode layer:
  cuBLAS score matmul (unchanged, only when the pool exceeds K)
  ONE Triton kernel: exact chronological TopK + append new KV + assemble IDs + gather K/V
  FlashAttention (unchanged inputs).
Every program of a row recomputes the identical radix selection over the (<=8192)
ranked keys, scatters only its own ranks into a scratch row, then gathers its own
contiguous slice of output slots; the new token is read from the input K/V.
All non-constexpr kernel arguments are unspecialized, so the compiled kernel is valid
for every argument value; after the first (normal, compiling) launch the cached
CompiledKernel is called directly, skipping Triton's per-call Python dispatch.
"""
import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_with_kvcache

from kvcompress.engine.w1_hybrid_gpu_select_cache import SELECT_BLOCK
from kvcompress.engine.w1_hybrid_v5_cache import HybridV5Layer
from kvcompress.engine.w1_hybrid_variants_cache import VARIANTS

PER_PROGRAM = 64
TOKENS = 8
ARGS = ('SCORES', 'KNEW', 'VNEW', 'CHOSEN', 'IDS', 'K', 'V', 'TK', 'TV', 'LENGTHS', 'POS', 'SEQLENS',
        'COUNT', 'START', 'END', 'EXTRA', 'N_MID', 'NEW_SLOT', 'ABSOLUTE',
        'SS', 'CS', 'IS', 'KS', 'VS', 'TS', 'NS', 'PS')


@triton.jit(do_not_specialize=list(ARGS))
def _select_append_gather(SCORES, KNEW, VNEW, CHOSEN, IDS, K, V, TK, TV, LENGTHS, POS, SEQLENS,
                          COUNT, START, END, EXTRA, N_MID, NEW_SLOT, ABSOLUTE,
                          SS, CS, IS, KS, VS, TS, NS, PS,
                          HD: tl.constexpr, HDP: tl.constexpr, PER: tl.constexpr, TOK: tl.constexpr,
                          BLOCK: tl.constexpr):
    row = tl.program_id(1)
    pid = tl.program_id(0)
    # HDP = next power of two >= HD (e.g. Phi-4: 10 KV heads x 128 = 1280 -> 2048).
    inner = tl.arange(0, HDP)
    hmask = inner < HD
    if pid == 0:
        # Append the new token to the pool and update per-row lengths.
        tl.store(K+row*KS+NEW_SLOT*HD+inner, tl.load(KNEW+row*NS+inner, hmask), hmask)
        tl.store(V+row*VS+NEW_SLOT*HD+inner, tl.load(VNEW+row*NS+inner, hmask), hmask)
        tl.store(POS+row*PS+NEW_SLOT, ABSOLUTE)
        tl.store(SEQLENS+row, NEW_SLOT+1)
        tl.store(LENGTHS+row, COUNT)
    s0 = pid*PER
    s1 = tl.minimum(s0+PER, COUNT)
    r0 = tl.maximum(s0-START, 0)
    r1 = tl.minimum(s1-START, EXTRA)
    if r1 > r0:
        # Identical keys and radix select to _select_chronological.
        j = tl.arange(0, BLOCK)
        valid = j < N_MID
        score = tl.load(SCORES+row*SS+START+j, valid, other=0)
        bits = score.to(tl.uint16, bitcast=True).to(tl.int64)
        order = tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
        order = tl.where((bits & 0x7FFF) == 0, 0x8000, order)
        key = tl.where(valid, (order << 16) | (0xFFFF-j.to(tl.int64)), -1)
        threshold = tl.sum(tl.zeros([BLOCK], dtype=tl.int64), axis=0)
        for bit in tl.static_range(31, -1, -1):
            candidate = threshold | (1 << bit)
            hits = tl.sum((key >= candidate).to(tl.int32), axis=0)
            threshold = tl.where(hits >= EXTRA, candidate, threshold)
        chosen = key >= threshold
        rank = tl.cumsum(chosen.to(tl.int32), axis=0)-1
        mine = chosen & (rank >= r0) & (rank < r1)
        tl.store(CHOSEN+row*CS+rank, j.to(tl.int64), mine)
        tl.debug_barrier()
    for t in range(s0, s1, TOK):
        slots = t+tl.arange(0, TOK)
        live = slots < s1
        middle = (slots >= START) & (slots < START+EXTRA) & live
        selected = tl.load(CHOSEN+row*CS+slots-START, middle, other=0)+START
        position = tl.where(slots < START, slots,
                            tl.where(slots < START+EXTRA, selected, END+slots-START-EXTRA))
        tl.store(IDS+row*IS+slots, position, live)
        is_new = (position == NEW_SLOT)[:, None]
        mask = live[:, None] & hmask[None, :]
        offsets = position[:, None]*HD+inner[None, :]
        out = slots[:, None]*HD+inner[None, :]
        kval = tl.load(K+row*KS+offsets, mask & (~is_new), other=0)
        vval = tl.load(V+row*VS+offsets, mask & (~is_new), other=0)
        kval = tl.where(is_new, tl.load(KNEW+row*NS+inner, hmask)[None, :], kval)
        vval = tl.where(is_new, tl.load(VNEW+row*NS+inner, hmask)[None, :], vval)
        tl.store(TK+row*TS+out, kval, mask)
        tl.store(TV+row*TS+out, vval, mask)


class DirectLaunch:
    """First call compiles via the JIT; later calls invoke the CompiledKernel directly."""
    def __init__(self, fn, num_warps, **constexprs):
        self.fn, self.num_warps, self.constexprs = fn, num_warps, constexprs
        self.kernel = None
        self.stream = None

    def __call__(self, grid, *args):
        kernel = self.kernel
        if kernel is None:
            self.kernel = self.fn[grid](*args, num_warps=self.num_warps, **self.constexprs)
            self.stream = torch.cuda.current_stream().cuda_stream
            return
        if torch.cuda.current_stream().cuda_stream != self.stream:
            raise RuntimeError('direct launch is bound to the original CUDA stream')
        kernel.run(grid[0], grid[1], 1, self.stream, kernel.function, kernel.packed_metadata,
                   None, None, None, *args)


_LAUNCHERS = {}


def launcher(hd):
    key = (hd, torch.cuda.current_device())
    if key not in _LAUNCHERS:
        _LAUNCHERS[key] = DirectLaunch(_select_append_gather, 8, HD=hd, HDP=triton.next_power_of_2(hd), PER=PER_PROGRAM,
                                       TOK=TOKENS, BLOCK=SELECT_BLOCK)
    return _LAUNCHERS[key]


class HybridV6Layer(HybridV5Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.implementation = self.implementation.replace('v5_fused_append', 'v6_single_launch')
        self._launch = None

    def attention(self, q, k, v, scale):
        if not self.cumulative_length or self.trace or k.device.type != 'cuda':
            return super().attention(q, k, v, scale)
        b, new, hkv, d = k.shape
        if new != 1 or not k.is_contiguous() or not v.is_contiguous():
            raise ValueError('expected contiguous single-token decode KV')
        slot = self._lengths[0]
        n = slot+1
        if n > self.capacity:
            raise RuntimeError('bounded cache capacity exceeded')
        old = self.cumulative_length
        self.cumulative_length = old+1
        self._lengths = [n]*b
        self.max_visible_tokens = old+1
        step = old+1-self.prompt_len
        hq = q.shape[2]
        rep = self.representative(step)
        kv = rep//(hq//hkv)
        self._rep_head, self._rep_kv = rep, kv
        budget = self.token_budget
        ids = self._ids
        if n <= budget:
            count, start, end, extra = n, n, n, 0
            scores = k  # never read: no middle range
            if ids is None or ids.shape != (b, count):
                ids = self._ids = torch.empty((b, count), device=self.device, dtype=torch.long)
        else:
            count = budget
            start = self._protect_len
            end = max(start, n-64)
            extra = budget-start-(n-end)
            if extra < 0:
                raise ValueError('protection exceeds total K')
            if ids is None or ids.shape != (b, count):
                ids = self._ids = torch.empty((b, count), device=self.device, dtype=torch.long)
            # Identical expression to v4: the ranked columns are bitwise unchanged.
            scores = (self.queries[:, rep, :1] @ self.k_cache[:, :n, kv].transpose(1, 2)).squeeze(1)
            self._last_v4_scores = scores
        chosen = self._chosen
        if chosen is None or chosen.shape[0] != b:
            chosen = self._chosen = torch.empty((b, SELECT_BLOCK), device=self.device, dtype=torch.long)
        lengths = self._selected_lengths
        if lengths is None or lengths.shape[0] != b:
            lengths = self._selected_lengths = torch.empty(b, device=self.device, dtype=torch.int32)
        launch = self._launch
        if launch is None:
            launch = self._launch = launcher(hkv*d)
        kc, vc, tk, tv, pos = self.k_cache, self.v_cache, self._temp_k, self._temp_v, self.slot_positions
        launch((triton.cdiv(count, PER_PROGRAM), b),
               scores, k, v, chosen, ids, kc, vc, tk, tv, lengths, pos, self.cache_seqlens,
               count, start, end, extra, end-start, slot, old,
               scores.stride(0), chosen.stride(0), ids.stride(0), kc.stride(0), vc.stride(0),
               tk.stride(0), k.stride(0), pos.stride(0))
        out = flash_attn_with_kvcache(q, tk, tv, cache_seqlens=lengths, causal=True, softmax_scale=scale)
        self._selected_count = count
        self._last_selected_ids = ids
        self._pool_scores = getattr(self, '_last_v4_scores', None) if self.residual_policy == 'top' else None
        bc = b*count
        self.selection_events += 1
        self.selection_token_sum += bc
        self.selection_token_observations += b
        self.attention_token_sum += bc
        self.attention_token_observations += b
        self.selected_pairs += hq*bc
        self.causal_pairs += hq*b*(old+1)
        if count > self.max_selected_tokens:
            self.max_selected_tokens = count
        if step % self.eviction_interval == 0:
            self.compact_candidate_pool(ids)
        return out


def install(rank_backend='packed_qk_topk'):
    """Process-local opt-in for the variant modes with the v6 decode path."""
    import kvcompress.engine.selector_grid_cache as grid
    from kvcompress.engine.w1_hybrid_variants_cache import install as install_variants
    install_variants(rank_backend)
    original_factory = grid.SelectorGridLayer

    def factory(*args, **kwargs):
        mode = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        if mode in VARIANTS:
            return HybridV6Layer(*args, rank_backend=rank_backend, **kwargs)
        return original_factory(*args, **kwargs)

    grid.SelectorGridLayer = factory
