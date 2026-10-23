"""Opt-in hybrid1000 W1 with less CPU work per layer and decode step (v4 candidate).

Same algorithm and, for finite scores, the same masks as hybrid v2/v3:
1. Representative heads. The private stream draws, per (layer, step),
   manual_seed(_stable_hash(0x47524550, rep_seed, sample, layer, step)) followed
   by one CPU torch.randint(Hq). Its first mt19937 output is reproduced for the
   whole generation in one vectorized pass (64-bit hash in NumPy uint64, mt19937
   on the GPU), then each layer/step only reads a CPU table entry.
2. Ranking. One Triton program per row packs the unchanged cuBLAS BF16 scores
   into unique 32-bit keys (BF16 order, earlier slot wins ties; identical to the
   v2 packed keys), finds the exact count-th largest key by 32-round radix
   select, and writes chosen slots in chronological order with a prefix sum.
   This replaces pack + TopK + sort (or the small-N stable argsort). The v3
   kernel then assembles prompt/chosen/recent64 and gathers K/V in one launch.
The selector matmul, attention, physical compaction and residual RNG are intact.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from kvcompress.engine.w1_interval1000_cache import HYBRID_MODE
from kvcompress.engine.w1_qk_fused_cache import FusedHybridW1Layer, _assemble_gather

_M64 = (1 << 64)-1
_GOLDEN = 0x9E3779B97F4A7C15
_TABLES = {}
_TABLE_LAYERS = 64
SELECT_BLOCK = 8192


def _python_hash_prefix(*vals):
    x = _GOLDEN
    for v in vals:
        x = (x ^ (int(v)+_GOLDEN+(x << 6)+(x >> 2))) & _M64
    return x


def stable_hash_steps(prefix_values, steps):
    """_stable_hash(*prefix_values, step) for every step in a uint64 array."""
    x = np.uint64(_python_hash_prefix(*prefix_values))
    a = np.uint64((_GOLDEN+(int(x) << 6)+(int(x) >> 2)) & _M64)
    s = steps.astype(np.uint64)
    with np.errstate(over='ignore'):
        h = x ^ (s+a)
        h = (h ^ (h >> np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        h = h ^ (h >> np.uint64(31))
    return h & np.uint64(0x7FFFFFFFFFFFFFFF)


def mt19937_first_output(seeds, device):
    """First uint32 of at::mt19937(seed) for each seed (seed truncated to 32 bits)."""
    s = torch.as_tensor((seeds & np.uint64(0xFFFFFFFF)).astype(np.int64), device=device)
    first = s
    second = None
    state = s
    for j in range(1, 398):
        # 1812433253 * x < 2**63 for x < 2**32: exact in int64.
        state = (1812433253*(state ^ (state >> 30))+j) & 0xFFFFFFFF
        if j == 1:
            second = state
    m397 = state
    y = (first & 0x80000000) | (second & 0x7FFFFFFF)
    y = m397 ^ (y >> 1) ^ torch.where((second & 1) == 1, 0x9908B0DF, 0)
    y = y ^ (y >> 11)
    y = y ^ ((y << 7) & 0x9D2C5680)
    y = y ^ ((y << 15) & 0xEFC60000)
    y = y ^ (y >> 18)
    return y & 0xFFFFFFFF


def representative_table(rep_seed, sample_id, hq, steps, device, layers=_TABLE_LAYERS):
    """CPU int16 table[layer, step] equal to the private representative draw."""
    key = (rep_seed, sample_id, hq, steps, layers, str(device))
    if key not in _TABLES:
        grid = np.arange(steps, dtype=np.uint64)
        seeds = np.concatenate([stable_hash_steps((0x47524550, rep_seed, sample_id, layer), grid)
                                for layer in range(layers)])
        draws = mt19937_first_output(seeds, device) % hq
        _TABLES[key] = draws.view(layers, steps).to(torch.int16).cpu().numpy()
    return _TABLES[key]


@triton.jit(do_not_specialize=['N_MID', 'START', 'COUNT', 'SS', 'CS'])
def _select_chronological(SCORES, CHOSEN, N_MID, START, COUNT, SS, CS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    valid = j < N_MID
    score = tl.load(SCORES+row*SS+START+j, valid, other=0)
    bits = score.to(tl.uint16, bitcast=True).to(tl.int64)
    order = tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
    order = tl.where((bits & 0x7FFF) == 0, 0x8000, order)
    key = tl.where(valid, (order << 16) | (0xFFFF-j.to(tl.int64)), -1)
    # Keys are unique, so the largest T with >=COUNT keys >= T selects exactly COUNT.
    threshold = tl.sum(tl.zeros([BLOCK], dtype=tl.int64), axis=0)
    for bit in tl.static_range(31, -1, -1):
        candidate = threshold | (1 << bit)
        hits = tl.sum((key >= candidate).to(tl.int32), axis=0)
        threshold = tl.where(hits >= COUNT, candidate, threshold)
    chosen = key >= threshold
    slot = tl.cumsum(chosen.to(tl.int32), axis=0)-1
    tl.store(CHOSEN+row*CS+slot, j.to(tl.int64), chosen)


def gpu_select_chronological(scores, count, start, end, out=None):
    """Zero-based chronological chosen slots in [start, end), same as exact_qk_topk."""
    b, n_mid = scores.shape[0], end-start
    if not 0 < count <= n_mid or n_mid > SELECT_BLOCK or scores.dtype != torch.bfloat16:
        raise ValueError('unsupported GPU selection range')
    if out is None:
        out = torch.empty((b, count), device=scores.device, dtype=torch.long)
    _select_chronological[(b,)](scores, out, n_mid, start, count, scores.stride(0), out.stride(0),
                                BLOCK=SELECT_BLOCK, num_warps=8)
    return out


class GpuSelectHybridW1Layer(FusedHybridW1Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.implementation = 'hybrid_w1_raw_qk_gpu_select_v4'
        if self.layer_idx == 0:
            # A new cache: representative-table preparation is timed per generation.
            _TABLES.clear()
        self._rep_table = None
        self._chosen = None

    def representative(self, step):
        table = self._rep_table
        if table is None or step >= table.shape[1]:
            steps = max(self.max_cache_length-self.prompt_len+1, step+1)
            table = self._rep_table = representative_table(
                self.rep_seed, self.sample_id or 0, self._num_query_heads, steps, self.device)
        return int(table[self.layer_idx, step])

    def select_indices(self):
        if self.device.type != 'cuda' or self.trace or self.layer_idx >= _TABLE_LAYERS:
            return super().select_indices()
        n, b = self._lengths[0], len(self._lengths)
        if not all(length == n for length in self._lengths):
            raise ValueError('equal live lengths required')
        hq, hkv = self._num_query_heads, self.k_cache.shape[2]
        if hq % hkv or self._q_valid != 1:
            raise ValueError('invalid GQA geometry or missing current query')
        rep = self.representative(self.cumulative_length-self.prompt_len)
        self._rep_head, self._rep_kv = rep, rep//(hq//hkv)
        count = min(self.token_budget, n)
        if self._ids is None or self._ids.shape != (b, count):
            self._ids = torch.empty((b, count), device=self.device, dtype=torch.long)
        if n <= self.token_budget:
            start, end, extra = n, n, 0
            chosen = self._ids  # Not read: the middle interval is empty.
        else:
            # Attention-protected prompt prefix; equals the prompt unless a pool
            # variant opts into a capped prefix for prompts longer than K-64.
            start = getattr(self, '_protect_len', self.prompt_len) if self.protection == 'protected' else 0
            end = max(start, n-64) if self.protection == 'protected' else n
            extra = count-start-(n-end)
            if extra < 0:
                raise ValueError('protection exceeds total K')
            # Exactly the v2 matmul expression, including protected keys.
            scores = (self.queries[:, rep, :1] @
                      self.k_cache[:, :n, self._rep_kv].transpose(1, 2)).squeeze(1)
            self._last_v4_scores = scores  # Reused only by opt-in pool variants.
            if self._chosen is None or self._chosen.shape[0] != b:
                self._chosen = torch.empty((b, SELECT_BLOCK), device=self.device, dtype=torch.long)
            chosen = self._chosen
            if extra:
                gpu_select_chronological(scores, extra, start, end, chosen)
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
        self._last_selected_ids = self._ids
        self.selection_events += 1
        self.selection_token_sum += b*count
        self.selection_token_observations += b
        return self._ids, [count]*b

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self._chosen = None


def install(rank_backend='packed_qk_topk'):
    """Process-local opt-in for the hybrid mode only; everything else unchanged."""
    import kvcompress.engine.selector_grid_cache as grid
    from kvcompress.engine.w1_qk_fused_cache import install as install_v3
    install_v3(rank_backend)
    original = grid.SelectorGridLayer

    def factory(*args, **kwargs):
        mode = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        if mode == HYBRID_MODE:
            return GpuSelectHybridW1Layer(*args, rank_backend=rank_backend, **kwargs)
        return original(*args, **kwargs)

    grid.SelectorGridLayer = factory
