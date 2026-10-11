"""Exact implementation optimizations for the full-history W1 TopK selector.

The historical SelectorGridLayer remains the reference. This class only accepts
its full/W1/TopK modes. All active rows must share one prompt and decode length,
as in batch_exist_generate. It retains every KV and selects anew at every step.
The default probability mode keeps the reference BF16 matmul/division and FP32
softmax. The opt-in qk mode ranks raw BF16 QK logits for a single current query;
finite-precision masks can differ from probability mode. Stable score ties prefer
the earlier original position; final IDs remain chronological.
"""
import math
import torch
import triton
import triton.language as tl

from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.selector_grid_cache import SelectorGridLayer, representative


@triton.jit(do_not_specialize=["POSITION", "KS", "VS", "CS", "PS"])
def _append_full(K, V, KC, VC, POS, LENGTHS, POSITION, HD: tl.constexpr,
                 KS, VS, CS, PS, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    inner = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = inner < HD
    kval = tl.load(K + row * KS + inner, valid, other=0)
    vval = tl.load(V + row * VS + inner, valid, other=0)
    offset = row * CS + POSITION * HD + inner
    tl.store(KC + offset, kval, valid)
    tl.store(VC + offset, vval, valid)
    tl.store(POS + row * PS + POSITION + inner, POSITION, inner == 0)
    tl.store(LENGTHS + row + inner, POSITION + 1, inner == 0)


@triton.jit(do_not_specialize=["N", "START", "SS", "OS"])
def _order_keys(SCORES, KEYS, N, START, SS, OS, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    score = tl.load(SCORES + row * SS + START + j, j < N, other=0)
    # Probabilities are finite, nonnegative float32. Their positive IEEE bits
    # have the same order as their values. The low32 bits give earlier positions
    # higher priority, making keys unique even for softmax underflow/ties.
    bits = score.to(tl.int32, bitcast=True).to(tl.int64)
    key = (bits << 32) | (0xFFFFFFFF - j.to(tl.int64))
    tl.store(KEYS + row * OS + j, key, j < N)


def chronological_topk(scores, count, start=0, end=None, backend='stable_sort', keys=None):
    """Batched exact stable TopK, returned in ascending original position."""
    end = scores.shape[1] if end is None else end
    n = end - start
    if not 0 <= count <= n:
        raise ValueError('invalid TopK budget')
    if count == 0:
        return torch.empty((scores.shape[0], 0), device=scores.device, dtype=torch.long)
    if backend == 'stable_sort':
        ids = torch.argsort(scores[:, start:end], dim=1, descending=True, stable=True)[:, :count]
    elif backend == 'packed_topk':
        if scores.dtype != torch.float32 or scores.device.type != 'cuda':
            raise ValueError('packed TopK requires CUDA float32 probabilities')
        if keys is None:
            keys = torch.empty((scores.shape[0], n), device=scores.device, dtype=torch.int64)
        _order_keys[(triton.cdiv(n, 256), scores.shape[0])](
            scores, keys, n, start, scores.stride(0), keys.stride(0), BLOCK=256)
        ids = keys[:, :n].topk(count, dim=1, sorted=False).indices
    else:
        raise ValueError(backend)
    return ids.sort(dim=1).values + start


class FullW1OptimizedLayer(SelectorGridLayer):
    implementation = 'full_w1_fixed_ranges_batched_exact_v1'

    def __init__(self, *args, rank_backend='stable_sort', score_mode='probability', **kwargs):
        super().__init__(*args, **kwargs)
        if (self.storage, self.scorer, self.selector) != ('full', 'w1', 'topk'):
            raise ValueError('optimized implementation only supports full W1 TopK')
        if rank_backend not in ('stable_sort', 'packed_topk'):
            raise ValueError(rank_backend)
        if score_mode not in ('probability', 'qk'):
            raise ValueError(score_mode)
        if score_mode == 'qk' and rank_backend != 'stable_sort':
            # The packed backend encodes nonnegative FP32 probabilities only;
            # its bit ordering cannot be used for signed raw QK logits.
            raise ValueError('raw QK requires the stable_sort rank backend')
        self.rank_backend = rank_backend
        self.score_mode = score_mode
        if score_mode == 'qk':
            self.implementation = 'full_w1_fixed_ranges_batched_qk_v1'
        self._representative_rng = torch.Generator()
        self._ids = None
        self._selected_lengths = None
        self._key_buffer = None

    def lazy_initialization(self, k, value_states=None):
        super().lazy_initialization(k, value_states)
        self._position_range = torch.arange(self.capacity, device=self.device)

    def update_queries(self, q):
        self._num_query_heads = q.shape[1]
        if not self.cumulative_length:
            return
        if q.shape[2] != 1:
            raise ValueError('single-query decode required')
        # W1 consumes this current post-RoPE query before it is replaced. No
        # model code mutates q afterwards; no history ring/copy is needed.
        self.queries = q
        self._q_valid = 1

    def select_indices(self):
        n = self.cumulative_length
        b = len(self._lengths)
        if not all(length == n for length in self._lengths):
            raise ValueError('full-history rows must have equal valid lengths')
        hq = self._num_query_heads
        hkv = self.k_cache.shape[2]
        if hq % hkv or self._q_valid != 1:
            raise ValueError('invalid GQA geometry or missing current query')
        step = n - self.prompt_len
        # Re-seeding the reused private generator produces exactly the same
        # first draw as representative() with a newly constructed generator.
        self._representative_rng.manual_seed(_stable_hash(
            0x47524550, self.rep_seed, self.sample_id or 0, self.layer_idx, step))
        rep = int(torch.randint(hq, (1,), generator=self._representative_rng))
        self._rep_head = rep
        self._rep_kv = rep // (hq // hkv)
        # One current query against every retained, legal historical key.
        logits = self.queries[:, rep, :1] @ self.k_cache[:, :n, self._rep_kv].transpose(1, 2)
        if self.score_mode == 'qk':
            # Positive scaling and softmax are unnecessary for single-query
            # TopK in exact arithmetic. Rank the existing BF16 logits directly,
            # without a division, probability tensor, or query-mean reduction.
            # BF16 scaling/FP32 softmax can create extra ties in the old mode,
            # so bitwise mask equivalence to probability mode is not promised.
            scores = logits.squeeze(1)
        else:
            # Preserve the default's historical arithmetic exactly.
            logits = logits / (math.sqrt(self.k_cache.shape[-1]) * self.temperature)
            scores = logits.softmax(-1, dtype=torch.float32).mean(1)
        if self.trace:
            self._last_scores = scores.clone()
        count = min(self.token_budget, n)
        if self._ids is None or self._ids.shape != (b, count):
            self._ids = torch.empty((b, count), device=self.device, dtype=torch.long)
        if n <= self.token_budget:
            self._ids.copy_(self._position_range[:n].expand(b, -1))
        else:
            start = self.prompt_len if self.protection == 'protected' else 0
            end = max(self.prompt_len, n - 64) if self.protection == 'protected' else n
            tail = n - end
            extra = count - start - tail
            if extra < 0:
                raise ValueError('protection exceeds total K')
            if self.rank_backend == 'packed_topk' and self._key_buffer is None:
                self._key_buffer = torch.empty((b, self.capacity), device=self.device, dtype=torch.int64)
            chosen = chronological_topk(scores, extra, start, end, self.rank_backend, self._key_buffer)
            self._ids[:, :start].copy_(self._position_range[:start].expand(b, -1))
            self._ids[:, start:start + extra].copy_(chosen)
            self._ids[:, start + extra:].copy_(self._position_range[end:n].expand(b, -1))
        counts = [count] * b
        self._last_selected_ids = self._ids
        self.selection_events += 1
        self.selection_token_sum += b * count
        self.selection_token_observations += b
        if self.trace:
            self.event_trace.append(dict(step=step, representative=rep, counts=counts,
                                         positions=self._ids.cpu().tolist()))
        return self._ids, counts

    def attention(self, q, k, v, scale):
        from flash_attn import flash_attn_with_kvcache
        if not self.cumulative_length or k.device.type != 'cuda':
            return super().attention(q, k, v, scale)
        b, new, hkv, d = k.shape
        if new != 1 or not k.is_contiguous() or not v.is_contiguous():
            raise ValueError('expected contiguous single-token decode KV')
        old = self.cumulative_length
        if old + 1 > self.capacity or not all(n == old for n in self._lengths):
            raise RuntimeError('full cache capacity/length contract violated')
        _append_full[(triton.cdiv(hkv * d, 256), b)](
            k, v, self.k_cache, self.v_cache, self.slot_positions, self.cache_seqlens,
            old, hkv * d, k.stride(0), v.stride(0), self.k_cache.stride(0),
            self.slot_positions.stride(0), BLOCK=256)
        self.cumulative_length = old + 1
        self._lengths = [old + 1] * b
        self.max_visible_tokens = self.cumulative_length
        ids, counts = self.select_indices()
        self.gather(ids)
        count = counts[0]
        if self._selected_lengths is None or self._selected_lengths.shape[0] != b:
            self._selected_lengths = torch.empty(b, device=self.device, dtype=torch.int32)
            self._selected_count = None
        if self._selected_count != count:
            self._selected_lengths.fill_(count)
            self._selected_count = count
        out = flash_attn_with_kvcache(q, self._temp_k, self._temp_v,
                                     cache_seqlens=self._selected_lengths,
                                     causal=True, softmax_scale=scale)
        hq = q.shape[2]
        self.selected_pairs += hq * b * count
        self.causal_pairs += hq * b * self.cumulative_length
        self.max_selected_tokens = max(self.max_selected_tokens, count)
        self.attention_token_sum += b * count
        self.attention_token_observations += b
        return out

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self._ids = None
        self._selected_lengths = None
        self._key_buffer = None


def install(rank_backend='stable_sort', score_mode='probability'):
    """Explicit process-local opt-in; existing launches remain unchanged."""
    import kvcompress.engine.selector_grid_cache as grid
    original = SelectorGridLayer
    if score_mode not in ('probability', 'qk'):
        raise ValueError(score_mode)
    if score_mode == 'qk' and rank_backend != 'stable_sort':
        raise ValueError('raw QK requires the stable_sort rank backend')

    def factory(*args, **kwargs):
        mode = kwargs.get('eviction_mode', args[3] if len(args) > 3 else '')
        if mode in ('grid_full_w1_topk_protected', 'grid_full_w1_topk_none'):
            return FullW1OptimizedLayer(*args, rank_backend=rank_backend,
                                        score_mode=score_mode, **kwargs)
        return original(*args, **kwargs)

    grid.SelectorGridLayer = factory
