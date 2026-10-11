"""Opt-in raw-QK W1 compact1000 and bounded hybrid1000 caches.

Hybrid attention selects K every decode forward. Only at multiples of 1000,
after attention, it retains that selected core plus Uniform-TopK 4096 positions
outside the core. New KV accumulates between compactions. Original positions,
GQA and chronological stable W1 ties are preserved. Historical modes are intact.
"""
import torch
import triton
import triton.language as tl

from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.selector_grid_cache import SelectorGridLayer
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer, chronological_topk
from kvcompress.engine.random_recent_shared_mask_cache import fused_gather_kv

COMPACT_MODE = 'grid_compact1000_w1_topk_protected'
HYBRID_MODE = 'grid_hybrid1000_w1_topk_protected'


def internal_arguments(args, kwargs, mode, interval):
    args, kwargs = list(args), dict(kwargs)
    actual = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
    if kwargs.get('eviction_interval', 1000) != 1000:
        raise ValueError('interval1000 mode requires eviction_interval=1000')
    if len(args) > 3:
        args[3] = mode
    else:
        kwargs['eviction_mode'] = mode
    kwargs['eviction_interval'] = interval
    return args, kwargs, actual


def uniform_residual_ids(n, core, count, generator):
    """Independent Uniform float32 priorities, secondary random tie breaker.

    Mask the core before TopK; return distinct chronological slots. Sampling is
    independent of global RNG. If all eligible positions fit, return all of them.
    """
    device = core.device
    eligible = torch.ones(n, device=device, dtype=torch.bool)
    eligible[core] = False
    count = min(count, n - core.numel())
    if count == n - core.numel():
        return eligible.nonzero().flatten()
    if not count:
        return core.new_empty(0)
    ib = max(1, (n-1).bit_length())
    sb = min(24, 63-24-ib)
    primary = (torch.rand(n, device=device, generator=generator)*2**24).long()
    secondary = torch.randint(2**sb, (n,), device=device, generator=generator, dtype=torch.long)
    priority = (primary << (sb+ib)) | (secondary << ib) | torch.arange(n, device=device)
    priority.masked_fill_(~eligible, -1)
    return priority.topk(count, sorted=False).indices.sort().values


@triton.jit(do_not_specialize=['SLOT', 'ABSOLUTE', 'KS', 'VS', 'CS', 'PS'])
def _append_bounded(K, V, KC, VC, POS, LENGTHS, SLOT, ABSOLUTE,
                    HD: tl.constexpr, KS, VS, CS, PS, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    j = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = j < HD
    kval = tl.load(K + row*KS + j, valid, other=0)
    vval = tl.load(V + row*VS + j, valid, other=0)
    tl.store(KC + row*CS + SLOT*HD + j, kval, valid)
    tl.store(VC + row*CS + SLOT*HD + j, vval, valid)
    tl.store(POS + row*PS + SLOT + j, ABSOLUTE, j == 0)
    tl.store(LENGTHS + row + j, SLOT+1, j == 0)


class W1Compact1000Layer(SelectorGridLayer):
    def __init__(self, *args, **kwargs):
        args, kwargs, actual = internal_arguments(args, kwargs, 'grid_compact500_w1_topk_protected', 500)
        if actual != COMPACT_MODE:
            raise ValueError(actual)
        super().__init__(*args, score_mode='qk', **kwargs)
        self.MODE = actual
        self.eviction_interval = 1000
        self.compaction_events = 0
        self.implementation = 'shared_w1_raw_qk_compact1000_v1'

    def compact(self):
        super().compact()
        self.compaction_events += 1


class W1Hybrid1000Layer(FullW1OptimizedLayer):
    def __init__(self, *args, extra_budget=4096, **kwargs):
        args, kwargs, actual = internal_arguments(args, kwargs, 'grid_full_w1_topk_protected', 1)
        if actual != HYBRID_MODE or extra_budget != 4096:
            raise ValueError('hybrid1000 requires protected W1 and random extra4096')
        super().__init__(*args, score_mode='qk', rank_backend='stable_sort', **kwargs)
        self.MODE = actual
        self.storage = 'hybrid'
        self.eviction_interval = 1000
        self.extra_budget = extra_budget
        self.candidate_budget = self.token_budget + extra_budget
        self.compaction_events = 0
        self.compaction_core_token_sum = 0
        self.compaction_residual_token_sum = 0
        self.compaction_token_observations = 0
        self.implementation = 'w1_raw_qk_bounded_random4096_interval1000_v1'

    def lazy_initialization(self, k, value_states=None):
        # Allocate bounded physical storage and a separate K-sized attention
        # scratch. The compaction scratch is bounded too and swapped, not copied.
        b, p, h, d = k.shape
        self.prompt_len, self.device, self.dtype = p, k.device, k.dtype
        if p+64 > self.token_budget:
            raise ValueError('full prompt + recent64 exceeds K')
        self.capacity = max(p, self.candidate_budget) + self.eviction_interval
        self.k_cache = k.new_empty((b, self.capacity, h, d))
        self.v_cache = torch.empty_like(self.k_cache)
        self._temp_k = k.new_empty((b, self.token_budget, h, d))
        self._temp_v = torch.empty_like(self._temp_k)
        self._compact_k = torch.empty_like(self.k_cache)
        self._compact_v = torch.empty_like(self.v_cache)
        self.slot_positions = torch.full((b, self.capacity), -1, device=k.device, dtype=torch.long)
        self._position_range = torch.arange(self.capacity, device=k.device)
        self.cache_seqlens = torch.zeros(b, device=k.device, dtype=torch.int32)
        self._lengths, self._batch_ids = [0]*b, list(range(b))
        self.is_initialized = True

    def select_indices(self):
        # Use optimized full W1 selection on equal, chronological live slots.
        # Prompt is a fixed prefix and protected recent64 is a fixed suffix even
        # after eviction. Only the physical length differs from absolute length.
        absolute = self.cumulative_length
        n = self._lengths[0]
        b = len(self._lengths)
        if not all(length == n for length in self._lengths) or self._q_valid != 1:
            raise ValueError('hybrid rows require equal live lengths/current query')
        hq, hkv = self._num_query_heads, self.k_cache.shape[2]
        if hq % hkv:
            raise ValueError('invalid GQA geometry')
        step = absolute - self.prompt_len
        self._representative_rng.manual_seed(_stable_hash(0x47524550, self.rep_seed,
                                                          self.sample_id or 0, self.layer_idx, step))
        rep = int(torch.randint(hq, (1,), generator=self._representative_rng))
        self._rep_head, self._rep_kv = rep, rep // (hq//hkv)
        scores = (self.queries[:, rep, :1] @ self.k_cache[:, :n, self._rep_kv].transpose(1, 2)).squeeze(1)
        count = min(self.token_budget, n)
        if self._ids is None or self._ids.shape != (b, count):
            self._ids = torch.empty((b, count), device=self.device, dtype=torch.long)
        if n <= self.token_budget:
            self._ids.copy_(self._position_range[:n].expand(b, -1))
        else:
            start, end = self.prompt_len, max(self.prompt_len, n-64)
            extra = count-start-(n-end)
            if extra < 0:
                raise ValueError('protection exceeds K')
            chosen = chronological_topk(scores, extra, start, end, 'stable_sort')
            self._ids[:, :start].copy_(self._position_range[:start].expand(b, -1))
            self._ids[:, start:start+extra].copy_(chosen)
            self._ids[:, start+extra:].copy_(self._position_range[end:n].expand(b, -1))
        self._last_selected_ids = self._ids
        self.selection_events += 1
        self.selection_token_sum += b*count
        self.selection_token_observations += b
        if self.trace:
            self._last_scores = scores.clone()
            self.event_trace.append(dict(step=step, representative=rep, counts=[count]*b,
                                         positions=self.slot_positions.gather(1, self._ids).cpu().tolist()))
        return self._ids, [count]*b

    def compact_candidate_pool(self, core_ids):
        n = self._lengths[0]
        if n <= self.candidate_budget:
            return
        rows = []
        step = self.cumulative_length-self.prompt_len
        for row in range(len(self._lengths)):
            rollout = self._batch_ids[row]
            if rollout not in self._generators:
                self._generators[rollout] = torch.Generator(device=self.device)
            rng = self._generators[rollout]
            rng.manual_seed(_stable_hash(0x48595253, self.mask_seed, self.sample_id or 0,
                                       self.layer_idx, step, rollout))
            residual = uniform_residual_ids(n, core_ids[row], self.extra_budget, rng)
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

    def attention(self, q, k, v, scale):
        from flash_attn import flash_attn_with_kvcache
        first = self.cumulative_length == 0
        b, new, hkv, d = k.shape
        if not first and new != 1:
            raise ValueError('single-token decode required')
        if not self.is_initialized:
            self.lazy_initialization(k, v)
        old, slot = self.cumulative_length, self._lengths[0]
        if slot+new > self.capacity or not all(n == slot for n in self._lengths):
            raise RuntimeError('bounded cache capacity/equal-length contract violated')
        if first:
            out = flash_attn_with_kvcache(q, self.k_cache, self.v_cache, k=k, v=v,
                                         cache_seqlens=self.cache_seqlens, causal=True, softmax_scale=scale)
            self.slot_positions[:, :new].copy_(self._position_range[:new].expand(b, -1))
            self.cache_seqlens.fill_(new)
        elif self.device.type == 'cuda':
            if not k.is_contiguous() or not v.is_contiguous():
                raise ValueError('contiguous single-token KV required')
            _append_bounded[(triton.cdiv(hkv*d, 256), b)](
                k, v, self.k_cache, self.v_cache, self.slot_positions, self.cache_seqlens,
                slot, old, hkv*d, k.stride(0), v.stride(0), self.k_cache.stride(0),
                self.slot_positions.stride(0), BLOCK=256)
        else:
            self.k_cache[:, slot:slot+new].copy_(k)
            self.v_cache[:, slot:slot+new].copy_(v)
            self.slot_positions[:, slot:slot+new] = old
            self.cache_seqlens.fill_(slot+new)
        self.cumulative_length = old+new
        self._lengths = [slot+new]*b
        self.max_visible_tokens = self.cumulative_length
        hq = q.shape[2]
        causal = hq*b*(new*old+new*(new+1)//2)
        if first:
            selected = causal
            self.prefill_pairs += selected
            visible = new
        else:
            ids, counts = self.select_indices()
            self.gather(ids)
            count = counts[0]
            if self._selected_lengths is None or len(self._selected_lengths) != b:
                self._selected_lengths = torch.empty(b, device=self.device, dtype=torch.int32)
                self._selected_count = None
            if self._selected_count != count:
                self._selected_lengths.fill_(count)
                self._selected_count = count
            out = flash_attn_with_kvcache(q, self._temp_k, self._temp_v,
                                         cache_seqlens=self._selected_lengths, causal=True, softmax_scale=scale)
            selected = hq*b*count
            visible = count
            self.attention_token_sum += b*count
            self.attention_token_observations += b
            step = self.cumulative_length-self.prompt_len
            if step % self.eviction_interval == 0:
                self.compact_candidate_pool(ids)
        self.selected_pairs += selected
        self.causal_pairs += causal
        self.max_selected_tokens = max(self.max_selected_tokens, visible)
        return out

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self._compact_k = self._compact_k[indices].contiguous()
        self._compact_v = self._compact_v[indices].contiguous()


def install():
    """Process-local installation only; historical modes retain their factory."""
    import kvcompress.engine.selector_grid_cache as grid
    original_decode, original_factory = grid.decode_mode, grid.SelectorGridLayer

    def decode(mode):
        if mode == COMPACT_MODE:
            return 'compact', 'w1', 'topk', 'protected'
        if mode == HYBRID_MODE:
            return 'hybrid', 'w1', 'topk', 'protected'
        return original_decode(mode)

    def factory(*args, **kwargs):
        mode = args[3] if len(args)>3 else kwargs.get('eviction_mode')
        if mode == COMPACT_MODE:
            return W1Compact1000Layer(*args, **kwargs)
        if mode == HYBRID_MODE:
            return W1Hybrid1000Layer(*args, **kwargs)
        return original_factory(*args, **kwargs)

    grid.decode_mode, grid.SelectorGridLayer = decode, factory
