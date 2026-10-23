"""Opt-in hybrid W1 variants on the v4 engine: pool-update interval and residual rule.

Attention is unchanged: every decode step the v4 W1 selector attends total K
(prompt + recent64 inside K) from the bounded candidate pool. After attention,
every `interval` decode steps (only if the pool exceeds K+4096) the pool keeps
the current W1 core plus 4096 further positions:
  random  - Uniform-TopK outside the core (exactly the existing hybrid stream);
  top     - the next 4096 by the same W1 raw-QK scores of that step, i.e. the
            pool becomes prompt + recent64 + top-(K-P-64+4096) middle slots,
            ranked by BF16 score with earlier-slot ties, kept chronological.
interval1000/random reproduces the existing v4 hybrid exactly.
"""
import torch

from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.w1_interval1000_cache import HYBRID_MODE
from kvcompress.engine.w1_qk_fast_cache import fast_uniform_residual_ids
from kvcompress.engine.w1_hybrid_gpu_select_cache import GpuSelectHybridW1Layer, gpu_select_chronological
from kvcompress.engine.random_recent_shared_mask_cache import fused_gather_kv

EXTRA = 4096


def mode_name(interval, residual):
    return f"grid_hybrid{interval}{'top' if residual == 'top' else ''}_w1_topk_protected"


VARIANTS = {mode_name(i, r): (i, r) for i in (64, 1000, 1024) for r in ('random', 'top')}


class HybridVariantLayer(GpuSelectHybridW1Layer):
    def __init__(self, *args, **kwargs):
        args, kwargs = list(args), dict(kwargs)
        actual = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        interval, residual = VARIANTS[actual]
        if kwargs.get('eviction_interval', interval) != interval:
            raise ValueError('eviction_interval must equal the mode interval')
        if len(args) > 3:
            args[3] = HYBRID_MODE
        else:
            kwargs['eviction_mode'] = HYBRID_MODE
        kwargs['eviction_interval'] = 1000
        super().__init__(*args, **kwargs)
        self.MODE = actual
        self.eviction_interval = interval
        self.residual_policy = residual
        self.implementation = f'hybrid_w1_raw_qk_gpu_select_v4_i{interval}_{residual}{EXTRA}'
        self._pool_scores = None

    def lazy_initialization(self, k, value_states=None):
        """As the hybrid pool (K+4096+interval), but long prompts are allowed.

        P <= K-64: the whole prompt is attention-protected (identical to v4).
        P > K-64: only the first K-65 prompt tokens are attention-protected
        (random_pp's cap, leaving at least one W1 slot); the full prompt stays in
        the pool permanently, occupying part of the 4096 extra positions.
        """
        b, p, h, d = k.shape
        self.prompt_len, self.device, self.dtype = p, k.device, k.dtype
        if p+64 > self.candidate_budget:
            raise ValueError('prompt + recent64 exceeds the K+4096 candidate pool')
        self._protect_len = p if p+64 <= self.token_budget else self.token_budget-65
        self.capacity = max(p, self.candidate_budget)+self.eviction_interval
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
        if self.trace and self._protect_len < self.prompt_len:
            raise RuntimeError('capped prompt protection requires the v4 CUDA path (trace off)')
        result = super().select_indices()
        if self.residual_policy == 'top':
            self._pool_scores = getattr(self, '_last_v4_scores', None)
        return result

    def _compact_random_keep_prompt(self, core_ids):
        """Long prompt: keep core + every prompt slot, then Uniform-fill to K+4096."""
        n = self._lengths[0]
        if n <= self.candidate_budget:
            return
        if self._random_priority_buffer is None:
            self._random_priority_buffer = core_ids.new_empty(self.capacity)
        step = self.cumulative_length-self.prompt_len
        prompt = self._position_range[:self.prompt_len]
        rows = []
        for row, rollout in enumerate(self._batch_ids):
            if rollout not in self._generators:
                self._generators[rollout] = torch.Generator(device=self.device)
            rng = self._generators[rollout]
            rng.manual_seed(_stable_hash(0x48595253, self.mask_seed, self.sample_id or 0,
                                       self.layer_idx, step, rollout))
            kept = torch.unique(torch.cat((core_ids[row], prompt)))  # sorted, unique
            residual = fast_uniform_residual_ids(n, kept, self.candidate_budget-kept.numel(),
                                                rng, self._random_priority_buffer)
            rows.append(torch.cat((kept, residual)).sort().values)
        ids = torch.stack(rows)
        fused_gather_kv(self.k_cache, self.v_cache, ids, self._compact_k, self._compact_v)
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

    def compact_candidate_pool(self, core_ids):
        if self.residual_policy == 'random':
            if self._protect_len == self.prompt_len:
                # The prompt is inside the core: identical to the v4 random stream.
                return super().compact_candidate_pool(core_ids)
            return self._compact_random_keep_prompt(core_ids)
        n = self._lengths[0]
        if n <= self.candidate_budget:
            return
        b = len(self._lengths)
        start = self.prompt_len
        end = max(start, n-64)
        count = self.candidate_budget-start-(n-end)
        scores = self._pool_scores
        if scores is None or scores.shape[1] < n:
            raise RuntimeError('top residual requires the current-step W1 scores')
        chosen = gpu_select_chronological(scores, count, start, end)
        prefix = self._position_range[:start].expand(b, -1)
        tail = self._position_range[end:n].expand(b, -1)
        ids = torch.cat((prefix, chosen+start, tail), dim=1)
        fused_gather_kv(self.k_cache, self.v_cache, ids, self._compact_k, self._compact_v)
        positions = self.slot_positions.gather(1, ids)
        self.k_cache, self._compact_k = self._compact_k, self.k_cache
        self.v_cache, self._compact_v = self._compact_v, self.v_cache
        self.slot_positions.fill_(-1)
        self.slot_positions[:, :ids.shape[1]].copy_(positions)
        self._lengths = [ids.shape[1]]*b
        self.cache_seqlens.fill_(ids.shape[1])
        self.compaction_events += 1
        self.compaction_core_token_sum += core_ids.numel()
        self.compaction_residual_token_sum += b*self.extra_budget
        self.compaction_token_observations += b
        if self.trace:
            self.last_compaction_ids = ids.clone()


def install(rank_backend='packed_qk_topk'):
    """Process-local opt-in for the variant modes; everything else unchanged."""
    import kvcompress.engine.selector_grid_cache as grid
    from kvcompress.engine.w1_hybrid_gpu_select_cache import install as install_v4
    install_v4(rank_backend)
    original_decode, original_factory = grid.decode_mode, grid.SelectorGridLayer

    def decode(mode):
        if mode in VARIANTS:
            return 'hybrid', 'w1', 'topk', 'protected'
        return original_decode(mode)

    def factory(*args, **kwargs):
        mode = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        if mode in VARIANTS:
            return HybridVariantLayer(*args, rank_backend=rank_backend, **kwargs)
        return original_factory(*args, **kwargs)

    grid.decode_mode, grid.SelectorGridLayer = decode, factory
