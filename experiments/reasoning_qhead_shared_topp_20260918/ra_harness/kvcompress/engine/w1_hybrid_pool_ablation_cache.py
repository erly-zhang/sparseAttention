"""Pool-update ablations of hybrid W1 on the v6 engine (per-step K attention unchanged).

Every decode step W1 still attends total K (prompt prefix + recent 64 + exact top-(K-pl-64) by the
representative head's raw QK) from the bounded candidate pool, exactly as v6. Only how the pool is
maintained changes:

  I   pool-step        total pool M = 4096 (K + N with N = 4096-K). After every decode step's attention,
                       if the pool exceeds M, keep that step's W1 core (incl. prompt and recent 64) and fill
                       Uniformly to M: exactly one Uniform token outside the core is removed per step.
                       This is v6's compaction code with interval 1 and pool M.
  II  (existing)       v6 pool K+4096 compacted every 64 decode steps: mode grid_hybrid64_w1_topk_protected.
  III pool-step-pre    total pool M = 4096. Before every decode step's selection, if the pool is full, one
                       Uniform token is removed among the slots outside the full prompt and the 63 most recent
                       slots (the new token makes the recent 64); the W1 core is NOT protected. The new token is
                       then appended and W1 selects K from the pool as in v6. No compaction after attention.

Slots stay chronological (removal compacts the pool with the fused gather), so prompt/recent protection
and earlier-slot tie breaking are the same as v6. Private RNG streams per (sample, layer, step, rollout).
"""
import torch

from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.random_recent_shared_mask_cache import fused_gather_kv
from kvcompress.engine.w1_hybrid_v6_cache import HybridV6Layer
from kvcompress.engine.w1_hybrid_variants_cache import mode_name

POOL_TOTAL = 4096
POOL_STEP = 'grid_hybridpoolstep_w1_topk_protected'       # I
POOL_STEP_PRE = 'grid_hybridpoolpre_w1_topk_protected'    # III
MODES = {POOL_STEP: 'after_attention', POOL_STEP_PRE: 'before_selection'}
_BASE = mode_name(64, 'random')   # parent constructor mode; interval/budget are overridden below


class HybridPoolStepLayer(HybridV6Layer):
    def __init__(self, *args, **kwargs):
        args, kwargs = list(args), dict(kwargs)
        actual = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        if actual not in MODES or kwargs.get('eviction_interval', 1) != 1:
            raise ValueError('pool-step ablations require their mode and eviction_interval=1')
        if len(args) > 3:
            args[3] = _BASE
        else:
            kwargs['eviction_mode'] = _BASE
        kwargs['eviction_interval'] = 64
        super().__init__(*args, **kwargs)
        if self.token_budget+64 >= POOL_TOTAL:
            raise ValueError('pool-step ablations need K+64 < 4096')
        self.MODE = actual
        self.pool_phase = MODES[actual]
        self.eviction_interval = 1
        self.extra_budget = POOL_TOTAL-self.token_budget
        self.candidate_budget = POOL_TOTAL
        self.pre_evictions = 0
        self.implementation = (f'hybrid_w1_raw_qk_v6_single_launch_pool{POOL_TOTAL}_'
                               f"{'step_after_attention_core_protected' if actual == POOL_STEP else 'step_before_selection_unprotected_core'}")

    def compact_candidate_pool(self, core_ids):
        if self.pool_phase == 'before_selection':
            return                          # III: the pool is maintained before selection only
        return super().compact_candidate_pool(core_ids)

    def _evict_one_before_selection(self):
        n, P = self._lengths[0], self.prompt_len
        lo, hi = P, n-63                    # outside the full prompt and the 63 newest slots
        if hi <= lo:
            raise ValueError('no evictable slot outside prompt and recent window')
        step = self.cumulative_length-self.prompt_len+1     # the decode step about to run
        draws = []
        for rollout in self._batch_ids:
            if rollout not in self._generators:
                self._generators[rollout] = torch.Generator(device=self.device)
            rng = self._generators[rollout]
            rng.manual_seed(_stable_hash(0x50524545, self.mask_seed, self.sample_id or 0,
                                         self.layer_idx, step, rollout))
            draws.append(torch.randint(lo, hi, (1,), device=self.device, generator=rng))
        evicted = torch.cat(draws)                                           # [b]
        base = self._position_range[:n-1].expand(len(draws), -1)
        ids = base+(base >= evicted[:, None]).long()                         # every slot but the evicted one
        fused_gather_kv(self.k_cache, self.v_cache, ids, self._compact_k, self._compact_v)
        positions = self.slot_positions.gather(1, ids)
        self.k_cache, self._compact_k = self._compact_k, self.k_cache
        self.v_cache, self._compact_v = self._compact_v, self.v_cache
        self.slot_positions.fill_(-1)
        self.slot_positions[:, :n-1].copy_(positions)
        self._lengths = [n-1]*len(draws)
        self.cache_seqlens.fill_(n-1)
        self.compaction_events += 1
        self.pre_evictions += 1
        if self.trace:
            self.last_evicted_slots = evicted.clone()

    def attention(self, q, k, v, scale):
        if (self.pool_phase == 'before_selection' and self.cumulative_length
                and self._lengths[0] >= self.candidate_budget):
            self._evict_one_before_selection()
        return super().attention(q, k, v, scale)


def install(rank_backend='packed_qk_topk'):
    """Process-local opt-in: v6 (incl. its variant modes) plus the two pool-step modes."""
    import kvcompress.engine.selector_grid_cache as grid
    from kvcompress.engine.w1_hybrid_v6_cache import install as install_v6
    install_v6(rank_backend)
    original_decode, original_factory = grid.decode_mode, grid.SelectorGridLayer

    def decode(mode):
        if mode in MODES:
            return 'hybrid', 'w1', 'topk', 'protected'
        return original_decode(mode)

    def factory(*args, **kwargs):
        mode = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        if mode in MODES:
            return HybridPoolStepLayer(*args, rank_backend=rank_backend, **kwargs)
        return original_factory(*args, **kwargs)

    grid.decode_mode, grid.SelectorGridLayer = decode, factory
