"""Opt-in hybrid W1 v5: fewer launches and less Python per layer and decode step.

Exactly the v4 hybrid variants (interval/residual/long-prompt rules, same masks,
same FlashAttention inputs and outputs). Per decode layer v4 launched: append KV,
score matmul, ranking, assemble+gather, FlashAttention. v5 launches the same
cuBLAS matmul and ranking kernel, then ONE kernel that appends the new KV to the
pool, assembles chronological IDs and gathers the attended K/V; FlashAttention is
unchanged. The decode path is a single method with the per-step bookkeeping inlined.

Ordering: the matmul now runs before the new key is written. Only the score of
the new slot (always inside recent64, never ranked) reads stale memory; every
ranked score is bitwise unchanged. The fused kernel reads the new token from the
input K/V instead of the pool, so no program reads a slot another program writes.
"""
import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_with_kvcache

from kvcompress.engine.w1_hybrid_gpu_select_cache import _select_chronological, SELECT_BLOCK
from kvcompress.engine.w1_hybrid_variants_cache import HybridVariantLayer, VARIANTS

BLOCK = 512


@triton.jit(do_not_specialize=['COUNT', 'START', 'END', 'EXTRA', 'NEW_SLOT', 'ABSOLUTE',
                               'CS', 'IS', 'KS', 'VS', 'TS', 'NS', 'PS'])
def _append_assemble_gather(KNEW, VNEW, CHOSEN, IDS, K, V, TK, TV, LENGTHS, POS, SEQLENS,
                            COUNT, START, END, EXTRA, NEW_SLOT, ABSOLUTE,
                            CS, IS, KS, VS, TS, NS, PS,
                            HD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    pid = tl.program_id(0)
    i = pid*BLOCK + tl.arange(0, BLOCK)
    # 1. Append the new token to the pool (the first HD/BLOCK programs share it).
    new_valid = i < HD
    knew = tl.load(KNEW+row*NS+i, new_valid, other=0)
    vnew = tl.load(VNEW+row*NS+i, new_valid, other=0)
    tl.store(K+row*KS+NEW_SLOT*HD+i, knew, new_valid)
    tl.store(V+row*VS+NEW_SLOT*HD+i, vnew, new_valid)
    first = pid == 0
    tl.store(POS+row*PS+NEW_SLOT, ABSOLUTE, first)
    tl.store(SEQLENS+row, NEW_SLOT+1, first)
    tl.store(LENGTHS+row, COUNT, first)
    # 2. Assemble prompt-prefix / chosen-middle / tail IDs and gather K/V.
    valid = i < COUNT*HD
    slot, inner = i//HD, i % HD
    middle = (slot >= START) & (slot < START+EXTRA) & valid
    selected = tl.load(CHOSEN+row*CS+slot-START, middle, other=0)+START
    position = tl.where(slot < START, slot,
                        tl.where(slot < START+EXTRA, selected, END+slot-START-EXTRA))
    tl.store(IDS+row*IS+slot, position, valid & (inner == 0))
    is_new = position == NEW_SLOT
    from_pool = valid & (~is_new)
    from_new = valid & is_new
    kval = tl.load(K+row*KS+position*HD+inner, from_pool, other=0)
    vval = tl.load(V+row*VS+position*HD+inner, from_pool, other=0)
    kval = tl.where(from_new, tl.load(KNEW+row*NS+inner, from_new, other=0), kval)
    vval = tl.where(from_new, tl.load(VNEW+row*NS+inner, from_new, other=0), vval)
    tl.store(TK+row*TS+i, kval, valid)
    tl.store(TV+row*TS+i, vval, valid)


class HybridV5Layer(HybridVariantLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.implementation = self.implementation.replace('gpu_select_v4', 'v5_fused_append')

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
        if n <= budget:
            count, start, end, extra = n, n, n, 0
        else:
            count = budget
            start = self._protect_len
            end = max(start, n-64)
            extra = budget-start-(n-end)
            if extra < 0:
                raise ValueError('protection exceeds total K')
        ids = self._ids
        if ids is None or ids.shape != (b, count):
            ids = self._ids = torch.empty((b, count), device=self.device, dtype=torch.long)
        chosen = self._chosen
        if chosen is None or chosen.shape[0] != b:
            chosen = self._chosen = torch.empty((b, SELECT_BLOCK), device=self.device, dtype=torch.long)
        if n > budget:
            # Identical expression to v4: the ranked columns are bitwise unchanged.
            scores = (self.queries[:, rep, :1] @ self.k_cache[:, :n, kv].transpose(1, 2)).squeeze(1)
            self._last_v4_scores = scores
            if extra:
                _select_chronological[(b,)](scores, chosen, end-start, start, extra,
                                            scores.stride(0), chosen.stride(0),
                                            BLOCK=SELECT_BLOCK, num_warps=8)
        lengths = self._selected_lengths
        if lengths is None or lengths.shape[0] != b:
            lengths = self._selected_lengths = torch.empty(b, device=self.device, dtype=torch.int32)
        hd = hkv*d
        kc, vc, tk = self.k_cache, self.v_cache, self._temp_k
        _append_assemble_gather[(triton.cdiv(count*hd, BLOCK), b)](
            k, v, chosen, ids, kc, vc, tk, self._temp_v, lengths, self.slot_positions,
            self.cache_seqlens, count, start, end, extra, slot, old,
            chosen.stride(0), ids.stride(0), kc.stride(0), vc.stride(0), tk.stride(0),
            k.stride(0), self.slot_positions.stride(0), HD=hd, BLOCK=BLOCK)
        out = flash_attn_with_kvcache(q, tk, self._temp_v, cache_seqlens=lengths,
                                     causal=True, softmax_scale=scale)
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
    """Process-local opt-in for the variant modes with the v5 decode path."""
    import kvcompress.engine.selector_grid_cache as grid
    from kvcompress.engine.w1_hybrid_variants_cache import install as install_variants
    install_variants(rank_backend)
    original_decode, original_factory = grid.decode_mode, grid.SelectorGridLayer

    def factory(*args, **kwargs):
        mode = args[3] if len(args) > 3 else kwargs.get('eviction_mode')
        if mode in VARIANTS:
            return HybridV5Layer(*args, rank_backend=rank_backend, **kwargs)
        return original_factory(*args, **kwargs)

    grid.SelectorGridLayer = factory
