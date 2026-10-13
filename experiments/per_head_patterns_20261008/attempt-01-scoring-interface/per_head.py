"""Representative Token support, independently augmented for every query head.

Reuse the frozen selector, probability aggregation and exact slash kernel.
Process one head at a time so full-context CSR indices are not replicated H times.
Only masks change; Q/K/V, RoPE, output projection and dense decoding are retained.
"""
import math
import torch
from selection import build_index, vertical_columns, merge_vertical
from patterns import slash_offsets, merge_slash, PatternIndex, allowed_mask
from attention import full_attention as original_attention

MODES = ('representative_token_vertical', 'representative_token_vertical_slash')


def shared_base(q, k, v, head, layer, options):
    return build_index(q, k, v, head, layer, 'representative_token', options)


def head_index(base, q, k, head, mode, options):
    if mode not in MODES:
        raise ValueError(mode)
    n = q.shape[1]
    columns, vertical = vertical_columns(q, k, head, options)
    index, stats = merge_vertical(base, columns, n)
    if mode.endswith('_slash'):
        offsets, slash = slash_offsets(q, k, head, options)
        index, added = merge_slash(index, offsets, n)
        stats.update(added)
        stats.update(slash)
    else:
        offsets = torch.empty(0, dtype=torch.long, device=q.device)
        stats.update(slash_added_pairs=0, slash_selected_offsets=0,
                     candidate_causal_pairs=stats['mask_selected_pairs'],
                     candidate_density=stats['mask_density'], exact_query_mask=True)
    stats.update(vertical)
    stats['slash_added'] = mode.endswith('_slash')
    stats.update(head=head, kv_head=head // (q.shape[2] // k.shape[2]),
                 pattern_source_head=head, pattern_shared_with_heads=[head])
    return index, stats, columns, offsets


def dispatch_head(q, k, v, head, index, options):
    from kernel import pattern_prefill_attention
    from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
    kv = head // (q.shape[2] // k.shape[2])
    dispatch = pattern_prefill_attention if isinstance(index, PatternIndex) else triton_token_compacted_prefill_attention
    # The one-head consumer reads group 0. The base group mapping is all zeros.
    # Explicitly selecting the original KV head preserves contiguous GQA mapping.
    return dispatch(q[:, :, head:head+1], k[:, :, kv:kv+1], v[:, :, kv:kv+1], index,
                    token_chunk_size=options['token_chunk_size'], gqa_interleave=False)


def full_attention(q, k, v, positions, mode, head=None, layer=None, options=None, calibration_callback=None):
    if mode not in MODES:
        return original_attention(q, k, v, positions, mode, head, layer, options, calibration_callback)
    n, heads = q.shape[2], q.shape[1]
    expected = torch.arange(n, device=q.device)[None]
    if not torch.equal(positions, expected):
        raise ValueError('Unmodified contiguous prompt positions required')
    qn, kn, vn = (x.transpose(1, 2) for x in (q, k, v))
    if calibration_callback is not None:
        raise ValueError('Use the existing calibration; no recalibration in this experiment')
    base, base_stats = shared_base(qn, kn, vn, head, layer, options)
    output = torch.empty_like(qn)
    records = []
    for h in range(heads):
        index, stats, columns, offsets = head_index(base, qn, kn, h, mode, options)
        result = dispatch_head(qn, kn, vn, h, index, options)
        output[:, :, h:h+1] = result
        records.append(stats)
        del index, result, columns, offsets
    if not bool(torch.isfinite(output).all()):
        raise ValueError('Nonfinite per-head sparse attention output')
    causal = n * (n+1) // 2
    stats = dict(mode=mode, layer=layer, base_source_head=head,
                 base_shared_with_heads=list(range(heads)), pattern_scope='per_head',
                 own_qk_renormalized=True, probabilities_shared=False,
                 query_tile_size=options['query_tile_size'], target_token_budget=options['target_token_budget'],
                 base_selected_pairs=base_stats['base_selected_pairs'], causal_pairs=causal,
                 total_causal_pairs=heads*causal, per_head=records,
                 mask_selected_pairs=sum(r['mask_selected_pairs'] for r in records),
                 vertical_added_pairs=sum(r['vertical_added_pairs'] for r in records),
                 slash_added_pairs=sum(r['slash_added_pairs'] for r in records),
                 candidate_causal_pairs=sum(r['candidate_causal_pairs'] for r in records),
                 selection_granularity='key token columns and exact query-key token distances')
    stats['mask_density'] = stats['mask_selected_pairs'] / stats['total_causal_pairs']
    stats['candidate_density'] = stats['candidate_causal_pairs'] / stats['total_causal_pairs']
    return output.flatten(-2), None, stats


def install(model, mode, *, heads, options, layers):
    import adapter
    original_dispatch = adapter.full_attention
    if original_dispatch is not original_attention:
        raise RuntimeError('Unexpected adapter dispatch; refuse overlapping patches')
    adapter.full_attention = full_attention
    try:
        state = adapter.install(model, mode, layers=layers, heads=heads, options=options)
    except BaseException:
        adapter.full_attention = original_dispatch
        raise
    close = state['close']
    def restore():
        try:
            close()
        finally:
            adapter.full_attention = original_dispatch
    state['close'] = restore
    return state


def literal_mask(base, columns, offsets, n):
    """Small independent mask oracle; never used by full-context execution."""
    mask = torch.zeros(n, n, dtype=torch.bool, device=base.token_indices.device)
    for i in range(n):
        qb = i // base.query_block_size
        a, b = int(base.row_starts[0,0,qb]), int(base.row_ends[0,0,qb])
        keys = set(base.token_indices[a:b].tolist()) | set(columns.tolist())
        keys.update(i-int(d) for d in offsets.tolist() if int(d) <= i)
        mask[i, [j for j in keys if 0 <= j <= i]] = True
    return mask


def reference_head(q, k, v, head, mask):
    kv = head // (q.shape[2] // k.shape[2])
    scores = q[0, :, head].float() @ k[0, :, kv].float().T / math.sqrt(q.shape[-1])
    return (scores.masked_fill(~mask, float('-inf')).softmax(-1) @ v[0, :, kv].float()).to(q.dtype)
