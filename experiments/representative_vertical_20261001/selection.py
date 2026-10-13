"""Single representative per layer; AutoBlock token targets plus vertical columns.

Only mask indices are shared. The existing Triton kernel applies each head's
own Q/K/V, including its own GQA KV head. Query tiles are scheduling units;
vertical columns and CSR key indices remain at token granularity.
"""
import math
import torch
from experiments.token_compacted_sparse import (
    RepresentativeTokenFirstBlockSelector, TokenCompactedIndex,
)
from experiments.shared_topp_mask_bench_20260930.attention import nucleus_mask


def normalized_attention(q, k, query_positions):
    """Q [H,R,D], corresponding K [H,N,D]; FP32 (FP64 for CPU references)."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = torch.matmul(q.to(dtype), k.to(dtype).transpose(-1, -2)) / math.sqrt(q.shape[-1])
    allowed = torch.arange(k.shape[-2], device=q.device)[None, :] <= query_positions[:, None]
    scores.masked_fill_(~allowed[None], float('-inf'))
    return scores.softmax(-1), allowed


def vertical_columns(q, k, head, options):
    """Algorithm 3's vertical portion, independently normalized before selection."""
    _, n, h, _ = q.shape
    hk = k.shape[2]
    count = min(options['vertical_queries'], n)
    pos = torch.arange(n-count, n, device=q.device)
    probs, _ = normalized_attention(q[0, -count:, head][None],
        k[0, :, head // (h // hk)][None], pos)
    # Every representative query contributes equal weight, including short inputs.
    scores = probs[0].double().mean(0)
    scores /= scores.sum()
    mask, mass, before = nucleus_mask(scores[None], torch.ones_like(scores[None], dtype=torch.bool),
        options['vertical_gamma'])
    ranked = scores.sort(descending=True, stable=True).indices
    requested = int(mask.sum())
    cap = options.get('vertical_max_tokens')
    if cap is not None and requested > cap:
        mask.zero_()
        mask[0, ranked[:cap]] = True
    selected = mask[0].nonzero().flatten()
    return selected, dict(vertical_query_count=count, vertical_gamma=options['vertical_gamma'],
        vertical_requested_tokens=requested, vertical_selected_tokens=len(selected),
        vertical_retained_mass=float(scores[selected].sum()), vertical_prefix_before_last=float(before[0]),
        vertical_budget_capped=len(selected) < requested, vertical_max_tokens=cap,
        vertical_source_head=head, vertical_granularity='key token', slash_added=False)


def make_selector(heads, method, options, num_heads=32):
    if method not in ('representative_autoblock', 'representative_token', 'representative_token_vertical'):
        raise ValueError(method)
    project = method == 'representative_autoblock'
    groups = {'layers': {str(layer): [{'representative': head, 'members': list(range(num_heads))}]
        for layer, head in heads.items()}}
    return RepresentativeTokenFirstBlockSelector(groups,
        logical_key_block_size=tuple(options['logical_key_block_sizes']),
        target_token_budget=options['target_token_budget'], target_token_top_p=options['target_token_top_p'],
        target_token_min_budget=None, target_top_p_start_layer=None,
        final_token_budget=options['autoblock_final_token_budget'] if project else None,
        maximum_selected_blocks=None, fill_final_token_budget=False,
        minimum_block_coverage_ratio=options['minimum_block_coverage_ratio'],
        minimum_block_target_probability_ratio=None, block_probability_top_p=None,
        block_target_probability_top_p=None, block_target_count_coverage_ratio=None,
        block_target_cost_aware=False, block_f_beta=None, measure_fixed_target_probability_mass=False,
        force_sink_block=options['autoblock_force_sink'] if project else options['token_force_sink'],
        force_diagonal_block=options['autoblock_force_diagonal'] if project else options['token_force_diagonal'],
        query_score_mode='four_probe_weighted', probe_weights=tuple(options['probe_weights']),
        adaptive_coverage_tolerance=options['adaptive_coverage_tolerance'], project_target_to_blocks=project)


def causal_pairs(indices, start, end):
    return int((end - torch.maximum(indices.long(), indices.new_tensor(start))).clamp_min(0).sum())


def merge_vertical(index, columns, n):
    if index.num_groups != 1 or index.row_starts.shape[0] != 1:
        raise ValueError('one representative/group, batch one required')
    if columns.ndim != 1 or bool(((columns < 0) | (columns >= n)).any()):
        raise ValueError('invalid vertical key positions')
    tiles = []
    base_pairs = final_pairs = added_slots = 0
    for qb in range(index.num_query_blocks):
        start, end = qb*index.query_block_size, min((qb+1)*index.query_block_size, n)
        old = index.token_indices[int(index.row_starts[0,0,qb]):int(index.row_ends[0,0,qb])].long()
        final = torch.cat((old, columns[columns < end])).unique(sorted=True)
        if not bool((final <= start).any()):
            raise ValueError('empty causal row; retain existing target budget >= query tile size')
        base_pairs += causal_pairs(old, start, end)
        final_pairs += causal_pairs(final, start, end)
        added_slots += len(final)-len(old)
        tiles.append(final.to(torch.int32))
    counts = torch.tensor([len(t) for t in tiles], device=columns.device, dtype=torch.int32)
    ends = counts.cumsum(0, dtype=torch.int32)
    merged = TokenCompactedIndex(row_starts=(ends-counts).view(1,1,-1).contiguous(),
        row_ends=ends.view(1,1,-1).contiguous(), token_indices=torch.cat(tiles).contiguous(),
        head_to_group=index.head_to_group, num_groups=1, num_query_blocks=index.num_query_blocks,
        query_block_size=index.query_block_size)
    return merged, dict(base_selected_pairs=base_pairs, mask_selected_pairs=final_pairs,
        causal_pairs=n*(n+1)//2, mask_density=final_pairs/(n*(n+1)//2),
        vertical_added_pairs=final_pairs-base_pairs, vertical_added_key_slots=added_slots)


def build_index(q, k, v, head, layer, method, options):
    if q.shape[0] != 1 or q.shape[1] != k.shape[1] or q.shape[2] % k.shape[2]:
        raise ValueError('single-sample, contiguous GQA prefill required')
    n, h = q.shape[1:3]
    if type(head) is not int or not 0 <= head < h:
        raise ValueError('representative head out of range')
    selector = make_selector({str(layer): head}, method, options, h)
    token = selector.current_layer.set(layer)
    try:
        index = selector(q,k,v,options['query_tile_size'],.95,1,n)
    finally:
        selector.current_layer.reset(token)
    columns, vs = (vertical_columns(q,k,head,options) if method.endswith('_vertical')
        else (torch.empty(0,device=q.device,dtype=torch.long), {}))
    index, stats = merge_vertical(index,columns,n)
    # Read base statistics directly; snapshot() can emit NaNs for inapplicable block fields.
    stats.update(mode=method, representative_head=head, shared_with_heads=list(range(h)),
        query_tile_size=options['query_tile_size'], selector='original AutoBlock',
        target_token_budget=options['target_token_budget'], target_token_top_p=options['target_token_top_p'],
        target_token_slots=selector.stats.target_mask_tokens//h,
        base_covered_target_token_slots=selector.stats.covered_target_tokens//h,
        base_target_token_coverage=selector.stats.covered_target_tokens/max(1,selector.stats.target_mask_tokens),
        base_chosen_block_size_rows=selector.stats.layer_chosen_block_size_rows.get(layer,{}),
        own_qk_renormalized=True, probabilities_shared=False, **vs)
    return index, stats


def reference_attention(q, k, v, index):
    """Independent GQA reference: every output head uses its OWN KV head."""
    _, n, h, d = q.shape
    output = torch.empty_like(q)
    for head in range(h):
        kv = head // (h // k.shape[2])
        for qb in range(index.num_query_blocks):
            start, end = qb*index.query_block_size, min((qb+1)*index.query_block_size, n)
            ids = index.token_indices[int(index.row_starts[0, 0, qb]):int(index.row_ends[0, 0, qb])].long()
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            scores = q[0, start:end, head].to(dtype) @ k[0, ids, kv].to(dtype).T / math.sqrt(d)
            scores.masked_fill_(ids[None] > torch.arange(start, end, device=q.device)[:, None], float('-inf'))
            output[0, start:end, head] = (scores.softmax(-1) @ v[0, ids, kv].to(dtype)).to(q.dtype)
    return output


def sampled_member_coverage(q, k, index, count=128, chunk=4):
    _, n, h, _ = q.shape
    sums = torch.zeros(h, dtype=torch.float64, device=q.device)
    minima = torch.ones(h, dtype=torch.float64, device=q.device)
    mapped_k = k[0].transpose(0, 1).repeat_interleave(h//k.shape[2], 0)
    for start in range(max(0, n-count), n, chunk):
        end = min(start+chunk, n)
        probs, _ = normalized_attention(q[0, start:end].transpose(0, 1), mapped_k,
            torch.arange(start, end, device=q.device))
        for row, position in enumerate(range(start, end)):
            qb = position//index.query_block_size
            ids = index.token_indices[int(index.row_starts[0, 0, qb]):int(index.row_ends[0, 0, qb])].long()
            mass = probs[:, row, ids].double().sum(-1)
            sums += mass
            minima = torch.minimum(minima, mass)
    return dict(member_coverage_queries=min(count, n), member_mean_retained_mass=(sums/min(count, n)).tolist(),
        member_min_retained_mass=minima.tolist(), member_coverage_scope='last queries only; not all rows')


def directional_coverage(q, k, top_p=.95, last_queries=128, chunk=4):
    """S[provider,target] = mean target mass retained by provider's nucleus mask."""
    _, n, h, _ = q.shape
    mapped_k = k[0].transpose(0, 1).repeat_interleave(h//k.shape[2], 0)
    similarity = torch.zeros(h, h, dtype=torch.float64, device=q.device)
    for start in range(max(0, n-last_queries), n, chunk):
        end = min(start+chunk, n)
        probabilities, allowed = normalized_attention(q[0, start:end].transpose(0, 1), mapped_k,
            torch.arange(start, end, device=q.device))
        masks, _, _ = nucleus_mask(probabilities, allowed[None].expand_as(probabilities), top_p)
        # Per-query batched matrix multiplication: provider x target, then sum queries.
        scores = masks.transpose(0, 1).to(probabilities.dtype) @ probabilities.transpose(0, 1).transpose(1, 2)
        similarity += scores.double().sum(0)
    similarity /= min(last_queries, n)
    outgoing = similarity.mean(-1)
    return dict(head=int(outgoing.argmax()), outgoing_scores=outgoing.tolist(), similarity=similarity.tolist())


def vote_representatives(samples, layers):
    if not samples:
        raise ValueError('no disjoint calibration samples')
    result = {}
    for layer in layers:
        votes = {}
        for sample in samples:
            record = sample[str(layer)]
            head = record['head']
            votes.setdefault(head, []).append(record['outgoing_scores'][head])
        result[str(layer)] = max(votes, key=lambda head: (len(votes[head]), sum(votes[head]), -head))
    return result
