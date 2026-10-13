"""Algorithm 3 token columns and absolute query-key slash distances.

Candidate CSR rows are only a scheduling superset. The actual mask is checked
per query, so a slash never becomes a vertical column or a whole block.
"""
from dataclasses import dataclass
import torch
from selection import normalized_attention, causal_pairs
from experiments.shared_topp_mask_bench_20260930.attention import nucleus_mask
from experiments.token_compacted_sparse import TokenCompactedIndex


@dataclass
class PatternIndex(TokenCompactedIndex):
    base_flags: torch.Tensor
    slash_lookup: torch.Tensor


def slash_scores(probabilities, positions, n):
    scores = probabilities.new_zeros(n, dtype=torch.float64)
    keys = torch.arange(n, device=probabilities.device)
    for row, position in enumerate(positions):
        valid = keys <= position
        scores.scatter_add_(0, (position-keys[valid]).long(), probabilities[row, valid].double())
    return scores / scores.sum()


def slash_offsets(q, k, head, options):
    _, n, h, _ = q.shape
    count = min(options['vertical_queries'], n)
    positions = torch.arange(n-count, n, device=q.device)
    probabilities, _ = normalized_attention(q[0, -count:, head][None],
        k[0, :, head // (h // k.shape[2])][None], positions)
    scores = slash_scores(probabilities[0], positions, n)
    mask, _, before = nucleus_mask(scores[None], torch.ones_like(scores[None], dtype=torch.bool),
        options['slash_gamma'])
    requested = int(mask.sum())
    cap = options.get('slash_max_offsets')
    if cap is not None and requested > cap:
        mask.zero_()
        mask[0, scores.sort(descending=True, stable=True).indices[:cap]] = True
    offsets = mask[0].nonzero().flatten()
    return offsets, dict(slash_added=True, slash_gamma=options['slash_gamma'],
        slash_query_count=count, slash_requested_offsets=requested, slash_selected_offsets=len(offsets),
        slash_retained_mass=float(scores[offsets].sum()), slash_prefix_before_last=float(before[0]),
        slash_budget_capped=requested>len(offsets), slash_max_offsets=cap, slash_source_head=head,
        slash_granularity='token distance d; key j=query i-d')


def merge_slash(index, offsets, n):
    if offsets.ndim != 1 or bool(((offsets < 0) | (offsets >= n)).any()):
        raise ValueError('invalid causal slash distances')
    offsets = offsets.long().unique(sorted=True)
    lookup = torch.zeros(n, device=index.token_indices.device, dtype=torch.bool)
    lookup[offsets] = True
    prefix = torch.cat((torch.zeros(1, device=lookup.device, dtype=torch.int64), lookup.cumsum(0)))
    rows, flags = [], []
    base_pairs = final_pairs = candidate_pairs = 0
    for qb in range(index.num_query_blocks):
        start, end = qb*index.query_block_size, min((qb+1)*index.query_block_size, n)
        old = index.token_indices[int(index.row_starts[0,0,qb]):int(index.row_ends[0,0,qb])].long()
        base = torch.zeros(end, device=lookup.device, dtype=torch.bool)
        base[old] = True
        # Each distance contributes an interval of candidate keys for this tile.
        left = (start-offsets).clamp_min(0)
        right = (end-offsets).clamp_min(0)
        difference = torch.zeros(end+1, device=lookup.device, dtype=torch.int32)
        ones = torch.ones_like(offsets, dtype=torch.int32)
        difference.scatter_add_(0, left, ones)
        difference.scatter_add_(0, right, -ones)
        ids = (base | (difference.cumsum(0)[:end]>0)).nonzero().flatten()
        rows.append(ids.to(torch.int32))
        flags.append(base[ids])
        before = causal_pairs(old, start, end)
        # Count exact slash pairs, subtract overlaps with base+vertical support.
        slash_pairs = (end-torch.maximum(offsets, offsets.new_tensor(start))).clamp_min(0).sum()
        low = (start-old).clamp(0,n)
        high = (end-old).clamp(0,n)
        overlaps = (prefix[high]-prefix[low]).sum()
        base_pairs += before
        final_pairs += before+int(slash_pairs-overlaps)
        candidate_pairs += causal_pairs(ids, start, end)
    counts = torch.tensor([len(r) for r in rows], device=lookup.device, dtype=torch.int32)
    ends = counts.cumsum(0, dtype=torch.int32)
    result = PatternIndex(row_starts=(ends-counts).view(1,1,-1), row_ends=ends.view(1,1,-1),
        token_indices=torch.cat(rows), head_to_group=index.head_to_group, num_groups=1,
        num_query_blocks=index.num_query_blocks, query_block_size=index.query_block_size,
        base_flags=torch.cat(flags), slash_lookup=lookup)
    return result, dict(slash_added_pairs=final_pairs-base_pairs, mask_selected_pairs=final_pairs,
        mask_density=final_pairs/(n*(n+1)//2), candidate_causal_pairs=candidate_pairs,
        candidate_density=candidate_pairs/(n*(n+1)//2), exact_query_mask=True,
        scheduling_note='candidate QK pairs may exceed actual mask; no speedup claim')


def allowed_mask(index, qb, positions, ids):
    causal = positions[:,None] >= ids[None]
    if not isinstance(index, PatternIndex):
        return causal
    begin, end = int(index.row_starts[0,0,qb]), int(index.row_ends[0,0,qb])
    distances = (positions[:,None]-ids[None]).clamp(0, len(index.slash_lookup)-1)
    return causal & (index.base_flags[begin:end][None] | index.slash_lookup[distances])
