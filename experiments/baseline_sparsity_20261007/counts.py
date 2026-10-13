"""Count effective causal support; never interpret padded/future pairs as density."""
import torch


def triangle(x):
    x = x.clamp_min(0)
    return x * (x + 1) // 2


def rectangle_pairs(q_start, q_end, k_start, k_end):
    return (triangle(q_end-k_start)-triangle(q_start-k_start)
            -triangle(q_end-k_end)+triangle(q_start-k_end))


def flex_pairs(indices, n, block_size):
    ids = indices.to(torch.int64)
    blocks = (n+block_size-1)//block_size
    row, col = ids//blocks, ids%blocks
    a, c = row*block_size, col*block_size
    b, d = (a+block_size).clamp(max=n), (c+block_size).clamp(max=n)
    pairs = rectangle_pairs(a, b, c, d).sum()
    invalid = ((ids < 0) | (ids >= blocks*blocks)).sum()
    return pairs, invalid


def minference_pairs(converted, n, block_m=64, block_n=64):
    """Converter emits sorted disjoint intervals and columns outside intervals.

    Verify these invariants on EVERY observed head. Count the actual converted
    64-wide intervals, including unaligned starts, rather than ideal slash lines.
    The Triton column branch has no causal predicate: require all columns to
    precede their query tile. Input batch/head must both be one.
    """
    bc, bo, cc, ci = converted
    assert bc.shape[:2] == (1,1) and cc.shape == bc.shape
    bc, bo, cc, ci = bc[0,0].long(), bo[0,0].long(), cc[0,0].long(), ci[0,0].long()
    rows = bc.numel()
    qs = torch.arange(rows, device=bo.device, dtype=torch.int64)*block_m
    qe = (qs+block_m).clamp(max=n)
    bv = torch.arange(bo.shape[1], device=bo.device)[None,:] < bc[:,None]
    cv = torch.arange(ci.shape[1], device=ci.device)[None,:] < cc[:,None]
    # Strictly disjoint sorted blocks and sorted unique columns.
    violations = ((bc<0)|(bc>bo.shape[1])|(cc<0)|(cc>ci.shape[1])).sum()
    violations += ((bo<0)&bv).sum() + (((ci<0)|(ci>=n)|(ci>=qs[:,None]))&cv).sum()
    violations += ((bo[:,1:] < bo[:,:-1]+block_n)&bv[:,1:]).sum()
    violations += ((ci[:,1:] <= ci[:,:-1])&cv[:,1:]).sum()
    # Search only valid interval starts; masked slots are an increasing sentinel.
    starts = torch.where(bv, bo, n+block_n).contiguous()
    at = torch.searchsorted(starts, ci.contiguous(), right=True)-1
    nearest = bo.gather(1, at.clamp(min=0, max=bo.shape[1]-1))
    overlaps = cv & (at>=0) & (at<bc[:,None]) & (ci<nearest+block_n)
    violations += overlaps.sum()
    kp = (bo+block_n).clamp(max=n)
    block_pairs = torch.where(bv, rectangle_pairs(qs[:,None],qe[:,None],bo.clamp(max=n),kp),0).sum()
    column_pairs = torch.where(cv,(qe[:,None]-torch.maximum(qs[:,None],ci)).clamp_min(0),0).sum()
    return block_pairs+column_pairs, violations, block_pairs, column_pairs
