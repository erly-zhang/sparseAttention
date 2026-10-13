"""Literal full attention per head, then query-row stitching; no speed shortcut."""
import math
import torch


def full_attention(q, k, v, positions, mode='shared_map'):
    """Return B,N,H*d output, optional shared B,N,N map, and diagnostics.

    Head ownership is original_position % H. With GQA, query head h uses
    KV head h // (H/Hkv), both before and after stitching. No output repetition.
    All H full N-by-N maps are computed, one at a time to bound peak memory.
    """
    if mode not in ('shared_map', 'dense_full'):
        raise ValueError('unknown attention mode')
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('expected Q B,H,N,d and K/V B,Hkv,N,d')
    b, h, n, d = q.shape
    hkv = k.shape[1]
    if min(b, h, n, d, hkv) < 1 or h % hkv or k.shape != (b, hkv, n, d):
        raise ValueError('invalid GQA or sequence shape')
    if positions.shape != (b, n) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError('explicit integer original positions required')
    if positions.device != q.device or k.device != q.device or v.device != q.device:
        raise ValueError('devices must match')
    if q.dtype != k.dtype or q.dtype != v.dtype or not q.is_floating_point():
        raise ValueError('Q/K/V floating dtypes must match')
    if bool((positions < 0).any()) or bool((positions[:, 1:] <= positions[:, :-1]).any()):
        raise ValueError('positions must be nonnegative and strictly increasing')
    if not all(bool(torch.isfinite(t).all()) for t in (q, k, v)):
        raise ValueError('nonfinite Q/K/V')
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    causal = positions[:, :, None] >= positions[:, None, :]
    owner = positions.remainder(h)
    shared = torch.zeros(b, n, n, dtype=dtype, device=q.device) if mode == 'shared_map' else None
    output = torch.empty(b, n, h, d, dtype=q.dtype, device=q.device)
    row_error = 0.0
    row_counts = []
    for head in range(h):
        kv = head // (h // hkv)
        # Deliberately evaluate every query row before selecting this head's rows.
        scores = torch.matmul(q[:, head].to(dtype), k[:, kv].to(dtype).transpose(-1, -2)) / math.sqrt(d)
        scores.masked_fill_(~causal, float('-inf'))
        probabilities = scores.softmax(dim=-1)
        row_error = max(row_error, float((probabilities.sum(-1)-1).abs().max()))
        selected = owner == head
        row_counts.append(selected.sum(-1).tolist())
        if shared is not None:
            shared[selected] = probabilities[selected]
        else:
            output[:, :, head] = torch.matmul(probabilities, v[:, kv].to(dtype)).to(q.dtype)
    if shared is not None:
        if not bool(torch.isfinite(shared).all()) or bool((shared.masked_select(~causal) != 0).any()):
            raise ValueError('invalid stitched causal map')
        row_error = max(row_error, float((shared.sum(-1)-1).abs().max()))
        for head in range(h):
            kv = head // (h // hkv)
            output[:, :, head] = torch.matmul(shared, v[:, kv].to(dtype)).to(q.dtype)
    if row_error > 2e-5 or not bool(torch.isfinite(output).all()):
        raise ValueError('normalization or finite-output failure')
    return output.flatten(-2), shared, dict(
        mode=mode, heads=h, kv_heads=hkv, full_maps_computed=h,
        score_elements=b*h*n*n, full_causal_pairs=b*h*n*(n+1)//2,
        stitched_causal_pairs=b*n*(n+1)//2 if shared is not None else None,
        rows_per_head=row_counts, max_row_sum_error=row_error)
