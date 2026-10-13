"""Output-only paired-head assembly. No attention edges are added."""
import torch


def head_pairs(heads):
    """Sorted cycle edges: H output slots, degree two, exact toy order at H=3."""
    if type(heads) is not int or heads < 3:
        raise ValueError("paired assembly requires at least three query heads")
    return tuple(sorted(tuple(sorted((h, (h + 1) % heads))) for h in range(heads)))


def assemble(merged, positions, valid, heads, mode):
    """B,N,d -> B,N,H*d; ownership is original_position % H, not row index."""
    if merged.ndim != 3 or positions.shape != merged.shape[:2] or valid.shape != positions.shape:
        raise ValueError("expected merged B,N,d and positions/valid B,N")
    if positions.dtype not in (torch.int32, torch.int64) or valid.dtype != torch.bool:
        raise ValueError("integer positions and boolean validity required")
    if mode not in ("repeat", "pair", "native"):
        raise ValueError(mode)
    clean = merged.masked_fill(~valid[..., None], 0)
    if mode == "repeat":
        return clean.repeat(1, 1, heads)
    if mode == "native":
        mask = positions.remainder(heads)[..., None] == torch.arange(heads, device=merged.device)
        return (clean[..., None, :] * mask[..., None].to(merged.dtype)).flatten(-2)
    pairs = torch.tensor(head_pairs(heads), device=merged.device)
    owner = positions.remainder(heads)
    mask = (owner[..., None] == pairs[:, 0]) | (owner[..., None] == pairs[:, 1])
    expanded = clean[..., None, :] * mask[..., None].to(merged.dtype)
    return expanded.flatten(-2)


def project(merged, positions, valid, projection, heads, mode):
    if projection.in_features != heads * merged.shape[-1]:
        raise ValueError("o_proj input width must equal H*d")
    # Original full projection, with its bias applied once. No amplitude rescaling.
    return projection(assemble(merged, positions, valid, heads, mode))
