"""Budget-matched, proxy-only cross-representative residual selection."""
import torch

RATIOS = [0, 1, 2, 3, 4]


def layout(p, starts=None, tile=128, sink=128):
    rows, n = p.shape
    if starts is None:
        starts = torch.arange(rows, device=p.device) * tile
    ids = torch.arange(n, device=p.device)[None, :]
    end = (starts[:, None] + tile).clamp(max=n)
    legal = ids < end
    protected = legal & ((ids < sink) | (ids >= starts[:, None]))
    return legal, protected


def ranked_mask(p, allowed, counts):
    width = min(p.shape[1], int(counts.max().item()))
    if width == 0:
        return torch.zeros_like(allowed)
    rank = p.masked_fill(~allowed, -torch.inf).topk(width, -1).indices
    take = torch.arange(width, device=p.device)[None, :] < counts[:, None]
    return torch.zeros_like(allowed).scatter(1, rank, take) & allowed


def select(p, kind, starts=None, budget=10240, minimum=1024, top_p=.99):
    legal, protected = layout(p, starts)
    if kind == 'fixed':
        counts = legal.sum(-1).clamp(max=budget)
        if (counts < protected.sum(-1)).any():
            raise ValueError('Budget smaller than protection')
        target = ranked_mask(p, legal, counts)
        final = protected | ranked_mask(p, legal & ~protected, counts-protected.sum(-1))
    else:
        values, rank = p.sort(dim=-1, descending=True, stable=True)
        counts = ((values.cumsum(-1) < top_p).sum(-1)+1).clamp(min=minimum)
        counts = torch.minimum(counts, legal.sum(-1))
        take = torch.arange(p.shape[1], device=p.device)[None, :] < counts[:, None]
        target = torch.zeros_like(legal).scatter(1, rank, take) & legal
        final = target | protected
    return target, final, protected


def candidates(p, allowed, residual):
    width = min(residual, p.shape[1])
    values, rank = p.masked_fill(~allowed, -torch.inf).topk(width, -1)
    return rank.masked_fill(~torch.isfinite(values), -1)


def first_unique(ids, count):
    """Keep original priority order while removing repeated/invalid indices."""
    ordered, permutation = ids.sort(dim=-1, stable=True)
    first = ordered >= 0
    first[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    valid = torch.zeros_like(first).scatter(1, permutation, first)
    return valid & (valid.cumsum(-1) <= count[:, None])


def prepare_group(ps, selections, group, kind, residual=1024, mode='replace'):
    p = ps[group]
    _, old, protected = selections[group]
    others = [i for i in range(len(ps)) if i != group]
    if len(others) != 2:
        raise ValueError('This experiment requires exactly three AE groups')
    if mode == 'append':
        if kind != 'topp':
            raise ValueError('Fixed policy must use replacement')
        core = old
        room = (layout(p)[0].sum(-1)-old.sum(-1)).clamp(max=residual)
    else:
        room = (old.sum(-1)-protected.sum(-1)).clamp(max=residual)
        core = protected | ranked_mask(p, old & ~protected, old.sum(-1)-room-protected.sum(-1))
    lists = [candidates(ps[s], selections[s][1] & ~core, residual) for s in others]
    dropped = candidates(p, old & ~core, residual)
    return {'core':core, 'old':old, 'protected':protected, 'room':room,
            'sources':others, 'lists':lists, 'dropped':dropped, 'mode':mode}


def apply_ratio(bundle, quarter):
    core, old, room = bundle['core'], bundle['old'], bundle['room']
    if quarter < 0:
        return old, torch.zeros_like(old)
    a, b = bundle['lists']
    q = torch.div(room*quarter, 4, rounding_mode='floor')
    front = torch.arange(a.shape[1], device=a.device)[None, :] < q[:, None]
    # Prefer q entries from A, fill from B, then remaining A and dropped own keys.
    ids = torch.cat((a.masked_fill(~front, -1), b,
                     a.masked_fill(front, -1), bundle['dropped']), -1)
    keep = first_unique(ids, room)
    added = torch.zeros_like(core)
    row = torch.arange(len(ids), device=ids.device)[:, None].expand_as(ids)
    added[row[keep], ids[keep]] = True
    final = core | added
    if bundle['mode'] == 'replace' and not torch.equal(final.sum(-1), old.sum(-1)):
        raise RuntimeError('Residual replacement failed to preserve final budget')
    return final, added & ~old
