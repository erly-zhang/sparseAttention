"""Independent probability-mean selectors; no edits to production methods."""
import contextvars
import math

import torch
import triton
import triton.language as tl

from experiments.token_compacted_sparse import TokenCompactedIndex, TokenCompactedSelectorStats


@triton.jit
def _lse(Q, K, L, N, SQ, SK, SQH, SKH,
         HQ, HK, D: tl.constexpr,
         M: tl.constexpr = 128, B: tl.constexpr = 128):
    r = tl.program_id(0)
    qi = r * M + tl.arange(0, M)
    dd = tl.arange(0, D)
    q = tl.load(Q + qi[:, None] * SQ + HQ * SQH + dd[None, :], qi[:, None] < N, 0)
    mx = tl.full((M,), -float('inf'), tl.float32)
    denom = tl.zeros((M,), tl.float32)
    for start in range(0, tl.minimum((r + 1) * M, N), B):
        ki = start + tl.arange(0, B)
        k = tl.load(K + ki[None, :] * SK + HK * SKH + dd[:, None], ki[None, :] < N, 0)
        s = tl.dot(q, k, input_precision='ieee') * (1.0 / math.sqrt(D))
        s = tl.where((ki[None, :] <= qi[:, None]) & (ki[None, :] < N), s, -float('inf'))
        new_mx = tl.maximum(mx, tl.max(s, 1))
        denom = denom * tl.exp(mx - new_mx) + tl.sum(tl.exp(s - new_mx[:, None]), 1)
        mx = new_mx
    tl.store(L + qi, mx + tl.log(denom), qi < N)


@triton.jit
def _prob(Q, K, L, P, N, SQ, SK, SQH, SKH,
          HQ, HK, D: tl.constexpr,
          M: tl.constexpr = 128, B: tl.constexpr = 128):
    r, c = tl.program_id(0), tl.program_id(1)
    qi = r * M + tl.arange(0, M)
    ki = c * B + tl.arange(0, B)
    dd = tl.arange(0, D)
    if c * B < tl.minimum((r + 1) * M, N):
        q = tl.load(Q + qi[:, None] * SQ + HQ * SQH + dd[None, :], qi[:, None] < N, 0)
        k = tl.load(K + ki[None, :] * SK + HK * SKH + dd[:, None], ki[None, :] < N, 0)
        lse = tl.load(L + qi, qi < N, 0)
        s = tl.dot(q, k, input_precision='ieee') * (1.0 / math.sqrt(D))
        valid = (qi[:, None] < N) & (ki[None, :] <= qi[:, None]) & (ki[None, :] < N)
        p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        avg = tl.sum(p, 0) / tl.minimum(M, N - r * M)
    else:
        avg = tl.full((B,), 0, tl.float32)
    tl.store(P + r * N + ki, avg, ki < N)


def probability_mean(q, k, head, kv_head, tile=128):
    assert q.shape[0] == k.shape[0] == 1 and q.shape[1] == k.shape[1]
    assert q.stride(-1) == k.stride(-1) == 1
    n, d = q.shape[1], q.shape[-1]
    lse = torch.empty(n, device=q.device, dtype=torch.float32)
    p = torch.empty((triton.cdiv(n, tile), n), device=q.device, dtype=torch.float32)
    _lse[(p.shape[0],)](q, k, lse, n, q.stride(1), k.stride(1), q.stride(2), k.stride(2), head, kv_head, d, tile, num_warps=8)
    _prob[(p.shape[0], triton.cdiv(n, 128))](q, k, lse, p, n, q.stride(1), k.stride(1), q.stride(2), k.stride(2), head, kv_head, d, tile, num_warps=8)
    return p


def logit_mean(q, k, head, kv_head, tile=128, return_scores=False):
    """Control: mean over i in tile with i>=j; denominator is visible-query count.

    This reproduces the historical causal-logit-mean definition without tail.
    Softmax is used only to report its proxy mass; fixed TopK ranks the logits.
    """
    n, d = q.shape[1], q.shape[-1]
    rows = triton.cdiv(n, tile)
    x = torch.nn.functional.pad(q[0, :, head].float(), (0, 0, 0, rows * tile - n)).view(rows, tile, d)
    starts = torch.arange(rows, device=q.device) * tile
    lengths = (n - starts).clamp(max=tile)
    s = (x.sum(1) / lengths[:, None]) @ k[0, :, kv_head].float().T / math.sqrt(d)
    for r in range(rows):
        a, b = r * tile, min((r + 1) * tile, n)
        suffix = x[r, :b-a].flip(0).cumsum(0).flip(0)
        s[r, a:b] = (suffix * k[0, a:b, kv_head].float()).sum(-1) / torch.arange(b-a, 0, -1, device=q.device) / math.sqrt(d)
        s[r, b:] = -torch.inf
    probability=s.softmax(-1)
    return (probability,s) if return_scores else probability


def choose(p, *, budget=None, top_p=.99, minimum=1024, tile=128, sink=128, ranking_scores=None):
    """Target is the unconstrained prefix; fixed-budget final reserves protection."""
    rows, n = p.shape
    ids = torch.arange(n, device=p.device)[None, :]
    a = torch.arange(rows, device=p.device)[:, None] * tile
    b = (a + tile).clamp(max=n)
    legal = ids < b
    protected = legal & ((ids < sink) | (ids >= a))
    ranking = p if ranking_scores is None else ranking_scores
    if budget is not None:
        desired = torch.minimum(b.squeeze(1), torch.full((rows,), budget, device=p.device))
        if budget < min(sink + tile, n):
            raise ValueError('Fixed budget must contain all sink/local protection')
        # TopK never re-softmaxes P. Targets are recorded before protection.
        rank = ranking.masked_fill(~legal, -torch.inf).topk(min(budget, n), -1).indices
        take = torch.arange(rank.shape[1], device=p.device)[None, :] < desired[:, None]
        target = torch.zeros_like(legal).scatter(1, rank, take)
        room = desired - protected.sum(-1)
        rank = ranking.masked_fill(~legal | protected, -torch.inf).topk(min(budget, n), -1).indices
        take = torch.arange(rank.shape[1], device=p.device)[None, :] < room[:, None]
        final = protected | torch.zeros_like(legal).scatter(1, rank, take)
    else:
        values, rank = p.sort(dim=-1, descending=True, stable=True)
        cumulative = values.cumsum(-1)
        # Count values strictly before crossing, then retain crossing token.
        desired = (cumulative < top_p).sum(-1) + 1
        desired = torch.minimum(torch.maximum(desired, torch.full_like(desired, minimum)), b.squeeze(1))
        take = torch.arange(n, device=p.device)[None, :] < desired[:, None]
        target = torch.zeros_like(legal).scatter(1, rank, take) & legal
        final = target | protected
    return target, final, protected


class DeferredStats(TokenCompactedSelectorStats):
    def __init__(self):
        super().__init__()
        self.pending = []

    def snapshot(self):
        if self.pending:
            totals = torch.stack(self.pending).sum(0).cpu().tolist()
            for name, val in zip(('selected_token_pairs', 'causal_token_pairs', 'compacted_key_tokens',
                                  'candidate_key_tokens', 'target_mask_tokens', 'covered_target_tokens',
                                  'target_probability_mass', 'covered_target_probability_mass'), totals):
                setattr(self, name, getattr(self, name) + (val if 'mass' in name else int(val)))
            self.pending.clear()
        return super().snapshot()


class ProbabilitySelector:
    def __init__(self, config, *, per_head=False, budget=10240, control=False):
        self.layers = config['layers']
        self.per_head, self.budget, self.control = per_head, budget, control
        self.current_layer = contextvars.ContextVar('probmean_layer', default=None)
        self.stats = DeferredStats()
        self.records = []
        self.ranges = ()
        self.collect = False
        self.ranked_probability_dump_dir = None

    def __call__(self, q, k, v, block_size, gamma, min_budget, max_budget, tau=0, gqa_interleave=False):
        n, heads = q.shape[1:3]
        layer = self.current_layer.get()
        groups = ([{'representative': h, 'members': [h]} for h in range(heads)] if self.per_head else self.layers[str(layer)])
        rows = triton.cdiv(n, 128)
        head_to_group = torch.empty(heads, dtype=torch.int32, device=q.device)
        parts, sizes = [], []
        ids = torch.arange(n, device=q.device)[None, :]
        a = torch.arange(rows, device=q.device)[:, None] * 128
        b = (a + 128).clamp(max=n)
        weights = (b - torch.maximum(a, ids)).clamp_min(0)
        layer_stats = []
        for g, group in enumerate(groups):
            h, members = int(group['representative']), group['members']
            kh = h % k.shape[2] if gqa_interleave else h // (heads // k.shape[2])
            if self.control:
                p,ranking=logit_mean(q,k,h,kh,return_scores=True)
            else:
                p=probability_mean(q,k,h,kh); ranking=None
            target, final, protection = choose(p, budget=self.budget,ranking_scores=ranking)
            counts = final.sum(-1)
            indices = final.nonzero()[:, 1].to(torch.int32)
            parts.append(indices)
            sizes.append(counts)
            head_to_group[members] = g
            mult = len(members)
            sums = torch.stack([x.double() for x in ((final * weights).sum(), torch.as_tensor(n * (n+1)//2, device=q.device),
                                final.sum(), (ids < b).sum(), target.sum(), (target & final).sum(),
                                (p * target).sum(), (p * (target & final)).sum())]) * mult
            layer_stats.append(sums)
            if self.collect:
                fields = [target.sum(-1), counts, (p*target).sum(-1), (p*final).sum(-1), p.sum(-1)]
                for lo, hi in self.ranges:
                    valid = ids[:,lo:hi] < b
                    fields += [valid.sum(-1), (target[:,lo:hi] & valid).sum(-1), (final[:,lo:hi] & valid).sum(-1)]
                self.records.append((layer, h, members, torch.stack(fields, 1)))
            del p, target, final, protection, ranking
        self.stats.pending.append(torch.stack(layer_stats).sum(0))
        self.stats.calls += 1
        counts = torch.stack(sizes).reshape(-1)
        end = counts.cumsum(0)
        return TokenCompactedIndex((end-counts).view(1,len(groups),rows).contiguous(),
                                   end.view(1,len(groups),rows).contiguous(), torch.cat(parts),
                                   head_to_group, len(groups), rows, 128)
