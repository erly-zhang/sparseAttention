"""Frozen NN member masks, unchanged representative masks and compact kernel."""
import contextvars
from contextlib import contextmanager
import torch
import triton
from experiments.probmean_compact_20260906.selector import probability_mean
from experiments.residual6040_20260908.indices import shared_indices, actual_counts
from experiments.token_compacted_sparse import TokenCompactedIndex, TokenCompactedSelectorStats
from .batched import (protection, fixed_mask, summaries, content_scores,
                      encode_masks, mask_scores, indices)


class Stats(TokenCompactedSelectorStats):
    def __init__(self):
        super().__init__()
        self.pending = []

    def snapshot(self):
        for counts, n, heads in self.pending:
            data = counts.cpu()
            ends = (torch.arange(data.shape[1])+1).mul(128).clamp(max=n)
            assert torch.equal(data[:, :, 0], ends.clamp(max=10240)[None].expand(heads, -1))
            self.selected_token_pairs += int(data[:, :, 1].sum())
            self.causal_token_pairs += heads*n*(n+1)//2
            self.compacted_key_tokens += int(data[:, :, 0].sum())
            self.candidate_key_tokens += heads*int(ends.sum())
        self.pending.clear()
        return super().snapshot()


class Selector:
    def __init__(self, config, predictor):
        assert config['layers'] == predictor.groups
        self.layers = config['layers']
        self.predictor = predictor
        self.current_layer = contextvars.ContextVar('learned_mask_layer', default=None)
        self.stats = Stats()
        self.ranked_probability_dump_dir = None
        self.validate = False
        self.validation = []

    @contextmanager
    def phase(self, _name):
        yield

    @torch.no_grad()
    def __call__(self, q, k, v, block_size, gamma, min_budget, max_budget,
                 tau=0, gqa_interleave=False):
        n, heads = q.shape[1:3]
        assert q.shape[0] == 1 and heads == 32 and k.shape[2] == 8
        assert not gqa_interleave and q.shape[-1] == 128
        layer = self.current_layer.get()
        assert layer is not None
        c = self.predictor.config
        rows, width = triton.cdiv(n, 128), min(n, c.budget)
        output = torch.empty((heads, rows, width), device=q.device, dtype=torch.int32)
        ends_all = [min((i+1)*128, n) for i in range(rows)]
        dense = torch.arange(width, device=q.device)[None].expand(rows, -1)
        dense = dense.masked_fill(dense >= torch.tensor(ends_all, device=q.device)[:, None], n).to(torch.int32)
        if n <= c.budget:
            output.copy_(dense[None])
        else:
            for group in self.layers[str(layer)]:
                rep = group['representative']
                p = probability_mean(q, k, rep, rep//4)
                # Representatives retain the historical selector, including ties.
                original = shared_indices(p, c.budget, c.core_budget)['old']
                output[rep].copy_(original)
                for h in group['members']:
                    if h != rep:
                        output[h, :c.budget//128].copy_(dense[:c.budget//128])
                projected = {}
                for h in group['members']:
                    if h == rep or self.predictor.variant == 'full_mask':
                        continue
                    name = self.predictor.routing[layer, h]
                    key = (name, h//4)
                    if key not in projected:
                        projected[key] = self.predictor.networks[name].key(k[0, :, h//4].float())
                # Prefix-padded batches keep the exact training definition while
                # bounding transient logits and convolution memory.
                for lo in range(c.budget//128, rows, 16):
                    hi = min(lo+16, rows)
                    starts = [i*128 for i in range(lo, hi)]
                    end_list = ends_all[lo:hi]
                    end = torch.tensor(end_list, device=q.device)
                    nn = end_list[-1]
                    pp = p[lo:hi, :nn]
                    protect = protection(end, torch.tensor(starts, device=q.device), nn, c.sink)
                    core = fixed_mask(pp, protect, end, c.core_budget)
                    full = fixed_mask(pp, protect, end, c.budget)
                    encoded = None
                    for h in group['members']:
                        if h == rep:
                            continue
                        name = self.predictor.routing[layer, h]
                        net = self.predictor.networks[name]
                        if self.predictor.variant == 'full_mask':
                            if encoded is None:
                                encoded = encode_masks(net, full, end)
                            scores = mask_scores(net, encoded, h)
                            fixed = protect
                        else:
                            summary = summaries(q[0, :, h], starts, end_list, c.query_segments)
                            scores = content_scores(net, projected[name, h//4][:nn], summary,
                                                    pp, core, end, h)
                            fixed = core
                        chosen = fixed_mask(scores, fixed, end, c.budget)
                        output[h, lo:hi].copy_(indices(chosen, width, n))
                        if self.validate:
                            legal = torch.arange(nn, device=q.device)[None] < end[:, None]
                            self.validation.append(torch.isfinite(scores[legal]).all())
                del projected, p
        counts = torch.stack([actual_counts(output[h], n) for h in range(heads)])
        if self.validate:
            ends = torch.tensor(ends_all, device=q.device)
            legal = output < ends[None, :, None]
            ordered = (output[:, :, 1:] > output[:, :, :-1]) | ~legal[:, :, 1:]
            # Exact protection check directly on indices, independent of masks.
            for h in range(heads):
                required = torch.cat((torch.arange(128, device=q.device)[None].expand(rows, -1),
                    torch.arange(rows, device=q.device)[:, None]*128 + torch.arange(128, device=q.device)[None]), -1)
                valid = required < ends[:, None]
                where = torch.searchsorted(output[h].contiguous(), required.to(torch.int32).contiguous())
                found = output[h].gather(1, where.clamp(max=width-1)) == required
                self.validation.append((found | ~valid).all())
            self.validation.append(ordered.all() & (output >= 0).all())
        self.stats.pending.append((counts, n, heads))
        self.stats.calls += 1
        starts = torch.arange(heads*rows, device=q.device, dtype=torch.int64).mul(width).view(1, heads, rows)
        lengths = torch.tensor(ends_all, device=q.device).clamp(max=c.budget)
        return TokenCompactedIndex(starts, starts+lengths[None, None], output.flatten(),
            torch.arange(heads, device=q.device, dtype=torch.int32), heads, rows, 128)
