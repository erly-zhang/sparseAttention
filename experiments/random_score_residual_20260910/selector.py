"""Reuse historical core/assembly/kernel; change only the random residual rule."""
import sys
from pathlib import Path
import torch
import triton

LEGACY = Path(__file__).resolve().parents[1] / 'residual6040_20260908'
sys.path.insert(0, str(LEGACY))
from selector6040 import Selector as LegacySelector
from experiments.residual6040_20260908.indices import shared_indices, assemble, actual_counts
from experiments.probmean_compact_20260906.selector import probability_mean
from experiments.token_compacted_sparse import TokenCompactedIndex
from experiments.random_score_residual_20260910.sampler import random_score_residual
from experiments.random_score_residual_20260910.diagnostics import diagnose, contains

METHOD = 'probmean_shared6144_randomscore4096_allmembers_seed42'


class Selector(LegacySelector):
    def __init__(self, config, mode='randomscore', *, budget=10240, core=6144):
        super().__init__(config, mode, budget=budget, core=core)

    def __call__(self, q, k, v, block_size, gamma, min_budget, max_budget,
                 tau=0, gqa_interleave=False):
        n, heads = q.shape[1:3]
        rows, width = triton.cdiv(n, 128), min(n, self.budget)
        layer = self.current_layer.get()
        output = torch.empty((heads, rows, width), device=q.device, dtype=torch.int32)
        counts = []
        for group in self.layers[str(layer)]:
            rep = int(group['representative'])
            kv = rep % k.shape[2] if gqa_interleave else rep // (heads // k.shape[2])
            with self.phase('representative_probability'):
                p = probability_mean(q, k, rep, kv)
            with self.phase('shared_selection_indices'):
                bundle = shared_indices(p, self.budget, self.core)
            for h in group['members']:
                residual = None
                if h != rep and n > self.budget:
                    residual = random_score_residual(bundle, self.input_hash, self.seed,
                                                     layer, h, phase=self.phase)
                with self.phase('final_indices_required_counts'):
                    final = assemble(bundle, residual)
                    output[h].copy_(final)
                    counts.append(actual_counts(final, n))
                if self.validate:
                    ends = bundle['ends'][:, None]
                    valid = final < ends
                    ordered = (final[:, 1:] > final[:, :-1]) | ~valid[:, 1:]
                    protected = torch.cat((torch.arange(128, device=q.device)[None, :].expand(rows, -1),
                        torch.arange(rows, device=q.device)[:, None] * 128 + torch.arange(128, device=q.device)[None, :]), 1).int()
                    protection = (contains(final, protected, n) | (protected >= ends)).all()
                    budgets = (valid.sum(-1) == bundle['lengths']).all()
                    core_kept = (contains(final, bundle['core'], n) | (bundle['core'] >= ends)).all()
                    self.validation.append(ordered.all() & (final >= 0).all() & protection & budgets & core_kept)
                if self.capture:
                    self.captured.append((layer, h, rep, bundle['core'].clone(), final.clone(), bundle['old'].clone(), None))
                if self.profile:
                    with self.phase('extra_diagnostics'):
                        data, fields = diagnose(bundle, final, h == rep, self.ranges)
                        self.diagnostics.append((layer, h, rep, data, fields))
                del residual, final
            del p, bundle
        self.stats.pending.append((torch.stack(counts), n, heads))
        self.stats.calls += 1
        starts = torch.arange(heads * rows, device=q.device, dtype=torch.int64).mul(width).view(1, heads, rows)
        lengths = (torch.arange(rows, device=q.device) + 1).mul(128).clamp(max=n).clamp(max=self.budget)
        return TokenCompactedIndex(starts, starts + lengths[None, None, :], output.flatten(),
            torch.arange(heads, device=q.device, dtype=torch.int32), heads, rows, 128)
