"""Profiling-only counts and evidence; no member-head QK scoring."""
import torch


def contains(reference, query, n):
    loc = torch.searchsorted(reference, query.contiguous()).clamp(max=reference.shape[1] - 1)
    return (reference.gather(1, loc) == query) & (query < n)


def diagnose(bundle, final, is_rep, ranges):
    n = bundle['n']
    end = bundle['ends'][:, None]
    start = torch.arange(len(final), device=final.device)[:, None] * 128
    old = bundle['old']
    core = old if is_rep else bundle['core']
    fields, data = [], []
    def add(name, x):
        fields.append(name)
        data.append(x.double())
    def valid(x): return x < end
    def weights(x): return torch.where(valid(x), end - torch.maximum(start, x), 0)
    residual = valid(final) & ~contains(core, final, n)
    novel = valid(final) & ~contains(old, final, n)
    add('core_count', valid(core).sum(-1))
    add('residual_count', residual.sum(-1))
    add('final_count', valid(final).sum(-1))
    add('novel_vs_shared10240_count', novel.sum(-1))
    add('selected_causal_pairs', weights(final).sum(-1))
    for name, (lo, hi) in zip(('value', 'kv_evidence'), ranges):
        tokens = torch.arange(lo, hi, device=final.device)[None, :].expand(len(final), -1)
        add(name + '_legal_slots', valid(tokens).sum(-1))
        add(name + '_legal_pairs', weights(tokens).sum(-1))
        for label, x in [('shared10240', old), ('core', core), ('residual', final), ('final', final)]:
            match = (x >= lo) & (x < hi) & valid(x)
            if label == 'residual': match &= residual
            add(name + '_' + label + '_slots', match.sum(-1))
            add(name + '_' + label + '_pairs', (weights(x) * match).sum(-1))
    return torch.stack(data, 1), fields
