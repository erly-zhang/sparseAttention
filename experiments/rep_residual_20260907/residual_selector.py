"""Online selector: three representative distributions, calibrated head routing."""
import torch
import triton
from experiments.probmean_compact_20260906 import selector as original
from policy import select, prepare_group, apply_ratio


class ResidualSelector(original.ProbabilitySelector):
    def __init__(self, config, *, kind, rules, mode='replace'):
        super().__init__(config, budget=10240 if kind == 'fixed' else None)
        self.kind, self.rules, self.mode = kind, rules, mode

    def __call__(self, q, k, v, block_size, gamma, min_budget, max_budget, tau=0, gqa_interleave=False):
        n, heads = q.shape[1:3]
        layer = self.current_layer.get()
        groups = self.layers[str(layer)]
        rows = triton.cdiv(n,128)
        ps = []
        for group in groups:
            h = int(group['representative'])
            kh = h % k.shape[2] if gqa_interleave else h // (heads // k.shape[2])
            ps.append(original.probability_mean(q,k,h,kh))
        selections = [select(p,self.kind) for p in ps]
        ids = torch.arange(n,device=q.device)[None,:]
        start = torch.arange(rows,device=q.device)[:,None]*128
        end = (start+128).clamp(max=n)
        weights = (end-torch.maximum(start,ids)).clamp_min(0)
        mapping = torch.empty(heads,device=q.device,dtype=torch.int32)
        parts, sizes, stats = [], [], []
        for g, group in enumerate(groups):
            bundle = prepare_group(ps,selections,g,self.kind,mode=self.mode)
            policies = {}
            for head in group['members']:
                quarter = int(self.rules[str(layer)][str(head)]['quarter'])
                policies.setdefault(quarter,[]).append(head)
            target, old, _ = selections[g]
            for quarter, members in sorted(policies.items()):
                final, novel = apply_ratio(bundle,quarter)
                counts = final.sum(-1)
                mapping[members] = len(parts)
                parts.append(final.nonzero()[:,1].to(torch.int32)); sizes.append(counts)
                p = ps[g]
                values = ((final*weights).sum(),torch.as_tensor(n*(n+1)//2,device=q.device),
                          final.sum(),(ids<end).sum(),target.sum(),(target&final).sum(),
                          (p*target).sum(),(p*(target&final)).sum())
                stats.append(torch.stack([x.double() for x in values])*len(members))
                if self.collect:
                    fields = [target.sum(-1),counts,(p*target).sum(-1),(p*final).sum(-1),p.sum(-1)]
                    for lo,hi in self.ranges:
                        valid=ids[:,lo:hi]<end
                        fields += [valid.sum(-1),(target[:,lo:hi]&valid).sum(-1),(final[:,lo:hi]&valid).sum(-1)]
                    fields += [old.sum(-1),(p*old).sum(-1),novel.sum(-1),
                               (old&~final).sum(-1),torch.full_like(counts,quarter)]
                    # The record contains representative proxy mass, not member P.
                    self.records.append((layer,int(group['representative']),members,torch.stack(fields,1)))
        self.stats.pending.append(torch.stack(stats).sum(0)); self.stats.calls+=1
        counts=torch.stack(sizes).reshape(-1); end=counts.cumsum(0)
        return original.TokenCompactedIndex((end-counts).view(1,len(parts),rows).contiguous(),
            end.view(1,len(parts),rows).contiguous(),torch.cat(parts),mapping,len(parts),rows,128)
