"""Three representative P distributions; random residual for every other Q head."""
import torch
import triton
from experiments.probmean_compact_20260906.selector import ProbabilitySelector,probability_mean,TokenCompactedIndex
from random_policy import prepare,random_final

FIELDS=['target_count','final_count','target_mass','final_mass','p_sum',
        'value_legal','value_target','value_final','pair_legal','pair_target','pair_final',
        'core_count','residual_count','novel_residual_count','representative']


class RandomSelector(ProbabilitySelector):
    def __init__(self,config,*,seed=42,final_budget=10240,core_budget=9216):
        super().__init__(config,budget=final_budget)
        self.seed=seed;self.core_budget=core_budget;self.input_hash='warmup'
        self.capture=False;self.captured=[]

    def __call__(self,q,k,v,block_size,gamma,min_budget,max_budget,tau=0,gqa_interleave=False):
        n,heads=q.shape[1:3];layer=self.current_layer.get();rows=triton.cdiv(n,128)
        groups=self.layers[str(layer)]
        mapping=torch.empty(heads,device=q.device,dtype=torch.int32)
        parts=[];counts=[];stats=[]
        ids=torch.arange(n,device=q.device)[None,:]
        start=torch.arange(rows,device=q.device)[:,None]*128
        end=(start+128).clamp(max=n);weights=(end-torch.maximum(start,ids)).clamp_min(0)
        seen=[]
        for group in groups:
            rep=int(group['representative'])
            kh=rep%k.shape[2] if gqa_interleave else rep//(heads//k.shape[2])
            p=probability_mean(q,k,rep,kh)
            bundle=prepare(p,self.budget,self.core_budget)
            target=bundle['target'];old=bundle['old']
            for h in group['members']:
                seen.append(h);is_rep=h==rep
                if is_rep:
                    final=old;residual=torch.zeros_like(old);core=old
                else:
                    final,residual=random_final(bundle,self.input_hash,self.seed,layer,h)
                    core=bundle['core']
                count=final.sum(-1)
                mapping[h]=len(parts);parts.append(final.nonzero()[:,1].int());counts.append(count)
                values=((final*weights).sum(),torch.as_tensor(n*(n+1)//2,device=q.device),
                    count.sum(),(ids<end).sum(),target.sum(),(target&final).sum(),
                    (p*target).sum(),(p*(target&final)).sum())
                stats.append(torch.stack([x.double() for x in values]))
                if self.collect:
                    fields=[target.sum(-1),count,(p*target).sum(-1),(p*final).sum(-1),p.sum(-1)]
                    for lo,hi in self.ranges:
                        valid=ids[:,lo:hi]<end
                        fields += [valid.sum(-1),(target[:,lo:hi]&valid).sum(-1),(final[:,lo:hi]&valid).sum(-1)]
                    fields += [core.sum(-1),residual.sum(-1),(residual&~old).sum(-1),
                               torch.full_like(count,int(is_rep))]
                    self.records.append((layer,rep,[h],torch.stack(fields,1)))
                if self.capture:self.captured.append((h,core.clone(),residual.clone(),final.clone(),old.clone()))
        assert sorted(seen)==list(range(heads))
        self.stats.pending.append(torch.stack(stats).sum(0));self.stats.calls+=1
        count=torch.stack(counts).reshape(-1);finish=count.cumsum(0)
        return TokenCompactedIndex((finish-count).view(1,heads,rows).contiguous(),
            finish.view(1,heads,rows).contiguous(),torch.cat(parts),mapping,heads,rows,128)
