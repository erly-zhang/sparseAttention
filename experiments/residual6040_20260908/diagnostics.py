"""Expensive, profiling-only member full-probability and causal-pair diagnostics."""
import torch
from indices import shared_indices


def contains(reference,query,n):
    loc=torch.searchsorted(reference,query.contiguous()).clamp(max=reference.shape[1]-1)
    return (reference.gather(1,loc)==query)&(query<n)


def diagnose(ph,phat,bundle,final,is_rep,ranges):
    n=bundle['n'];rows=len(ph);end=bundle['ends'][:,None]
    start=torch.arange(rows,device=ph.device)[:,None]*128
    old=bundle['old'];core=old if is_rep else bundle['core']
    own=shared_indices(ph,bundle['budget'],bundle['core_budget'])['old']
    fields=[];data=[]
    def add(name,x):fields.append(name);data.append(x.double())
    def valid(x):return x<end
    def weights(x):return torch.where(valid(x),end-torch.maximum(start,x),0).double()
    def mass(x,mask=None,p=ph):
        value=p.gather(1,x.long().clamp(max=n-1))*valid(x)
        if mask is not None:value=value*mask
        return value.sum(-1)
    def count(x):return valid(x).sum(-1)
    residual=valid(final)&~contains(core,final,n)
    novel=valid(final)&~contains(old,final,n)
    lost=valid(old)&~contains(final,old,n)
    add('core_count',count(core));add('residual_count',residual.sum(-1));add('final_count',count(final))
    add('novel_vs_shared10240_count',novel.sum(-1))
    add('member_p_sum',ph.sum(-1));add('core_member_mass',mass(core));add('final_member_mass',mass(final))
    add('residual_member_mass',mass(final,residual));add('new_vs_shared10240_member_mass',mass(final,novel))
    add('replaced_shared10240_member_mass',mass(old,lost));add('original_shared10240_member_mass',mass(old))
    add('member2q_p_sum',phat.sum(-1) if phat is not None else torch.full((rows,),float('nan'),device=ph.device))
    add('member2q_final_proxy_mass',mass(final,p=phat) if phat is not None else torch.full((rows,),float('nan'),device=ph.device))
    den=weights(own).sum(-1);add('perhead10240_causal_pairs',den)
    for name,x in [('shared10240',old),('shared_core',core),('final',final)]:
        intersection=(weights(x)*contains(own,x,n)).sum(-1)
        union=weights(x).sum(-1)+den-intersection
        add(name+'_intersection_pairs',intersection);add(name+'_union_pairs',union)
        add(name+'_causal_pair_recall',intersection/den.clamp(min=1))
        add(name+'_causal_pair_jaccard',intersection/union.clamp(min=1))
    for name,(lo,hi) in zip(('value','kv_evidence'),ranges):
        tokens=torch.arange(lo,hi,device=ph.device)[None,:].expand(rows,-1)
        add(name+'_legal_slots',valid(tokens).sum(-1));add(name+'_legal_pairs',weights(tokens).sum(-1))
        for label,x in [('shared10240',old),('core',core),('final',final)]:
            match=(x>=lo)&(x<hi)&valid(x)
            add(name+'_'+label+'_slots',match.sum(-1))
            add(name+'_'+label+'_pairs',(weights(x)*match).sum(-1))
    return torch.stack(data,1),fields
