"""Stateless uniform sampling without replacement from a shared-core complement."""
import hashlib
import struct

import torch
import triton
import triton.language as tl
from experiments.rep_residual_20260907.policy import select


@triton.jit
def _draw(Seeds, Bounds, Out, WIDTH:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    offsets=tl.arange(0,BLOCK)
    seed=tl.load(Seeds+row).to(tl.uint64)
    bound=tl.load(Bounds+row).to(tl.uint64)
    # Reject the incomplete upper interval before modulo, avoiding modulo bias.
    limit=(4294967296//bound)*bound
    value=tl.randint(seed,offsets.to(tl.uint32)).to(tl.uint32)
    invalid=value.to(tl.uint64)>=limit
    attempt=0
    while tl.sum(invalid.to(tl.int32),0)>0:
        attempt+=1
        nxt=tl.randint(seed,(offsets+attempt*1048576).to(tl.uint32)).to(tl.uint32)
        value=tl.where(invalid,nxt,value)
        invalid=value.to(tl.uint64)>=limit
    tl.store(Out+row*WIDTH+offsets,(value.to(tl.uint64)%bound).to(tl.int64),offsets<WIDTH)


def stable_seeds(input_hash,seed,layer,head,tiles,device):
    prefix=f'random-residual-v1|{seed}|{input_hash}|{layer}|{head}|'.encode()
    values=[struct.unpack('<q',hashlib.sha256(prefix+str(r).encode()).digest()[:8])[0] for r in tiles]
    return torch.tensor(values,device=device,dtype=torch.int64)


def prepare(p,final_budget=10240,core_budget=9216):
    if core_budget>final_budget:raise ValueError('Core exceeds final budget')
    target,old,protected=select(p,'fixed',budget=final_budget)
    _,core,_=select(p,'fixed',budget=core_budget)
    rows,n=p.shape
    ends=(torch.arange(rows,device=p.device)+1).mul(128).clamp(max=n)
    legal=torch.arange(n,device=p.device)[None,:]<ends[:,None]
    complement=legal&~core
    active=(ends>final_budget).nonzero().flatten()
    active_complement=complement[active]
    counts=active_complement.sum(-1)
    keys=active_complement.nonzero()[:,1]
    starts=counts.cumsum(0)-counts
    return {'target':target,'old':old,'protected':protected,'core':core,'ends':ends,
            'complement':complement,'active':active,'counts':counts,'keys':keys,'starts':starts,
            'residual_size':final_budget-core_budget,'final_budget':final_budget}


def random_final(bundle,input_hash,seed,layer,head):
    core=bundle['core'];active=bundle['active'];n=core.shape[1]
    # When history fits the final budget, all complement positions must be kept.
    final=bundle['old'].clone()
    if not len(active):return final,final&~core
    k=bundle['residual_size']
    if k==0:return core.clone(),torch.zeros_like(core)
    seeds=stable_seeds(input_hash,seed,layer,head,active.cpu().tolist(),core.device)
    width=triton.next_power_of_2(max(4096,4*k))
    while True:
        draws=torch.empty((len(active),width),device=core.device,dtype=torch.int64)
        _draw[(len(active),)](seeds,bundle['counts'],draws,width,width,num_warps=8)
        sorted_draws,order=draws.sort(dim=-1,stable=True)
        first=torch.ones_like(sorted_draws,dtype=torch.bool)
        first[:,1:]=sorted_draws[:,1:]!=sorted_draws[:,:-1]
        unique=torch.zeros_like(first).scatter(1,order,first)
        if bool((unique.sum(-1)>=k).all()):break
        width*=2
        if width>65536:raise RuntimeError('Uniform sampler could not fill budget; no biased fallback')
    keep=unique&(unique.cumsum(-1)<=k)
    positions=bundle['keys'][bundle['starts'][:,None]+draws]
    residual=torch.zeros_like(core)
    rows=active[:,None].expand_as(draws)
    residual[rows[keep],positions[keep]]=True
    residual|=bundle['complement']&(bundle['ends'][:,None]<=bundle['final_budget'])
    final=core|residual
    if not torch.equal(final.sum(-1),bundle['ends'].clamp(max=bundle['final_budget'])):
        raise RuntimeError('Uniform residual count mismatch')
    return final,residual
