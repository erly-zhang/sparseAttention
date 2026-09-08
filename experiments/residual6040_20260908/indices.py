"""Short-index construction; no per-member dense boolean masks."""
import torch
import triton
import triton.language as tl
from experiments.random_residual_20260908.random_policy import _draw, stable_seeds


@triton.jit
def _rank(P, Out, Core, N, WIDTH:tl.constexpr, HAS_CORE:tl.constexpr,
          PROTECT:tl.constexpr, BLOCK:tl.constexpr=256):
    r,c=tl.program_id(0),tl.program_id(1)
    j=c*BLOCK+tl.arange(0,BLOCK)
    p=tl.load(P+r*N+j,j<N,0)
    legal=j<tl.minimum((r+1)*128,N)
    if PROTECT:legal=legal&~((j<128)|(j>=r*128))
    if HAS_CORE:
        # Group-shared membership, not a fresh per-member dense mask.
        legal=legal&~tl.load(Core+r*N+j,j<N,0)
    bits=p.to(tl.int32,bitcast=True).to(tl.int64)
    code=(bits<<32)+(4294967295-j.to(tl.int64))
    tl.store(Out+r*N+j,tl.where(legal,code,-1),j<N)


def ranked(p,k,core_mask=None,protect=False):
    rows,n=p.shape
    code=torch.empty((rows,n),dtype=torch.int64,device=p.device)
    _rank[(rows,triton.cdiv(n,256))](p,code,core_mask if core_mask is not None else p,
                                   n,0,core_mask is not None,protect)
    return code.topk(min(k,n),dim=-1,sorted=True).indices.to(torch.int32)


def shared_indices(p,budget=10240,core_budget=6144):
    rows,n=p.shape;device=p.device
    ends=(torch.arange(rows,device=device)+1).mul(128).clamp(max=n)
    lengths=ends.clamp(max=budget);core_lengths=lengths.clamp(max=core_budget)
    # Preserve the original representative's torch.topk tie behavior.
    ids=torch.arange(n,device=device)[None,:]
    protected=(ids<ends[:,None])&((ids<128)|(ids>=torch.arange(rows,device=device)[:,None]*128))
    scores=p.masked_fill(protected|(ids>=ends[:,None]),-torch.inf)
    rank=scores.topk(min(budget,n),dim=-1,sorted=True).indices.to(torch.int32)
    prot=torch.cat((torch.arange(128,device=device)[None,:].expand(rows,-1),
                    torch.arange(rows,device=device)[:,None]*128+torch.arange(128,device=device)[None,:]),1)
    valid=(prot<ends[:,None])&~((torch.arange(rows,device=device)[:,None]==0)&(torch.arange(256,device=device)[None,:]>=128))
    prot=prot.masked_fill(~valid,n).sort(-1).values.to(torch.int32)
    pc=valid.sum(-1)
    def build(cap,counts):
        width=min(cap,n);out=torch.full((rows,width),n,device=device,dtype=torch.int32)
        col=torch.arange(width,device=device)[None,:].expand(rows,-1)
        from_prot=col<pc[:,None]
        pi=prot.gather(1,col.clamp(max=255))
        ri=rank.gather(1,(col-pc[:,None]).clamp(min=0,max=rank.shape[1]-1))
        out=torch.where(from_prot,pi,ri).masked_fill(col>=counts[:,None],n)
        return out.sort(-1).values.contiguous()
    old=build(budget,lengths);core=build(core_budget,core_lengths)
    mask=torch.zeros((rows,n+1),device=device,dtype=torch.bool)
    mask.scatter_(1,core.long(),True)
    # Padding sentinel is deliberately excluded from the actual key dimension.
    mask=mask[:,:n].contiguous()
    return {'old':old,'core':core,'mask':mask,'ends':ends,'lengths':lengths,
            'core_lengths':core_lengths,'n':n,'budget':budget,'core_budget':core_budget}


def random_residual(bundle,input_hash,seed,layer,head):
    n=bundle['n'];core=bundle['core'];rows=len(core)
    k=min(bundle['budget']-bundle['core_budget'],n)
    out=torch.full((rows,k),n,device=core.device,dtype=torch.int32)
    first=bundle['budget']//128
    if n<=bundle['budget']:return out
    active=list(range(first,rows))
    seeds=stable_seeds(input_hash,seed,layer,head,active,core.device)
    bounds=(bundle['ends'][first:]-bundle['core_lengths'][first:]).long()
    width=triton.next_power_of_2(4*k)
    while True:
        draws=torch.empty((len(active),width),device=core.device,dtype=torch.int64)
        _draw[(len(active),)](seeds,bounds,draws,width,width,num_warps=8)
        ordered,permutation=draws.sort(dim=-1,stable=True)
        first_occ=torch.ones_like(ordered,dtype=torch.bool)
        first_occ[:,1:]=ordered[:,1:]!=ordered[:,:-1]
        unique=torch.zeros_like(first_occ).scatter(1,permutation,first_occ)
        if bool((unique.sum(-1)>=k).all()):break
        width*=2
        if width>131072:raise RuntimeError('Uniform sampler exhausted draws; no biased fallback')
    positions=torch.arange(width,device=core.device)[None,:].expand_as(draws)
    take=positions.masked_fill(~unique,width).topk(k,-1,largest=False,sorted=True).indices
    ranks=draws.gather(1,take).contiguous()
    # kth complement key = k + upper_bound(core[i]-i, k).
    adjusted=(core[first:].long()-torch.arange(core.shape[1],device=core.device)[None,:]).contiguous()
    keys=ranks+torch.searchsorted(adjusted,ranks,right=True)
    out[first:]=keys.to(torch.int32).sort(-1).values
    return out


def scored_residual(p,bundle):
    k=min(bundle['budget']-bundle['core_budget'],bundle['n'])
    return ranked(p,k,core_mask=bundle['mask']).sort(-1).values


def assemble(bundle,residual=None):
    n=bundle['n'];width=min(n,bundle['budget']);rows=len(bundle['core'])
    if residual is None:return bundle['old']
    joined=torch.cat((bundle['core'],residual),-1).sort(-1).values[:,:width].contiguous()
    # Short histories are completely dense, including partial tiles.
    ids=torch.arange(width,device=joined.device)[None,:].expand(rows,-1)
    dense=ids.masked_fill(ids>=bundle['ends'][:,None],n).to(torch.int32)
    return torch.where(bundle['ends'][:,None]<=bundle['budget'],dense,joined)


@triton.jit
def _counts(Ids,Out,N,W:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0);a=r*128;b=tl.minimum(a+128,N)
    x=tl.arange(0,BLOCK);j=tl.load(Ids+r*W+x,x<W,N)
    valid=(x<W)&(j<b)
    pairs=tl.where(valid,b-tl.maximum(a,j),0)
    tl.store(Out+r*2,tl.sum(valid.to(tl.int32),0))
    tl.store(Out+r*2+1,tl.sum(pairs.to(tl.int64),0))


def actual_counts(indices,n):
    out=torch.empty((len(indices),2),device=indices.device,dtype=torch.int64)
    _counts[(len(indices),)](indices,out,n,indices.shape[1],triton.next_power_of_2(indices.shape[1]))
    return out
