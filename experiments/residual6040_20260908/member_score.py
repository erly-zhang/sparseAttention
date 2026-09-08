"""Two actual member queries; normalize each over its entire causal history."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _sample_lse(Q,K,L,N,SQ,SK,SQH,SKH,H,KH,D:tl.constexpr,B:tl.constexpr=128):
    group=tl.program_id(0)
    u=group*16+tl.arange(0,16);r=u//2;sub=u%2
    m=tl.minimum(128,N-r*128);s=tl.minimum(2,m)
    qi=r*128+((2*sub+1)*m)//(2*tl.maximum(s,1))
    valid=(m>0)&(sub<s);d=tl.arange(0,D)
    q=tl.load(Q+qi[:,None]*SQ+H*SQH+d[None,:],valid[:,None],0)
    mx=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32)
    for c in range(0,tl.minimum((group+1)*8*128,N),B):
        j=c+tl.arange(0,B)
        k=tl.load(K+j[None,:]*SK+KH*SKH+d[:,None],j[None,:]<N,0)
        score=tl.dot(q,k,input_precision='ieee')/math.sqrt(D)
        score=tl.where((j[None,:]<=qi[:,None])&(j[None,:]<N),score,-float('inf'))
        new=tl.maximum(mx,tl.max(score,1))
        den=den*tl.exp(mx-new)+tl.sum(tl.exp(score-new[:,None]),1);mx=new
    tl.store(L+u,mx+tl.log(den),u<tl.cdiv(N,128)*2)


@triton.jit
def _sample_prob(Q,K,L,P,N,SQ,SK,SQH,SKH,H,KH,D:tl.constexpr,B:tl.constexpr=128):
    group,c=tl.program_id(0),tl.program_id(1)
    u=group*16+tl.arange(0,16);r=u//2;sub=u%2
    m=tl.minimum(128,N-r*128);s=tl.minimum(2,m)
    qi=r*128+((2*sub+1)*m)//(2*tl.maximum(s,1));valid=(m>0)&(sub<s)
    d=tl.arange(0,D);j=c*B+tl.arange(0,B)
    if c*B<tl.minimum((group+1)*1024,N):
        q=tl.load(Q+qi[:,None]*SQ+H*SQH+d[None,:],valid[:,None],0)
        k=tl.load(K+j[None,:]*SK+KH*SKH+d[:,None],j[None,:]<N,0)
        l=tl.load(L+u,u<tl.cdiv(N,128)*2,0)
        score=tl.dot(q,k,input_precision='ieee')/math.sqrt(D)
        p=tl.where(valid[:,None]&(j[None,:]<=qi[:,None])&(j[None,:]<N),tl.exp(score-l[:,None]),0.)
        avg=tl.sum(tl.reshape(p,(8,2,B)),1)/tl.maximum(tl.minimum(2,N-(group*8+tl.arange(0,8))*128),1)[:,None]
    else:avg=tl.full((8,B),0.,tl.float32)
    rr=group*8+tl.arange(0,8)
    tl.store(P+rr[:,None]*N+j[None,:],avg,(rr[:,None]<tl.cdiv(N,128))&(j[None,:]<N))


def member_probability(q,k,head,kv_head):
    n=q.shape[1];rows=triton.cdiv(n,128)
    p=torch.empty((rows,n),device=q.device,dtype=torch.float32)
    l=torch.empty(rows*2,device=q.device,dtype=torch.float32)
    args=(q,k,l,n,q.stride(1),k.stride(1),q.stride(2),k.stride(2),head,kv_head,q.shape[-1])
    _sample_lse[(triton.cdiv(rows,8),)](*args,num_warps=4)
    _sample_prob[(triton.cdiv(rows,8),triton.cdiv(n,128))](q,k,l,p,*args[3:],num_warps=4)
    return p


def sample_positions(a,b):
    m=b-a;s=min(2,m)
    return [a+((2*u+1)*m)//(2*s) for u in range(s)]
