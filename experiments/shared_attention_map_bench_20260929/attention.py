"""Exact query-row-blocked full attention and shared-map assembly."""
import math
import torch


def full_attention(q, k, v, positions, mode='shared_map', block_size=256, return_map=False):
    if mode not in ('shared_map','dense_full') or type(block_size) is not int or block_size<1:
        raise ValueError('invalid mode or block size')
    if q.ndim!=4 or k.ndim!=4 or k.shape!=v.shape:
        raise ValueError('expected Q B,H,N,d and K/V B,Hkv,N,d')
    b,h,n,d=q.shape
    hk=k.shape[1]
    if min(b,h,n,d,hk)<1 or h%hk or k.shape!=(b,hk,n,d):
        raise ValueError('invalid GQA shapes')
    if positions.shape!=(b,n) or positions.dtype not in (torch.int32,torch.int64):
        raise ValueError('integer original positions required')
    if any(t.device!=q.device for t in (k,v,positions)) or q.dtype!=k.dtype or q.dtype!=v.dtype or not q.is_floating_point():
        raise ValueError('device/dtype mismatch')
    if bool((positions<0).any()) or bool((positions[:,1:]<=positions[:,:-1]).any()):
        raise ValueError('positions must strictly increase')
    if not all(bool(torch.isfinite(t).all()) for t in (q,k,v)):
        raise ValueError('nonfinite QKV')
    if return_map and n>4096:
        raise ValueError('full-map materialization is for small parity tests only')
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    group=h//hk
    output=torch.empty((b,n,h,d),dtype=q.dtype,device=q.device)
    full_map=torch.empty((b,n,n),dtype=dtype,device=q.device) if return_map and mode=='shared_map' else None
    error=torch.zeros((),dtype=dtype,device=q.device)
    owners=positions.remainder(h)
    for lo in range(0,n,block_size):
        hi=min(lo+block_size,n)
        allowed=positions[:,lo:hi,None]>=positions[:,None,:]
        shared=torch.zeros((b,hi-lo,n),dtype=dtype,device=q.device) if mode=='shared_map' else None
        for g in range(hk):
            first=g*group
            # Every head evaluates every query row against ALL keys. Softmax
            # spans the full key dimension; query blocking changes no context.
            queries=q[:,first:first+group,lo:hi].to(dtype)
            keys=k[:,g:g+1].to(dtype)
            scores=torch.matmul(queries,keys.transpose(-1,-2)) / math.sqrt(d)
            scores.masked_fill_(~allowed[:,None],float('-inf'))
            probs=scores.softmax(-1)
            error=torch.maximum(error,(probs.sum(-1)-1).abs().max())
            if shared is not None:
                for local in range(group):
                    selected=owners[:,lo:hi]==first+local
                    shared[selected]=probs[:,local][selected]
            else:
                values=v[:,g:g+1].to(dtype)
                output[:,lo:hi,first:first+group]=torch.matmul(probs,values).transpose(1,2).to(q.dtype)
            del queries,keys,scores,probs
        if shared is not None:
            error=torch.maximum(error,(shared.sum(-1)-1).abs().max())
            if not bool(torch.isfinite(shared).all()) or bool((shared.masked_select(~allowed)!=0).any()):
                raise ValueError('nonfinite or noncausal shared map')
            if full_map is not None:
                full_map[:,lo:hi]=shared
            for g in range(hk):
                probabilities=shared[:,None].expand(-1,group,-1,-1)
                output[:,lo:hi,g*group:(g+1)*group]=torch.matmul(
                    probabilities,v[:,g:g+1].to(dtype)).transpose(1,2).to(q.dtype)
    max_error=float(error)
    if max_error>2e-5 or not bool(torch.isfinite(output).all()):
        raise ValueError('normalization/finite-output failure')
    return output.flatten(-2),full_map,dict(mode=mode,heads=h,kv_heads=hk,
        full_maps_computed=h,score_elements=b*h*n*n,full_causal_pairs=b*h*n*(n+1)//2,
        stitched_causal_pairs=b*n*(n+1)//2 if mode=='shared_map' else None,
        rows_per_head=[(owners==head).sum(-1).tolist() for head in range(h)],
        max_row_sum_error=max_error,query_block_size=block_size,all_keys_per_block=n)
