"""Owner-row top-p mask shared across heads, with head-specific attention."""
import math
import torch


def nucleus_mask(probabilities, allowed, p):
    if not 0 < p <= 1:
        raise ValueError('top-p must be in (0,1]')
    if probabilities.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError('mask shape/dtype mismatch')
    if not bool(allowed.any(-1).all()) or not bool(torch.isfinite(probabilities).all()):
        raise ValueError('empty causal row or nonfinite probabilities')
    if bool((probabilities < 0).any()) or bool((probabilities[~allowed] != 0).any()):
        raise ValueError('invalid probabilities')
    if p == 1:
        mask = allowed.clone()
        previous = torch.zeros_like(probabilities[..., 0], dtype=torch.float64)
    else:
        # Stable sort breaks equal-probability ties by ascending key position.
        values, indices = probabilities.sort(dim=-1, descending=True, stable=True)
        cumulative = values.cumsum(-1, dtype=torch.float64)
        previous_mass = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1)
        keep = (previous_mass < p) & allowed.gather(-1, indices)
        mask = torch.zeros_like(allowed).scatter_(-1, indices, keep)
        counts = keep.sum(-1)
        previous = previous_mass.gather(-1, (counts-1).unsqueeze(-1)).squeeze(-1)
    mass = (probabilities.double()*mask).sum(-1)
    if not bool(mask.any(-1).all()) or bool((mask & ~allowed).any()):
        raise ValueError('invalid selected mask')
    if bool((mass < p-2e-5).any()) or (p < 1 and bool((previous >= p).any())):
        raise ValueError('top-p coverage/minimal-prefix failure')
    return mask, mass, previous


def full_attention(q, k, v, positions, mode='shared_topp', block_size=256, return_map=False, top_p=0.95):
    if mode not in ('shared_topp', 'dense_full') or type(block_size) is not int or block_size < 1:
        raise ValueError('invalid mode or block size')
    if not 0 < top_p <= 1:
        raise ValueError('invalid top-p')
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('expected Q B,H,N,d and K/V B,Hkv,N,d')
    b,h,n,d = q.shape
    hk = k.shape[1]
    if min(b,h,n,d,hk) < 1 or h % hk or k.shape != (b,hk,n,d):
        raise ValueError('invalid GQA shapes')
    if positions.shape != (b,n) or positions.dtype not in (torch.int32,torch.int64):
        raise ValueError('integer original positions required')
    if any(t.device != q.device for t in (k,v,positions)) or q.dtype != k.dtype or q.dtype != v.dtype or not q.is_floating_point():
        raise ValueError('device/dtype mismatch')
    if bool((positions < 0).any()) or bool((positions[:,1:] <= positions[:,:-1]).any()):
        raise ValueError('positions must strictly increase')
    if not all(bool(torch.isfinite(t).all()) for t in (q,k,v)):
        raise ValueError('nonfinite QKV')
    if return_map and n > 4096:
        raise ValueError('materialized mask is for small parity tests only')
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    group = h//hk
    output = torch.empty((b,n,h,d), dtype=q.dtype, device=q.device)
    full_mask = torch.empty((b,n,n), dtype=torch.bool, device=q.device) if return_map and mode == 'shared_topp' else None
    error = torch.zeros((), dtype=dtype, device=q.device)
    owners = positions.remainder(h)
    selected_pairs = 0
    min_mass, max_previous, mass_sum = 1.0, 0.0, 0.0
    for lo in range(0,n,block_size):
        hi = min(lo+block_size,n)
        allowed = positions[:,lo:hi,None] >= positions[:,None,:]
        owned = torch.zeros((b,hi-lo,n), dtype=dtype, device=q.device) if mode == 'shared_topp' else None
        for g in range(hk):
            first = g*group
            queries = q[:,first:first+group,lo:hi].to(dtype)
            keys = k[:,g:g+1].to(dtype)
            scores = torch.matmul(queries,keys.transpose(-1,-2)) / math.sqrt(d)
            scores.masked_fill_(~allowed[:,None],float('-inf'))
            probs = scores.softmax(-1)
            error = torch.maximum(error,(probs.sum(-1)-1).abs().max())
            if owned is not None:
                for local in range(group):
                    selected = owners[:,lo:hi] == first+local
                    owned[selected] = probs[:,local][selected]
            else:
                values = v[:,g:g+1].to(dtype)
                output[:,lo:hi,first:first+group] = torch.matmul(probs,values).transpose(1,2).to(q.dtype)
            del queries,keys,scores,probs
        if owned is not None:
            mask,mass,previous = nucleus_mask(owned,allowed,top_p)
            selected_pairs += int(mask.sum())
            min_mass = min(min_mass,float(mass.min()))
            max_previous = max(max_previous,float(previous.max()))
            mass_sum += float(mass.sum())
            del owned,mass,previous
            if full_mask is not None:
                full_mask[:,lo:hi] = mask
            # Share support only. Recompute each head's own scores and softmax;
            # no averaging, probability copying, or repetition of head outputs.
            for g in range(hk):
                first = g*group
                scores = torch.matmul(q[:,first:first+group,lo:hi].to(dtype),
                    k[:,g:g+1].to(dtype).transpose(-1,-2)) / math.sqrt(d)
                scores.masked_fill_(~mask[:,None],float('-inf'))
                probs = scores.softmax(-1)
                error = torch.maximum(error,(probs.sum(-1)-1).abs().max())
                output[:,lo:hi,first:first+group] = torch.matmul(
                    probs,v[:,g:g+1].to(dtype)).transpose(1,2).to(q.dtype)
                del scores,probs
    max_error = float(error)
    if max_error > 2e-5 or not bool(torch.isfinite(output).all()):
        raise ValueError('normalization/finite-output failure')
    shared = mode == 'shared_topp'
    return output.flatten(-2), full_mask, dict(mode=mode,heads=h,kv_heads=hk,
        full_maps_computed=h,score_elements=b*h*n*n,
        application_score_elements=b*h*n*n if shared else 0,
        full_causal_pairs=b*h*n*(n+1)//2,
        stitched_causal_pairs=b*n*(n+1)//2 if shared else None,
        rows_per_head=[(owners==head).sum(-1).tolist() for head in range(h)],
        max_row_sum_error=max_error,query_block_size=block_size,all_keys_per_block=n,
        top_p=top_p if shared else None,mask_selected_pairs=selected_pairs if shared else None,
        mask_density=selected_pairs/(b*n*(n+1)//2) if shared else None,
        owner_min_retained_mass=min_mass if shared else None,
        owner_mean_retained_mass=mass_sum/(b*n) if shared else None,
        owner_max_prefix_before_last=max_previous if shared else None,
        own_qk_renormalized=shared,tie_break='ascending original key position' if shared else None)
