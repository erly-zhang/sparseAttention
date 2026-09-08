"""Independent reference checks for residual sets and original-position kernel."""
import json
import math
from pathlib import Path

import torch
from policy import select, prepare_group, apply_ratio
from experiments.probmean_compact_20260906.selector import probability_mean, choose
from experiments.token_compacted_sparse import TokenCompactedIndex, triton_token_compacted_prefill_attention


def main():
    torch.manual_seed(123)
    n=385;heads=8
    q=torch.randn(1,n,heads,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,n,2,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    reps=[4,2,7];member_groups=[0,1,1,2,0,0,1,2]
    ps=[probability_mean(q,k,h,h//4) for h in reps]
    checks=[]
    for kind in ('fixed','topp'):
        selections=[select(p,kind,budget=320,minimum=16) for p in ps]
        for p,s in zip(ps,selections):
            expected=choose(p,budget=320 if kind=='fixed' else None,minimum=16)
            assert all(torch.equal(a,b) for a,b in zip(s,expected))
        bundles=[prepare_group(ps,selections,g,kind,residual=64) for g in range(3)]
        final_masks=[]
        for h,g in enumerate(member_groups):
            bundle=bundles[g];quarter=h%5
            final,_=apply_ratio(bundle,quarter)
            for r in range(len(final)):
                core=set(bundle['core'][r].nonzero().flatten().tolist())
                room=int(bundle['room'][r]);quota=room*quarter//4
                a,b=[x[r].tolist() for x in bundle['lists']]
                drop=bundle['dropped'][r].tolist()
                residual=[]
                for j in (a[:quota]+b+a[quota:]+drop if room else []):
                    if j>=0 and j not in core and j not in residual:
                        residual.append(j)
                    if len(residual)==room:break
                assert set(final[r].nonzero().flatten().tolist())==core|set(residual)
            assert torch.equal(final.sum(-1),bundle['old'].sum(-1))
            assert (final|bundle['protected']).equal(final)
            assert not final[0,128:].any()
            assert apply_ratio(bundle,-1)[0].equal(bundle['old'])
            final_masks.append(final)
        counts=torch.stack([x.sum(-1) for x in final_masks]).reshape(-1)
        end=counts.cumsum(0)
        index=TokenCompactedIndex((end-counts).view(1,heads,4),end.view(1,heads,4),
            torch.cat([x.nonzero()[:,1].int() for x in final_masks]),
            torch.arange(heads,device='cuda',dtype=torch.int32),heads,4,128)
        got=triton_token_compacted_prefill_attention(q,k,v,index,token_chunk_size=32)
        errors=[]
        for h,mask in enumerate(final_masks):
            scores=q[0,:,h].float()@k[0,:,h//4].float().T/math.sqrt(128)
            pos=torch.arange(n,device='cuda')
            allowed=mask[pos//128]&(pos[:,None]>=pos[None,:])
            expected=scores.masked_fill(~allowed,-torch.inf).softmax(-1)@v[0,:,h//4].float()
            error=float((got[0,:,h].float()-expected).norm()/expected.norm())
            assert error<.005,error
            errors.append(error)
        # Sampled calibration rows must use original tile starts, not renumbering.
        subset=torch.tensor([1,3],device='cuda')
        for p,selection in zip(ps,selections):
            sampled=select(p[subset],kind,starts=subset*128,budget=320,minimum=16)
            assert all(a.equal(b[subset]) for a,b in zip(sampled,selection))
        checks.append({'kind':kind,'budget_and_unique_and_causal':'passed','max_kernel_relative_error':max(errors)})
    document={'passed':True,'checks':checks}
    Path(__file__).with_name('correctness.json').write_text(json.dumps(document,indent=2))
    print(json.dumps(document,indent=2))


if __name__=='__main__':main()
