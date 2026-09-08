"""Independent set invariants, RNG controls and explicit masked-attention reference."""
import hashlib
import json
from pathlib import Path
import math

import torch
from random_policy import prepare,random_final,stable_seeds,_draw
from random_selector import RandomSelector
from experiments.probmean_compact_20260906.selector import probability_mean,choose
from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention


def main():
    torch.manual_seed(9)
    checks=[]
    for n in (1,127,128,129,255,256,257,383,384,385,513):
        q=torch.randn(1,n,8,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,n,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        config={'layers':{'0':[{'representative':4,'members':[0,4,5]},
                               {'representative':2,'members':[1,2,6]},
                               {'representative':7,'members':[3,7]}]}}
        selector=RandomSelector(config,final_budget=384,core_budget=256)
        selector.current_layer.set(0);selector.input_hash='abc123';selector.capture=True
        before=torch.cuda.get_rng_state().clone()
        index=selector(q,k,v,128,.95,1,100)
        assert torch.equal(before,torch.cuda.get_rng_state()),'Sampler changed global RNG'
        out=triton_token_compacted_prefill_attention(q,k,v,index,token_chunk_size=32)
        for h,core,residual,final,old in selector.captured:
            group=next(g for g in config['layers']['0'] if h in g['members'])
            rep=group['representative'];p=probability_mean(q,k,rep,rep//4)
            _,expected_old,protection=choose(p,budget=384)
            _,expected_core,_=choose(p,budget=256)
            assert old.equal(expected_old)
            assert core.equal(expected_old if h==rep else expected_core)
            assert not (core&residual).any()
            assert final.equal(core|residual)
            assert (final|protection).equal(final)
            ends=torch.arange(len(final),device='cuda').add(1).mul(128).clamp(max=n)
            assert torch.equal(final.sum(-1),ends.clamp(max=384))
            if h==rep:assert not residual.any() and final.equal(old)
            else:
                assert torch.equal(core.sum(-1),ends.clamp(max=256))
                assert torch.equal(residual.sum(-1),ends.clamp(max=384)-ends.clamp(max=256))
                bundle=prepare(p,384,256)
                repeat=random_final(bundle,'abc123',42,0,h)[0]
                assert repeat.equal(final)
                if n>384:
                    if n>512:assert not random_final(bundle,'abc123',42,0,h+100)[0].equal(final)
                    # Verify first-unique stream with a separate Python set loop.
                    active=bundle['active'];seeds=stable_seeds('abc123',42,0,h,active.cpu().tolist(),'cuda')
                    draws=torch.empty((len(active),4096),device='cuda',dtype=torch.int64)
                    _draw[(len(active),)](seeds,bundle['counts'],draws,4096,4096,num_warps=8)
                    for i,r in enumerate(active.tolist()):
                        candidates=bundle['complement'][r].nonzero().flatten().tolist()
                        unique=[];seen=set()
                        for rank in draws[i].tolist():
                            if rank not in seen:unique.append(candidates[rank]);seen.add(rank)
                            if len(unique)==128:break
                        assert set(unique)==set(residual[r].nonzero().flatten().tolist())
            pos=torch.arange(n,device='cuda')
            mask=final[pos//128]&(pos[:,None]>=pos[None,:])
            scores=q[0,:,h].float()@k[0,:,h//4].float().T/math.sqrt(128)
            expected=scores.masked_fill(~mask,-torch.inf).softmax(-1)@v[0,:,h//4].float()
            err=float((out[0,:,h].float()-expected).norm()/expected.norm())
            assert err<.005,err
            g=int(index.head_to_group[h])
            for r in range(len(final)):
                ids=index.token_indices[int(index.row_starts[0,g,r]):int(index.row_ends[0,g,r])]
                assert ids.long().equal(final[r].nonzero().flatten())
        checks.append({'n':n,'invariants_and_independent_kernel':'passed'})
    # Actual budgets, partial tile, independent GPUs and call ordering.
    n=10503;ends=(torch.arange((n+127)//128,device='cuda')+1).mul(128).clamp(max=n)
    legal=torch.arange(n,device='cuda')[None,:]<ends[:,None]
    p=legal.float()/ends[:,None];bundle=prepare(p)
    one=random_final(bundle,'fixed-input-sha256',42,17,21)[0]
    random_final(bundle,'other-input',42,17,21)
    assert one.equal(random_final(bundle,'fixed-input-sha256',42,17,21)[0])
    if torch.cuda.device_count()>1:
        # Copy the same core masks to isolate RNG from score tie behavior.
        other={k:v.to('cuda:1') if isinstance(v,torch.Tensor) else v for k,v in bundle.items()}
        with torch.cuda.device(1):two=random_final(other,'fixed-input-sha256',42,17,21)[0]
        assert one.cpu().equal(two.cpu())
    else:raise RuntimeError('Two idle GPUs required for cross-GPU reproducibility test')
    digest=hashlib.sha256(one.cpu().numpy().tobytes()).hexdigest()
    # Distribution sanity, not an additional benchmark seed sweep.
    bounds=torch.full((512,),16,device='cuda',dtype=torch.int64)
    seeds=stable_seeds('uniformity-unit-test',42,0,0,list(range(512)),'cuda')
    draws=torch.empty((512,4096),device='cuda',dtype=torch.int64)
    _draw[(512,)](seeds,bounds,draws,4096,4096,num_warps=8)
    hist=torch.bincount(draws[:,0],minlength=16).cpu().tolist()
    assert min(hist)>12 and max(hist)<55,hist
    result={'passed':True,'checks':checks,'actual_budget_cross_gpu_reproducible':True,
        'mask_sha256':digest,'first_draw_histogram_16bins_512_unit_streams':hist,
        'independent_kernel_relative_tolerance':.005}
    Path(__file__).with_name('correctness.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
