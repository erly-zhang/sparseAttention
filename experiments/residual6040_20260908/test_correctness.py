"""Independent small-reference gates, before any full generation."""
import hashlib
import json
import math
from pathlib import Path
import torch
from selector6040 import Selector
from member_score import member_probability,sample_positions
from indices import shared_indices,random_residual,assemble,scored_residual
from experiments.probmean_compact_20260906.selector import probability_mean,choose
from experiments.token_compacted_sparse import TokenCompactedIndex,triton_token_compacted_prefill_attention

RELATIVE_LIMIT=.005
ABSOLUTE_LIMIT=.02


def reference(q,k,h,kh,sampled):
    n=q.shape[1];out=[]
    for a in range(0,n,128):
        b=min(a+128,n);positions=sample_positions(a,b) if sampled else list(range(a,b))
        logits=q[0,positions,h].float()@k[0,:,kh].float().T/math.sqrt(128)
        mask=torch.arange(n,device=q.device)[None,:]<=torch.tensor(positions,device=q.device)[:,None]
        probs=logits.masked_fill(~mask,-torch.inf).softmax(-1)
        assert torch.allclose(probs.sum(-1),torch.ones(len(positions),device=q.device),atol=1e-6)
        out.append(probs.mean(0))
    return torch.stack(out)


def main():
    torch.manual_seed(42);checks=[];max_rel=0.;max_abs=0.;max_p=0.
    config={'layers':{'0':[{'representative':4,'members':[0,4,5]},
                         {'representative':2,'members':[1,2,6]},
                         {'representative':7,'members':[3,7]}]}}
    for n in (1,127,128,129,255,257,511,512,513,641):
        # Slice a padded allocation: padding must not enter query averages.
        q=torch.randn(1,8,n+9,128,device='cuda',dtype=torch.bfloat16).transpose(1,2)[:,:n]
        k=torch.randn(1,2,n+9,128,device='cuda',dtype=torch.bfloat16).transpose(1,2)[:,:n];v=torch.randn_like(k)
        for interleave in (False,True):
            kv=lambda h:h%2 if interleave else h//4
            for h in (0,1,4,7):
                pp=member_probability(q,k,h,kv(h));expected=reference(q,k,h,kv(h),True)
                error=float((pp-expected).abs().max());max_p=max(max_p,error)
                assert error<2e-5,error
                assert torch.allclose(pp.sum(-1),torch.ones(len(pp),device='cuda'),atol=1e-5)
                for r,row in enumerate(pp):assert not row[min((r+1)*128,n):].any()
                full=probability_mean(q,k,h,kv(h));expected_full=reference(q,k,h,kv(h),False)
                assert (full-expected_full).abs().max()<2e-5
            for mode in ('random','member2q'):
                sel=Selector(config,mode,budget=512,core=256);sel.current_layer.set(0)
                sel.capture=True;sel.input_hash='fixed-reference-id';before=torch.cuda.get_rng_state().clone()
                index=sel(q,k,v,128,.95,1,None,gqa_interleave=interleave)
                assert torch.equal(before,torch.cuda.get_rng_state())
                out=triton_token_compacted_prefill_attention(q,k,v,index,token_chunk_size=32,gqa_interleave=interleave)
                parts=[];lengths=[]
                for layer,h,rep,core,final,old,phat in sel.captured:
                    p=probability_mean(q,k,rep,kv(rep));_,oldmask,protection=choose(p,budget=512)
                    end=(torch.arange(len(final),device='cuda')+1).mul(128).clamp(max=n)
                    fm=torch.zeros((len(final),n+1),device='cuda',dtype=torch.bool).scatter_(1,final.long(),True)[:,:n]
                    assert torch.equal(fm.sum(-1),end.clamp(max=512))
                    assert (fm|protection).equal(fm)
                    if h==rep:assert fm.equal(oldmask)
                    for r in range(len(final)):
                        ids=final[r][final[r]<end[r]]
                        assert len(ids)==len(ids.unique())
                        assert bool((ids[1:]>ids[:-1]).all())
                        ci=core[r][core[r]<end[r]]
                        assert set(ci.tolist())<=set(ids.tolist())
                        if mode=='member2q' and h!=rep and int(end[r])>512:
                            candidates=[j for j in range(int(end[r])) if j not in set(ci.tolist())]
                            expected=sorted(candidates,key=lambda j:(-float(phat[r,j]),j))[:256]
                            assert set(ids.tolist())-set(ci.tolist())==set(expected)
                    pos=torch.arange(n,device='cuda');mask=fm[pos//128]&(pos[:,None]>=pos[None,:])
                    logits=q[0,:,h].float()@k[0,:,kv(h)].float().T/math.sqrt(128)
                    ref=logits.masked_fill(~mask,-torch.inf).softmax(-1)@v[0,:,kv(h)].float()
                    rel=float((out[0,:,h].float()-ref).norm()/ref.norm());ab=float((out[0,:,h].float()-ref).abs().max())
                    assert rel<=RELATIVE_LIMIT and ab<=ABSOLUTE_LIMIT,(rel,ab)
                    max_rel=max(max_rel,rel);max_abs=max(max_abs,ab)
                # Original ragged CSR vs optimized fixed-stride short-index CSR.
                width=min(512,n);flat=index.token_indices.view(8,-1,width)
                for h in range(8):
                    for r in range(flat.shape[1]):
                        valid=flat[h,r]<min((r+1)*128,n)
                        parts.append(flat[h,r][valid]);lengths.append(int(valid.sum()))
                counts=torch.tensor(lengths,device='cuda');finish=counts.cumsum(0)
                original=TokenCompactedIndex((finish-counts).view(1,8,-1),finish.view(1,8,-1),torch.cat(parts),index.head_to_group,8,len(final),128)
                other=triton_token_compacted_prefill_attention(q,k,v,original,token_chunk_size=32,gqa_interleave=interleave)
                assert torch.equal(other,out)
        checks.append({'length':n,'gqa_contiguous_and_interleaved':True,'both_methods':True})
    n=10503;end=(torch.arange((n+127)//128,device='cuda')+1).mul(128).clamp(max=n)
    p=(torch.arange(n,device='cuda')[None,:]<end[:,None]).float()/end[:,None]
    bundle=shared_indices(p);first=random_residual(bundle,'identity',42,17,21)
    assert first.equal(random_residual(bundle,'identity',42,17,21))
    assert not first.equal(random_residual(bundle,'identity',42,17,22))
    other={k:v.to('cuda:1') if isinstance(v,torch.Tensor) else v for k,v in bundle.items()}
    with torch.cuda.device(1):second=random_residual(other,'identity',42,17,21)
    assert first.cpu().equal(second.cpu())
    final=assemble(bundle,first)
    assert torch.equal((final<end[:,None]).sum(-1),end.clamp(max=10240))
    # Every probability ties: B must choose ascending original positions.
    tied=scored_residual(p,bundle)
    for r in range(80,len(p)):
        core=set(bundle['core'][r].tolist());expected=[j for j in range(int(end[r])) if j not in core][:4096]
        assert tied[r].tolist()==expected
    # Uniformity check over unit-test streams, not extra full experiment seeds.
    from experiments.random_residual_20260908.random_policy import stable_seeds,_draw
    bounds=torch.full((512,),16,device='cuda',dtype=torch.int64);seeds=stable_seeds('unit',42,0,0,list(range(512)),'cuda')
    draws=torch.empty((512,4096),device='cuda',dtype=torch.int64);_draw[(512,)](seeds,bounds,draws,4096,4096,num_warps=8)
    hist=torch.bincount(draws[:,0],minlength=16).cpu().tolist();assert min(hist)>12 and max(hist)<55
    result={'passed':True,'checks':checks,'relative_l2_limit':RELATIVE_LIMIT,'absolute_limit':ABSOLUTE_LIMIT,
        'max_relative_l2':max_rel,'max_absolute':max_abs,'max_member2q_probability_error':max_p,
        'original_vs_optimized_index_kernel_bitwise_equal':True,'cross_gpu_mask_equal':True,
        'random_mask_sha256':hashlib.sha256(first.cpu().numpy().tobytes()).hexdigest(),'unit_uniformity_histogram':hist,
        'no_attention_kernel_modification':True}
    Path(__file__).with_name('correctness.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
