import json
import math
from pathlib import Path
import torch
from selector import probability_mean, choose, ProbabilitySelector
from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
checks = []
for n in (1, 127, 128, 129, 385):
    # Head-major allocation deliberately exercises non-contiguous BNHD strides.
    q = torch.randn(1, 8, n, 64, device='cuda', dtype=torch.bfloat16).transpose(1,2)
    k = torch.randn(1, 2, n, 64, device='cuda', dtype=torch.bfloat16).transpose(1,2)
    for h in (0,3,4,7):
        scores = q[0,:,h].float() @ k[0,:,h//4].float().T / 8
        causal = torch.arange(n,device='cuda')[:,None] >= torch.arange(n,device='cuda')[None,:]
        attention = scores.masked_fill(~causal,-torch.inf).softmax(-1)
        ref = torch.stack([attention[a:min(a+128,n)].mean(0) for a in range(0,n,128)])
        got = probability_mean(q,k,h,h//4)
        err = (got-ref).abs().max().item()
        assert torch.allclose(got,ref,atol=2e-5,rtol=2e-3), (n,h,err)
        assert torch.allclose(got.sum(-1),torch.ones(got.shape[0],device='cuda'),atol=2e-5)
        for r in range(got.shape[0]):
            assert torch.count_nonzero(got[r,min((r+1)*128,n):]) == 0
        for budget in (None,256):
            target, final, protected = choose(got,budget=budget,minimum=16)
            assert (final | protected).equal(final)
            assert not (final & ~causal[torch.tensor([min(a+127,n-1) for a in range(0,n,128)],device='cuda')]).any()
            if budget is not None:
                want=torch.tensor([min(a+128,n,budget) for a in range(0,n,128)],device='cuda')
                assert final.sum(-1).equal(want)
            else:
                assert torch.all((got*target).sum(-1)>=.99-2e-5)
                sorted_p=got.sort(-1,descending=True).values
                for r,count in enumerate(target.sum(-1).tolist()):
                    if count>16: assert sorted_p[r,:count-1].sum()<.99+2e-5
        checks.append({'n':n,'head':h,'max_probability_error':err})

# Test actual shared/per-head CSR, original positions and GQA kernel mapping.
n=385
q=torch.randn(1,n,8,128,device='cuda',dtype=torch.bfloat16)
k=torch.randn(1,n,2,128,device='cuda',dtype=torch.bfloat16)
v=torch.randn_like(k)
config={'layers':{'0':[{'representative':6,'members':[0,2,4,6]}, {'representative':1,'members':[1,3,5,7]}]}}
for independent in (False,True):
    selector=ProbabilitySelector(config,per_head=independent,budget=256)
    selector.current_layer.set(0)
    index=selector(q,k,v,128,.95,1,100)
    out=triton_token_compacted_prefill_attention(q,k,v,index,token_chunk_size=32)
    for h in range(8):
        group=int(index.head_to_group[h]); source=h if independent else config['layers']['0'][group]['representative']
        p=probability_mean(q,k,source,source//4)
        _,expected,_=choose(p,budget=256)
        for r in range(4):
            lo=int(index.row_starts[0,group,r]); hi=int(index.row_ends[0,group,r])
            ids=index.token_indices[lo:hi].long()
            assert ids.equal(expected[r].nonzero().flatten())
        scores=q[0,:,h].float() @ k[0,:,h//4].float().T / math.sqrt(128)
        mask=expected[torch.arange(n,device='cuda')//128] & (torch.arange(n,device='cuda')[:,None]>=torch.arange(n,device='cuda')[None,:])
        ref=scores.masked_fill(~mask,-torch.inf).softmax(-1) @ v[0,:,h//4].float()
        relative=float((out[0,:,h].float()-ref).norm()/ref.norm())
        assert relative<.005, relative
    checks.append({'per_head':independent,'kernel':'PASS'})

p0=probability_mean(q,k,0,0)
q2=q.clone(); q2[:,:,1]*=3
assert torch.equal(p0,probability_mean(q2,k,0,0))
assert not torch.allclose(probability_mean(q,k,1,0),probability_mean(q2,k,1,0))
n=12000
end=(torch.arange(math.ceil(n/128),device='cuda')+1).mul(128).clamp(max=n)
legal=torch.arange(n,device='cuda')[None,:]<end[:,None]
uniform=legal.float()/end[:,None]
target,final,_=choose(uniform,budget=None)
assert target.sum(-1)[-1]>=11880 and final.sum(-1)[-1]>10240
assert target.sum(-1)[7]==1024
assert torch.all((target & final)==target)
checks.append({'topp_unbounded_and_minimum':'PASS'})
Path(__file__).with_name('correctness.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
print('PASS',json.dumps(checks))
