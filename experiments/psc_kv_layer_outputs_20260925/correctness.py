import os
"""Single-H100 gates. Synthetic fixtures are never used to produce AE groups."""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import LlamaConfig,LlamaForCausalLM,DynamicCache
from adapters import install, METHODS, ROOT
HERE=Path(__file__).resolve().parent
RUN=Path(os.environ.get('PSC_RUN_DIR', str(HERE/'run')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);a=p.parse_args()
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()[0]>=9
    torch.manual_seed(42);torch.set_num_threads(4)
    cfg=LlamaConfig(hidden_size=512,intermediate_size=768,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=64,max_position_embeddings=512)
    cfg._attn_implementation='flash_attention_2'
    model=LlamaForCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16).eval()
    # Fixed structural fixture, never an experimental/calibration group file.
    groups={'layers':{str(i):[{'representative':0,'members':[0,3]},
        {'representative':1,'members':[1]},{'representative':2,'members':[2]}] for i in range(2)}}
    ids=torch.arange(129,device='cuda').remainder(63)[None]
    def forward(m,tokens):
        cache=DynamicCache()
        with torch.inference_mode():
            out=m.model(tokens,position_ids=torch.arange(tokens.shape[1],device='cuda')[None],
                        attention_mask=torch.ones_like(tokens),past_key_values=cache,use_cache=True)
        assert all(tuple(x.shape)==(1,2,tokens.shape[1],128) for x in cache.key_cache)
        return out.last_hidden_state
    expected=forward(model,ids)
    model,state=install(model,a.method,groups)
    actual=forward(model,ids)
    assert torch.isfinite(actual).all()
    changed=ids.clone();changed[:,65:]=(changed[:,65:]+7)%63
    altered=forward(model,changed)
    torch.testing.assert_close(actual[:,:65],altered[:,:65],rtol=.01,atol=.02)
    if a.method!='routing_c':
        rel=float((actual.float()-expected.float()).norm()/expected.float().norm())
        assert rel<.02,rel  # These short fixtures are dense fallback / full budget.
    else:
        from routing import routed_attention,repeat_project
        for n in (1,3,17,129):
            q=torch.randn(1,4,n,128,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(1,2,n,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
            pos=torch.arange(n,device='cuda')[None];valid=torch.ones_like(pos,dtype=torch.bool)
            m,counts=routed_attention(q,k,v,pos,valid,'q_kv',128**-.5,backend='flash_attention_2')
            oracle=torch.empty_like(m,dtype=torch.float32);pairs=0
            for i in range(n):
                h=i%4;keys=list(range(h,i+1,4));pairs+=len(keys)
                prob=(q[0,h,i].float()@k[0,h//2,keys].float().T/128**.5).softmax(-1)
                oracle[0,i]=prob@v[0,h//2,keys].float()
            assert counts['eligible_pairs']==pairs
            assert float((m.float()-oracle).norm()/oracle.norm())<.006
            proj=torch.nn.Linear(512,512,bias=True).cuda().bfloat16()
            explicit=repeat_project(m,proj,4)
            expected_proj=F.linear(torch.cat([m]*4,-1),proj.weight,proj.bias)
            torch.testing.assert_close(explicit,expected_proj,rtol=0,atol=0)
        rel=None
    if a.method=='shared6144_member2q4096':
        from selector6040 import Selector
        from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
        for n in (129,513,641):
            q=torch.randn(1,n,4,128,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(1,n,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
            sel=Selector(groups,'member2q',budget=512,core=256)
            sel.current_layer.set(0);sel.capture=True
            index=sel(q,k,v,128,.95,1,None)
            out=triton_token_compacted_prefill_attention(q,k,v,index,token_chunk_size=32)
            pos=torch.arange(n,device='cuda')
            for layer,h,rep,core,final,old,phat in sel.captured:
                ends=((torch.arange(len(final),device='cuda')+1)*128).clamp(max=n)
                assert torch.equal((final<ends[:,None]).sum(-1),ends.clamp(max=512))
                fm=torch.zeros(len(final),n+1,device='cuda',dtype=torch.bool).scatter_(1,final.long(),True)[:,:n]
                mask=fm[pos//128] & (pos[:,None]>=pos[None,:])
                prob=(q[0,:,h].float()@k[0,:,h//2].float().T/128**.5).masked_fill(~mask,-torch.inf).softmax(-1)
                ref=prob@v[0,:,h//2].float()
                assert float((out[0,:,h].float()-ref).norm()/ref.norm())<.005
                for r in range(len(final)):
                    visible=set(final[r][final[r]<ends[r]].tolist())
                    core_set=set(core[r][core[r]<ends[r]].tolist())
                    assert core_set<=visible
                    if h!=rep and int(ends[r])>512:
                        candidates=[j for j in range(int(ends[r])) if j not in core_set]
                        expected=set(sorted(candidates,key=lambda j:(-float(phat[r,j]),j))[:256])
                        assert visible-core_set==expected
    result={'method':a.method,'status':'passed','fixture_only':True,'full_prompt_cache':True,
            'future_invariance':True,'short_dense_relative_l2':rel}
    (RUN/f'correctness_{a.method}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
