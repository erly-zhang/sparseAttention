"""Diagnose frozen model parity without changing the failed admission gate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import worker as w
import torch
import numpy as np
from transformers import DynamicCache
from flash_attn import flash_attn_func
import llama_adapter
from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention

def metrics(a,b):
    a=a.float();b=b.float();delta=a-b
    return dict(max_abs=float(delta.abs().max()),mean_abs=float(delta.abs().mean()),
                rms=float(delta.square().mean().sqrt()),
                relative_rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt()),
                cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
                fraction_outside_original_gate=float((delta.abs()>(.15+.05*b.abs())).float().mean()))

c,p=w.verify();model,tok=w.load_model(c,p)
report=dict(status='diagnosing',lock_sha256=w.sha(ROOT/'lock.json'),cases=[])
qref={}
# Frozen evaluation content is used here only for short numerical checks.
item=p['evaluation'][0]
raw=np.fromfile(Path(p['input_directory'])/(item['key']+'.ids'),dtype='<i8')
cases=[('synthetic',torch.arange(513,device='cuda')[None]),
       ('natural_prefix',torch.tensor(raw[:513].copy()[None],device='cuda')),
       ('natural_tail',torch.tensor(raw[-513:].copy()[None],device='cuda'))]
with torch.inference_mode():
 for name,ids in cases:
    kwargs=dict(input_ids=ids,position_ids=torch.arange(ids.shape[1],device='cuda')[None],attention_mask=torch.ones_like(ids),use_cache=True,return_dict=True)
    captured={}
    hooks=[]
    for i,layer in enumerate(model.model.layers):
        def hook(module,args,output,i=i):captured[i]=output[0].detach().clone()
        hooks.append(layer.register_forward_hook(hook))
    native=model.model(past_key_values=DynamicCache(),**kwargs).last_hidden_state
    for hook in hooks:hook.remove()
    controls={};kernel_layers={}
    def dense_callback(q,k,v,*args,layer=None,**kw):
        qn,kn,vn=[x.transpose(1,2) for x in (q,k,v)]
        out=flash_attn_func(qn,kn,vn,causal=True)
        return out.flatten(-2),None,{}
    llama_adapter.full_attention=dense_callback
    state=llama_adapter.install(model,layers=c['layers'])
    try: adapter_fa=model.model(past_key_values=DynamicCache(),**kwargs).last_hidden_state
    finally:state['close']()
    def kernel_callback(q,k,v,*args,layer=None,**kw):
        qn,kn,vn=[x.transpose(1,2) for x in (q,k,v)]
        s=w.make_selector(c,w.read(ROOT/'groups.json'));ctx=s.current_layer.set(layer)
        try:index=s(qn,kn,vn,128,.9,1,100000)
        finally:s.current_layer.reset(ctx)
        out=triton_token_compacted_prefill_attention(qn,kn,vn,index,token_chunk_size=32)
        reference=flash_attn_func(qn,kn,vn,causal=True)
        # Independent literal dense mask reference, with FP32 accumulation.
        logits=qn.permute(0,2,1,3).float()@kn.repeat_interleave(4,dim=2).permute(0,2,3,1).float()/(128**.5)
        logits.masked_fill_(~torch.ones(ids.shape[1],ids.shape[1],dtype=torch.bool,device='cuda').tril(),-torch.inf)
        oracle=(logits.softmax(-1)@vn.repeat_interleave(4,dim=2).permute(0,2,1,3).float()).permute(0,2,1,3)
        kernel_layers[layer]=dict(kernel_vs_fa2=metrics(out,reference),kernel_vs_fp32=metrics(out,oracle),fa2_vs_fp32=metrics(reference,oracle))
        return out.flatten(-2),None,{}
    llama_adapter.full_attention=kernel_callback
    state=llama_adapter.install(model,layers=c['layers'])
    try: kernel=model.model(past_key_values=DynamicCache(),**kwargs).last_hidden_state
    finally:state['close']()
    a=model.lm_head(native[:,-1]).float();b=model.lm_head(kernel[:,-1]).float()
    na=a.argmax(-1);nb=b.argmax(-1)
    row=dict(case=name,adapter_fa2_vs_native=metrics(adapter_fa,native),kernel_vs_native=metrics(kernel,native),
             logits=metrics(b,a),native_top1=int(na[0]),kernel_top1=int(nb[0]),
             native_top10=a.topk(10).indices[0].tolist(),kernel_top10=b.topk(10).indices[0].tolist(),
             logit_kl=float(torch.nn.functional.kl_div(b.log_softmax(-1),a.softmax(-1),reduction='batchmean')),
             layer_kernel_metrics=kernel_layers)
    report['cases'].append(row)
    w.save(ROOT/'diagnostics'/'dense-parity.json',report)
    print(name,row['adapter_fa2_vs_native'],row['kernel_vs_native'],row['logits'],flush=True)
report['status']='complete'
w.save(ROOT/'diagnostics'/'dense-parity.json',report)
