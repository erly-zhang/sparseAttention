import os
"""One method/process, genuine end-to-end prefill; two explicitly separate passes."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, DynamicCache
from adapters import install, METHODS, ROOT
sys.path.insert(0,str(ROOT/'experiments/prefill_token_head_routing_20260924/layer_output_comparison'))
from metrics import compare_outputs
HERE=Path(__file__).resolve().parent
RUN=Path(os.environ.get('PSC_RUN_DIR', str(HERE/'run')))
MODEL=os.environ.get('PSC_MODEL_DIR', '/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/models/Llama-3.1-8B-Instruct')


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--groups',required=True)
    a=p.parse_args();torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    out=RUN/a.method;out.mkdir(parents=True,exist_ok=False)
    raw=(HERE/'inputs/3.ids.i64le').read_bytes()
    import numpy as np
    ids=torch.tensor(np.frombuffer(raw,dtype='<i8').copy(),device='cuda')[None]
    n=ids.shape[1];pos=torch.arange(n,device='cuda')[None];mask=torch.ones_like(ids)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2').cuda().eval()
    model,state=install(model,a.method,json.loads(Path(a.groups).read_text()))
    # Full forward warmup shape is intentionally not duplicated; small full-model warmup only.
    def run(tokens,positions,attention_mask):
        cache=DynamicCache()
        with torch.inference_mode():
            result=model.model(input_ids=tokens,position_ids=positions,attention_mask=attention_mask,
                past_key_values=cache,use_cache=True,return_dict=True)
        return result,cache
    result,cache=run(ids[:,:129],pos[:,:129],mask[:,:129]);del result,cache
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    wall=time.perf_counter();begin.record();result,cache=run(ids,pos,mask);end.record();torch.cuda.synchronize()
    timing={'cuda_elapsed_seconds':begin.elapsed_time(end)/1000,'wall_seconds':time.perf_counter()-wall,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'warmup_tokens':129,
        'first_full_shape_compilation_may_be_included':True,'diagnostic_pass_excluded':True,
        'cache_shapes':[list(x.shape) for x in cache.key_cache]}
    assert all(x==[1,8,n,128] for x in timing['cache_shapes'])
    (out/'timing.json').write_text(json.dumps(timing,indent=2)+'\n');del result,cache
    instrumentation=None
    if a.method in ('flexprefill','minference'):
        from experiments.baseline_sparsity import install_baseline_sparsity_instrumentation
        instrumentation=install_baseline_sparsity_instrumentation(model,a.method)
        instrumentation.begin_sample(call_index=0,input_ids_sha256=hashlib.sha256(raw).hexdigest(),input_tokens=n,context={'task':'kv_retrieval'})
    pairs={};previous=0
    def count_hook(layer):
        def hook(module,inp,output):
            nonlocal previous
            if instrumentation:
                instrumentation.flush_stats();now=instrumentation.stats.selected_token_pairs
                pairs[layer]=now-previous;previous=now
            elif a.method=='dense':pairs[layer]=32*n*(n+1)//2
            else:pairs[layer]=int(state['pairs'][layer])
        return hook
    handles=[];rows=[]
    refdir=RUN/'dense/tensors'
    if a.method=='dense':refdir.mkdir()
    def capture(layer,point):
        def hook(module,inp,output):
            value=output[0] if isinstance(output,tuple) else output
            cpu=value.detach().cpu();path=refdir/f'{point}_{layer:02}.pt'
            if a.method=='dense':
                torch.save(cpu,path);ref=cpu
            else:ref=torch.load(path,map_location='cpu',weights_only=True)
            metrics=compare_outputs(ref,cpu,torch.ones(cpu.shape[:2],dtype=torch.bool))
            metrics['rmse']=(metrics['squared_error']/cpu.numel())**0.5
            rows.append(dict(method=a.method,layer=layer,point=point,shape=list(cpu.shape),
                source_index=3,input_ids_sha256=hashlib.sha256(raw).hexdigest(),
                trajectory='method_own_end_to_end',padding_tokens=0,**metrics))
        return hook
    for i,layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(count_hook(i)))
        handles.append(layer.self_attn.o_proj.register_forward_hook(capture(i,'attention_post_o_proj')))
        handles.append(layer.register_forward_hook(capture(i,'block_output')))
    diag=time.perf_counter();result,cache=run(ids,pos,mask);torch.cuda.synchronize()
    for h in handles:h.remove()
    for row in rows:row['retained_causal_pairs']=pairs[row['layer']]
    assert len(rows)==64 and len(pairs)==32
    (out/'layers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'diagnostics.json').write_text(json.dumps({'diagnostic_seconds':time.perf_counter()-diag,'not_pure_inference_timing':True},indent=2))
    print(a.method,'completed',flush=True)
if __name__=='__main__':main()
