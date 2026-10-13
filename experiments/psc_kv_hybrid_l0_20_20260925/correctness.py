"""GPU-only tiny32-layer integration; actual production split at20/21."""
import copy,json,os
from pathlib import Path
import torch
from transformers import LlamaConfig,LlamaForCausalLM,DynamicCache
from hybrid_adapter import install,base

def main():
    assert os.environ.get('SLURM_JOB_ID') and 'H100' in torch.cuda.get_device_name()
    torch.manual_seed(42);torch.set_num_threads(4)
    cfg=LlamaConfig(hidden_size=512,intermediate_size=768,num_hidden_layers=32,num_attention_heads=4,
        num_key_value_heads=2,vocab_size=64,max_position_embeddings=512,torch_dtype=torch.bfloat16)
    cfg._attn_implementation='flash_attention_2'
    model=LlamaForCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16).eval()
    all_c=copy.deepcopy(model); all_c,_=base.install(all_c,'routing_c')
    original=[x.self_attn.forward for x in model.model.layers]
    model,state=install(model)
    assert all(model.model.layers[i].self_attn.forward==original[i] for i in range(21,32))
    def run(m,ids):
        captures={};handles=[];first_dense_inputs=[]
        for i,layer in enumerate(m.model.layers):
            def capture(_,args,out,index=i):captures[index]=out[0].detach().clone()
            handles.append(layer.register_forward_hook(capture))
        # Block input before normalization, so it must equal prior block output.
        def pre(_,args,kwargs):first_dense_inputs.append(kwargs.get('hidden_states',args[0] if args else None).detach().clone())
        handles.append(m.model.layers[21].register_forward_pre_hook(pre,with_kwargs=True))
        cache=DynamicCache()
        with torch.inference_mode():
            output=m.model(input_ids=ids,position_ids=torch.arange(ids.shape[1],device='cuda')[None],
                attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=True).last_hidden_state
        for h in handles:h.remove()
        assert len(cache.key_cache)==32
        assert all(tuple(k.shape)==(1,2,ids.shape[1],128) for k in cache.key_cache)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(first_dense_inputs[0],captures[20],rtol=0,atol=0)
        return output,captures,cache
    ids=torch.arange(65,device='cuda').remainder(63)[None]
    full,caps_full,cache_full=run(all_c,ids)
    hybrid,caps,cache=run(model,ids)
    for i in range(21):
        torch.testing.assert_close(caps[i],caps_full[i],rtol=0,atol=0)
        torch.testing.assert_close(cache.key_cache[i],cache_full.key_cache[i],rtol=0,atol=0)
        torch.testing.assert_close(cache.value_cache[i],cache_full.value_cache[i],rtol=0,atol=0)
    assert not torch.equal(caps[21],caps_full[21])
    changed=ids.clone();changed[:,33:]=(changed[:,33:]+7)%63
    altered,_,_=run(model,changed)
    torch.testing.assert_close(hybrid[:,:33],altered[:,:33],rtol=.01,atol=.02)
    result={'status':'passed','synthetic_fixture':True,'layers':32,'split_zero_based':[20,21],
        'prefix21_layers_and_KV_bitwise_equal_to_full_C':True,'tail_original_forwards_preserved':True,
        'dense_tail_uses_own_prefix_output':True,'full_prompt_cache':True,'future_invariance':True}
    run_dir=Path(os.environ['PSC_RUN_DIR']);(run_dir/'correctness.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':main()
