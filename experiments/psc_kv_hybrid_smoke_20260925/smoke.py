"""Hybrid prefill followed by original dense decode, own full KV cache."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,LlamaConfig,LlamaForCausalLM,DynamicCache
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'psc_kv_layer_outputs_20260925'
sys.path.insert(0,str(HERE.parent/'psc_kv_hybrid_l0_20_20260925'))
from hybrid_adapter import install
RUN=HERE/'run'

def save(name,value): (RUN/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

@torch.inference_mode()
def generate(model,ids,max_new,eos):
    original=[layer.self_attn.forward for layer in model.model.layers]
    model,state=install(model)
    cache=DynamicCache();n=ids.shape[1];generated=[];calls=[0]*32
    def cache_check(length):
        assert len(cache.key_cache)==32
        for k,v in zip(cache.key_cache,cache.value_cache):
            assert k.shape==v.shape and k.shape[2]==length and k.shape[1]==model.config.num_key_value_heads
    def step(tokens,pos):
        result=model.model(input_ids=tokens,position_ids=pos,attention_mask=torch.ones((1,int(pos[0,-1])+1),dtype=torch.long,device='cuda'),
            past_key_values=cache,use_cache=True,return_dict=True)
        # Only project the final position, avoiding prompt_length*vocab_size logits.
        logits=model.lm_head(result.last_hidden_state[:,-1:]).float()
        assert torch.isfinite(logits).all()
        return int(logits[0,0].argmax())
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    try:
        token=step(ids,torch.arange(n,device='cuda')[None]);torch.cuda.synchronize()
        prefill=time.perf_counter()-start;cache_check(n)
    finally:
        for i,layer in enumerate(model.model.layers):layer.self_attn.forward=original[i]
    assert all(layer.self_attn.forward==original[i] for i,layer in enumerate(model.model.layers))
    handles=[]
    for i,layer in enumerate(model.model.layers):
        def hook(_,args,out,index=i):calls[index]+=1
        handles.append(layer.self_attn.register_forward_hook(hook))
    decode_start=time.perf_counter();reason='max_new_tokens';steps=0
    try:
        for i in range(max_new):
            generated.append(token)
            if token in eos:reason='eos_token';break
            if i+1==max_new:break
            token=step(torch.tensor([[token]],device='cuda'),torch.tensor([[n+i]],device='cuda'))
            steps+=1;cache_check(n+steps)
        torch.cuda.synchronize();decode=time.perf_counter()-decode_start
    finally:
        for h in handles:h.remove()
    assert all(c==steps for c in calls)
    return {'generated_ids':generated,'generated_tokens':len(generated),'stop_reason':reason,'terminated':reason=='eos_token',
        'terminated_definition':'generation selected a model EOS token; length cap is not terminated',
        'dense_decode_steps':steps,'dense_decode_layer_calls':calls,'cache_length':n+steps,'cache_shapes':[list(k.shape) for k in cache.key_cache],
        'prefill_seconds':prefill,'decode_seconds':decode,'total_generation_seconds':prefill+decode,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'timing_includes_cache_checks_and_greedy_selection':True,
        'timing_excludes_loading_warmup':True,'no_KL_computation':True}

def main():
    assert os.environ.get('SLURM_JOB_ID') and 'H100' in torch.cuda.get_device_name()
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    cfg=LlamaConfig(hidden_size=512,intermediate_size=768,num_hidden_layers=32,num_attention_heads=4,num_key_value_heads=2,
        vocab_size=64,max_position_embeddings=512,torch_dtype=torch.bfloat16)
    cfg._attn_implementation='flash_attention_2'
    tiny=LlamaForCausalLM(cfg).cuda().bfloat16().eval()
    check=generate(tiny,torch.arange(33,device='cuda')[None],5,[])
    assert check['dense_decode_steps']==4 and all(x==4 for x in check['dense_decode_layer_calls'])
    save('correctness.json',{'status':'passed','tiny32layer_hybrid_prefill_then4dense_decode_steps':True,'full_cache_growth':True,'details':check})
    del tiny;torch.cuda.empty_cache()
    path=os.environ['PSC_MODEL_DIR']
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    raw=(BASE/'inputs/3.ids.i64le').read_bytes();ids=torch.tensor(np.frombuffer(raw,dtype='<i8').copy(),device='cuda')[None]
    assert ids.shape==(1,123959)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    # Same procedure warmup, then fresh full-input cache for measured generation.
    generate(model,ids[:,:129],3,[])
    result=generate(model,ids,128,eos)
    result.update(method='C_layers0_20_dense_layers21_31_prefill_all_dense_decode',input_source_index=3,input_tokens=ids.shape[1],
        input_ids_sha256=hashlib.sha256(raw).hexdigest(),seed=42,do_sample=False,max_new_tokens=128,eos_token_ids=eos,
        prediction=tok.decode(result['generated_ids'],skip_special_tokens=True),raw_prediction=tok.decode(result['generated_ids'],skip_special_tokens=False),
        exploratory_posthoc=True,not_formal_accuracy_evaluation=True)
    # Record ground truth after generation, without feeding it to model or adapting config.
    with (BASE/'data/kv_retrieval.jsonl').open() as f:
        for i,line in enumerate(f):
            if i==3:result['reference_answers']=json.loads(line)['answer'];break
    save('generation.json',result)
    (RUN/'prediction.txt').write_text(result['prediction']+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
