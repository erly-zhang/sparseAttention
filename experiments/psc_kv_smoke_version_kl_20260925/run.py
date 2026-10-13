"""Same-L40S dense versus exact smoke-version hybrid; prefill output KL."""
import csv,hashlib,importlib.util,json,math,os,sys,time,traceback
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM,DynamicCache
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=HERE.parent/'psc_kv_layer_outputs_20260925';RUN=HERE/'run'
SMOKE=HERE.parent/'psc_kv_hybrid_smoke_20260925/vscode_l40s/smoke.py'
spec=importlib.util.spec_from_file_location('verified_smoke_version',SMOKE);smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)
sys.path.insert(0,str(HERE.parent/'prefill_token_head_routing_20260924/layer_output_comparison'))
from metrics import compare_outputs
START=time.monotonic()
def save(n,o):(RUN/n).write_text(json.dumps(o,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    RUN.mkdir(exist_ok=False)
    save('state.json',{'status':'running','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ.get('SLURM_STEP_ID')})
    try:
        assert torch.cuda.device_count()==1 and 'L40S' in torch.cuda.get_device_name()
        torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
        local=Path('/local/psc-hybrid-smoke-47082259-sbm8_zpy')
        prov=json.loads((BASE/'model_provenance.json').read_text())
        hashes={name:sha(local/name) for name in prov['files']}
        assert all(hashes[name]==v['sha256'] for name,v in prov['files'].items())
        raw=(BASE/'inputs/3.ids.i64le').read_bytes();identity=hashlib.sha256(raw).hexdigest()
        expected=json.loads((BASE/'inputs/manifest.json').read_text())['records'][3]
        assert identity==expected['input_ids_sha256']
        ids=torch.tensor(np.frombuffer(raw,dtype='<i8').copy(),device='cuda')[None];n=ids.shape[1];assert n==123959
        save('provenance.json',{'gpu':torch.cuda.get_device_name(),'node':os.uname().nodename,'torch':torch.__version__,
            'input_index':3,'input_ids_sha256':identity,'input_tokens':n,'model_files':hashes,'model_revision':prov['revision'],
            'smoke_source_sha256':sha(SMOKE),'same_inference_MLP_buffer_reuse_both_conditions':True,
            'hybrid_routing_layers':list(range(21)),'hybrid_dense_layers':list(range(21,32)),
            'trajectory':'own_end_to_end','exploratory_posthoc':True,'no_decode_KL':True})
        model=AutoModelForCausalLM.from_pretrained(local,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
        smoke.install_memory_mlp(model)
        all_rows=[];references=RUN/'dense_tensors';references.mkdir();pair_state=None
        @torch.inference_mode()
        def forward(tokens):
            cache=DynamicCache();result=model.model(input_ids=tokens,position_ids=torch.arange(tokens.shape[1],device='cuda')[None],
                attention_mask=torch.ones_like(tokens),past_key_values=cache,use_cache=True,return_dict=True)
            assert len(cache.key_cache)==32 and all(list(x.shape)==[1,8,tokens.shape[1],128] for x in cache.key_cache)
            assert torch.isfinite(result.last_hidden_state).all()
            return result,cache
        for method in ['dense','hybrid_smoke']:
            if method=='hybrid_smoke':model,pair_state=smoke.install(model)
            warm,cache=forward(ids[:,:129]);del warm,cache
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            handles=[];rows=[];pairs={};method_start=time.perf_counter()
            def capture(layer,point):
                def hook(module,args,out):
                    value=out[0] if isinstance(out,tuple) else out
                    cpu=value.detach().cpu();path=references/('%s_%02d.pt'%(point,layer))
                    assert list(cpu.shape)==[1,n,4096]
                    if method=='dense':torch.save(cpu,path);ref=cpu
                    else:ref=torch.load(path,map_location='cpu',weights_only=True)
                    m=compare_outputs(ref,cpu,torch.ones(cpu.shape[:2],dtype=torch.bool))
                    m['rmse']=(m['squared_error']/cpu.numel())**.5
                    row={'method':method,'layer':layer,'point':point,'input_ids_sha256':identity,'input_source_index':3,
                        'shape':list(cpu.shape),'padding_tokens':0,'trajectory':'own_end_to_end','mlp_buffer_reuse':True,**m}
                    rows.append(row)
                    # Incremental metrics survive interruption; completion validated below.
                    with (RUN/(method+'_partial.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
                return hook
            def pair_hook(layer):
                def hook(module,args,out):
                    pairs[layer]=32*n*(n+1)//2 if method=='dense' or layer>=21 else int(pair_state['pairs'][layer])
                return hook
            for i,layer in enumerate(model.model.layers):
                handles.append(layer.self_attn.o_proj.register_forward_hook(capture(i,'attention_post_o_proj')))
                handles.append(layer.self_attn.register_forward_hook(pair_hook(i)))
                handles.append(layer.register_forward_hook(capture(i,'block_output')))
            result,cache=forward(ids);torch.cuda.synchronize()
            for h in handles:h.remove()
            assert len(rows)==64 and len(pairs)==32
            for row in rows:row['retained_causal_pairs']=pairs[row['layer']]
            save(method+'_diagnostics.json',{'seconds':time.perf_counter()-method_start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'includes_tensor_IO_and_metrics':True,'not_pure_inference_timing':True})
            (RUN/(method+'_layers.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows));all_rows.extend(rows)
            del result,cache
            print(method,'finished',flush=True)
        assert len(all_rows)==128
        (RUN/'all_layers.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in all_rows))
        fields=['method','layer','point','input_source_index','input_ids_sha256','valid_tokens','kl_dense_to_method','relative_l2','rmse','mean_token_cosine','retained_causal_pairs']
        with (RUN/'all_layers.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(all_rows)
        summary=[]
        for point in ['attention_post_o_proj','block_output']:
            rr=sorted([x for x in all_rows if x['method']=='hybrid_smoke' and x['point']==point],key=lambda x:x['layer'])
            assert [x['layer'] for x in rr]==list(range(32))
            assert all(math.isfinite(x['kl_dense_to_method']) for x in rr)
            summary.append({'point':point,'mean_kl_all_layers':sum(x['kl_dense_to_method'] for x in rr)/32,
                'mean_kl_layers0_20':sum(x['kl_dense_to_method'] for x in rr[:21])/21,
                'mean_kl_layers21_31':sum(x['kl_dense_to_method'] for x in rr[21:])/11,'final_layer_kl':rr[-1]['kl_dense_to_method']})
        save('summary.json',summary)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,point in zip(axes,['attention_post_o_proj','block_output']):
            for method,label in [('dense','Dense'),('hybrid_smoke','Smoke: C0-20 / Dense21-31')]:
                rr=sorted([x for x in all_rows if x['method']==method and x['point']==point],key=lambda x:x['layer'])
                ax.plot([x['layer'] for x in rr],[x['kl_dense_to_method'] for x in rr],label=label)
            ax.axvline(20.5,color='gray',linestyle='--');ax.set(title=point,xlabel='Layer (zero-based)',ylabel='KL(dense || method), nats');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Same L40S + MLP buffer reuse | KV retrieval index3 | Own trajectories; exploratory')
        fig.tight_layout();fig.savefig(RUN/'layerwise_kl.png',dpi=180)
        save('state.json',{'status':'complete','rows':128,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ.get('SLURM_STEP_ID'),'elapsed_seconds':time.monotonic()-START})
    except BaseException:
        save('state.json',{'status':'incomplete','error':traceback.format_exc(),'elapsed_seconds':time.monotonic()-START});raise
if __name__=='__main__':main()
