"""Read-only observer on the unchanged historical prefill dispatch."""
import argparse
import gc
import time
from pathlib import Path

from common import HERE,config,read,save,sha,require_approval,runtime,load_model,Llama44LensModel,legacy,atomic_torch_save
from metrics import layer_metrics,vector_metrics,to_lists,distribution_metrics
from progress import PeriodicSaver


def run(method,shard):
    require_approval()
    import numpy as np
    import torch
    from jlens import JacobianLens
    runtime()
    c=config()
    if method not in c['replay']['methods'] or not 0<=shard<c['replay']['workers_per_method']:
        raise ValueError('Invalid replay worker')
    old=legacy()
    prior,historical_manifest=old.preflight()
    heads=old.load_heads(prior,historical_manifest)
    from adapter import install
    manifest=read(HERE/'input-manifest.json')
    items=[x for x in manifest['evaluation'] if x['analysis_shard']==shard]
    directory=HERE/'outputs'/('replay-'+method+'-'+str(shard))
    with PeriodicSaver(directory):
        try:
            model,tokenizer=load_model()
            old.runtime_quality().install_mlp(model)
            wrapper=Llama44LensModel(model,tokenizer)
            lens=JacobianLens.load(str(HERE/'outputs/lens.pt'))
            lens_hash=sha(HERE/'outputs/lens.pt')
            if lens_hash != read(HERE/'outputs/lens-provenance.json')['lens_sha256']:
                raise ValueError('Merged lens changed')
            for item in sorted(items,key=lambda x:-x['input_tokens']):
                output=directory/(item['key']+'.json')
                if output.exists():
                    completed=read(output)
                    if completed['artifact_sha256'] != sha(directory/completed['artifact_file']):
                        raise ValueError('Previous result artifact changed')
                    continue
                dense=None
                if method!='dense_full':
                    reference=HERE/'outputs'/('replay-dense_full-'+str(shard))/(item['key']+'.json')
                    meta=read(reference)
                    artifact=reference.parent/meta['artifact_file']
                    if meta['artifact_sha256'] != sha(artifact): raise ValueError('Dense reference changed')
                    dense=torch.load(artifact,map_location='cpu',weights_only=True)
                    if dense['positions']!=item['analysis_positions']: raise ValueError('Misaligned positions')
                residuals,layerwise={},{}
                indices=torch.tensor(item['analysis_positions'],device='cuda')
                handles=[]
                def make_hook(layer):
                    def hook(module,inputs,output):
                        tensor=output if torch.is_tensor(output) else output[0]
                        selected=tensor[0].index_select(0,indices).detach()
                        residuals[layer]=selected.cpu()
                        if dense is not None:
                            layerwise[str(layer)]=layer_metrics(wrapper,lens,layer,dense['residuals'][layer],selected,
                                c['replay']['top_k'],c['replay']['readout_chunk_size'])
                        else:
                            layerwise[str(layer)]=layer_metrics(wrapper,lens,layer,selected,selected,
                                c['replay']['top_k'],c['replay']['readout_chunk_size'])
                    return hook
                for layer in c['replay']['layers']:
                    handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
                spec=prior['scenarios'].get(method)
                state=install(model,spec['selector_method'] if spec else method,layers=prior['layers'],
                    heads=heads[spec['head_strategy']] if spec else None,
                    options=prior['baselines'] if method in prior['baselines'] else prior['selection'])
                try:
                    start=time.monotonic()
                    with torch.no_grad(): logits,cache,ids=old.forward_prompt(model,item,state,lambda:None)
                    torch.cuda.synchronize()
                    seconds=time.monotonic()-start
                    # Final row uses J=I. A single-row readout avoids changing GEMM shape.
                    with torch.no_grad(): final_readout=wrapper.unembed(residuals[31][-1:].cuda().float()).float()[0]
                    torch.testing.assert_close(final_readout,logits,rtol=.02,atol=.15)
                    reconstruction_delta=float((final_readout-logits).abs().max())
                    sampled_k,sampled_v={},{}
                    cache_metrics={}
                    for layer,(k,v) in enumerate(zip(cache.key_cache,cache.value_cache)):
                        sampled_k[layer]=k.index_select(2,indices).transpose(1,2)[0].cpu()
                        sampled_v[layer]=v.index_select(2,indices).transpose(1,2)[0].cpu()
                        if dense is not None:
                            cache_metrics[str(layer)]={
                                'key':to_lists(vector_metrics(dense['keys'][layer],sampled_k[layer])),
                                'value':to_lists(vector_metrics(dense['values'][layer],sampled_v[layer]))}
                    stats={str(k):v for k,v in state['prefill'].items()}
                finally:
                    state['close']()
                    for handle in handles: handle.remove()
                if set(residuals)!=set(c['replay']['layers']): raise ValueError('Incomplete observations')
                history=item['history'][method]
                old_logits=torch.from_numpy(np.load(history['logits'],allow_pickle=False).copy()).float()
                current=logits.detach().float().cpu()
                history_delta=float((current-old_logits).abs().max())
                replay_match=dict(bitwise_equal=bool(torch.equal(current,old_logits)),
                    max_abs_logit_difference=history_delta,
                    first_token_equal=int(current.argmax())==int(old_logits.argmax()),
                    distribution=to_lists(distribution_metrics(old_logits[None],current[None],10)))
                artifact=directory/(item['key']+'.pt')
                atomic_torch_save(artifact,dict(positions=item['analysis_positions'],residuals=residuals,
                        keys=sampled_k,values=sampled_v,final_logits=current))
                vocab_ids={token for values in layerwise.values() for field in ['dense_top_ids','sparse_top_ids']
                           for row in values[field] for token in row}
                save(output,dict(key=item['key'],task=item['task'],method=method,input_tokens=len(ids),
                    positions=item['analysis_positions'],position_groups=item['position_groups'],
                    context_tokens={str(p):tokenizer.decode([ids[p]]) for p in item['analysis_positions']},
                    vocab_fragment={str(t):tokenizer.decode([t]) for t in sorted(vocab_ids)},
                    historical_score=history['score'],historical_score_is_new_evaluation=False,
                    history_replay_match=replay_match,layer_metrics=layerwise,sampled_kv_metrics=cache_metrics,
                    final_identity_readout_max_abs_difference=reconstruction_delta,
                    prefill_statistics=stats,observed_prefill_seconds=seconds,
                    timings_include_observer=True,artifact_file=artifact.name,artifact_sha256=sha(artifact),
                    lens_sha256=lens_hash,config_sha256=sha(HERE/'config.json')))
                del dense,residuals,sampled_k,sampled_v,cache,logits
                gc.collect()
                torch.cuda.empty_cache()
                print('COMPLETE',method,item['key'],history_delta,flush=True)
            save(directory/'complete.json',dict(status='complete',method=method,shard=shard,records=len(items),
                files={item['key']+'.json':sha(directory/(item['key']+'.json')) for item in items},
                snapshot_error=read(directory/'snapshot-error.json') if (directory/'snapshot-error.json').exists() else None))
            del model,lens
            gc.collect()
            torch.cuda.empty_cache()
            from aggregate import finalize_if_ready
            finalize_if_ready()
        except Exception as exc:
            save(directory/'failure.json',dict(error=repr(exc)))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',required=True)
    p.add_argument('--shard',required=True,type=int)
    a=p.parse_args()
    run(a.method,a.shard)
