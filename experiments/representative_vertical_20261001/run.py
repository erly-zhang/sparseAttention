"""Approval-gated calibration and evaluation; CPU preflight is separately callable."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
from protocol import RUN,ORIGIN,PRIOR,TASKS,sha,digest,save,official,repetition
from protocol import configuration,verify_sources,verify_inputs,require_approval


def preflight():
    import importlib.metadata as metadata
    lock=verify_sources()
    c,m=verify_inputs()
    for package,version in lock['versions'].items():
        if metadata.version(package)!=version:
            raise ValueError('dependency version changed: '+package)
    runtime_quality()
    return c,m


def runtime_quality():
    spec=importlib.util.spec_from_file_location('frozen_quality',PRIOR/'quality.py')
    q=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    q.verify_sources()
    return q


def gpu_parity(c):
    import torch
    from selection import build_index,reference_attention
    from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
    torch.manual_seed(61001)
    q=torch.randn(1,513,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,513,8,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    options=dict(c['selection'],target_token_budget=256,autoblock_final_token_budget=256)
    for method in c['methods'][1:]:
        index,stats=build_index(q,k,v,17,0,method,options)
        actual=triton_token_compacted_prefill_attention(q,k,v,index,token_chunk_size=options['token_chunk_size'])
        expected=reference_attention(q,k,v,index)
        torch.testing.assert_close(actual,expected,atol=.025,rtol=.02)
        assert stats['shared_with_heads']==list(range(32))
    save(RUN/'outputs'/'gpu-parity-'+os.environ['SLURM_JOB_ID']+'.json',dict(status='passed',
        methods=c['methods'][1:],scope='BF16 original Triton vs independent FP32 GQA reference, same CSR mask',
        config_sha256=sha(RUN/'config.json')))


def inputs(item,calibration=False):
    directory=RUN/'inputs' if calibration else ORIGIN/'inputs'
    p=directory/item['key']
    return (np.fromfile(str(p)+'.ids',dtype='<i8').tolist(),
        np.fromfile(str(p)+'.pos',dtype='<i8').tolist(),np.fromfile(str(p)+'.mask',dtype='u1').tolist())


def forward_prompt(model,item,state,deadline,calibration=False):
    import torch
    from transformers import DynamicCache
    ids,pos,mask=inputs(item,calibration)
    cache=DynamicCache()
    deadline()
    with torch.no_grad():
        result=model.model(input_ids=torch.tensor([ids],device='cuda'),
            position_ids=torch.tensor([pos],device='cuda'),attention_mask=torch.tensor([mask],device='cuda'),
            use_cache=True,past_key_values=cache,return_dict=True)
        logits=model.lm_head(result.last_hidden_state[:,-1]).float()[0]
    assert bool(torch.isfinite(logits).all())
    assert len(cache.key_cache)==len(cache.value_cache)==32
    assert all(k.shape==v.shape==(1,8,len(ids),128) for k,v in zip(cache.key_cache,cache.value_cache))
    return logits,cache,ids


def calibrate(model,c,m,out,deadline):
    import torch
    from adapter import install
    from selection import directional_coverage,vote_representatives
    samples=[]
    for item in m['calibration']:
        per_layer={}
        def callback(layer,q,k,v):
            deadline()
            per_layer[str(layer)]=directional_coverage(q,k,top_p=.95,
                last_queries=c['calibration']['queries'],chunk=c['calibration']['query_chunk_size'])
        state=install(model,'dense_full',layers=c['layers'],calibration_callback=callback)
        try:
            with torch.no_grad():
                logits,cache,_=forward_prompt(model,item,state,deadline,calibration=True)
            assert set(per_layer)==set(map(str,c['layers']))
            save(out/(item['key']+'.json'),dict(input=item,layers=per_layer))
            samples.append(per_layer)
        finally:
            state['close']()
        del logits,cache
        gc.collect()
        torch.cuda.empty_cache()
        print('CALIBRATION',item['key'],'complete',flush=True)
    heads=vote_representatives(samples,c['layers'])
    save(out/'representative-heads.json',dict(representative_heads=heads,
        input_manifest_sha256=sha(RUN/'inputs/manifest.json'),config_sha256=sha(RUN/'config.json'),
        source_lock_sha256=sha(RUN/'source-lock.json'),calibration_keys=[r['key'] for r in m['calibration']],
        selection='mean outgoing directional coverage, sample majority vote, voter-score tie break, lowest head'))
    return heads


def load_heads(c,m):
    r=json.loads((RUN/'outputs/pilot/representative-heads.json').read_text())
    for field,path in (('config_sha256',RUN/'config.json'),('input_manifest_sha256',RUN/'inputs/manifest.json'),
        ('source_lock_sha256',RUN/'source-lock.json')):
        assert r[field]==sha(path)
    assert r['calibration_keys']==[x['key'] for x in m['calibration']]
    heads=r['representative_heads']
    assert set(heads)==set(map(str,c['layers'])) and all(type(h) is int and 0<=h<32 for h in heads.values())
    return heads


def generate(model,tok,item,method,c,heads,out,metadata,deadline,scorer):
    import torch
    from adapter import install
    torch.manual_seed(c['generation_seed'])
    torch.cuda.manual_seed_all(c['generation_seed'])
    state=install(model,method,layers=c['layers'],heads=heads,options=c['selection'])
    start=time.monotonic()
    try:
        with torch.no_grad():
            logits,cache,ids=forward_prompt(model,item,state,deadline)
            torch.cuda.synchronize()
            prefill_seconds=time.monotonic()-start
            first=logits.cpu().numpy().copy()
            eos=model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else list(eos)
            generated=[]
            for j in range(item['max_new_tokens']):
                token=int(logits.argmax())
                generated.append(token)
                if token in eos or j==item['max_new_tokens']-1:
                    break
                deadline()
                offset=len(ids)+j
                result=model.model(input_ids=torch.tensor([[token]],device='cuda'),
                    position_ids=torch.tensor([[offset]],device='cuda'),
                    attention_mask=torch.ones(1,offset+1,dtype=torch.long,device='cuda'),
                    use_cache=True,past_key_values=cache,return_dict=True)
                logits=model.lm_head(result.last_hidden_state[:,-1]).float()[0]
                assert bool(torch.isfinite(logits).all())
                assert all(k.shape==v.shape==(1,8,offset+1,128) for k,v in zip(cache.key_cache,cache.value_cache))
            torch.cuda.synchronize()
        total=time.monotonic()-start
        assert set(state['prefill'])==set(c['layers'])
        assert all(state['decode_calls'].get(i,0)==len(generated)-1 for i in c['layers'])
        prediction=tok.decode(generated,skip_special_tokens=True)
        path=out/(item['key']+'-'+method+'.npy')
        with path.open('xb') as f:
            np.save(f,first,allow_pickle=False)
        row=dict(item,**metadata,method=method,generated_ids=generated,prediction=prediction,eos_ids=eos,
            score=float(scorer(prediction,item['answers'],'llama')),stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',
            full_kv_cache=True,dense_decode=True,finite_all_steps=True,
            prefill={str(k):v for k,v in state['prefill'].items()},decode_calls={str(i):state['decode_calls'].get(i,0) for i in c['layers']},
            repetition=repetition(generated),prefill_seconds=prefill_seconds,total_seconds=total,
            decode_seconds=total-prefill_seconds,logits_file=path.name,logits_sha256=sha(path))
        save(path.with_suffix('.json'),row)
        return row
    finally:
        state['close']()
        gc.collect()
        torch.cuda.empty_cache()


def projected_seconds(rows,remaining,elapsed,methods):
    prediction=elapsed
    for method in methods:
        seen=[r for r in rows if r['method']==method]
        prefill=max(r['prefill_seconds']/r['input_tokens']**2 for r in seen)
        decode=max(r['decode_seconds']/max(1,len(r['generated_ids'])-1) for r in seen)
        prediction+=1.5*sum(prefill*r['input_tokens']**2+decode*(r['max_new_tokens']-1) for r in remaining)
    return prediction+600


def run(profile):
    # This check intentionally precedes GPU imports, parity tests, or model loading.
    require_approval(profile)
    c,m=preflight()
    if 'SLURM_JOB_ID' not in os.environ:
        raise PermissionError('experiments require the approved Slurm allocation')
    submission=RUN/('submitted-'+profile+'.json')
    for _ in range(30):
        if submission.exists():
            break
        time.sleep(1)
    submitted=json.loads(submission.read_text())
    expected=os.environ.get('SLURM_ARRAY_JOB_ID',os.environ['SLURM_JOB_ID'])
    assert submitted['job_id']==expected
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name()
    assert torch.cuda.get_device_properties(0).total_memory>75*1024**3
    assert transformers.__version__=='4.44.0'
    torch.set_num_threads(c[profile]['cpus'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.monotonic()
    limit=c[profile]['walltime_hours']*3600-600
    def deadline():
        if time.monotonic()-started>=limit:
            raise TimeoutError('approved allocation budget exhausted; preserve artifacts, no retry')
    (RUN/'outputs').mkdir(exist_ok=True)
    suffix='pilot' if profile=='pilot' else 'formal-'+os.environ['SLURM_ARRAY_TASK_ID']
    out=RUN/'outputs'/suffix
    out.mkdir(exist_ok=False)
    metadata=dict(config_sha256=sha(RUN/'config.json'),input_manifest_sha256=sha(RUN/'inputs/manifest.json'),
        source_lock_sha256=sha(RUN/'source-lock.json'),profile=profile,job=os.environ['SLURM_JOB_ID'],
        node=os.uname().nodename,gpu=torch.cuda.get_device_name(),generation_seed=c['generation_seed'])
    save(out/'run-start.json',metadata)
    try:
        gpu_parity(c)
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
        runtime_quality().install_mlp(model)
        assert (model.config.num_hidden_layers,model.config.num_attention_heads,model.config.num_key_value_heads)==(32,32,8)
        versions={name:p._version for name,p in model.named_parameters()}
        heads=calibrate(model,c,m,out,deadline) if profile=='pilot' else load_heads(c,m)
        metadata['representative_heads']=heads
        items=[r for r in m['evaluation'] if r['key'] in m['pilot_keys']] if profile=='pilot' else [
            r for r in m['evaluation'] if r['shard']==int(os.environ['SLURM_ARRAY_TASK_ID'])]
        items=sorted(items,key=lambda r:(-r['input_tokens']**2,r['key']))
        _,_,scorers=official(ORIGIN/'official')
        history={r['key']:r['methods']['dense_full'] for r in json.loads((ORIGIN/'independent-audit.json').read_text())['examples']}
        rows=[]
        for i,item in enumerate(items):
            for method in c['methods']:
                row=generate(model,tok,item,method,c,heads,out,metadata,deadline,scorers[item['task']])
                if method=='dense_full':
                    assert all(row[k]==history[item['key']][k] for k in ('generated_ids','prediction','score','stop_reason')), 'historical Dense regression'
                rows.append(row)
                print('RESULT',item['key'],method,row['score'],repr(row['prediction']),flush=True)
            projection=projected_seconds(rows,items[i+1:],time.monotonic()-started,c['methods'])
            save(out/f'projection-{i:03d}.json',dict(projected_seconds=projection,limit_seconds=limit))
            assert projection<limit,'runtime projection exceeds approved budget; no automatic expansion'
        assert versions=={name:p._version for name,p in model.named_parameters()}
        save(out/'complete.json',dict(status='complete',records=len(rows),frozen_base_unchanged=True,
            **metadata,elapsed_seconds=time.monotonic()-started,files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    except Exception as e:
        save(out/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('check','run'))
    parser.add_argument('--profile',choices=('pilot','formal'),default='pilot')
    args=parser.parse_args()
    if args.action=='check':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        c,m=preflight()
        print('CPU PREFLIGHT PASS',len(m['evaluation']),len(m['calibration']),len(m['pilot_keys']))
    else:
        run(args.profile)
