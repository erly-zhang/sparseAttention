"""Observe official prefill support on the frozen 200 full prompts per method."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'representative_patterns_fast_retry03_20261001'
sys.path.insert(0,str(ORIGINAL))
# Import our observer without importing original run as __main__.
sys.path.insert(0,str(HERE))
from protocol import sha
from observer import SparsityObserver


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def verify():
    for name,digest in json.loads((HERE/'source-lock.json').read_text()).items():
        assert sha(name)==digest,name
    approval=json.loads((HERE/'approval.json').read_text())
    assert approval['source_lock_sha256']==sha(HERE/'source-lock.json')
    assert approval['config_sha256']==sha(HERE/'config.json')
    from run import preflight
    return preflight()


def converter_test():
    import torch
    from counts import minference_pairs
    ops=importlib.import_module('minference.ops.pit_sparse_flash_attention_v2')
    assert ops.convert_vertical_slash_indexes_opt is None
    checked=0
    for n in (127,128,129,257,513):
        vi=torch.tensor([[[0,1,7,29,65,91,110]]],device='cuda',dtype=torch.int32)
        si=torch.tensor([[[95,64,40,3,1,0]]],device='cuda',dtype=torch.int32)
        lens=torch.tensor([n],device='cuda',dtype=torch.int32)
        selected=ops.convert_vertical_slash_indexes(lens,vi,si,n,64,64)
        total,invalid,_,_=minference_pairs(selected,n)
        assert int(invalid)==0
        bc,bo,cc,ci=[x.cpu() for x in selected]
        dense=torch.zeros(n,n,dtype=torch.bool)
        for r in range(bc.shape[-1]):
            start=r*64;end=min(start+64,n)
            for j in bo[0,0,r,:int(bc[0,0,r])].tolist():dense[start:end,j:min(j+64,n)]=True
            for j in ci[0,0,r,:int(cc[0,0,r])].tolist():dense[start:end,j]=True
        assert int(total)==int(dense.tril().sum())
        checked+=1
    return checked


def parity_synthetic(method,c):
    import torch
    import baselines
    torch.manual_seed(20261007)
    q=torch.randn(1,4097,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,4097,8,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    tested=[]
    for layer in (0,15,31):
        original,_=baselines.baseline_attention(q,k,v,method,layer,c['baselines'])
        with SparsityObserver(method) as observer:
            observed,_=baselines.baseline_attention(q,k,v,method,layer,c['baselines'])
        assert torch.equal(original,observed),(method,layer,'observer changed output')
        tested.append({'layer':layer,'bitwise_equal':True,'head_count':len(observer.records[layer]['heads'])})
    del q,k,v,original,observed
    gc.collect();torch.cuda.empty_cache()
    return tested


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','worker']);parser.add_argument('--index',type=int)
    args=parser.parse_args();c,manifest=verify()
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM
    from run import runtime_quality,forward_prompt
    from adapter import install
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,'/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/FlexPrefill')
    index=args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    plan=json.loads((HERE/'work-plan.json').read_text())
    method=['flexprefill','minference'][index] if args.phase=='smoke' else plan[index]['method']
    if args.phase=='worker':
        for baseline in ['flexprefill','minference']:
            proof=json.loads((HERE/'outputs'/('smoke-'+baseline)/'complete.json').read_text())
            assert proof['status']=='PASS' and proof['source_lock_sha256']==sha(HERE/'source-lock.json')
    out=HERE/'outputs'/('smoke-'+method if args.phase=='smoke' else 'worker-'+str(index))
    out.mkdir(parents=True,exist_ok=True)
    metadata={'method':method,'node':os.uname().nodename,'job':os.environ['SLURM_JOB_ID'],
        'source_lock_sha256':sha(HERE/'source-lock.json'),'original_source_lock_sha256':sha(ORIGINAL/'source-lock.json'),
        'original_config_sha256':sha(ORIGINAL/'config.json'),'input_manifest_sha256':sha(ORIGINAL/'inputs/manifest.json'),
        'baseline_options':c['baselines'][method]}
    save(out/'start.json',metadata);start=time.monotonic()
    try:
        if args.phase=='smoke':
            converter_cases=converter_test() if method=='minference' else 0
            synthetic=parity_synthetic(method,c)
            save(out/'synthetic-parity.json',dict(status='PASS',converter_dense_reference_cases=converter_cases,tests=synthetic))
        keys=[min(manifest['evaluation'],key=lambda x:x['input_tokens'])['key']] if args.phase=='smoke' else plan[index]['keys']
        items=sorted([x for x in manifest['evaluation'] if x['key'] in keys],key=lambda x:(-x['input_tokens']**2,x['key']))
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
        runtime_quality().install_mlp(model)
        versions={name:p._version for name,p in model.named_parameters()}
        original_plans=json.loads((ORIGINAL/'work-plan.json').read_text())
        for item in items:
            target=out/(item['key']+'.json')
            if target.exists():
                previous=json.loads(target.read_text());assert previous['source_lock_sha256']==metadata['source_lock_sha256']
                assert len(previous['layers'])==32
                continue
            oldplan=next(x for x in original_plans if x['method']==method and item['key'] in x['keys'])
            oldfile=ORIGINAL/'outputs'/('formal-'+str(oldplan['worker']))/(item['key']+'-'+method+'.json')
            old=json.loads(oldfile.read_text());oldlogits=oldfile.parent/old['logits_file']
            assert sha(oldlogits)==old['logits_sha256'];expected=np.load(oldlogits,allow_pickle=False)
            torch.manual_seed(c['generation_seed']);torch.cuda.manual_seed_all(c['generation_seed'])
            reference=None
            if args.phase=='smoke':
                state=install(model,method,layers=c['baseline_layers'],options=c['baselines'])
                try:
                    logits,cache,ids=forward_prompt(model,item,state,lambda:None)
                    reference=logits.cpu().numpy().copy()
                finally:state['close']()
                del logits,cache;gc.collect();torch.cuda.empty_cache()
            state=install(model,method,layers=c['baseline_layers'],options=c['baselines'])
            sample_start=time.monotonic();last_save=[0.]
            def progress(layers):
                if time.monotonic()-last_save[0]>=60 or len(layers)==32:
                    save(out/'progress.json',dict(metadata,key=item['key'],completed_layers=len(layers),layers={str(k):v for k,v in layers.items()},elapsed_seconds=time.monotonic()-sample_start))
                    last_save[0]=time.monotonic()
                print('LAYER',item['key'],len(layers),'density',layers[max(layers)]['mask_density'],flush=True)
            try:
                with SparsityObserver(method,progress) as observer:
                    logits,cache,ids=forward_prompt(model,item,state,lambda:None)
                actual=logits.cpu().numpy().copy()
                assert set(observer.records)==set(range(32)) and len(ids)==item['input_tokens']
                assert versions=={name:p._version for name,p in model.named_parameters()}
                if reference is not None:assert np.array_equal(actual,reference),'observer changes full-model logits'
                layers={str(k):v for k,v in observer.records.items()}
                path=out/(item['key']+'.npy');np.save(path,actual,allow_pickle=False)
                row=dict(metadata,key=item['key'],task=item['task'],input_tokens=item['input_tokens'],layers=layers,
                    mask_density=sum(v['mask_density'] for v in layers.values())/32,
                    observer_full_model_bitwise_equal=bool(np.array_equal(actual,reference)) if reference is not None else None,
                    historical_logits_bitwise_equal=bool(np.array_equal(actual,expected)),historical_logits_max_abs_error=float(np.abs(actual-expected).max()),
                    historical_first_token_equal=int(actual.argmax())==int(expected.argmax()),
                    original_record_sha256=sha(oldfile),original_record_path=str(oldfile),original_logits_sha256=old['logits_sha256'],
                    logits_file=path.name,logits_sha256=sha(path),elapsed_seconds=time.monotonic()-sample_start,
                    scope='same-configuration full-prefill replay; actual historical masks not saved',kernel=getattr(observer,'kernel','flex_block_wise'))
                save(target,row)
                print('RECORDED',item['key'],'density',row['mask_density'],'seconds',row['elapsed_seconds'],flush=True)
            finally:state['close']()
            del logits,cache,observer,actual;gc.collect();torch.cuda.empty_cache()
        save(out/'complete.json',dict(metadata,status='PASS',samples=len(items),elapsed_seconds=time.monotonic()-start,
            files={p.name:sha(p) for p in out.glob('*.json') if p.name not in ['complete.json','failure.json']}))
        if args.phase=='worker':
            from aggregate import aggregate_if_ready
            aggregate_if_ready()
    except Exception as error:
        save(out/'failure.json',dict(metadata,type=type(error).__name__,message=str(error)))
        raise

if __name__=='__main__':main()
