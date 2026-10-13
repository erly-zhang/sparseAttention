"""Approval-gated GPU correctness, matched controls and four-task evaluation."""
import argparse
import gc
import os
import threading
import time
from support import HERE, gate, legacy, read, save, sha


def runtime():
    import torch
    import transformers
    assert transformers.__version__=='4.44.0'
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)


def proof():
    prepared=gate()
    runtime()
    import torch
    from per_head import full_attention, shared_base, head_index, literal_mask, reference_head, MODES
    c=read(HERE/'config.json')
    torch.manual_seed(102008)
    q=torch.randn(1,513,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,513,8,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    opts=dict(c['selection'],target_token_budget=256,autoblock_final_token_budget=256,
              vertical_max_tokens=3,slash_max_offsets=5)
    checks=[]
    for mode in MODES:
        for constant in [False,True]:
            values=v if not constant else torch.arange(1,9,device='cuda',dtype=q.dtype)[None,None,:,None].expand_as(k).contiguous()
            actual,_,stats=full_attention(q.transpose(1,2),k.transpose(1,2),values.transpose(1,2),
                                         torch.arange(513,device='cuda')[None],mode,17,0,opts)
            actual=actual.view_as(q)
            base,_=shared_base(q,k,values,17,0,opts)
            for h in range(32):
                idx,hs,columns,offsets=head_index(base,q,k,h,mode,opts)
                mask=literal_mask(base,columns,offsets,513)
                expected=reference_head(q,k,values,h,mask)
                torch.testing.assert_close(actual[0,:,h],expected,atol=.025,rtol=.02)
                assert hs['mask_selected_pairs']==int(mask.sum())
                assert stats['per_head'][h]['pattern_source_head']==h
            checks.append(dict(mode=mode,constant_kv=constant))
    save(HERE/'kernel-proof.json',dict(status='passed',lock_sha256=sha(HERE/'lock.json'),checks=checks,
        scope='BF16 kernels vs independent literal token mask and FP32 GQA attention, N=513 H=32 KV=8 D=128',job=os.environ['SLURM_JOB_ID']))


def model_and_inputs():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    old=legacy(); c,m=old.preflight()
    heads=old.load_heads(c,m)
    model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    old.runtime_quality().install_mlp(model)
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    return old,c,m,heads,model,tok


def history(index):
    p=gate(); runtime()
    kernel_check=read(HERE/'kernel-proof.json')
    assert kernel_check['status']=='passed' and kernel_check['lock_sha256']==sha(HERE/'lock.json')
    old,c,m,heads,model,tok=model_and_inputs()
    reference=p['sentinels'][index]
    item=next(x for x in m['evaluation'] if x['key']==reference['key'])
    directory=HERE/'outputs'/('history-%d'%index);directory.mkdir(parents=True,exist_ok=True)
    method=reference['method'];path=directory/(item['key']+'-'+method+'.json')
    if path.exists(): row=read(path)
    else:
        row=old.generate(model,tok,item,method,c,heads,directory,
            dict(experiment='per_head_patterns_20261008',lock_sha256=sha(HERE/'lock.json'),
                 job=os.environ['SLURM_JOB_ID'],gpu=__import__('torch').cuda.get_device_name()),lambda:None,old.official()[item['task']])
    historical=read(reference['path'])
    assert row['input_ids_sha256']==historical['input_ids_sha256']
    assert row['generated_ids']==historical['generated_ids'], 'Historical output differs; inspect before formal launch'
    save(directory/'complete.json',dict(status='passed',lock_sha256=sha(HERE/'lock.json'),method=method,
        key=item['key'],historical_generated_ids_equal=True,historical_result_sha256=sha(reference['path'])))


def generate(model,tok,item,method,spec,c,heads,old,directory,progress):
    import numpy as np
    import torch
    from per_head import install
    chosen=heads[spec['head_strategy']]
    torch.manual_seed(c['generation_seed']);torch.cuda.manual_seed_all(c['generation_seed'])
    state=install(model,spec['selector_method'],heads=chosen,options=c['selection'],layers=c['layers'])
    started=time.monotonic();generated=[]
    progress.update(stage='prefill',key=item['key'],method=method,completed_decode_steps=0)
    try:
        with torch.no_grad():
            logits,cache,ids=old.forward_prompt(model,item,state,lambda:None)
            torch.cuda.synchronize();prefill_seconds=time.monotonic()-started
            first=logits.cpu().numpy().copy()
            eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
            progress.update(stage='decode')
            for j in range(item['max_new_tokens']):
                token=int(logits.argmax());generated.append(token)
                if (j+1)%8==0:
                    progress.update(completed_decode_steps=j+1,generated_ids=list(generated))
                    save(directory/'progress.json',dict(progress))
                if token in eos or j==item['max_new_tokens']-1:break
                offset=len(ids)+j
                result=model.model(input_ids=torch.tensor([[token]],device='cuda'),
                    position_ids=torch.tensor([[offset]],device='cuda'),attention_mask=torch.ones(1,offset+1,dtype=torch.long,device='cuda'),
                    use_cache=True,past_key_values=cache,return_dict=True)
                logits=model.lm_head(result.last_hidden_state[:,-1]).float()[0]
                assert bool(torch.isfinite(logits).all())
                assert all(k.shape==v.shape==(1,8,offset+1,128) for k,v in zip(cache.key_cache,cache.value_cache))
        torch.cuda.synchronize();total=time.monotonic()-started
        assert set(state['prefill'])==set(range(32))
        assert all(state['decode_calls'].get(i,0)==len(generated)-1 for i in range(32))
        prediction=tok.decode(generated,skip_special_tokens=True)
        logits_path=directory/(item['key']+'-'+method+'.npy')
        temp=logits_path.with_suffix('.npy.tmp')
        with temp.open('wb') as f:np.save(f,first,allow_pickle=False)
        temp.replace(logits_path)
        row=dict(item,status='complete',method=method,pattern_scope='per_head',base_selected_heads=chosen,
            generated_ids=generated,prediction=prediction,score=float(old.official()[item['task']](prediction,item['answers'],'llama')),
            stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',full_kv_cache=True,dense_decode=True,
            prefill={str(k):v for k,v in state['prefill'].items()},repetition=old.repetition(generated),
            prefill_seconds=prefill_seconds,total_seconds=total,finite_all_steps=True,
            lock_sha256=sha(HERE/'lock.json'),logits_file=logits_path.name,logits_sha256=sha(logits_path),
            job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name())
        save(logits_path.with_suffix('.json'),row)
        return row
    finally:
        state['close']();gc.collect();torch.cuda.empty_cache()


def formal(index):
    p=gate(); runtime()
    for i in range(len(p['sentinels'])):
        proof=read(HERE/'outputs'/('history-%d'%i)/'complete.json')
        assert proof['status']=='passed' and proof['lock_sha256']==sha(HERE/'lock.json')
    old,prior,manifest,heads,model,tok=model_and_inputs()
    c=read(HERE/'config.json');worker=p['workers'][index]
    directory=HERE/'outputs'/('worker-%03d'%index);directory.mkdir(parents=True,exist_ok=True)
    progress=dict(worker=index,completed_samples=0,stage='loading')
    stop=threading.Event()
    def snapshots():
        while not stop.wait(300):save(directory/'progress.json',dict(progress))
    thread=threading.Thread(target=snapshots,daemon=True);thread.start()
    try:
        for count,key in enumerate(worker['keys']):
            item=next(x for x in manifest['evaluation'] if x['key']==key)
            path=directory/(key+'-'+worker['method']+'.json')
            if path.exists():
                row=read(path)
                assert row['lock_sha256']==sha(HERE/'lock.json') and row['input_ids_sha256']==item['input_ids_sha256']
                assert row['status']=='complete' and sha(directory/row['logits_file'])==row['logits_sha256']
            else:generate(model,tok,item,worker['method'],c['scenarios'][worker['method']],c,heads,old,directory,progress)
            progress.update(completed_samples=count+1,stage='between_samples')
            save(directory/'progress.json',dict(progress))
            print('COMPLETE',index,key,worker['method'],flush=True)
        save(directory/'complete.json',dict(status='complete',worker=worker,lock_sha256=sha(HERE/'lock.json')))
    finally:stop.set();thread.join(timeout=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['proof','history','formal']);parser.add_argument('--index',type=int)
    args=parser.parse_args()
    try:
        if args.stage=='proof':proof()
        elif args.stage=='history':history(args.index)
        else:formal(args.index)
    except Exception as error:
        save(HERE/('failure-'+args.stage+'-'+str(args.index)+'.json'),dict(error=repr(error),stage=args.stage,index=args.index))
        raise

if __name__=='__main__':main()
