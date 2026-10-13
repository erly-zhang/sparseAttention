"""Frozen-source MASA over every query in each tile, using matched IB inputs."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import string
import threading
import time
import types

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp-' + str(os.getpid()))
    with temp.open('w') as f:
        json.dump(x, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    temp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def verify():
    for p, expected in read(HERE/'lock.json')['files'].items():
        assert sha(p) == expected, p
    return read(HERE/'config.json'), read(HERE/'prepared.json')


def scorer(task, prepared):
    path = Path(prepared['official_directory'])/'compute_scores.py'
    assert sha(path) == prepared['source_sha256'][str(path)]
    names = {'normalize_answer', 'f1_score', 'qa_f1_score', 'get_score_one_'+task}
    funcs = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(re=re, string=string, Counter=Counter)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), ns)
    return ns['get_score_one_'+task]


def make_selector(c, groups, budget=None):
    from experiments.masa_token_scoring import RepresentativeMASATokenSelector
    return RepresentativeMASATokenSelector(groups,
        logical_key_block_size=(32,64,128), target_token_budget=budget or c['target_token_budget'],
        target_token_top_p=None, target_token_min_budget=None, target_top_p_start_layer=None,
        final_token_budget=None, maximum_selected_blocks=None, fill_final_token_budget=False,
        minimum_block_coverage_ratio=.125, minimum_block_target_probability_ratio=None,
        block_probability_top_p=None, block_target_probability_top_p=None,
        block_target_count_coverage_ratio=None, block_target_cost_aware=False, block_f_beta=None,
        measure_fixed_target_probability_mass=False, force_sink_block=False,
        force_diagonal_block=False, project_target_to_blocks=False,
        masa_norm_alpha=c['masa_norm_alpha'],
        masa_clip_percentile=c['masa_clip_percentile'], masa_query_chunk_size=c['masa_query_chunk_size'],
        masa_query_mode=c['masa_query_mode'],masa_full_query_chunk_size=c['masa_full_query_chunk_size'])


def install(model, c, progress):
    import torch
    import llama_adapter
    from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
    groups = read(HERE/'groups.json')
    selectors = {}
    def attention(q, k, v, positions, mode, head=None, layer=None, **kwargs):
        progress.update(stage='prefill', layer=layer)
        s = make_selector(c, groups); selectors[layer] = s
        qn, kn, vn = [x.transpose(1,2) for x in (q,k,v)]
        token = s.current_layer.set(layer)
        try: index = s(qn,kn,vn,128,.9,1,100000)
        finally: s.current_layer.reset(token)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = triton_token_compacted_prefill_attention(qn,kn,vn,index,token_chunk_size=32)
        end.record(); s.stats.record_kernel_events(start,end)
        return output.flatten(-2), None, dict(groups=3, representatives=[g['representative'] for g in groups['layers'][str(layer)]])
    llama_adapter.full_attention = attention
    state = llama_adapter.install(model, 'dense_full', layers=c['layers'])
    state['selectors'] = selectors
    return state


def load_model(c, prepared):
    import torch
    import transformers
    assert transformers.__version__ == '4.44.0'
    assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name()
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Compile the actual CSR attention path before timing long prompts.
    from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
    warm_q=torch.randn(1,513,32,128,device='cuda',dtype=torch.bfloat16)
    warm_k=torch.randn(1,513,8,128,device='cuda',dtype=torch.bfloat16)
    warm_v=torch.randn_like(warm_k)
    warm_selector=make_selector(c,read(HERE/'groups.json'))
    context=warm_selector.current_layer.set(0)
    warm_index=warm_selector(warm_q,warm_k,warm_v,128,.9,1,10000)
    warm_selector.current_layer.reset(context)
    triton_token_compacted_prefill_attention(warm_q,warm_k,warm_v,warm_index,token_chunk_size=32)
    torch.cuda.synchronize()
    del warm_q,warm_k,warm_v,warm_index,warm_selector
    for p, metadata in prepared['model_files'].items():
        assert Path(p).stat().st_size == metadata['bytes'], p
    model = transformers.AutoModelForCausalLM.from_pretrained(c['model'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    # Exact historical in-place MLP evaluation order reduces long-prompt memory.
    for layer in model.model.layers:
        def mlp(self, x):
            gate=self.act_fn(self.gate_proj(x)); up=self.up_proj(x)
            gate.mul_(up); del up
            return self.down_proj(gate)
        layer.mlp.forward=types.MethodType(mlp,layer.mlp)
    return model, transformers.AutoTokenizer.from_pretrained(c['model'], local_files_only=True)


def generate(model, tok, item, c, prepared, progress):
    import numpy as np
    import torch
    from transformers import DynamicCache
    path=HERE/'outputs'/(item['key']+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        row=read(path); assert row['lock_sha256']==sha(HERE/'lock.json')
        assert row['input_ids_sha256']==item['input_ids_sha256']
        return
    directory=Path(prepared['input_directory'])
    arrays={}
    for suffix, field, dtype in [('ids','input_ids_sha256','<i8'), ('pos','position_ids_sha256','<i8'), ('mask','attention_mask_sha256','u1')]:
        p=directory/(item['key']+'.'+suffix); assert sha(p)==item[field]
        arrays[suffix]=np.fromfile(p,dtype=dtype).copy()
    assert len(arrays['ids'])==item['input_tokens']
    torch.manual_seed(c['generation_seed']); torch.cuda.manual_seed_all(c['generation_seed'])
    progress.update(key=item['key'],stage='prefill',generated_ids=[])
    state=install(model,c,progress); cache=DynamicCache(); generated=[]
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    try:
        with torch.inference_mode():
            result=model.model(input_ids=torch.tensor(arrays['ids'][None],device='cuda'),
                position_ids=torch.tensor(arrays['pos'][None],device='cuda'),
                attention_mask=torch.tensor(arrays['mask'][None],device='cuda',dtype=torch.long),
                past_key_values=cache,use_cache=True,return_dict=True)
            logits=model.lm_head(result.last_hidden_state[:,-1]).float()[0]
            del result
            torch.cuda.synchronize(); prefill=time.monotonic()-started
            assert bool(torch.isfinite(logits).all())
            first=logits.cpu().numpy().copy()
            eos=model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else list(eos)
            progress.update(stage='decode')
            for j in range(item['max_new_tokens']):
                token=int(logits.argmax()); generated.append(token)
                progress.update(generated_ids=list(generated),completed_decode_steps=j+1)
                if token in eos or j==item['max_new_tokens']-1: break
                offset=item['input_tokens']+j
                result=model.model(input_ids=torch.tensor([[token]],device='cuda'),
                    position_ids=torch.tensor([[offset]],device='cuda'),
                    attention_mask=torch.ones(1,offset+1,dtype=torch.long,device='cuda'),
                    past_key_values=cache,use_cache=True,return_dict=True)
                logits=model.lm_head(result.last_hidden_state[:,-1]).float()[0]; del result
                assert bool(torch.isfinite(logits).all())
            torch.cuda.synchronize(); total=time.monotonic()-started
        assert set(state['prefill'])==set(range(32))
        assert all(state['decode_calls'].get(i,0)==len(generated)-1 for i in range(32))
        assert all(k.shape==v.shape==(1,8,item['input_tokens']+len(generated)-1,128)
                   for k,v in zip(cache.key_cache,cache.value_cache))
        stats=[s.stats.snapshot() for s in state['selectors'].values()]
        selected=sum(s['selected_token_pairs'] for s in stats); causal=sum(s['causal_token_pairs'] for s in stats)
        prediction=tok.decode(generated,skip_special_tokens=True)
        score=float(scorer(item['task'],prepared)(prediction,item['answers'],'llama'))
        lp=path.with_suffix('.npy'); temp=lp.with_suffix('.npy.tmp')
        with temp.open('wb') as f: np.save(f,first,allow_pickle=False)
        temp.replace(lp)
        row=dict(item,method=c['method'],status='complete',query_weight_rule=c['query_weight_rule'],
            masa_query_mode=c['masa_query_mode'],query_aggregation=c['query_aggregation'],
            actual_queries_per_full_tile=128,masa_full_query_chunk_size=c['masa_full_query_chunk_size'],
            generated_ids=generated,prediction=prediction,score=score,
            stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',
            prefill_seconds=prefill,total_seconds=total,decode_seconds=total-prefill,
            selector_seconds=sum(s['selection_latency_sec'] for s in stats),
            sparse_attention_seconds=sum(s['kernel_latency_sec'] for s in stats),
            selected_token_pairs=selected,causal_token_pairs=causal,pair_sparsity=1-selected/causal,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),full_kv_cache=True,dense_decode=True,
            applied_layers=c['layers'],groups_sha256=sha(HERE/'groups.json'),
            lock_sha256=sha(HERE/'lock.json'),job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),
            first_logits_sha256=sha(lp),prefill_layers=state['prefill'])
        save(path,row); print('COMPLETE',item['key'],score,prefill,flush=True)
    finally:
        state['close'](); del cache
        import gc
        gc.collect(); torch.cuda.empty_cache()


def proof(c, prepared, progress):
    import torch
    from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
    torch.manual_seed(61012)
    torch.backends.cuda.matmul.allow_tf32=False
    q=torch.randn(1,513,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,513,8,128,device='cuda',dtype=torch.bfloat16); v=torch.randn_like(k)
    groups=read(HERE/'groups.json'); checks=[]
    for layer in (0,15,31):
        s=make_selector(c,groups,budget=256); context=s.current_layer.set(layer)
        index=s(q,k,v,128,.9,1,10000); s.current_layer.reset(context)
        reps=torch.tensor([g['representative'] for g in groups['layers'][str(layer)]],device='cuda')
        qr=q.index_select(2,reps);kr=k.index_select(2,reps//4);vr=v.index_select(2,reps//4)
        value_context=s._profile_values.set(vr)
        profile,starts,ends,lengths=s._score_representative_keys(qr,kr,block_size=128)
        s._profile_values.reset(value_context)
        dense_logits=qr.permute(0,2,1,3).float() @ kr.permute(0,2,3,1).float() / (128**.5)
        dense_logits.masked_fill_(~torch.ones(513,513,dtype=torch.bool,device='cuda').tril(),-torch.inf)
        dense_a=dense_logits.softmax(-1);dense_x=vr.permute(0,2,1,3).float()
        dense_output=dense_a@dense_x
        dense_d=2*dense_a*(dense_output@dense_x.transpose(-1,-2))-dense_a.square()*dense_x.square().sum(-1).unsqueeze(-2)
        for tile in range(index.num_query_blocks):
            start=tile*128;end=min(start+128,513)
            torch.testing.assert_close(profile[:,:,tile,:end],dense_d[:,:,start:end,:end].mean(-2),atol=5e-5,rtol=3e-4)
        actual=triton_token_compacted_prefill_attention(q,k,v,index,token_chunk_size=32)
        for h in range(32):
            group=int(index.head_to_group[h]); mask=torch.zeros(513,513,dtype=torch.bool,device='cuda')
            for tile in range(index.num_query_blocks):
                begin=int(index.row_starts[0,group,tile]); end=int(index.row_ends[0,group,tile])
                ids=index.token_indices[begin:end].long()
                mask[tile*128:min((tile+1)*128,513),ids]=True
            mask &= torch.ones_like(mask).tril()
            assert bool(mask.any(-1).all())
            logits=q[0,:,h].float() @ k[0,:,h//4].float().T / (128**.5)
            expected=logits.masked_fill(~mask,-torch.inf).softmax(-1) @ v[0,:,h//4].float()
            torch.testing.assert_close(actual[0,:,h].float(),expected,atol=.025,rtol=.02)
        checks.append(dict(layer=layer,heads=32,groups=3,independent_gqa_reference=True,
                           all_query_score_vs_dense_matrix=True))
    save(HERE/'kernel-proof.json',dict(status='passed',checks=checks,lock_sha256=sha(HERE/'lock.json')))
    # Verify model bytes once, before admitting the parallel evaluation array.
    for p, expected in prepared['model_sha256'].items(): assert sha(p)==expected['sha256'],p
    model,tok=load_model(c,prepared)
    # With budget >= short prompt length, the selected mask must reproduce FA2.
    from transformers import DynamicCache
    with torch.inference_mode():
        ids=torch.arange(513,device='cuda')[None] % int(model.config.vocab_size)
        kwargs=dict(input_ids=ids,position_ids=torch.arange(513,device='cuda')[None],
                    attention_mask=torch.ones_like(ids),use_cache=True,return_dict=True)
        dense=model.model(past_key_values=DynamicCache(),**kwargs).last_hidden_state
        state=install(model,c,progress)
        try: sparse=model.model(past_key_values=DynamicCache(),**kwargs).last_hidden_state
        finally:state['close']()
        torch.testing.assert_close(sparse.float(),dense.float(),atol=.15,rtol=.05)
        del dense,sparse
    item=next(r for r in prepared['evaluation'] if r['key']==prepared['smoke_key'])
    generate(model,tok,item,c,prepared,progress)
    save(HERE/'proof.json',dict(status='passed',lock_sha256=sha(HERE/'lock.json'),
        model_sha256_verified=True,short_prompt_dense_parity=True,query_mode=c['masa_query_mode'],
        smoke_key=item['key'],job=os.environ['SLURM_JOB_ID']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['proof','formal'])
    parser.add_argument('--index',type=int);args=parser.parse_args()
    c,p=verify();progress=dict(stage='starting',worker=args.index,completed_samples=0)
    pp=HERE/'progress'/('%s-%s.json'%(args.stage,args.index))
    stop=threading.Event()
    def snapshots():
        while not stop.wait(c['periodic_save_seconds']): save(pp,dict(progress))
    thread=threading.Thread(target=snapshots,daemon=True);thread.start()
    try:
        if args.stage=='proof': proof(c,p,progress)
        else:
            check=read(HERE/'proof.json');assert check['status']=='passed' and check['lock_sha256']==sha(HERE/'lock.json')
            model,tok=load_model(c,p);worker=p['workers'][args.index]
            for count,key in enumerate(worker['keys']):
                item=next(r for r in p['evaluation'] if r['key']==key)
                generate(model,tok,item,c,p,progress)
                progress.update(completed_samples=count+1,stage='between_samples');save(pp,dict(progress))
            save(HERE/'completed'/('%d.json'%args.index),dict(status='complete',worker=worker))
    except Exception as e:
        save(HERE/'failures'/('%s-%s.json'%(args.stage,args.index)),dict(error=repr(e),progress=progress));raise
    finally: stop.set();thread.join(timeout=2)


if __name__=='__main__': main()
