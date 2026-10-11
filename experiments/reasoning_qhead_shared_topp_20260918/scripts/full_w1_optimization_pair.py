"""Three gated same-GPU pairs, unchanged HMMT generation and full KV storage."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

root = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(root/'source/scripts'), str(root/'source/ra_harness'),
                str(root/'source/ra_harness/kvcompress/harness')]
from selector_grid import sha, read, rows, write, env_for, grade_bounded


def verify():
    for rel, digest in read(root/'manifest.json').items():
        assert sha(root/rel) == digest, rel
    assert read(root/'correctness.json')['status'] == 'passed'
    assert read(root/'microbenchmark.json')['status'] == 'completed'
    assert read(root/'model_correctness.json')['status'] == 'passed'
    return read(root/'config.json')


def validate(idx, implementation):
    cfg=read(root/'config.json');out=root/'pairs'/f'input_{idx}'/implementation
    assert read(out/'exit.json')['exit_code'] == 0
    mm=rows(out/'cache_metrics.jsonl');assert len(mm)==1
    m=mm[0];ref={r['idx']:r for r in rows(root/'inputs/hmmt.jsonl')}[idx]
    assert (m['idx'],m['input_tokens'],m['input_ids_sha256']) == (idx,ref['input_tokens'],ref['input_ids_sha256'])
    assert m['full_w1_implementation']==implementation
    assert m['grid_storage']=='full' and m['grid_protection']=='protected'
    assert m['query_window']==1 and m['token_budget']==cfg['K'] and m['eviction_interval']==1
    p=m['input_tokens'];ts=m['generated_tokens_per_rollout'];factor=36*32
    assert len(ts)==len(m['eos_reached'])==2
    assert m['decode_forward_calls']==max(ts)-1 and m['decode_query_tokens']==sum(ts)-2
    pf=2*factor*p*(p+1)//2
    counts=0
    for t,eos in zip(ts,m['eos_reached']):
        assert 1<=t<=cfg['max_tokens']-p
        if not eos:assert t==cfg['max_tokens']-p
        f=t-1;early=min(f,max(0,cfg['K']-p))
        counts+=early*p+early*(early+1)//2+(f-early)*cfg['K']
    assert m['selected_pairs']==pf+factor*counts
    assert m['causal_pairs']==factor*sum((p+t-1)*(p+t)//2 for t in ts)
    assert m['selection_events']==36*(max(ts)-1)
    assert all(all(n==p+max(ts)-1 for n in layer) for layer in m['final_cache_seqlens'])
    token_rows=read(out/'token_ids.json');assert len(token_rows)==1 and token_rows[0]['idx']==idx
    assert len(token_rows[0]['token_ids'])==2
    texts=[]
    for run in (0,1):
        ff=list((out/f'run_{run}').glob('completions*.jsonl'));assert len(ff)==1
        rr=rows(ff[0]);assert len(rr)==1 and rr[0]['idx']==idx
        assert rr[0]['completion'].startswith(ref['prompt_text'])
        texts.append(rr[0]['completion'][len(ref['prompt_text']):])
    return m,token_rows[0],texts


def infer(index):
    cfg=verify();idx=cfg['sample_ids'][index]
    for rel,digest in read(root/'model_assets_sha256.json').items():
        assert sha(Path(cfg['model'])/rel)==digest,rel
    import torch
    gpu=torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count()==1 and 'A100' in gpu.name and gpu.total_memory>75*1024**3
    work=root/'pairs'/f'input_{idx}';work.mkdir(parents=True)
    write(work/'hardware.json',dict(gpu=gpu.name,memory=gpu.total_memory,node=platform.node(),job=os.environ['SLURM_JOB_ID'],
                                   packages={k:importlib.metadata.version(k) for k in ['torch','transformers','flash-attn']}))
    backend=read(root/'microbenchmark.json')['selected_backend']
    order=['baseline','optimized'] if index%2==0 else ['optimized','baseline']
    write(work/'order.json',dict(order=order,rank_backend=backend))
    for implementation in order:
        out=work/implementation;out.mkdir()
        env=env_for(root,'hmmt');env.update(RA_FULL_W1_TRIAL_IMPL=implementation,RA_FULL_W1_RANK_BACKEND=backend,
                                          RA_FULL_W1_TOKEN_OUTPUT=str(out/'token_ids.json'))
        args=[sys.executable,'-u',str(root/'source/scripts/run_full_w1_trial.py'),
              '--model_name_or_path',cfg['model'],'--data_name','hmmt','--split','test',
              '--prompt_type','qwen-instruct','--surround_with_messages','--attention_implementation','flash_attention_2',
              '--sparsity_method','eviction','--eviction_mode',cfg['method'],'--token_budget',str(cfg['K']),
              '--residual_length','0','--top_p','.95','--query_window','1','--mask_seed','42','--rep_seed','0',
              '--seed','42','--eviction_interval','1','--batch_size','2','--run_id','0','--max_tokens',str(cfg['max_tokens']),
              '--warmup_new_tokens','2','--use_batch_exist','--output_dir',str(out),
              '--ex_start_i',str(idx),'--ex_end_i',str(idx+1)]
        write(out/'command.json',args)
        with (out/'runner.log').open('x') as log:
            proc=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(out/'exit.json',dict(exit_code=proc.returncode));assert proc.returncode==0
        validate(idx,implementation)
    old,old_ids,_=validate(idx,'baseline');new,new_ids,_=validate(idx,'optimized')
    assert old_ids['token_ids']==new_ids['token_ids'],'generated token IDs diverged'
    for name in ['input_tokens','input_ids_sha256','generated_tokens_per_rollout','eos_reached','selected_pairs','causal_pairs',
                 'per_layer_selected_pairs','per_layer_causal_pairs','selection_events','final_cache_seqlens','final_cumulative_length']:
        assert old[name]==new[name],name
    write(work/'pair_validation.json',dict(status='passed',token_ids_bitwise_equal=True,input_alignment='passed',
                                          pair_counts_equal=True,full_kv_storage=True,
                                          baseline_decode_seconds=old['decode_seconds'],optimized_decode_seconds=new['decode_seconds']))


def collect():
    cfg=verify();refs={r['idx']:r for r in rows(root/'inputs/hmmt.jsonl')};all_metrics={};details=[]
    for idx in cfg['sample_ids']:
        assert read(root/'pairs'/f'input_{idx}'/'pair_validation.json')['status']=='passed'
        for implementation in ['baseline','optimized']:
            m,tokens,texts=validate(idx,implementation)
            all_metrics.setdefault(implementation,[]).append(m)
            for run,text in enumerate(texts):
                prediction,correct=grade_bounded(text,refs[idx]['gold'])
                details.append(dict(idx=idx,implementation=implementation,run=run,correct=int(correct),prediction=prediction))
    summaries={}
    for implementation,mm in all_metrics.items():
        seconds=sum(m['decode_seconds'] for m in mm);calls=sum(m['decode_forward_calls'] for m in mm)
        summaries[implementation]=dict(questions=len(mm),answers=2*len(mm),
            correct=sum(r['correct'] for r in details if r['implementation']==implementation),
            mean_decode_seconds=seconds/len(mm),aggregate_ms_per_forward=1000*seconds/calls,
            mean_generated_tokens=statistics.mean(t for m in mm for t in m['generated_tokens_per_rollout']),
            peak_allocated_gib=max(m['peak_memory_allocated'] for m in mm)/2**30,
            selected_pairs=sum(m['selected_pairs'] for m in mm),causal_pairs=sum(m['causal_pairs'] for m in mm))
    fraction=1-summaries['optimized']['mean_decode_seconds']/summaries['baseline']['mean_decode_seconds']
    result=dict(status='completed_paired_pilot',sample_ids=cfg['sample_ids'],K=cfg['K'],total_max_length=cfg['max_tokens'],
                input_alignment='passed/full_within_pilot',generated_token_ids='passed/exact',pair_counts='passed/exact',
                results=summaries,decode_time_reduction=fraction,per_question=[dict(idx=idx,**read(root/'pairs'/f'input_{idx}'/'pair_validation.json')) for idx in cfg['sample_ids']],
                caveat='Three paired inputs only; one measured run/implementation/input. Same allocated A10080GB for each pair; different pairs may use different nodes. No full-HMMT accuracy or whole-dataset speedup claim.')
    write(root/'paired_summary.json',result);write(root/'grades.json',details)
    (root/'analysis/paired_results_en.txt').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if sys.argv[1]=='infer':
    infer(int(sys.argv[3]))
elif sys.argv[1]=='collect':
    collect()
else:
    raise ValueError(sys.argv[1])
