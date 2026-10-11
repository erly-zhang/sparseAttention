"""Frozen official random_pp HMMT baseline, smoke reuse, technical gates and grading."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import unittest

root=Path(sys.argv[2]).resolve()
sys.path[:0]=[str(root/'source/scripts'),str(root/'source/tests'),str(root/'source/ra_harness'),
              str(root/'source/ra_harness/kvcompress/harness')]
from selector_grid import sha,read,rows,write,env_for,grade_bounded


def verify():
    for rel,digest in read(root/'manifest.json').items():assert sha(root/rel)==digest,rel
    return read(root/'config.json')


def check():
    cfg=verify();import torch
    assert torch.cuda.device_count()==1
    suite=unittest.TestSuite()
    for name in ['test_official_random_pp','test_official_random_pp_hmmt']:
        spec=importlib.util.spec_from_file_location(name,root/'source/tests'/f'{name}.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    with (root/'correctness.log').open('x') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    ref=rows(root/'inputs/hmmt.jsonl')
    assert len(ref)==60 and {r['idx'] for r in ref}==set(range(60))
    assert max(r['input_tokens'] for r in ref)<min(c['K']-64 for c in cfg['cells'])
    proof=dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',tests=result.testsRun,
        skipped=result.skipped,log_sha256=sha(root/'correctness.log'),gpu=torch.cuda.get_device_name(),
        node=platform.node(),job=os.environ.get('SLURM_JOB_ID'),max_prompt_tokens=max(r['input_tokens'] for r in ref))
    write(root/'correctness.json',proof);print(json.dumps(proof),flush=True);assert proof['status']=='passed'


def directory(cell,phase,shard=0):
    return root/'cells'/cell['name']/phase/(f'shard_{shard}' if phase=='formal' else 'first3')


def validate(cell,phase,shard=0):
    cfg=read(root/'config.json');out=directory(cell,phase,shard)
    assert read(out/'exit.json')['exit_code']==0
    expected=set(range(3)) if phase=='smoke' else {i for i in range(3,60) if i%cfg['shards']==shard}
    refs={r['idx']:r for r in rows(root/'inputs/hmmt.jsonl')};mm=rows(out/'cache_metrics.jsonl')
    assert len(mm)==len(expected) and {m['idx'] for m in mm}==expected
    hq=cfg['model_geometry']['num_attention_heads'];layers=cfg['model_geometry']['num_hidden_layers'];factor=hq*layers
    for m in mm:
        ref=refs[m['idx']];p=m['input_tokens'];k=cell['K'];ts=m['generated_tokens_per_rollout']
        assert (p,m['input_ids_sha256'])==(ref['input_tokens'],ref['input_ids_sha256'])
        assert len(ts)==len(m['eos_reached'])==2
        assert m['selection_strategy']=='official_random_pp' and m['token_budget']==k and m['residual_length']==64
        assert m['cache_semantics']=='persistent_eviction' and m['random_scope']=='independent_per_kv_head'
        assert m['decode_forward_calls']==max(ts)-1 and m['decode_query_tokens']==sum(ts)-2
        pf=2*factor*p*(p+1)//2;selected=0
        for t,eos in zip(ts,m['eos_reached']):
            assert 1<=t<=cfg['max_tokens']-p
            if not eos:assert t==cfg['max_tokens']-p
            stored=p
            for step in range(1,t):
                stored+=1;selected+=stored
                if stored>k and stored%64==0:stored=k
        causal=factor*sum((p+t-1)*(p+t)//2 for t in ts)
        assert m['prefill_selected_pairs']==m['prefill_causal_pairs']==pf
        assert m['selected_pairs']==pf+factor*selected and m['causal_pairs']==causal
        stored=p;events=0
        for step in range(1,max(ts)):
            stored+=1
            if stored>k and stored%64==0:stored=k;events+=1
        assert m['selection_events']==layers*events
        assert all(all(n==stored for n in layer) for layer in m['final_cache_seqlens'])
        assert all(n==p+max(ts)-1 for n in m['final_cumulative_length'])
        assert m['max_selected_tokens']<=k+64 and m['max_capacity']==k+64
    completions={}
    for run in [0,1]:
        files=list((out/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1
        rr=rows(files[0]);assert len(rr)==len(expected) and {r['idx'] for r in rr}==expected
        for r in rr:
            prefix=refs[r['idx']]['prompt_text'];assert r['completion'].startswith(prefix)
            completions[run,r['idx']]=r['completion'][len(prefix):]
    return mm,completions


def infer(phase,index):
    cfg=verify();proof=read(root/'correctness.json')
    assert proof['status']=='passed' and sha(root/'correctness.log')==proof['log_sha256']
    if phase=='formal':
        gate=read(root/'smoke_gate.json');assert gate['status']=='passed'
        for rel,digest in gate['validated_files'].items():assert sha(root/rel)==digest,rel
        work=read(root/'formal_work.json')[index];ci=work['cell'];shard=work['shard']
    else:ci=index;shard=0
    cell=cfg['cells'][ci];out=directory(cell,phase,shard);out.mkdir(parents=True)
    for rel,digest in read(root/'model_assets_sha256.json').items():assert sha(Path(cfg['model'])/rel)==digest,rel
    import torch
    gpu=torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count()==1 and 'A100' in gpu.name and gpu.total_memory>75*1024**3
    packages={k:importlib.metadata.version(k) for k in cfg['packages']};assert packages==cfg['packages']
    write(out/'hardware.json',dict(gpu=gpu.name,memory=gpu.total_memory,node=platform.node(),job=os.environ['SLURM_JOB_ID'],packages=packages))
    args=[sys.executable,'-u',str(root/'source/ra_harness/kvcompress/harness/eval_hf.py'),
        '--model_name_or_path',cfg['model'],'--data_name','hmmt','--split','test','--prompt_type','qwen-instruct',
        '--surround_with_messages','--attention_implementation','flash_attention_2','--sparsity_method','eviction',
        '--eviction_mode','official_random_pp','--token_budget',str(cell['K']),'--residual_length','64',
        '--query_window','1','--eviction_interval','1','--top_p','.95','--mask_seed','42','--rep_seed','0',
        '--seed','42','--batch_size','2','--run_id','0','--max_tokens','32768','--warmup_new_tokens','2',
        '--use_batch_exist','--output_dir',str(out)]
    if phase=='smoke':args+=['--ex_start_i','0','--ex_end_i','3']
    else:args+=['--ex_start_i','3','--ex_end_i','60','--num_shards',str(cfg['shards']),'--shard_id',str(shard)]
    write(out/'command.json',args)
    with (out/'runner.log').open('x') as stream:
        proc=subprocess.run(args,env=env_for(root,'hmmt'),stdout=stream,stderr=subprocess.STDOUT)
    write(out/'exit.json',dict(exit_code=proc.returncode));assert proc.returncode==0
    validate(cell,phase,shard)
    write(out/'validation.json',dict(status='passed',phase=phase,cell=cell['name'],input_alignment='passed/full_within_shard'))


def smoke_gate():
    cfg=verify();validated={};scope=[]
    for cell in cfg['cells']:
        mm,_=validate(cell,'smoke');scope.append(dict(cell=cell['name'],questions=len(mm)))
        out=directory(cell,'smoke')
        for name in ['cache_metrics.jsonl','exit.json','validation.json']:
            p=out/name;validated[str(p.relative_to(root))]=sha(p)
    proof=dict(status='passed',scope=scope,input_alignment='passed/full_within_smoke',validated_files=validated,
               reuse='Smoke IDs0,1,2 count as the first3 inputs of each full60 result; formal generation starts at ID3.')
    write(root/'smoke_gate.json',proof);print(json.dumps(proof),flush=True)


def collect():
    cfg=verify();assert read(root/'smoke_gate.json')['status']=='passed'
    refs={r['idx']:r for r in rows(root/'inputs/hmmt.jsonl')};summaries=[];all_grades=[]
    for cell in cfg['cells']:
        mm,texts=validate(cell,'smoke')
        for shard in range(cfg['shards']):
            more,comps=validate(cell,'formal',shard);mm+=more
            assert not set(texts)&set(comps);texts.update(comps)
        assert len(mm)==60 and {m['idx'] for m in mm}==set(range(60))
        assert len(texts)==120
        grades=[]
        for (run,idx),text in sorted(texts.items()):
            prediction,correct=grade_bounded(text,refs[idx]['gold'])
            grades.append(dict(cell=cell['name'],run=run,idx=idx,prediction=prediction,correct=int(correct)))
        all_grades+=grades
        seconds=sum(m['decode_seconds'] for m in mm);calls=sum(m['decode_forward_calls'] for m in mm)
        selected=sum(m['selected_pairs'] for m in mm);causal=sum(m['causal_pairs'] for m in mm)
        values=[t for m in mm for t in m['generated_tokens_per_rollout']]
        record=dict(method='official_random_pp',K=cell['K'],recent64=True,questions=60,answers=120,
            correct=sum(g['correct'] for g in grades),accuracy=sum(g['correct'] for g in grades)/120,
            input_alignment='passed/full',pair_count_crosscheck='passed/full',
            mean_decode_seconds=seconds/60,median_decode_seconds=statistics.median(m['decode_seconds'] for m in mm),
            mean_prefill_seconds=statistics.mean(m['prefill_seconds'] for m in mm),
            mean_total_seconds=statistics.mean(m['batch_wall_seconds'] for m in mm),
            aggregate_ms_per_decode_forward=1000*seconds/calls,
            active_decode_tokens_per_second=sum(m['decode_query_tokens'] for m in mm)/seconds,
            mean_generated_tokens=statistics.mean(values),median_generated_tokens=statistics.median(values),
            cap_hits=sum(not eos for m in mm for eos in m['eos_reached']),
            selected_pairs=selected,causal_pairs=causal,global_pair_sparsity=1-selected/causal,
            maximum_peak_allocated_gib=max(m['peak_memory_allocated'] for m in mm)/2**30,
            cache_semantics='persistent_eviction',mask_scope='independent_per_kv_head')
        summaries.append(record)
    result=dict(status='completed',input_alignment='passed/full',cells=summaries,
        protocol='60 canonical HMMT questions x2 sampled answers per K; smoke first3 reused; generation settings matched to Dense and W1.',
        caveats=cfg['comparison_caveats'])
    write(root/'summary.json',result)
    (root/'grades.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in all_grades))
    (root/'analysis/results_en.txt').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


action=sys.argv[1]
if action=='check':check()
elif action in ['smoke','formal']:infer(action,int(sys.argv[3]))
elif action=='gate':smoke_gate()
elif action=='collect':collect()
else:raise ValueError(action)
