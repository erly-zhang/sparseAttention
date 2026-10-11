"""Frozen 40-cell selector grid: smoke gate, sharded inference, bounded grading."""
import datetime,hashlib,importlib.metadata,json,multiprocessing,os,platform,statistics,subprocess,sys
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def verify(root):
    for rel,h in read(root/'manifest.json').items():assert sha(root/rel)==h,rel
    return read(root/'config.json')
def env_for(root,task=None):
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),RA_DATA_DIR=str(root/'data'),QHEAD_DEBUG='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    for k in ('QHEAD_LOG','KEEPLOG','KEEPLOG_POS','FORCE_KEEP_RANGE','RA_GRID_REFERENCE'):env.pop(k,None)
    if task:env['RA_GRID_REFERENCE']=str(root/'inputs'/f'{task}.jsonl')
    return env

def inference(root,phase,index):
    cfg=verify(root);work=read(root/('smoke_work.json' if phase=='smoke' else 'formal_work.json'))[index]
    cell=cfg['cells'][work['cell']];out=root/'cells'/cell['name']/phase/f"shard_{work['shard']}";out.mkdir(parents=True)
    proof=read(root/'correctness.json');assert proof['status']=='passed' and sha(root/'correctness.log')==proof['log_sha256']
    if phase=='formal':
        gate=read(root/'smoke_gate.json');assert gate['status']=='passed'
        for rel,h in gate['summary_hashes'].items():assert sha(root/rel)==h,rel
    for rel,h in read(root/'model_assets_sha256.json').items():assert sha(Path(cfg['model'])/rel)==h,rel
    import torch
    assert torch.cuda.device_count()==1 and os.environ.get('SLURM_JOB_ID')
    gpu=torch.cuda.get_device_properties(0);assert 'A100' in gpu.name and gpu.total_memory>75*1024**3
    packages={k:importlib.metadata.version(k) for k in cfg['packages']};assert packages==cfg['packages']
    write(out/'hardware.json',dict(node=platform.node(),gpu=gpu.name,memory=gpu.total_memory,packages=packages,job=os.environ.get('SLURM_JOB_ID'),array=os.environ.get('SLURM_ARRAY_JOB_ID'),task_id=index))
    args=[sys.executable,'-u',str(root/'source/ra_harness/kvcompress/harness/eval_hf.py'),'--model_name_or_path',cfg['model'],'--data_name',cell['task'],'--split','test','--prompt_type','qwen-instruct','--surround_with_messages','--attention_implementation','flash_attention_2','--sparsity_method','eviction','--eviction_mode',cell['method'],'--token_budget',str(cell['K'] if cell['selector']=='topk' else 0),'--residual_length','0','--top_p','.95','--query_window',str(cell['window']),'--mask_seed','42','--rep_seed','0','--seed','42','--eviction_interval','64' if cell['storage']=='compact' else '1','--batch_size','2','--run_id','0','--max_tokens','32768','--warmup_new_tokens','2','--use_batch_exist','--output_dir',str(out),'--num_shards',str(work['num_shards']),'--shard_id',str(work['shard'])]
    if phase=='smoke':args+=['--ex_start_i','0','--ex_end_i','3']
    write(out/'command.json',args)
    write(out/'status.json',dict(phase='generating',started=datetime.datetime.now().astimezone().isoformat(),work=work))
    with (out/'runner.log').open('x') as f:proc=subprocess.run(args,env=env_for(root,cell['task']),stdout=f,stderr=subprocess.STDOUT)
    write(out/'exit.json',dict(exit_code=proc.returncode,ended=datetime.datetime.now().astimezone().isoformat()))
    if proc.returncode:raise RuntimeError(f'runner exited {proc.returncode}')
    # Technical validation only on GPU. Grading happens on CPU after release.
    validate_shard(root,cell,phase,work)
    write(out/'status.json',dict(phase='completed',ended=datetime.datetime.now().astimezone().isoformat()))


def validate_shard(root,cell,phase,work):
    cfg=read(root/'config.json');d=root/'cells'/cell['name']/phase/f"shard_{work['shard']}"
    assert read(d/'exit.json')['exit_code']==0,d
    expected={r['idx']:r for r in rows(root/'inputs'/f"{cell['task']}.jsonl")}
    ids=set(range(3)) if phase=='smoke' else set(range(work['shard'],cell['count'],work['num_shards']))
    metrics=rows(d/'cache_metrics.jsonl');assert len(metrics)==len(ids) and {r['idx'] for r in metrics}==ids
    h=cfg['model_geometry']['num_attention_heads'];layers=cfg['model_geometry']['num_hidden_layers'];factor=h*layers
    for r in metrics:
        P=r['input_tokens'];ref=expected[r['idx']]
        assert P==ref['input_tokens'] and r['input_ids_sha256']==ref['input_ids_sha256']
        assert r['grid_method']==cell['method'] and r['grid_storage']==cell['storage'] and r['grid_protection']==cell['protection']
        assert r['grid_selector']==cell['selector'] and r['grid_scorer']==cell['scorer']
        ts=r['generated_tokens_per_rollout'];assert len(ts)==len(r['eos_reached'])==2
        for t,eos in zip(ts,r['eos_reached']):
            assert 1<=t<=cfg['max_tokens']-P
            if not eos:assert t==cfg['max_tokens']-P
        pf=2*factor*P*(P+1)//2
        ca=factor*sum((P+t-1)*(P+t)//2 for t in ts)
        assert r['prefill_selected_pairs']==r['prefill_causal_pairs']==pf
        assert r['causal_pairs']==ca and r['decode_causal_pairs']==ca-pf
        assert r['decode_query_tokens']==sum(ts)-2 and r['decode_forward_calls']==max(ts)-1
        assert r['selected_pairs']==sum(r['per_layer_selected_pairs'])==pf+r['decode_selected_pairs']
        assert r['causal_pairs']==sum(r['per_layer_causal_pairs'])
        assert r['decode_selected_pairs']==h*r['attention_token_sum']
        assert r['attention_token_observations']==layers*(sum(ts)-2)
        assert pf<=r['selected_pairs']<=ca
        if cell['selector']=='topk':
            total=0
            for t in ts:
                cached=P
                for step in range(1,t):
                    cached+=1
                    total+=min(cell['K'],cached) if cell['storage']=='full' else cached
                    if cell['storage']=='compact' and step%64==0 and cached>cell['K']:cached=cell['K']
            assert r['decode_selected_pairs']==factor*total,(r['idx'],r['decode_selected_pairs'],factor*total)
            if cell['storage']=='compact':assert r['max_selected_tokens']<=max(P,cell['K']+64)
        if cell['storage']=='full':
            assert all(all(n==P+max(ts)-1 for n in row) for row in r['final_cache_seqlens'])
        assert all(c==P+max(ts)-1 for c in r['final_cumulative_length'])
        assert r['prefill_seconds']>0 and r['decode_seconds']>=0
    comps={}
    for run in (0,1):
        fs=list((d/f'run_{run}').glob('completions*.jsonl'));assert len(fs)==1
        rr=rows(fs[0]);assert len(rr)==len(ids) and {r['idx'] for r in rr}==ids
        for r in rr:
            prefix=expected[r['idx']]['prompt_text'];assert r['completion'].startswith(prefix)
            comps[run,r['idx']]=r['completion'][len(prefix):]
    return metrics,comps


def grade_child(conn,text,gold):
    try:
        from Utils.parser import extract_answer
        from Utils.grader import check_is_correct
        pred=extract_answer(text)
        conn.send(dict(prediction=pred,correct=bool(check_is_correct(pred,gold))))
    except BaseException as e:conn.send(dict(error=repr(e)))
    finally:conn.close()

def grade_bounded(text,gold):
    # A process boundary is necessary: upstream sympy wrappers may catch alarms.
    ctx=multiprocessing.get_context('fork');parent,child=ctx.Pipe(duplex=False)
    proc=ctx.Process(target=grade_child,args=(child,text,gold));proc.start();child.close()
    try:
        if not parent.poll(30):raise TimeoutError('math grader exceeded 30 seconds')
        result=parent.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result['prediction'],result['correct']
    finally:
        parent.close();proc.join(1)
        if proc.is_alive():proc.terminate();proc.join(2)
        if proc.is_alive():proc.kill();proc.join()


def summarize(root,phase):
    cfg=verify(root)
    sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
    from Utils.parser import extract_answer
    from Utils.grader import check_is_correct
    import numpy as np
    summary_hashes={};all_summaries=[]
    def stats(v):return dict(mean=statistics.mean(v),median=statistics.median(v),p90=float(np.percentile(v,90)),total=sum(v))
    for ci,cell in enumerate(cfg['cells']):
        out=root/'cells'/cell['name']/phase
        metrics=[];comps={};hardware=[];raw={}
        work=[w for w in read(root/('smoke_work.json' if phase=='smoke' else 'formal_work.json')) if w['cell']==ci]
        for w in work:
            rr,cc=validate_shard(root,cell,phase,w);metrics+=rr
            assert not (set(comps)&set(cc));comps.update(cc)
            d=out/f"shard_{w['shard']}";hardware.append(read(d/'hardware.json'))
            for p in [d/'cache_metrics.jsonl',*d.glob('run_*/completions*.jsonl')]:raw[str(p.relative_to(root))]=sha(p)
        expected={r['idx']:r for r in rows(root/'inputs'/f"{cell['task']}.jsonl")}
        byid={r['idx']:r for r in metrics};ids=set(range(3 if phase=='smoke' else cell['count']))
        assert len(metrics)==len(ids) and set(byid)==ids and set(comps)=={(run,i) for run in (0,1) for i in ids}
        assert sum(r['selection_events'] for r in metrics)>0, 'smoke must exercise selection'
        grades=[];unresolved=[]
        for (run,idx),text in sorted(comps.items()):
            try:
                pred,correct=grade_bounded(text,expected[idx]['gold'])
                grades.append(dict(idx=idx,run=run,prediction=pred,correct=int(correct)))
            except Exception as e:unresolved.append(dict(idx=idx,run=run,error=repr(e)))
        if unresolved:
            write(out/'unresolved_grades.json',unresolved)
            raise RuntimeError(f'{cell["name"]}: unresolved grading; no incomplete accuracy published')
        for g in grades:g.update(generated_tokens=byid[g['idx']]['generated_tokens_per_rollout'][g['run']],eos_reached=byid[g['idx']]['eos_reached'][g['run']])
        selected=sum(r['selected_pairs'] for r in metrics);causal=sum(r['causal_pairs'] for r in metrics);decode=sum(r['decode_seconds'] for r in metrics)
        summary=dict(status='passed' if phase=='smoke' else 'completed',cell=cell,questions=len(ids),rollouts=len(grades),correct=sum(g['correct'] for g in grades),accuracy=statistics.mean(g['correct'] for g in grades),runs=[dict(run=run,correct=sum(g['correct'] for g in grades if g['run']==run),count=len(ids)) for run in (0,1)],input_alignment='passed/full' if phase=='formal' else 'passed/3 smoke inputs',pair_accounting='passed/fixed_budget_analytic' if cell['selector']=='topk' else 'passed/variable_actual_pairs',generated_tokens=stats([g['generated_tokens'] for g in grades]),length_limited_rollouts=sum(not g['eos_reached'] for g in grades),prefill_seconds=stats([r['prefill_seconds'] for r in metrics]),decode_seconds=stats([r['decode_seconds'] for r in metrics]),total_seconds=stats([r['batch_wall_seconds'] for r in metrics]),decode_tokens_per_second=sum(r['decode_query_tokens'] for r in metrics)/decode if decode else None,selected_pairs=selected,causal_pairs=causal,global_pair_sparsity=1-selected/causal,mean_attention_positions=sum(r['attention_token_sum'] for r in metrics)/max(1,sum(r['attention_token_observations'] for r in metrics)),mean_selected_positions_at_selection=sum(r['selection_token_sum'] for r in metrics)/max(1,sum(r['selection_token_observations'] for r in metrics)),peak_memory_allocated=max(r['peak_memory_allocated'] for r in metrics),hardware=hardware,raw_files_sha256=raw,timing_scope=cfg['timing'],caveats=cfg['caveats'])
        with (out/'grades.jsonl').open('x') as f:f.writelines(json.dumps(g)+'\n' for g in grades)
        with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
        summary_hashes[str((out/'summary.json').relative_to(root))]=sha(out/'summary.json');all_summaries.append(summary)
        print(json.dumps(dict(cell=cell['name'],phase=phase,accuracy=summary['accuracy'],questions=len(ids))),flush=True)
    if phase=='smoke':write(root/'smoke_gate.json',dict(status='passed',cells=len(cfg['cells']),summary_hashes=summary_hashes))
    else:
        comparisons=[]
        for task in ('math','hmmt'):
            for storage in ('compact','full'):
                for protection in ('protected','none'):
                    base=f'{task}_grid_{storage}_random_topk_{protection}'
                    baseline={(g['run'],g['idx']):g['correct'] for g in rows(root/'cells'/base/'formal/grades.jsonl')}
                    for head in ('w1','w16'):
                        name=f'{task}_grid_{storage}_{head}_topk_{protection}'
                        candidate={(g['run'],g['idx']):g['correct'] for g in rows(root/'cells'/name/'formal/grades.jsonl')}
                        assert set(candidate)==set(baseline)
                        wins=sum(candidate[k]>baseline[k] for k in baseline)
                        losses=sum(candidate[k]<baseline[k] for k in baseline)
                        comparisons.append(dict(baseline=base,candidate=name,paired_outputs=len(baseline),head_only_correct=wins,random_only_correct=losses,both_correct=sum(candidate[k] and baseline[k] for k in baseline),both_incorrect=sum(not candidate[k] and not baseline[k] for k in baseline),accuracy_difference=(wins-losses)/len(baseline)))
        write(root/'paired_topk_comparisons.json',comparisons)
        write(root/'summary.json',dict(status='completed',cells=all_summaries,paired_topk_comparisons=comparisons,source_manifest_sha256=sha(root/'manifest.json')))


if __name__=='__main__':
    action=sys.argv[1];root=Path(sys.argv[2]).resolve()
    if action in ('smoke','formal'):inference(root,action,int(sys.argv[3]))
    elif action=='gate':summarize(root,'smoke')
    elif action=='collect':summarize(root,'formal')
    else:raise ValueError(action)
