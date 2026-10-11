"""Resume only missing samples, preserving original outputs and inference bytes."""
import importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path
from bernoulli_math500 import read,write,sha,verify,env_for


def verify_recovery(root):
    cfg=verify(root);plan=read(root/'recovery_plan.json');old=Path(plan['original_root'])
    for rel,h in read(root/'historical_output_hashes.json').items():assert sha(old/rel)==h,rel
    for rel,h in read(root/'inference_source_hashes.json').items():assert sha(root/rel)==h and sha(old/rel)==h,rel
    proof=read(root/'correctness.json');assert proof['status']=='passed' and sha(root/'correctness.log')==proof['log_sha256']
    gate=read(root/'smoke_gate.json');assert gate['status']=='passed' and sha(root/'smoke/summary.json')==gate['summary_sha256']
    return cfg,plan,old


def load_outputs(d):
    metrics=[json.loads(s) for s in (d/'cache_metrics.jsonl').read_text().splitlines()]
    assert len({x['idx'] for x in metrics})==len(metrics)
    comps={}
    for run in (0,1):
        files=list((d/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1
        rows=[json.loads(s) for s in files[0].read_text().splitlines()]
        assert len({x['idx'] for x in rows})==len(rows)
        assert {x['idx'] for x in rows}=={x['idx'] for x in metrics}
        comps[run]=rows
    return metrics,comps


def worker(root,shard):
    cfg,plan,old=verify_recovery(root)
    missing=plan['missing_by_shard'][str(shard)]
    assert missing==list(range(missing[0],500,8)) and all(x%8==shard for x in missing)
    out=root/f'repair/shard_{shard}';out.mkdir()
    try:
        for rel,h in read(root/'model_assets_sha256.json').items():assert sha(Path(cfg['model'])/rel)==h,rel
        import torch
        assert os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_ARRAY_TASK_ID'])==shard
        assert torch.cuda.device_count()==1
        gpu=torch.cuda.get_device_properties(0);assert 'A100' in gpu.name and gpu.total_memory>75*1024**3
        assert platform.node().split('.')[0] not in plan['excluded_nodes']
        packages={k:importlib.metadata.version(k) for k in cfg['packages']};assert packages==cfg['packages']
        write(out/'hardware.json',dict(node=platform.node(),gpu_name=gpu.name,gpu_memory=gpu.total_memory,packages=packages,job_id=os.environ.get('SLURM_JOB_ID'),array_task_id=shard,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        write(out/'preflight.json',dict(status='passed',source_identical_to_original=True,model_hashes='passed',sample_ids=missing))
        cmd=read(old/f'random/shard_{shard}/command.json')
        cmd[0]=sys.executable;cmd[2]=str(root/'source/ra_harness/kvcompress/harness/eval_hf.py')
        cmd[cmd.index('--output_dir')+1]=str(out)
        assert '--ex_start_i' not in cmd and '--ex_end_i' not in cmd
        cmd+=['--ex_start_i',str(missing[0]),'--ex_end_i','500']
        write(out/'command.json',cmd)
        with (out/'runner.log').open('x') as f:
            proc=subprocess.run(cmd,env=env_for(root),stdout=f,stderr=subprocess.STDOUT)
        write(out/'exit.json',dict(exit_code=proc.returncode));assert proc.returncode==0
        metrics,comps=load_outputs(out);assert sorted(x['idx'] for x in metrics)==missing
        expected={x['idx']:x for x in map(json.loads,(root/'inputs.jsonl').read_text().splitlines())}
        for row in metrics:
            ref=expected[row['idx']]
            assert row['input_tokens']==ref['input_tokens'] and row['input_ids_sha256']==ref['input_ids_sha256']
        write(out/'validation.json',dict(status='passed',input_alignment='passed/missing subset',questions=len(missing),rollouts=2*len(missing),sample_ids=missing))
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc)))
        if not (out/'exit.json').exists():write(out/'exit.json',dict(exit_code=1))
        raise


def collect(root):
    try:
        cfg,plan,old=verify_recovery(root)
        for shard in plan['missing_by_shard']:
            d=root/f'repair/shard_{shard}'
            assert read(d/'exit.json')['exit_code']==0 and read(d/'validation.json')['status']=='passed'
        write(root/'collection_status.json',dict(status='merging_validated_outputs'))
        provenance=[]
        for shard in range(8):
            sources=[old/f'random/shard_{shard}']
            if str(shard) in plan['missing_by_shard']:sources.append(root/f'repair/shard_{shard}')
            else:assert read(sources[0]/'exit.json')['exit_code']==0
            all_metrics=[];all_comps={0:[],1:[]};seen=set();hardware=[]
            for d in sources:
                metrics,comps=load_outputs(d)
                ids={x['idx'] for x in metrics};assert not seen.intersection(ids)
                if d==sources[0]:assert sorted(ids)==plan['completed_by_shard'][str(shard)]
                else:assert sorted(ids)==plan['missing_by_shard'][str(shard)]
                seen.update(ids);all_metrics.extend(metrics)
                for run in (0,1):all_comps[run].extend(comps[run])
                hardware.append(dict(source=str(d),hardware=read(d/'hardware.json'),sample_ids=sorted(ids)))
                provenance.extend(dict(idx=idx,source_directory=str(d),recovered=d!=sources[0]) for idx in sorted(ids))
            assert seen==set(range(shard,500,8))
            out=root/f'random/shard_{shard}';out.mkdir()
            with (out/'cache_metrics.jsonl').open('x') as f:
                f.writelines(json.dumps(x)+'\n' for x in sorted(all_metrics,key=lambda x:x['idx']))
            for run in (0,1):
                (out/f'run_{run}').mkdir()
                with (out/f'run_{run}/completions.jsonl').open('x') as f:
                    f.writelines(json.dumps(x)+'\n' for x in sorted(all_comps[run],key=lambda x:x['idx']))
            write(out/'hardware.json',dict(assembly='original successful samples plus independent recovery',sources=hardware))
            write(out/'exit.json',dict(exit_code=0,meaning='validated assembly; original failed job exits preserved at their source',source_directories=[str(d) for d in sources]))
        write(root/'sample_provenance.json',sorted(provenance,key=lambda x:x['idx']))
        write(root/'collection_status.json',dict(status='grading_full500'))
        subprocess.run([sys.executable,str(root/'source/scripts/check_bernoulli_math500.py'),str(root),'--phase','formal'],env=env_for(root),check=True)
        s=read(root/'random/summary.json')
        s.update(recovery=dict(original_complete_questions=471,recovered_questions=29,inference_source_identical=True,original_failed_nodes=plan['excluded_nodes']),timing_note='471 original plus29 recovered samples; recovery hardware recorded per sample in sample_provenance.json and shard hardware.json. No failed attempt time is included in successful-generation latency.')
        write(root/'random/summary.json',s)
        write(root/'collection_status.json',dict(status='completed',questions=s['questions'],rollouts=s['rollouts'],accuracy=s['accuracy'],input_alignment=s['input_alignment']))
    except Exception as exc:
        write(root/'collection_status.json',dict(status='failed_or_incomplete',error=repr(exc)));raise

if __name__=='__main__':
    root=Path(sys.argv[2]).resolve()
    if sys.argv[1]=='worker':worker(root,int(sys.argv[3]))
    elif sys.argv[1]=='collect':collect(root)
    else:raise ValueError('unknown action')
