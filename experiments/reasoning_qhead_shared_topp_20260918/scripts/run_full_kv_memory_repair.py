"""Frozen, gated replay of full-KV fixed-budget MATH500; never modifies old runs."""
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def validate(d,ids,root):
    cfg=read(root/'config.json');expected={r['idx']:r for r in rows(root/'inputs.jsonl')}
    mm=rows(d/'cache_metrics.jsonl');assert len(mm)==len(ids) and {r['idx'] for r in mm}==set(ids)
    for r in mm:
        e=expected[r['idx']];P=r['input_tokens'];ts=r['generated_tokens_per_rollout']
        assert P==e['input_tokens'] and r['input_ids_sha256']==e['input_ids_sha256']
        assert r['max_capacity']==32768 and r['cache_semantics']=='full_history_temporary_mask'
        assert r['reasoning_budget']==1024 and r['recent_window']==64
        assert r['memory_allocated_after_generation'] <= r['memory_allocated_before_generation']+256*1024**2
        assert len(ts)==2
        for t,eos in zip(ts,r['eos_reached']):
            assert 1<=t<=32768-P
            if not eos:assert t==32768-P
        factor=36*32;pf=2*factor*P*(P+1)//2
        ds=factor*sum((t-1)*P+sum(min(j,1024) for j in range(1,t)) for t in ts)
        causal=factor*sum((P+t-1)*(P+t)//2 for t in ts)
        assert r['selected_pairs']==pf+ds and r['causal_pairs']==causal
        assert all(n==P+max(ts)-1 for n in r['final_cumulative_length'])
    for run in (0,1):
        pp=list((d/f'run_{run}').glob('completions*.jsonl'));assert len(pp)==1
        rr=rows(pp[0]);assert len(rr)==len(ids) and {r['idx'] for r in rr}==set(ids)
        assert all(r['completion'].startswith(expected[r['idx']]['prompt_text']) for r in rr)
    write(d/'validation.json',dict(status='passed',ids=ids,input_alignment='passed/full subset',exact_pair_budget='passed',memory_release='passed/boundary allocated memory',peak_memory_allocated=max(r['peak_memory_allocated'] for r in mm)))

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('phase',choices=['smoke','oom','repair']);p.add_argument('--worker',type=int,default=0);a=p.parse_args();root=a.root.resolve();cfg=read(root/'config.json');plan=read(root/'repair_plan.json')
for rel,h in read(root/'manifest.json').items():assert sha(root/rel)==h,rel
assert read(root/'correctness.json')['status']=='passed'
for rel,h in read(root/'model_assets_sha256.json').items():assert sha(Path(cfg['model'])/rel)==h,rel
assert sha(Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl'))==cfg['dataset_sha256']
for k,v in cfg['packages'].items():assert importlib.metadata.version(k)==v,(k,v)
if a.phase=='smoke':
    d=root/'smoke';start,end,mod,rem=0,3,1,0;ids=[0,1,2]
else:
    assert read(root/'smoke_regression.json')['status']=='passed'
    if a.phase=='oom':
        start=plan['failed_ids'][a.worker];end=start+1;mod,rem=1,0;ids=[start];d=root/f'oom_replay/worker_{a.worker}'
    else:
        for i in (0,1):assert read(root/f'oom_replay/worker_{i}/validation.json')['status']=='passed'
        w=plan['workers'][a.worker];start=w['start'];mod=w['modulus'];rem=w['remainder'];end=500
        if rem in (8,12):start+=16  # reuse successful OOM replay records
        ids=[i for i in range(start,end) if i%mod==rem]
        assert set(ids)<=set(plan['missing_ids'])-set(plan['failed_ids'])
        d=root/f'repair/worker_{a.worker}'
d.mkdir(parents=True,exist_ok=False)
rc=1
try:
    import torch
    assert torch.cuda.device_count()==1
    g=torch.cuda.get_device_properties(0);assert 'A100' in g.name and g.total_memory>75*1024**3
    write(d/'hardware.json',dict(node=platform.node(),gpu_name=g.name,gpu_total_memory=g.total_memory,job_id=os.environ.get('SLURM_JOB_ID'),cuda=torch.version.cuda,packages={k:importlib.metadata.version(k) for k in cfg['packages']}))
    cmd=[sys.executable,'-u',str(root/'source/ra_harness/kvcompress/harness/eval_hf.py'),
        '--model_name_or_path',cfg['model'],'--data_name','math','--split','test','--prompt_type','qwen-instruct','--surround_with_messages',
        '--attention_implementation','flash_attention_2','--sparsity_method','eviction','--eviction_mode',cfg['method'],
        '--token_budget','0','--residual_length','0','--reasoning_budget','1024','--recent_window','64','--mask_seed','42','--eviction_interval','1','--rep_seed','0','--seed','42',
        '--batch_size','2','--run_id','0','--max_tokens','32768','--warmup_new_tokens','2','--num_shards',str(mod),'--shard_id',str(rem),
        '--ex_start_i',str(start),'--ex_end_i',str(end),'--use_batch_exist','--output_dir',str(d)]
    write(d/'invocation.json',dict(command=cmd,expected_ids=ids))
    with (d/'runner.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    validate(d,ids,root)
    rc=0
finally:write(d/'exit.json',dict(exit_code=rc))
if a.phase=='smoke':
    old=Path(plan['old_root']);comparisons=[]
    om={r['idx']:r for i in range(8) for r in rows(old/f'random/shard_{i}/cache_metrics.jsonl')}
    for r in rows(d/'cache_metrics.jsonl'):
        for key in ('generated_tokens_per_rollout','eos_reached','selected_pairs','causal_pairs','input_tokens','input_ids_sha256'):
            assert r[key]==om[r['idx']][key],(r['idx'],key)
    for run in (0,1):
        old_text={r['idx']:r['completion'] for i in range(8) for f in (old/f'random/shard_{i}/run_{run}').glob('completions*.jsonl') for r in rows(f)}
        for f in (d/f'run_{run}').glob('completions*.jsonl'):
            for r in rows(f):
                assert r['completion']==old_text[r['idx']],(run,r['idx'],'output mismatch')
                comparisons.append(dict(idx=r['idx'],run=run,identical=True))
    subprocess.run([sys.executable,str(root/'source/scripts/check_random_fixed_recent_smoke.py'),str(root),'--phase','smoke'],check=True)
    write(root/'smoke_regression.json',dict(status='passed',comparisons=comparisons,token_counts_pairs_eos='identical',memory_release='passed'))
print(a.phase,ids,'passed',flush=True)
