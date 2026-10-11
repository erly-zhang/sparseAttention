"""Frozen Bernoulli MATH500 smoke, workers, validation and collection."""
import datetime,hashlib,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path
MODE='qhead_shared_mask_bernoulli_recent_gpu'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def verify(root):
    for rel,h in read(root/'manifest.json').items():assert sha(root/rel)==h,rel
    cfg=read(root/'config.json')
    assert cfg['method']==MODE and cfg['random_keep_ratio']==.2 and cfg['full_run_authorized']
    assert sha('/project/yl258/jz648/Random-Attention/data/math/test.jsonl')==cfg['dataset_sha256']
    return cfg

def env_for(root):
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),RA_DATA_DIR='/project/yl258/jz648/Random-Attention/data',QHEAD_DEBUG='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    for k in ('KEEPLOG','QHEAD_LOG','KEEPLOG_POS','FORCE_KEEP_RANGE'):env.pop(k,None)
    return env

def inference(root,phase,shard=0):
    cfg=verify(root)
    out=root/'smoke' if phase=='smoke' else root/f'random/shard_{shard}'
    out.mkdir()
    try:
        proof=read(root/'correctness.json');assert proof['status']=='passed' and sha(root/'correctness.log')==proof['log_sha256']
        if phase!='smoke':
            gate=read(root/'smoke_gate.json');assert gate['status']=='passed' and sha(root/'smoke/summary.json')==gate['summary_sha256']
        for rel,h in read(root/'model_assets_sha256.json').items():assert sha(Path(cfg['model'])/rel)==h,rel
        import torch
        assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
        gpu=torch.cuda.get_device_properties(0);assert 'A100' in gpu.name and gpu.total_memory>75*1024**3
        packages={k:importlib.metadata.version(k) for k in cfg['packages']}
        assert packages==cfg['packages']
        hardware=dict(node=platform.node(),gpu_name=gpu.name,gpu_memory=gpu.total_memory,packages=packages,job_id=os.environ.get('SLURM_JOB_ID'),array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        write(out/'hardware.json',hardware);write(out/'preflight.json',dict(status='passed',manifest_sha256=sha(root/'manifest.json'),model_hashes='passed',input_dataset_hash='passed'))
        cmd=[sys.executable,'-u',str(root/'source/ra_harness/kvcompress/harness/eval_hf.py'),'--model_name_or_path',cfg['model'],'--data_name','math','--split','test','--prompt_type','qwen-instruct','--surround_with_messages','--attention_implementation','flash_attention_2','--sparsity_method','eviction','--eviction_mode',MODE,'--token_budget','0','--residual_length','0','--random_keep_ratio','.2','--recent_window','64','--mask_seed','42','--rep_seed','0','--seed','42','--eviction_interval','1','--mask_precompute_window','16','--batch_size','2','--run_id','0','--max_tokens','32768','--warmup_new_tokens','2','--use_batch_exist','--output_dir',str(out)]
        cmd+=['--ex_start_i','0','--ex_end_i','3'] if phase=='smoke' else ['--num_shards','8','--shard_id',str(shard)]
        write(out/'command.json',cmd)
        with (out/'runner.log').open('x') as f:proc=subprocess.run(cmd,env=env_for(root),stdout=f,stderr=subprocess.STDOUT)
        write(out/'exit.json',dict(exit_code=proc.returncode));assert proc.returncode==0,proc.returncode
        if phase=='smoke':
            subprocess.run([sys.executable,str(root/'source/scripts/check_bernoulli_math500.py'),str(root),'--phase','smoke'],env=env_for(root),check=True)
    except Exception as e:
        write(out/'failure.json',dict(error=repr(e)))
        if not (out/'exit.json').exists():write(out/'exit.json',dict(exit_code=1,error=repr(e)))
        raise

def collect(root):
    verify(root)
    try:
        for i in range(8):assert read(root/f'random/shard_{i}/exit.json')['exit_code']==0,i
        subprocess.run([sys.executable,str(root/'source/scripts/check_bernoulli_math500.py'),str(root),'--phase','formal'],env=env_for(root),check=True)
        s=read(root/'random/summary.json')
        write(root/'collection_status.json',dict(status='completed',questions=s['questions'],rollouts=s['rollouts'],accuracy=s['accuracy'],input_alignment=s['input_alignment']))
    except Exception as e:
        write(root/'collection_status.json',dict(status='failed_or_incomplete',error=repr(e)));raise

if __name__=='__main__':
    root=Path(sys.argv[2]).resolve()
    if sys.argv[1]=='smoke':inference(root,'smoke')
    elif sys.argv[1]=='worker':inference(root,'formal',int(sys.argv[3]))
    elif sys.argv[1]=='collect':collect(root)
    else:raise ValueError('unknown action')
