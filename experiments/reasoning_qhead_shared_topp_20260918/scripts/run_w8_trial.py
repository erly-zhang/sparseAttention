import datetime,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path
from rep_compact_math_common import sha
root=Path(sys.argv[1]).resolve();stage=sys.argv[2];cfg=json.loads((root/'config.json').read_text())
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
assert cfg['sample_ids']==list(range(20)) and not cfg['auto_full_submission']
env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),RA_DATA_DIR=str(root/'data'),QHEAD_DEBUG='0')
for k in ('QHEAD_LOG','KEEPLOG','KEEPLOG_POS','FORCE_KEEP_RANGE'):env.pop(k,None)
import torch
assert torch.cuda.device_count()==1
gpu=torch.cuda.get_device_properties(0);assert 'A100' in gpu.name and gpu.total_memory>75*1024**3
packages={k:importlib.metadata.version(k) for k in cfg['packages']};assert packages==cfg['packages']
if stage=='check':out=root/'correctness'
else:out=root/f'shard_{int(os.environ["SLURM_ARRAY_TASK_ID"])}'
out.mkdir();(out/'hardware.json').write_text(json.dumps(dict(node=platform.node(),gpu=gpu.name,gpu_memory=gpu.total_memory,packages=packages,job_id=os.environ['SLURM_JOB_ID'],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2)+'\n')
def status(phase,**kw):
 x=dict(phase=phase,time=datetime.datetime.now().astimezone().isoformat(),**kw);(out/'status.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
try:
 if stage=='check':
  status('gpu_correctness')
  with (out/'tests.log').open('x') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'source/tests'),'-p','test*shared*mask*.py','-v'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 else:
  assert json.loads((root/'correctness/status.json').read_text())['phase']=='completed'
  for rel,h in json.loads((root/'model_assets_sha256.json').read_text()).items():assert sha(Path(cfg['model'])/rel)==h,rel
  shard=int(os.environ['SLURM_ARRAY_TASK_ID']);order=['w1','w8'] if shard%2==0 else ['w8','w1']
  for name in order:
   status('full_generation',active_method=name,order=order);d=out/name;d.mkdir()
   mode=cfg['baseline_method'] if name=='w1' else cfg['method'];w='1' if name=='w1' else '8'
   args=[sys.executable,'-u',str(root/'source/ra_harness/kvcompress/harness/eval_hf.py'),'--model_name_or_path',cfg['model'],'--data_name','math','--split','test','--prompt_type','qwen-instruct','--surround_with_messages','--attention_implementation','flash_attention_2','--sparsity_method','eviction','--query_window',w,'--eviction_mode',mode,'--token_budget','0','--residual_length','0','--reasoning_budget','1024','--recent_window','64','--mask_seed','42','--eviction_interval','64','--rep_seed','0','--seed','42','--batch_size','2','--run_id','0','--max_tokens','32768','--warmup_new_tokens','2','--ex_start_i','0','--ex_end_i','20','--num_shards','4','--shard_id',str(shard),'--use_batch_exist','--output_dir',str(d)]
   (d/'command.json').write_text(json.dumps(args,indent=2)+'\n')
   with (d/'runner.log').open('x') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
   (d/'exit.json').write_text(json.dumps(dict(exit_code=r.returncode))+'\n');assert r.returncode==0,name
 status('completed')
except Exception as exc:status('failed',error=repr(exc));raise
