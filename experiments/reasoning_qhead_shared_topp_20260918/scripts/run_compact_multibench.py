"""One allocated GPU: immutable implementation, stable global question IDs, no resume."""
import datetime,importlib.metadata,json,os,platform,subprocess,sys,time
from pathlib import Path
from compact_multibench_common import verify,sha
root=Path(sys.argv[1]).resolve();phase=sys.argv[2];index=int(os.environ['SLURM_ARRAY_TASK_ID'])
assert phase in ('smoke','formal')
cfg=verify(root,weights=True)
if phase=='smoke':
 task=cfg['tasks'][index];work=dict(task=task['name'],shard=0,num_shards=1)
else:
 gate=json.loads((root/'smoke_gate.json').read_text());assert gate['status']=='passed'
 for rel,h in gate['summaries'].items():assert sha(root/rel)==h
 work=json.loads((root/'formal_work.json').read_text())[index]
 task=next(t for t in cfg['tasks'] if t['name']==work['task'])
out=root/work['task']/phase/f"shard_{work['shard']}";out.mkdir(parents=True)
rc=1
try:
 assert cfg['reasoning_budget']==1024 and cfg['recent_window']==64 and cfg['prompt_outside_budget'] and cfg['recent_inside_budget']
 assert cfg['cache_semantics']=='persistent_eviction' and cfg['compaction_phase']=='after_attention'
 assert cfg['full_run_authorized'] and cfg['rollouts']==cfg['batch_size']==2 and cfg['seed']==cfg['mask_seed']==42
 assert json.loads((root/'correctness.json').read_text())['status']=='passed'
 assert sha(root/'correctness.log')==json.loads((root/'correctness.json').read_text())['log_sha256']
 import torch
 assert torch.cuda.device_count()==1
 gpu=torch.cuda.get_device_properties(0);assert 'A100' in gpu.name and gpu.total_memory>75*1024**3
 packages={k:importlib.metadata.version(k) for k in ('torch','transformers','flash-attn','numpy','triton')}
 for k,v in cfg['packages'].items():assert packages[k]==v,(k,packages[k],v)
 hardware=dict(node=platform.node(),gpu_name=gpu.name,gpu_total_memory=gpu.total_memory,packages=packages,cuda=torch.version.cuda,job_id=os.environ['SLURM_JOB_ID'],array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],array_task_id=index,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cpu_affinity=sorted(os.sched_getaffinity(0)),timestamp=datetime.datetime.now().astimezone().isoformat())
 (out/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
 sample_ids=[0,1,2] if phase=='smoke' else list(range(work['shard'],task['count'],work['num_shards']))
 (out/'preflight.json').write_text(json.dumps(dict(status='passed',manifest_sha256=sha(root/'manifest.json'),model_assets='passed',sample_ids=sample_ids,work=work),indent=2)+'\n')
 env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),RA_DATA_DIR=str(root/'data'),QHEAD_DEBUG='0')
 for k in ('QHEAD_LOG','KEEPLOG','KEEPLOG_POS','FORCE_KEEP_RANGE'):env.pop(k,None)
 args=[sys.executable,'-u',str(root/'source/ra_harness/kvcompress/harness/eval_hf.py'),'--model_name_or_path',cfg['model'],'--data_name',work['task'],'--split','test','--prompt_type','qwen-instruct','--surround_with_messages','--attention_implementation','flash_attention_2','--sparsity_method','eviction','--eviction_mode',cfg['method'],'--token_budget','0','--residual_length','0','--reasoning_budget','1024','--recent_window','64','--mask_seed','42','--eviction_interval','64','--rep_seed','0','--seed','42','--batch_size','2','--run_id','0','--max_tokens','32768','--warmup_new_tokens','2','--num_shards',str(work['num_shards']),'--shard_id',str(work['shard']),'--use_batch_exist','--output_dir',str(out)]
 if phase=='smoke':args+=['--ex_start_i','0','--ex_end_i','3']
 (out/'command.json').write_text(json.dumps(args,indent=2)+'\n')
 for when in ('before','after'):
  if when=='after':
   with (out/'runner.log').open('x') as f:rc=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  with (out/f'gpu_{when}.csv').open('x') as f:subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used','--format=csv'],stdout=f,check=True)
finally:
 (out/'exit.json').write_text(json.dumps(dict(exit_code=rc,ended_epoch=time.time()))+'\n')
sys.exit(rc)
