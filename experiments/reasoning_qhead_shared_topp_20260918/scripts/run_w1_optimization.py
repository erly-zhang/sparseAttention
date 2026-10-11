"""Mandatory exactness tests, then same-GPU ABBA on three complete generations."""
import datetime,importlib.metadata,json,os,platform,subprocess,sys,time
from pathlib import Path
from rep_compact_math_common import sha,rows
root=Path(sys.argv[1]).resolve();cfg=json.loads((root/'config.json').read_text())
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
for rel,h in json.loads((root/'model_assets_sha256.json').read_text()).items():assert sha(Path(cfg['model'])/rel)==h,rel
assert cfg['query_window']==1 and not cfg['auto_full_submission']
with (root/'pipeline.lock').open('x') as f:f.write(str(os.getpid()))
env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),RA_DATA_DIR=str(root/'data'),QHEAD_DEBUG='0')
for k in ('QHEAD_LOG','KEEPLOG','KEEPLOG_POS','FORCE_KEEP_RANGE'):env.pop(k,None)
def status(phase,**kw):
 x=dict(phase=phase,time=datetime.datetime.now().astimezone().isoformat(),**kw);(root/'pipeline_status.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
try:
 import torch
 assert torch.cuda.device_count()==1
 gpu=torch.cuda.get_device_properties(0);assert 'A100' in gpu.name and gpu.total_memory>75*1024**3
 packages={k:importlib.metadata.version(k) for k in cfg['packages']}
 assert packages==cfg['packages']
 (root/'hardware.json').write_text(json.dumps(dict(node=platform.node(),gpu=gpu.name,gpu_memory=gpu.total_memory,packages=packages,job_id=os.environ['SLURM_JOB_ID'],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2)+'\n')
 status('gpu_correctness')
 with (root/'correctness.log').open('x') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'source/tests'),'-p','test*shared*mask*.py','-v'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'correctness.json').write_text(json.dumps(dict(status='passed',log_sha256=sha(root/'correctness.log')))+'\n')
 for name in cfg['pair_order']:
  status('full_generation',active_pass=name,order=cfg['pair_order']);out=root/name;out.mkdir()
  mode=cfg['baseline_method'] if name.startswith('baseline') else cfg['method']
  args=[sys.executable,'-u',str(root/'source/ra_harness/kvcompress/harness/eval_hf.py'),'--model_name_or_path',cfg['model'],'--data_name','math','--split','test','--prompt_type','qwen-instruct','--surround_with_messages','--attention_implementation','flash_attention_2','--sparsity_method','eviction','--query_window','1','--eviction_mode',mode,'--token_budget','0','--residual_length','0','--reasoning_budget','1024','--recent_window','64','--mask_seed','42','--eviction_interval','64','--rep_seed','0','--seed','42','--batch_size','2','--run_id','0','--max_tokens','32768','--warmup_new_tokens','2','--ex_start_i','0','--ex_end_i','3','--use_batch_exist','--output_dir',str(out)]
  (out/'command.json').write_text(json.dumps(args,indent=2)+'\n')
  with (out/'runner.log').open('x') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
  (out/'exit.json').write_text(json.dumps(dict(exit_code=r.returncode))+'\n');assert r.returncode==0,name
  # Fail immediately on divergent complete outputs; do not proceed with a
  # timing claim if this execution change altered generation.
  if name!='baseline_a':
   for run in (0,1):
    current=rows(out/f'run_{run}/completions_shard0.jsonl');ref=rows(root/f'baseline_a/run_{run}/completions_shard0.jsonl')
    assert current==ref,('generation diverged',name,run)
 status('collecting')
 with (root/'collection.log').open('x') as f:subprocess.run([sys.executable,str(root/'source/scripts/collect_w1_optimization.py'),str(root)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 status('completed',report=str(root/'REPORT.md'))
except Exception as exc:status('failed',error=repr(exc));raise
