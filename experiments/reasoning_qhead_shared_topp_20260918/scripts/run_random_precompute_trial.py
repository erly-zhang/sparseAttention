"""Exactness gate, selection microbench, then three-input paired full generations."""
import datetime,hashlib,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();cfg=json.loads((r/'config.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def status(phase,**kw):
 x=dict(phase=phase,timestamp=datetime.datetime.now().astimezone().isoformat(),**kw);write(r/'pipeline_status.json',x);print(json.dumps(x),flush=True)
try:
 for rel,h in json.loads((r/'manifest.json').read_text()).items():assert sha(r/rel)==h,rel
 for rel,h in json.loads((r/'model_assets_sha256.json').read_text()).items():assert sha(Path(cfg['model'])/rel)==h,rel
 with (r/'pipeline.lock').open('x') as f:f.write(str(os.getpid()))
 import torch
 assert torch.cuda.device_count()==1
 gpu=torch.cuda.get_device_properties(0);assert 'A100' in gpu.name and gpu.total_memory>75*1024**3
 write(r/'hardware.json',dict(node=platform.node(),gpu_name=gpu.name,gpu_memory=gpu.total_memory,slurm_job_id=os.environ.get('SLURM_JOB_ID'),packages={k:importlib.metadata.version(k) for k in cfg['packages']}))
 env=os.environ.copy();env.update(PYTHONPATH=str(r/'source/ra_harness'),RA_DATA_DIR='/project/yl258/jz648/Random-Attention/data',QHEAD_DEBUG='0')
 for k in ('KEEPLOG','QHEAD_LOG','KEEPLOG_POS','FORCE_KEEP_RANGE'):env.pop(k,None)
 status('correctness')
 with (r/'correctness.log').open('x') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r/'source/tests'),'-p','test*shared*mask*.py','-v'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 write(r/'correctness.json',dict(status='passed',log_sha256=sha(r/'correctness.log')))
 status('selection_microbenchmark')
 with (r/'microbenchmark.log').open('x') as f:subprocess.run([sys.executable,str(r/'source/scripts/benchmark_precomputed_random_masks.py'),str(r/'microbenchmark.json')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 expected={x['idx']:x for x in rows(r/'inputs.jsonl')}
 summaries={}
 for name,mode in cfg['passes']:
  status('full_generation',active_pass=name)
  out=r/name;out.mkdir()
  compact='compact' in mode
  cmd=[sys.executable,'-u',str(r/'source/ra_harness/kvcompress/harness/eval_hf.py'),'--model_name_or_path',cfg['model'],'--data_name','math','--split','test','--prompt_type','qwen-instruct','--surround_with_messages','--attention_implementation','flash_attention_2','--sparsity_method','eviction','--eviction_mode',mode,'--token_budget','0','--residual_length','0','--recent_window','64','--mask_seed','42','--rep_seed','0','--seed','42','--batch_size','2','--run_id','0','--max_tokens','32768','--warmup_new_tokens','2','--ex_start_i','0','--ex_end_i','3','--use_batch_exist','--output_dir',str(out)]
  cmd+=['--reasoning_budget','1024','--eviction_interval','64'] if compact else ['--random_keep_ratio','.4','--eviction_interval','1']
  write(out/'command.json',cmd)
  with (out/'runner.log').open('x') as log:proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  write(out/'exit.json',dict(exit_code=proc.returncode));assert proc.returncode==0,name
  mm=rows(out/'cache_metrics.jsonl');assert len(mm)==3 and {x['idx'] for x in mm}=={0,1,2}
  old=Path(cfg['historical_compact'] if compact else cfg['historical_full40'])
  for x in mm:
   idx=x['idx'];assert x['input_tokens']==expected[idx]['input_tokens'] and x['input_ids_sha256']==expected[idx]['input_ids_sha256']
   historical=next(y for y in rows(old/f'shard_{idx}/cache_metrics.jsonl') if y['idx']==idx)
   for field in ('generated_tokens_per_rollout','eos_reached','selected_pairs','causal_pairs','selection_events'):assert x[field]==historical[field],(name,idx,field)
   for run in (0,1):
    now=next(y['completion'] for y in rows(out/f'run_{run}/completions_shard0.jsonl') if y['idx']==idx)
    orig=next(y['completion'] for y in rows(old/f'shard_{idx}/run_{run}/completions.jsonl') if y['idx']==idx)
    assert now==orig,(name,idx,run,'complete output differs')
  summaries[name]=dict(questions=3,rollouts=6,exact_historical_outputs=True,mean_decode_seconds=sum(x['decode_seconds'] for x in mm)/3,mean_batch_wall_seconds=sum(x['batch_wall_seconds'] for x in mm)/3,mean_mask_precompute_seconds=sum(x.get('mask_precompute_seconds',0) for x in mm)/3,decode_tokens_per_second=sum(x['decode_query_tokens'] for x in mm)/sum(x['decode_seconds'] for x in mm),peak_memory_allocated=max(x['peak_memory_allocated'] for x in mm),mask_plan_bytes=max(x.get('mask_plan_bytes',x.get('mask_plan_peak_bytes',0)) for x in mm))
  write(out/'validation.json',summaries[name])
  write(r/'partial_results.json',summaries)
 comparisons={}
 for kind in ('compact','full40'):
  a=summaries[kind+'_baseline'];b=summaries[kind+'_precomputed']
  comparisons[kind]=dict(decode_time_reduction=1-b['mean_decode_seconds']/a['mean_decode_seconds'],total_time_reduction_including_precompute=1-b['mean_batch_wall_seconds']/a['mean_batch_wall_seconds'])
 result=dict(status='completed',scope='three matched inputs, two rollouts each, one paired pass per implementation; preliminary speed only',input_alignment='passed/all3',full_generation_outputs='identical to corresponding historical method',results=summaries,comparison=comparisons,hardware=json.loads((r/'hardware.json').read_text()))
 write(r/'summary.json',result);status('completed',summary=str(r/'summary.json'))
except Exception as e:status('failed',error=repr(e));raise
