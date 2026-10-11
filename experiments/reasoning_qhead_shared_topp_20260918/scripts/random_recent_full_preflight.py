"""Verify the frozen passed implementation, inputs and model before a GPU shard."""
import datetime,hashlib,json,os,sys,platform,importlib.metadata
from pathlib import Path
root,out=map(Path,sys.argv[1:3]);shard=int(sys.argv[3])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
cfg=json.loads((root/'config.json').read_text());assert cfg['method']=='qhead_shared_mask_random_recent_gpu'
assert cfg['random_keep_ratio']==.4 and cfg['recent_window']==64 and cfg['recent_inside_budget']
assert cfg['query_window'] is None and cfg['selection_p'] is None and cfg['full_run_authorized']
assert json.loads((root/'correctness.json').read_text())['status']=='passed'
assert sha(root/'correctness.log')==json.loads((root/'correctness.json').read_text())['log_sha256']
gate=json.loads((root/'smoke_gate.json').read_text());assert gate['status']=='passed'
assert sha(root/'smoke/summary.json')==gate['summary_sha256']
for rel,h in json.loads((root/'inference_source_hashes.json').read_text()).items():assert sha(root/rel)==h,rel
for rel,h in json.loads((root/'model_assets_sha256.json').read_text()).items():assert sha(Path(cfg['model'])/rel)==h,rel
assert sha(Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl'))==cfg['dataset_sha256']
assert os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_ARRAY_TASK_ID'])==shard
import torch
assert torch.cuda.device_count()==1,'Exactly one allocated GPU must be visible'
p=torch.cuda.get_device_properties(0);assert 'A100' in p.name and p.total_memory>75*1024**3
packages={k:importlib.metadata.version(k) for k in ('torch','transformers','flash-attn','numpy','triton')}
for k,v in cfg['packages'].items():assert packages[k]==v,(k,packages[k],v)
h=dict(node=platform.node(),gpu_name=p.name,gpu_total_memory=p.total_memory,torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,packages=packages,job_id=os.environ['SLURM_JOB_ID'],array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=shard,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cpu_affinity=sorted(os.sched_getaffinity(0)),slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),timestamp=datetime.datetime.now().astimezone().isoformat())
(out/'hardware.json').write_text(json.dumps(h,indent=2)+'\n')
(out/'preflight.json').write_text(json.dumps(dict(status='passed',manifest_sha256=sha(root/'manifest.json'),inference_identical_to_passed_smoke=True,model_assets_sha256='passed/all tokenizer and weight assets',shard=shard,sample_ids=list(range(shard,500,8)),timestamp=h['timestamp']),indent=2)+'\n')
(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(h),flush=True)
