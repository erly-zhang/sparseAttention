"""Freeze precisely the passed W1 variant, never alter the inference implementation."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--smoke',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();smoke=a.smoke.resolve();E=Path(__file__).resolve().parents[1]
cfg=json.loads((smoke/'config.json').read_text());assert cfg['method'] in ('qhead_shared_mask_topp','qhead_shared_mask_w1_recent_gpu')
assert cfg['query_window']==1 and cfg['selection_p']==.95
assert json.loads((smoke/'pipeline_status.json').read_text())['phase']=='completed'
gate=json.loads((smoke/'smoke_gate.json').read_text());assert gate['status']=='passed' and gate['summary_sha256']==sha(smoke/'smoke/summary.json')
summary=json.loads((smoke/'smoke/summary.json').read_text());assert summary['status']=='passed' and summary['questions']==3 and summary['rollouts']==6
assert summary['grading_errors']==0 and summary['full_history_continuity']=='passed' and summary['pair_count_crosscheck']=='passed' and summary['input_alignment'].startswith('passed')
assert json.loads((smoke/'correctness.json').read_text())['status']=='passed'
assert json.loads((smoke/'correctness.json').read_text())['log_sha256']==sha(smoke/'correctness.log')
old=json.loads((smoke/'manifest.json').read_text())
for rel,h in old.items():assert sha(smoke/rel)==h,rel
root.mkdir();(root/'source/scripts').mkdir(parents=True);(root/'formal').mkdir();(root/'logs').mkdir()
for name in ('ra_harness','tests'):shutil.copytree(smoke/'source'/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('inputs.jsonl','smoke_gate.json','correctness.json','correctness.log'):shutil.copyfile(smoke/name,root/name)
shutil.copytree(smoke/'smoke',root/'smoke')
for name in ('prepare_w1_full.py','submit_w1_full.py','w1_full_array.sbatch','w1_full_worker.sh','w1_full_collect.sbatch','w1_full_preflight.py','collect_w1_full.py'):
 shutil.copyfile(E/'scripts'/name,root/'source/scripts'/name)
linkage={rel:h for rel,h in old.items() if rel.startswith(('source/ra_harness/','source/tests/'))}
for rel,h in linkage.items():assert sha(root/rel)==h,rel
(root/'inference_source_hashes.json').write_text(json.dumps(linkage,indent=2)+'\n')
(root/'smoke_provenance.json').write_text(json.dumps(dict(source=str(smoke),manifest_sha256=sha(smoke/'manifest.json'),summary_sha256=sha(smoke/'smoke/summary.json'),inference_and_tests_identical=True),indent=2)+'\n')
prior=E/'results/full_kv_random_r040_recent64_gpu_20260924'
for name in ('model_assets_sha256.json','dataset_metadata.jsonl'):shutil.copyfile(prior/name,root/name)
for name,h in json.loads((root/'model_assets_sha256.json').read_text()).items():assert sha(Path(cfg['model'])/name)==h,name
assert sha(Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl'))==cfg['dataset_sha256']
shutil.copytree(prior/'reference_baselines',root/'reference_baselines')
for name,src in [('w64',E/'results/full_kv_w16_w64_20260922/w64'),('random40_recent64',prior/'random')]:
 d=root/'reference_baselines'/name;d.mkdir();shutil.copyfile(src/'grades.jsonl',d/'grades.jsonl')
 (d/'provenance.json').write_text(json.dumps(dict(source=str(src),grades_sha256=sha(src/'grades.jsonl')),indent=2)+'\n')
cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_run_authorized=True,
 scope='Full MATH500; two rollouts each; user authorized both W1 and W1 Top-p union recent64 after technical smoke passes',
 smoke_source=str(smoke),num_shards=8,array_tasks=8,max_concurrent_gpus=4,walltime_per_worker_hours=48,
 auto_full_submission=False,comparison_caveat='Two independent experiments; W1 preserves historical gather, recent64 uses fused reusable gather. Retention masks and lengths differ; dense historical timing is not a synchronized phase comparison.')
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(root/'README.md').write_text(f'''# Full MATH500 {cfg['method']}

Explicitly authorized by the user after the technical smoke gate. Source and tests are byte-identical to {smoke}. W1 Top-p0.95, all prompt KV protected, full persistent KV retained, per-layer/per-step representative Q-head shared masks. The recent64 variant unions the Top-p set with the last64 non-prompt positions; no fixed40% budget. Original W1 retains its smoke-tested gather implementation.

500 unique inputs, two sampled rollouts each, Qwen3-4B BF16, seed42, rep_seed0, temperature0.6, generation Top-p0.95, batch2, max total length32768, untimed2-token warmup. Eight shards with one A10080GB each, maximum4 concurrent GPUs per experiment (at most8 across both experiments), 8CPU,64GB host memory,48h per worker. Slurm handles allocation; existing jobs are not cancelled or reused.

Source/data/model/package preflight before every worker. Formal raw outputs under formal/shard_i. Actual input token count/SHA256, synchronized prefill/decode/total time, tokens, EOS, full-cache lengths, raw selected/causal pairs and peak allocated GPU memory are recorded. Recent64 records extra recent vs Top-p-core pairs. Per-step diagnostic logging is off.

Collector runs after all workers end, rejects missing/duplicate/misaligned/failed outputs, grades all1000answers, reports per-run/pooled/level accuracy, termination, length distributions, weighted throughput, pair sparsity, timing percentiles, memory and paired outcomes against dense,W16,W64,random40recent64. CSV and JSON outputs are retained. Dense speed comparisons have historical instrumentation limitations. Smoke answer accuracy is recorded but is not used as a correctness gate or a full500 performance estimate.

submission.json holds job IDs; collection_status.json and formal/REPORT.md hold final outcome. No source or results from prior experiments are overwritten.
''')
manifest={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(root,flush=True)
