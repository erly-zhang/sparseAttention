"""Freeze the passed smoke inference implementation for the explicitly requested full run."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();E=Path(__file__).resolve().parents[1]
smoke=E/'results/full_kv_random_r040_recent64_gpu_smoke_20260924'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
old_manifest=json.loads((smoke/'manifest.json').read_text())
for rel,h in old_manifest.items():assert sha(smoke/rel)==h,rel
assert json.loads((smoke/'pipeline_status.json').read_text())['phase']=='completed'
assert json.loads((smoke/'smoke_gate.json').read_text())['summary_sha256']==sha(smoke/'smoke/summary.json')
assert json.loads((smoke/'correctness.json').read_text())['log_sha256']==sha(smoke/'correctness.log')
root.mkdir();(root/'logs').mkdir();(root/'random').mkdir();(root/'source/scripts').mkdir(parents=True)
for name in ('ra_harness','tests'):shutil.copytree(smoke/'source'/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('inputs.jsonl','correctness.json','correctness.log','smoke_gate.json'):shutil.copyfile(smoke/name,root/name)
shutil.copytree(smoke/'smoke',root/'smoke')
for name in ('random_recent_full_worker.sh','random_recent_full_array.sbatch','random_recent_full_collect.sbatch','random_recent_full_preflight.py','collect_random_recent_full.py','summarize_recent_metrics.py','prepare_random_recent_full.py','submit_random_recent_full.py'):
 shutil.copyfile(E/'scripts'/name,root/'source/scripts'/name)
shutil.copyfile(smoke/'source/scripts/check_random_recent_smoke.py',root/'source/scripts/check_random_recent_smoke.py')
linkage={rel:h for rel,h in old_manifest.items() if rel.startswith(('source/ra_harness/','source/tests/'))}
for rel,h in linkage.items():assert sha(root/rel)==h
(root/'inference_source_hashes.json').write_text(json.dumps(linkage,indent=2)+'\n')
(root/'smoke_provenance.json').write_text(json.dumps(dict(source=str(smoke),manifest_sha256=sha(smoke/'manifest.json'),correctness_log_sha256=sha(smoke/'correctness.log'),inference_and_tests_identical=True),indent=2)+'\n')
# Freeze dataset metadata and reference audits for post-run comparisons.
data=Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl')
metadata=[{k:r[k] for k in ('idx','level') if k in r} for r in map(json.loads,data.read_text().splitlines())]
assert len(metadata)==500 and {r['idx'] for r in metadata}==set(range(500))
(root/'dataset_metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata))
for name,audit in [('w16','w16_full_20260924'),('dense','dense_baseline_20260923')]:
 src=E/'audits'/audit;dst=root/'reference_baselines'/name;dst.mkdir(parents=True)
 shutil.copyfile(src/'summary.json',dst/'summary.json')
 files=sorted(src.glob('grades*.jsonl'));rows=[];originals=dst/'originals';originals.mkdir()
 for f in files:
  shutil.copyfile(f,originals/f.name)
  for line in f.read_text().splitlines():
   r=json.loads(line)
   if 'run' not in r:r['run']=int(f.stem.rsplit('_',1)[-1])
   rows.append(r)
 assert len(rows)==1000 and {(r['run'],r['idx']) for r in rows}=={(run,i) for run in (0,1) for i in range(500)}
 (dst/'grades.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 (dst/'provenance.json').write_text(json.dumps(dict(source=str(src),source_files_sha256={f.name:sha(f) for f in [src/'summary.json',*files]},normalization='Dense run ID added from original grade filename; grades unchanged.'),indent=2)+'\n')
cfg=json.loads((smoke/'config.json').read_text());cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_run_authorized=True,scope='Full 500 MATH inputs, two rollouts each; explicit user submission after passed three-input smoke',smoke_source=str(smoke),auto_full_submission=False,comparison_caveat='Same canonical inputs and generation settings as W16; masks and output lengths differ. Dense has historical timing scope and no saved input token hashes.',model_assets_verification='Full SHA256 of weights and tokenizer assets at preparation and on every worker before inference')
assert sha(data)==cfg['dataset_sha256']
assets={};model=Path(cfg['model'])
for f in sorted(model.iterdir()):
 if f.is_file() and f.suffix in ('.json','.safetensors','.txt') and f.name not in ('README.md',):
  before=f.stat();assets[f.name]=sha(f);after=f.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  print('Hashed model asset',f.name,flush=True)
(root/'model_assets_sha256.json').write_text(json.dumps(assets,indent=2)+'\n')
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(root/'README.md').write_text('''# Full MATH500 random40 + recent64 GPU experiment

Explicitly authorized after the passed three-input smoke. 500 questions, two rollouts each; eight shards, one A100 80GB each, at most eight simultaneous workers, 48-hour limit per worker. Slurm assigns GPUs when resources and priority permit. Existing experiments are untouched.

Inference source and tests are byte-identical to the passed smoke: Qwen3-4B BF16, full prompt protection, 40% non-prompt budget with recent64 inside that budget, private GPU uniform score TopK, fused reusable KV gather, per-layer/per-step head-shared masks, full persistent KV. No W16/W64 scoring, no permanent eviction, no 1024 cap. Existing chat preprocessing, two sampled rollouts, seed42, temperature0.6, generation Top-p0.95, total length32768 and two-token untimed warmup are retained.

Each worker verifies source, input/data, model/tokenizer SHA256, package versions and A100 memory. Raw completions, input hashes, exact pair counters, cache lengths, EOS outcomes, generated lengths, synchronized prefill/decode/batch timings and peak allocated memory are saved in random/shard_i. No per-step diagnostic serialization is added to inference.

The dependent CPU collector runs after all workers terminate, including failures. It refuses full accuracy if any worker failed or if outputs have missing/duplicate IDs, misaligned input hashes, incorrect pair budgets or inconsistent cache lengths. A successful collector writes random/REPORT.md, comprehensive_summary.json, summary.json, grades.jsonl, question_metrics.jsonl/csv and rollout_metrics.csv. Distributions include mean/median/P90/P95/P99/min/max. Per-level accuracy, per-run and pooled accuracy, correct-and-EOS rate, length-limit exits, generated lengths, weighted throughput, prompt/recent/random pair totals and baseline paired outcomes are included. Baseline audit snapshots are frozen separately. Model loading/warmup and output-file serialization are outside decode timing. Temporary KV copying remains part of decode time.

Monitor submission.json, collection_status.json, logs/, each worker exit.json, and squeue. Slurm accounting is saved after completion. Snapshot hashes protect the source; no existing smoke results or W16/W64 outputs are overwritten. An OOM, timeout or signal remains visible in Slurm state/logs even if the worker cannot write exit.json; the collector then marks the run incomplete rather than silently omitting that shard.
''')
manifest={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Prepared',root,flush=True)
