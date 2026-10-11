"""Freeze the validated memory repair for a fresh, fully regenerated MATH500 run."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve()
E=Path(__file__).resolve().parents[1]
passed=E/'results/full_kv_random_reason1024_memory_repair_20260928'
old=E/'results/full_kv_random_reason1024_recent64_20260925'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
for rel,h in read(passed/'manifest.json').items():assert sha(passed/rel)==h,rel
proof=read(passed/'correctness.json');assert proof['status']=='passed'
assert sha(passed/'correctness.log')==proof['log_sha256']
assert sha(passed/'memory_correctness.log')==proof['memory_log_sha256']
assert read(passed/'smoke_regression.json')['status']=='passed'
assert sha(passed/'smoke/summary.json')==read(passed/'smoke_gate.json')['summary_sha256']
assert read(passed/'collection_status.json')['status']=='completed'
for i in (0,1):assert read(passed/f'oom_replay/worker_{i}/validation.json')['status']=='passed'
root.mkdir();(root/'source/scripts').mkdir(parents=True);(root/'random').mkdir();(root/'logs').mkdir()
for name in ('ra_harness','tests'):shutil.copytree(passed/'source'/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('inputs.jsonl','model_assets_sha256.json','correctness.json','correctness.log','memory_correctness.log','smoke_gate.json','smoke_regression.json'):
 shutil.copy2(passed/name,root/name)
shutil.copytree(passed/'smoke',root/'smoke')
shutil.copytree(old/'reference_baselines',root/'reference_baselines')
shutil.copy2(old/'dataset_metadata.jsonl',root/'dataset_metadata.jsonl')
for name in ('random_fixed_recent_full_worker.sh','random_fixed_recent_full_array.sbatch','random_fixed_recent_full_collect.sbatch','random_fixed_recent_full_preflight.py','collect_random_fixed_recent_full.py','summarize_fixed_recent_metrics.py','check_random_fixed_recent_smoke.py'):
 shutil.copy2(passed/'source/scripts'/name,root/'source/scripts'/name)
shutil.copy2(Path(__file__),root/'source/scripts'/Path(__file__).name)
# Only orchestration/validation changes below; inference bytes are unchanged.
f=root/'source/scripts/random_fixed_recent_full_array.sbatch';f.write_text(f.read_text().replace('rand1024-rec64','rand1024-repaired'))
f=root/'source/scripts/collect_random_fixed_recent_full.py';s=f.read_text();needle=" status('validating_and_grading')"
insert=''' # Check the full fresh run's release and bounded storage independently.
 rows=[json.loads(line) for i in range(8) for line in (root/f'random/shard_{i}/cache_metrics.jsonl').read_text().splitlines()]
 assert len(rows)==500 and {r['idx'] for r in rows}==set(range(500))
 assert all(r['max_capacity']==32768 for r in rows)
 assert all(r['memory_allocated_after_generation']<=r['memory_allocated_before_generation']+256*1024**2 for r in rows)
 memory=dict(status='passed',questions=500,all_samples_fresh=True,historical_outputs_reused=0,full_history_storage_capacity=32768,
             max_allocated_after_generation=max(r['memory_allocated_after_generation'] for r in rows),
             peak_memory_allocated=max(r['peak_memory_allocated'] for r in rows))
 (root/'memory_validation.json').write_text(json.dumps(memory,indent=2)+'\\n')
'''
assert needle in s;s=s.replace(needle,insert+needle)
needle=" from summarize_fixed_recent_metrics import summarize"
insert=" summary_path=root/'random/summary.json'\n summary=json.loads(summary_path.read_text())\n summary.update(implementation='full_kv_memory_repair_v1',all_samples_fresh=True,historical_outputs_reused=0,memory_validation=memory)\n summary_path.write_text(json.dumps(summary,indent=2)+'\\n')\n"
assert needle in s;s=s.replace(needle,insert+needle);compile(s,str(f),'exec');f.write_text(s)
linkage={rel:h for rel,h in read(passed/'manifest.json').items() if rel.startswith(('source/ra_harness/','source/tests/'))}
for rel,h in linkage.items():assert sha(root/rel)==h,rel
write(root/'inference_source_hashes.json',linkage)
write(root/'smoke_provenance.json',dict(source=str(passed),manifest_sha256=sha(passed/'manifest.json'),inference_and_tests_identical=True,correctness_log_sha256=sha(passed/'correctness.log'),memory_log_sha256=sha(passed/'memory_correctness.log'),smoke_regression_sha256=sha(passed/'smoke_regression.json'),historical_oom_replays_passed=[232,204]))
cfg=read(passed/'config.json');cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Fresh full MATH500 rerun: regenerate all 500 questions / 1000 rollouts with validated full-KV memory repair',all_samples_fresh=True,historical_outputs_reused=0,full_run_authorized=True,num_shards=8,array_tasks=8,max_concurrent_gpus=8,walltime_per_worker_hours=48,smoke_source=str(passed),comparison_caveat='All 500 question timings come from this repaired run. GPUs/nodes, outputs and algorithm budgets may differ from other methods; report these when comparing speed.')
assert sha(Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl'))==cfg['dataset_sha256']
write(root/'config.json',cfg)
(root/'README.md').write_text('''# Repaired Random reasoning1024 + recent64: fresh MATH500

All 500 questions (two rollouts each) are regenerated with the inference code byte-identical to the passed memory-repair smoke. Previously completed 429/71 outputs are not reused. Source/test hashes, three-input exact text regression and the 32K historical OOM replays are preserved as validation provenance.

Qwen3-4B BF16; full prompt protection outside the reasoning budget; reasoning1024 includes recent64; private uniform-score TopK; full persistent KV retained, temporary attention masks each decoding step. Full cache storage is reserved up to the 32768 total length cap. No permanent eviction or periodic compression. Seed42, batch2, sampled decoding temperature0.6 / Top-p0.95 and original chat preprocessing remain unchanged.

Eight round-robin shards (62/63 questions), one A100 80GB per shard, at most eight GPUs, Slurm scheduled without stopping other work. Each shard checks source/input/model hashes, packages, hardware and passed smoke evidence. Each sample records input alignment, exact pair counts, synchronized prefill/decode timings, peak memory and memory before/after generation. Loading, warmup and diagnostic serialization are excluded from phase timing.

The dependent CPU collector runs after all shards terminate, refuses incomplete/failed runs, requires 500 unique questions and 1000 unique outputs with full input alignment and exact pair budgets, and validates cache release for all 500 samples. It writes random/summary.json, comprehensive_summary.json, question/rollout metrics, grades and REPORT.md. This fresh run provides timing for the repaired implementation across all 500 inputs.
''')
write(root/'preparation_validation.json',dict(status='passed',inference_and_tests_identical_to_passed_repair=True,source_and_input_hashes='passed',three_input_regression='passed/6 identical outputs',historical_oom_replay='passed/232,204',fresh_question_count=500,rollouts=1000))
write(root/'manifest.json',{str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()})
print(root,flush=True)
