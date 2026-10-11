import datetime,json,shutil,sys
from pathlib import Path
from rep_compact_math_common import sha,rows
E=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve();ref=E/'results/representative_w1_compact_math_20260926';explore=E/'results/w1_optimization_exploration_20260927'
assert json.loads((ref/'formal_collection_status.json').read_text())['status']=='completed'
assert (explore/'microbenchmark.json').exists()
root.mkdir();(root/'logs').mkdir();(root/'source/scripts').mkdir(parents=True)
for name in ('ra_harness','tests'):shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in (E/'scripts').glob('*w1_optimization*'):
 if p.is_file():shutil.copyfile(p,root/'source/scripts'/p.name)
for name in ('rep_compact_math_common.py','check_rep_compact_math.py'):shutil.copyfile(E/'scripts'/name,root/'source/scripts'/name)
for rel in ('source/ra_harness/kvcompress/engine/representative_compact_recent_cache.py','source/ra_harness/kvcompress/engine/random_compact_recent_cache.py','source/ra_harness/kvcompress/engine/modify_forward.py','source/ra_harness/kvcompress/harness/generation_utils.py'):
 assert sha(root/rel)==sha(ref/rel),rel
for name in ('baseline_profile.txt','baseline_profile.json','microbenchmark.json','optimization_correctness.log'):shutil.copyfile(explore/name,root/name)
shutil.copyfile(ref/'model_assets_sha256.json',root/'model_assets_sha256.json')
shutil.copyfile(ref/'inputs/math.jsonl',root/'inputs.jsonl');(root/'data/math').mkdir(parents=True);shutil.copyfile(ref/'data/math/test.jsonl',root/'data/math/test.jsonl')
cfg=json.loads((ref/'config.json').read_text());cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='qhead_shared_representative_compact_optimized_gpu',full_run_authorized=False,auto_full_submission=False,scope='Same GPU ABBA: baseline/optimized/optimized/baseline, each MATH500 inputs0,1,2 with identical seed42 and two rollouts; no new full500 run',baseline_method='qhead_shared_representative_compact_recent_gpu',implementation_reference=str(ref),pair_order=['baseline_a','optimized_a','optimized_b','baseline_b'],comparison_caveat='Controlled same-input/same-GPU paired timing; complete decoded outputs, lengths, EOS and pair totals must match exactly. Three inputs do not establish full500 speedup.',optimizations=['save current Q only at compaction events','reasoning-only fused KV gather into spare capacity','swap two full-capacity buffers; copy protected prompt only once'],unchanged=['representative RNG draw sequence','original QK/scale/FP32 softmax','stable score and chronological sort','protection/budget/interval','attention and generation runner'])
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(root/'manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},indent=2)+'\n')
print(root,flush=True)
