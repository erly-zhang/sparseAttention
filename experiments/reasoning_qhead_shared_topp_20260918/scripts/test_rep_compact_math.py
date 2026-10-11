"""Pre-submission validation; GPU tests are mandatory inside the smoke worker."""
import ast,json,subprocess,sys
from pathlib import Path
from rep_compact_math_common import verify,sha,rows
from check_rep_compact_math import check_metric
root=Path(sys.argv[1]).resolve();cfg=verify(root,weights=True)
for p in (root/'source').rglob('*.py'):ast.parse(p.read_text())
for name in ('rep_compact_math_gpu.sbatch','rep_compact_math_cpu.sbatch'):subprocess.run(['bash','-n',str(root/'source/scripts'/name)],check=True)
ref=Path(cfg['implementation_reference']);expected={r['idx']:r for r in rows(root/'inputs/math.jsonl')};seen=set()
for d in sorted((ref/'random').glob('shard_*')):
 for record in rows(d/'cache_metrics.jsonl'):
  # Map only selector metadata: the budget checker must accept the same actual
  # attention counts/lengths and reject corrupt counters under the new method.
  r=dict(record);r.update(query_window=1,selection_strategy='persistent_shared_representative_topk_recent',representative_policy='private_sample_layer_stream_draw_at_compaction',selection_ties='earlier_original_position')
  check_metric(r,expected[r['idx']],cfg);assert r['idx'] not in seen;seen.add(r['idx'])
assert seen==set(range(500))
for key,value in [('selected_pairs',r['selected_pairs']+1),('query_window',16),('input_ids_sha256','bad')]:
 bad=dict(r);bad[key]=value
 try:check_metric(bad,expected[r['idx']],cfg)
 except AssertionError:pass
 else:raise AssertionError(('failed to reject',key))
import os
env=os.environ.copy();env.update(PYTHONPATH=str(root/'source/ra_harness'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
with (root/'cpu_correctness.log').open('x') as f:
 subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'source/tests'),'-p','test_representative_compact_shared_mask.py','-v'],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
result=dict(status='passed',manifest_sha256=sha(root/'manifest.json'),model_assets='passed',metric_budget_regression_questions=len(seen),corruption_rejected=3,cpu_selector_tests='passed; GPU test deferred to mandatory allocated smoke correctness',cpu_log_sha256=sha(root/'cpu_correctness.log'))
(root/'pipeline_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
