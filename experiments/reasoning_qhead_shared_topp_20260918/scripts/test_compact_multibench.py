"""Meaningful offline regression: index/budget checker against real MATH output, graders."""
import json,subprocess,sys
from pathlib import Path
from compact_multibench_common import verify,rows,sha
from check_compact_multibench import gpqa_answer,check_metric,lcb_module
root=Path(sys.argv[1]).resolve();cfg=verify(root,weights=True,tests=True)
for text,want in [(r'Analysis A, B, C. Final \\boxed{D}','D'),(r'\\boxed{\\text{B}}','B'),(r'\\boxed{A} then \\boxed{C}','C'),('Analysis: A then B',''),(r'\\boxed{AB}',''),(r'\\boxed{A',''),(r'\\boxed{\\mathrm{A}}','A')]:
 # Strings above contain escaped literals to remain readable in this fixture.
 text=text.replace('\\\\','\\');assert gpqa_answer(text)==want,(text,want)
ref=Path(cfg['implementation_reference']);oldcfg=json.loads((ref/'config.json').read_text());expected={r['idx']:r for r in rows(ref/'inputs.jsonl')};seen=set()
for d in sorted((ref/'random').glob('shard_*')):
 for r in rows(d/'cache_metrics.jsonl'):
  assert r['idx'] not in seen;seen.add(r['idx']);check_metric(r,expected[r['idx']],oldcfg)
assert seen==set(range(500))
# Budget and alignment corruption must fail closed.
for key,value in [('selected_pairs',r['selected_pairs']+1),('input_ids_sha256','bad'),('reasoning_budget',1025)]:
 bad=dict(r);bad[key]=value
 try:check_metric(bad,expected[r['idx']],oldcfg)
 except AssertionError:pass
 else:raise AssertionError(('failed to reject',key))
m=lcb_module(root)
assert m.extract_code('No solution')==''
with (root/'grader_selftest.log').open('x') as f:
 result=subprocess.run([sys.executable,str(root/'source/ra_harness/kvcompress/eval/grader_lcb.py'),'--selftest'],stdout=f,stderr=subprocess.STDOUT)
assert result.returncode==0,'LCB execution selftest failed'
sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
from Utils.parser import extract_answer
from Utils.grader import check_is_correct
assert check_is_correct(extract_answer(r'Answer \\boxed{007}'.replace('\\\\','\\')),'7')
assert not check_is_correct('8','7')
out=dict(status='passed',manifest_sha256=sha(root/'manifest.json'),model_assets='passed',lcb_test_assets='passed/all383',metric_regression_questions=len(seen),corruption_rejected=3,gpqa_parser_cases=7,lcb_execution_selftest_log_sha256=sha(root/'grader_selftest.log'))
(root/'pipeline_tests.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
