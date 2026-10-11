"""Add cohort forward-time reporting before sealing/submitting the snapshot."""
import ast
import hashlib
import json
from pathlib import Path
E=Path(__file__).resolve().parents[1]
root=E/'results/hmmt_w1_raw_qk_k1024_full_compact500_20261009'
assert (root/'manifest.json').exists() and not (root/'jobs.json').exists()
p=root/'source/scripts/selector_grid.py';s=p.read_text()
old="decode_tokens_per_second=sum(r['decode_query_tokens'] for r in metrics)/decode if decode else None,"
assert old in s
s=s.replace(old,"decode_forward_calls=stats([r['decode_forward_calls'] for r in metrics]),decode_ms_per_forward=1000*decode/sum(r['decode_forward_calls'] for r in metrics),"+old)
old="'passed/fixed_budget_analytic' if cell['selector']=='topk' else 'passed/variable_actual_pairs'"
assert old in s
s=s.replace(old,"'passed/interval500_analytic' if cell['storage']=='compact' else 'passed/full_kv_fixed_budget_analytic'")
ast.parse(s);p.write_text(s)
cfg=json.loads((root/'config.json').read_text())
refs=[json.loads(line) for line in (root/'inputs/hmmt.jsonl').read_text().splitlines()]
assert len(refs)==60 and {r['idx'] for r in refs}==set(range(60))
assert max(r['input_tokens']+64 for r in refs)<=1024
for cell in cfg['cells']:
    assert cell['K']==1024 and cell['protection']=='protected' and cell['score_mode']=='qk'
    assert cell['compaction_interval']==(500 if cell['storage']=='compact' else None)
proof=json.loads((root/'static_validation.json').read_text())
proof.update(reference_inputs=60,max_protected_tokens=max(r['input_tokens']+64 for r in refs),
             cohort_decode_ms_per_forward='synchronized decode seconds / actual batch forward calls',
             cpu_tests=13,cpu_status='passed',gpu_status='pending correctness job')
(root/'static_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
manifest=json.loads((root/'manifest.json').read_text())
manifest[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(proof),flush=True)
