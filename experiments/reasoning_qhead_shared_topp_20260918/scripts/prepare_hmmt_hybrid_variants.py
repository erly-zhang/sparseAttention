"""Freeze HMMT hybrid v4 variants: pool update every 64/1024 steps x random/top-score 4096.

Algorithm changes (no exact reference). Comparator: v4 hybrid, interval1000 random,
same K (identical to the formal v2 outputs). One question per shard, no GPU cap.
Formal starts after GPU correctness only; every shard is still fully validated.
"""
import ast
import datetime
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

E = Path(__file__).resolve().parents[1]
BASE = E/'results/hmmt_w1_raw_qk_hybrid_v4_k1024_k2048_20261015'
ROOT = E/'results/hmmt_w1_raw_qk_hybrid_variants_k1024_k2048_20261016'
NEW_FILES = ('ra_harness/kvcompress/engine/w1_hybrid_gpu_select_cache.py',
             'ra_harness/kvcompress/engine/w1_hybrid_variants_cache.py',
             'scripts/run_w1_hybrid_variants.py', 'tests/test_w1_hybrid_variants.py',
             'tests/test_w1_hybrid_gpu_select.py')
SHARDS = 60


def read(p): return json.loads(p.read_text())
def write(p, value): p.write_text(json.dumps(value, indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


def main():
    import sys
    sys.path.insert(0, str(E/'ra_harness'))
    from kvcompress.engine.w1_hybrid_variants_cache import mode_name
    assert not ROOT.exists(), 'Never overwrite a run'
    ROOT.mkdir()
    for folder in ('source', 'inputs', 'data'):
        files = [p for p in (BASE/folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix != '.pyc']
        for parent in {(ROOT/folder/p.relative_to(BASE/folder)).parent for p in files}:
            parent.mkdir(parents=True, exist_ok=True)

        def copy(p):
            with p.open('rb') as src, (ROOT/folder/p.relative_to(BASE/folder)).open('xb') as dst:
                shutil.copyfileobj(src, dst)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy, files))
    for rel in NEW_FILES:
        target = ROOT/'source'/rel
        target.unlink(missing_ok=True)
        shutil.copyfile(E/rel, target)
    for name in ('model_assets_sha256.json', 'prompt_protection_stats.json'):
        shutil.copyfile(BASE/name, ROOT/name)
    (ROOT/'logs').mkdir()
    (ROOT/'reference_previous').mkdir()

    cfg = read(BASE/'config.json')
    template = dict(cfg['cells'][0])
    cells = []
    for K in (1024, 2048):
        for interval in (64, 1024):
            for residual in ('random', 'top'):
                cell = dict(template)
                cell.update(name=f'hmmt_k{K}_w1_raw_qk_hybrid_i{interval}_{residual}4096_protected', K=K,
                            method=mode_name(interval, residual), compaction_interval=interval,
                            candidate_budget=K+4096, residual_policy=residual, shards=SHARDS,
                            implementation=f'hybrid_w1_raw_qk_gpu_select_v4_i{interval}_{residual}4096')
                cells.append(cell)
                shutil.copyfile(BASE/f'cells/hmmt_k{K}_w1_raw_qk_hybrid_v4_protected/formal/grades.jsonl',
                                ROOT/'reference_previous'/f"{cell['name']}_grades.jsonl")
    cfg.update(created=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(),
               experiment='hmmt_w1_raw_qk_hybrid_variants_k1024_k2048', cells=cells,
               max_concurrent_gpus=None, compaction_interval=None,
               scope_authorization='2026-10-16 user requests pool updates every 1024/64 steps and top-score 4096 residual; no GPU limit.',
               rank_backend_selection='v4 GPU chronological radix select (same masks as v2).',
               caveats=[
                   '60 canonical questions x2 sampled rollouts; accuracy per rollout, not pass@2.',
                   'Every decode step: W1 attends total K (prompt + recent64 inside K) from the candidate pool (v4 engine).',
                   'Every interval decode steps after attention, if pool > K+4096: keep the W1 core plus 4096 others.',
                   'random: Uniform-TopK outside the core (existing hybrid stream). top: next 4096 by that step\'s W1 raw-QK score.',
                   'Comparator: v4 hybrid interval1000 random at the same K (identical to formal v2 outputs).',
                   'GPU-synchronized timings; different A100 allocations limit speed comparisons.',
                   f'8 cells x{SHARDS} shards x1 question; formal starts after GPU correctness (no smoke gate).'
               ])
    write(ROOT/'config.json', cfg)
    write(ROOT/'smoke_work.json', [])
    write(ROOT/'formal_work.json', [dict(cell=c, shard=s, num_shards=SHARDS)
                                    for c in range(len(cells)) for s in range(SHARDS)])

    p = ROOT/'source/scripts/selector_grid.py'
    text = p.read_text()
    text = replace(text, 'Frozen HMMT hybrid1000 W1 v4 exact-output run (K1024/K2048)',
                   'Frozen HMMT hybrid W1 v4 variants run (interval64/1024 x random/top4096)')
    text = replace(text, "'source/scripts/run_w1_hybrid_v4.py'", "'source/scripts/run_w1_hybrid_variants.py'")
    text = replace(text, """        gate=read(root/'smoke_gate.json');assert gate['status']=='passed'
        for rel,h in gate['summary_hashes'].items():assert sha(root/rel)==h,rel
""", """        pass  # No smoke gate: GPU correctness gates formal; each shard is fully validated.
""")
    text = replace(text, "r['compaction_interval']==r['eviction_interval']==1000",
                   "r['compaction_interval']==r['eviction_interval']==cell['compaction_interval']")
    text = replace(text, "r['max_capacity']==cell['K']+5096",
                   "r['max_capacity']==cell['K']+4096+cell['compaction_interval']")
    text = replace(text, "step%1000==0", "step%cell['compaction_interval']==0", 2)
    text = replace(text, """    refs={(r['run'],r['idx']):r['completion'] for r in read(root/'reference_previous'/f"{cell['name']}_completions.json")}
""", "")
    text = replace(text, """    for (run,idx),text in comps.items():
        assert expected[idx]['prompt_text']+text==refs[run,idx], ('raw-QK output divergence',cell['name'],idx,run)
""", "")
    text = replace(text, "comparisons.append(compare(name+'_previous_same_storage',name,reference,candidate_path,'implementation_only_same_raw_QK_algorithm'))",
                   "comparisons.append(compare(name+'_vs_v4_interval1000_random',name,reference,candidate_path,'pool_variant_vs_v4_hybrid_interval1000_random_matched_K'))")
    p.write_text(text)

    check = '''"""Variant GPU validation plus v4 regressions."""
import hashlib, importlib.util, json, os, sys, unittest
from pathlib import Path
import torch
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'source/ra_harness'),str(root/'source/tests')]
suite=unittest.TestSuite()
for name in ('test_w1_hybrid_variants','test_w1_hybrid_gpu_select'):
    spec=importlib.util.spec_from_file_location(name,root/'source/tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
with (root/'correctness.log').open('x') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
proof=dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
 tests=result.testsRun,skipped=[str(s) for s in result.skipped],gpu=torch.cuda.get_device_name(),
 node=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID'),
 log_sha256=hashlib.sha256((root/'correctness.log').read_bytes()).hexdigest())
(root/'correctness.json').write_text(json.dumps(proof,indent=2)+'\\n')
print(json.dumps(proof),flush=True)
assert proof['status']=='passed'
'''
    (ROOT/'source/scripts/benchmark_w1_qk_fast.py').write_text(check)
    n = len(cells)*SHARDS
    for stage in ('correctness', 'formal', 'collect'):
        p = ROOT/'source/scripts'/f'{stage}.sbatch'
        text = p.read_text().replace('hmmt-v4-', 'hmmt-var-')
        if stage == 'formal':
            text = replace(text, '--array=0-39%12', f'--array=0-{n-1}')
            text = replace(text, '--time=03:00:00', '--time=02:00:00')
        p.write_text(text)
    write(ROOT/'provenance.json', dict(base=str(BASE), comparator='v4 hybrid interval1000 random, same K',
          variants_sha256=sha(ROOT/'source/ra_harness/kvcompress/engine/w1_hybrid_variants_cache.py'),
          v4_sha256=sha(ROOT/'source/ra_harness/kvcompress/engine/w1_hybrid_gpu_select_cache.py'),
          input_sha256=sha(ROOT/'inputs/hmmt.jsonl')))
    (ROOT/'README.md').write_text('''# HMMT hybrid W1 v4 variants: pool interval x residual rule (K1024 / K2048)

Every decode step W1 attends total K from the bounded pool (v4 engine). Every
64 or 1024 decode steps the pool keeps the W1 core plus 4096 positions chosen
uniformly at random (existing stream) or by that step's W1 raw-QK score (top).
Comparator: v4 hybrid interval1000 random at the same K. One question per shard.
''')
    for p in (ROOT/'source').rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    for stage in ('correctness', 'formal', 'collect'):
        subprocess.run(['bash', '-n', str(ROOT/'source/scripts'/f'{stage}.sbatch')], check=True)
    files = [ROOT/p for p in ('config.json', 'smoke_work.json', 'formal_work.json',
                             'model_assets_sha256.json', 'inputs/hmmt.jsonl', 'data/hmmt/test.jsonl', 'provenance.json')]
    files += [p for folder in ('source', 'reference_previous') for p in (ROOT/folder).rglob('*') if p.is_file()]
    write(ROOT/'manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})
    write(ROOT/'static_validation.json', dict(status='passed', cells=len(cells), shards_per_cell=SHARDS,
          formal_tasks=n, max_concurrent_gpus=None))
    print(json.dumps(dict(root=str(ROOT), cells=len(cells), formal_tasks=n)), flush=True)


if __name__ == '__main__':
    main()
