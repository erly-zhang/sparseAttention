"""Freeze the HMMT hybrid1000 W1 v4 run: exact-output check + speed, K1024 and K2048.

v4 is an implementation change only (precomputed representative table and one
GPU ranking kernel). Every generated answer must equal the existing formal
hybrid output byte for byte (K1024: v2 packed32 run; K2048: interval1000 run).
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
BASE = E/'results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32'
REFERENCES = {
    1024: E/'results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32/cells/hmmt_k1024_w1_raw_qk_fast_hybrid_protected/formal',
    2048: E/'results/hmmt_w1_raw_qk_k2048_hybrid4096_compact1000_shortshards_20261012/cells/hmmt_k2048_w1_raw_qk_hybrid1000_protected/formal',
}
ROOT = E/'results/hmmt_w1_raw_qk_hybrid_v4_k1024_k2048_20261015'
NEW_FILES = ('ra_harness/kvcompress/engine/w1_qk_fused_cache.py',
             'ra_harness/kvcompress/engine/w1_hybrid_gpu_select_cache.py',
             'scripts/run_w1_hybrid_v4.py', 'tests/test_w1_hybrid_gpu_select.py')
MAX_GPUS = 12


def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p, value): p.write_text(json.dumps(value, indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


def main():
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
        with (E/rel).open('rb') as src, (ROOT/'source'/rel).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    for name in ('model_assets_sha256.json', 'prompt_protection_stats.json'):
        shutil.copyfile(BASE/name, ROOT/name)
    (ROOT/'logs').mkdir()
    (ROOT/'reference_previous').mkdir()

    cfg = read(BASE/'config.json')
    template = next(c for c in cfg['cells'] if c['storage'] == 'hybrid')
    cells, references = [], {}
    for K, ref in REFERENCES.items():
        cell = dict(template)
        cell.update(name=f'hmmt_k{K}_w1_raw_qk_hybrid_v4_protected', K=K,
                    candidate_budget=K+4096, implementation='hybrid_w1_raw_qk_gpu_select_v4')
        cells.append(cell)
        completions, grades = [], []
        for shard in sorted(ref.glob('shard_*')):
            for run in (0, 1):
                (f,) = list((shard/f'run_{run}').glob('completions*.jsonl'))
                completions += [dict(idx=r['idx'], run=run, completion=r['completion']) for r in rows(f)]
        grades = rows(ref/'grades.jsonl')
        assert len(completions) == 120 and len({(c['run'], c['idx']) for c in completions}) == 120
        assert len(grades) == 120
        write(ROOT/'reference_previous'/f"{cell['name']}_completions.json", completions)
        with (ROOT/'reference_previous'/f"{cell['name']}_grades.jsonl").open('x') as f:
            f.writelines(json.dumps(g)+'\n' for g in grades)
        shutil.copyfile(ref/'summary.json', ROOT/'reference_previous'/f"{cell['name']}_summary.json")
        references[cell['name']] = dict(root=str(ref), summary_sha256=sha(ref/'summary.json'),
                                        grades_sha256=sha(ref/'grades.jsonl'))
    cfg.update(created=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(),
               experiment='hmmt_w1_raw_qk_hybrid_v4_k1024_k2048', cells=cells,
               max_concurrent_gpus=MAX_GPUS, compaction_interval=1000,
               scope_authorization='2026-10-15 user requests optimizing hybrid speed and more parallel GPUs; v4 must reproduce existing hybrid outputs exactly.',
               rank_backend_selection='v4 GPU chronological radix select; recorded backend label packed_qk_topk (same masks).',
               caveats=[
                   '60 canonical questions x2 sampled rollouts; accuracy per rollout, not pass@2.',
                   'Implementation-only change: every answer must equal the existing formal hybrid output exactly.',
                   'K1024 reference: v2 packed32 run; K2048 reference: interval1000 hybrid run (stable_sort, same masks).',
                   'GPU-synchronized timings include selection/gather/attention/sampling; exclude loading/warmup/serialization.',
                   'Different A100 allocations limit speed comparisons; same outputs make forward counts identical.',
                   f'2 cells x20 formal shards x3 questions, at most{MAX_GPUS} GPUs.'
               ])
    write(ROOT/'config.json', cfg)
    write(ROOT/'smoke_work.json', [dict(cell=c, shard=0, num_shards=1) for c in range(2)])
    write(ROOT/'formal_work.json', [dict(cell=c, shard=s, num_shards=20) for c in range(2) for s in range(20)])

    p = ROOT/'source/scripts/selector_grid.py'
    text = p.read_text()
    text = replace(text, 'Frozen2-cell HMMT W1 K1024 interval1000 grid', 'Frozen HMMT hybrid1000 W1 v4 exact-output run (K1024/K2048)')
    text = replace(text, "'source/scripts/run_w1_qk_fast.py'", "'source/scripts/run_w1_hybrid_v4.py'")
    text = replace(text, "read(root/'microbenchmark.json')['selected_backends'][cell['storage']]", "'packed_qk_topk'", 2)
    text = replace(text, "r['candidate_budget']==5120", "r['candidate_budget']==cell['K']+4096")
    text = replace(text, "r['max_capacity']==6120", "r['max_capacity']==cell['K']+5096")
    text = replace(text, "r['compaction_core_token_sum']==1024*", "r['compaction_core_token_sum']==cell['K']*")
    text = replace(text, "cached>5120:", "cached>cell['K']+4096:", 2)
    text = replace(text, "cached=5120;", "cached=cell['K']+4096;", 2)
    text = replace(text, "max(P,1024)", "max(P,cell['K'])")
    assert '5120' not in text and '6120' not in text
    p.write_text(text)

    check = '''"""v4 validation against v2/v3, plus the established v2 regression suite."""
import hashlib, importlib.util, json, os, sys, unittest
from pathlib import Path
import torch
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'source/ra_harness'),str(root/'source/tests')]
suite=unittest.TestSuite()
for name in ('test_w1_hybrid_gpu_select','test_w1_qk_fast'):
    spec=importlib.util.spec_from_file_location(name,root/'source/tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
with (root/'correctness.log').open('x') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
proof=dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
 tests=result.testsRun,skipped=[str(s) for s in result.skipped],gpu=torch.cuda.get_device_name(),
 node=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID'),implementation='hybrid_w1_raw_qk_gpu_select_v4',
 log_sha256=hashlib.sha256((root/'correctness.log').read_bytes()).hexdigest())
(root/'correctness.json').write_text(json.dumps(proof,indent=2)+'\\n')
print(json.dumps(proof),flush=True)
assert proof['status']=='passed'
'''
    (ROOT/'source/scripts/benchmark_w1_qk_fast.py').write_text(check)
    for stage in ('correctness', 'smoke', 'gate', 'formal', 'collect'):
        p = ROOT/'source/scripts'/f'{stage}.sbatch'
        text = p.read_text().replace('hmmt-qk-fast-', 'hmmt-v4-')
        if stage == 'formal':
            text = replace(text, '--array=0-39%2', f'--array=0-39%{MAX_GPUS}')
        p.write_text(text)
    write(ROOT/'provenance.json', dict(base=str(BASE), references=references,
          implementation='hybrid_w1_raw_qk_gpu_select_v4',
          implementation_sha256=sha(ROOT/'source/ra_harness/kvcompress/engine/w1_hybrid_gpu_select_cache.py'),
          v2_sha256=sha(ROOT/'source/ra_harness/kvcompress/engine/w1_qk_fast_cache.py'),
          input_sha256=sha(ROOT/'inputs/hmmt.jsonl')))
    (ROOT/'README.md').write_text(f'''# HMMT hybrid1000 W1 v4: exact outputs, speed (K1024 / K2048)

Implementation-only optimization of W1 + random4096 (every-step W1 over the
bounded pool, compaction every1000 steps). v4 precomputes the private
representative stream in one vectorized pass and replaces pack/TopK/sort with
one exact GPU radix-select kernel. Every generated answer must match the
existing formal hybrid output byte for byte, so forward counts are identical
and decode times compare the implementations directly (different A100s).
GPU tests -> two3-question exact smokes -> gate -> 2x20 shards (<= {MAX_GPUS} GPUs) -> collect.
''')
    for p in (ROOT/'source').rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    for stage in ('correctness', 'smoke', 'gate', 'formal', 'collect'):
        subprocess.run(['bash', '-n', str(ROOT/'source/scripts'/f'{stage}.sbatch')], check=True)
    for f in ('w1_qk_fast_cache.py', 'w1_interval1000_cache.py', 'cache_utils.py', 'selector_grid_cache.py'):
        assert sha(ROOT/'source/ra_harness/kvcompress/engine'/f) == sha(E/'ra_harness/kvcompress/engine'/f), f
    write(ROOT/'static_validation.json', dict(status='passed', K=[1024, 2048], cells=2,
          questions=60, answers_per_cell=120, formal_shards=40, max_concurrent_gpus=MAX_GPUS,
          exact_reference_outputs=True))
    files = [ROOT/p for p in ('config.json', 'smoke_work.json', 'formal_work.json',
                             'model_assets_sha256.json', 'inputs/hmmt.jsonl', 'data/hmmt/test.jsonl', 'provenance.json')]
    files += [p for folder in ('source', 'reference_previous') for p in (ROOT/folder).rglob('*') if p.is_file()]
    write(ROOT/'manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})
    print(json.dumps(dict(root=str(ROOT), static_validation='passed', manifest_files=len(files))), flush=True)


if __name__ == '__main__':
    main()
