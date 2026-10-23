"""Freeze the HMMT Full-KV raw-QK W1 interval64 run (K1024 and K2048).

Algorithm change requested 2026-10-15: like official random_pp, build the shared
token mask once every 64 decode steps instead of every step. All KV is retained;
each refresh ranks the full history with the per-step v2 W1 selector. Between
refreshes, new tokens are appended (attention K..K+63, random_pp-matched).
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
BASE = E/'results/hmmt_full_w1_raw_qk_fast_k2048_20261014'
K1024_PER_STEP = E/'results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32/cells/hmmt_k1024_w1_raw_qk_fast_full_protected/formal/grades.jsonl'
K2048_PER_STEP = BASE/'cells/hmmt_k2048_w1_raw_qk_fast_full_protected/formal/grades.jsonl'
RANDOM_PP = E/'results/hmmt_official_random_pp_k1024_k2048_20261008/grades.jsonl'
ROOT = E/'results/hmmt_full_w1_raw_qk_interval64_k1024_k2048_20261015'
NEW_FILES = ('ra_harness/kvcompress/engine/w1_full_interval_cache.py',
             'scripts/run_w1_full_interval.py', 'tests/test_w1_full_interval.py')


def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p, value): p.write_text(json.dumps(value, indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
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
    template = dict(cfg['cells'][0])
    cells = []
    for K in (1024, 2048):
        cell = dict(template)
        cell.update(name=f'hmmt_k{K}_w1_raw_qk_full64_protected', K=K,
                    method='grid_full64_w1_topk_protected', storage='full',
                    mask_interval=64, compaction_interval=None,
                    implementation='full_w1_raw_qk_interval64_v1')
        cells.append(cell)
    cfg.update(created=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(),
               experiment='hmmt_full_w1_raw_qk_interval64_k1024_k2048_v1', cells=cells,
               max_concurrent_gpus=8,
               scope_authorization='2026-10-15 user requests changing Full-KV W1 to build the token mask every64 steps like random_pp, at K1024 and K2048, with random_pp-matched budget growth.',
               rank_backend_selection='Fixed stable_sort, identical to the completed per-step Full-KV v2 runs.',
               compaction_interval=None, mask_interval=64,
               baseline_root=str(BASE), baseline_budget=None,
               compaction=('None; retain all KV. At decode steps 1,65,129,... select total K (prompt+recent64 inside K) '
                           'from the full history with the per-step v2 W1 selector; append each new token to that '
                           'attention set until the next refresh (K..K+63 attended positions).'),
               caveats=[
                   '60 canonical questions x2 sampled rollouts; accuracy per rollout, not pass@2.',
                   'Algorithm change: the W1 mask is refreshed every64 decode steps, not every step.',
                   'At refresh, total K includes prompt and recent64; then up to63 newer tokens are appended (random_pp-matched).',
                   'Refresh selection is the frozen v2 raw-QK Full-KV selector; same private representative stream at that step.',
                   'Full KV retains all history; no permanent eviction; any token can return at a later refresh.',
                   'GPU-synchronized timings include selection/gather/attention/sampling; exclude loading/warmup/serialization.',
                   'Batch2 seconds are per question batch, not per individual answer.',
                   'Different A100 allocations and freely generated lengths limit isolated speed comparisons.',
                   'Per-step Full-KV W1 and official random_pp at the same K are accuracy comparators, not exact-output references.',
                   '2 cells x20 formal shards x3 questions, at most8 GPUs; formal work waits for both3-question smoke gates.'
               ])
    write(ROOT/'config.json', cfg)
    write(ROOT/'smoke_work.json', [dict(cell=c, shard=0, num_shards=1) for c in range(2)])
    write(ROOT/'formal_work.json', [dict(cell=c, shard=s, num_shards=20) for c in range(2) for s in range(20)])
    references = {}
    for cell, per_step in zip(cells, (K1024_PER_STEP, K2048_PER_STEP)):
        name = cell['name']
        shutil.copyfile(per_step, ROOT/'reference_previous'/f'{name}_per_step_grades.jsonl')
        rp = [dict(idx=r['idx'], run=r['run'], correct=r['correct']) for r in rows(RANDOM_PP)
              if r['cell'] == f"official_random_pp_k{cell['K']}_recent64"]
        assert len(rp) == 120 and len({(r['run'], r['idx']) for r in rp}) == 120
        with (ROOT/'reference_previous'/f'{name}_random_pp_grades.jsonl').open('x') as f:
            f.writelines(json.dumps(r)+'\n' for r in rp)
        references[name] = dict(per_step=str(per_step), per_step_sha256=sha(per_step),
                                random_pp=str(RANDOM_PP), random_pp_sha256=sha(RANDOM_PP))

    p = ROOT/'source/scripts/selector_grid.py'
    text = p.read_text()
    text = replace_once(text, 'Frozen Full-KV raw-QK W1 K2048 run', 'Frozen Full-KV raw-QK W1 interval64 run (K1024/K2048)')
    text = replace_once(text, "'source/scripts/run_w1_qk_fast.py'", "'source/scripts/run_w1_full_interval.py'")
    text = replace_once(text, "str(cell.get('compaction_interval',cfg['compaction_interval'])) if cell['storage'] in ('compact','hybrid') else '1'",
                        "str(cell['mask_interval'])")
    text = replace_once(text, "            assert r['compaction_interval'] is None and r['eviction_interval']==1\n",
                        "            assert r['compaction_interval'] is None and r['eviction_interval']==cell['mask_interval']==64\n")
    start = text.index('        total=0;events=0\n')
    end = text.index("        for row in r['final_cache_seqlens']:assert all(n==cached for n in row)")
    text = text[:start]+'''        # Analytic interval64 attention sizes: refresh at decode steps1,65,...
        # to min(K, history); otherwise previous size + the appended token.
        I=cell['mask_interval'];total=0
        for t in ts:
            current=None
            for step in range(1,t):
                current=min(cell['K'],P+step) if (step-1)%I==0 else current+1
                total+=current
        assert r['decode_selected_pairs']==factor*total
        longest=max(ts);cached=P+longest-1
        assert r['compaction_events']==0
        assert r['selection_events']==layers*len(range(1,longest,I))
        assert r['max_selected_tokens']<=max(P,cell['K']+I-1)
'''+text[end:]
    text = replace_once(text, "'passed/full_kv_fixed_budget_analytic' if cell['storage']=='full' else 'passed/bounded_hybrid_fixed_attention_analytic'",
                        "'passed/full_kv_interval64_analytic'")
    text = replace_once(text, '''            reference=root/'reference_previous'/f'{name}_grades.jsonl'
            comparisons.append(compare(name+'_probability_scored',name,reference,candidate_path,'raw_QK_vs_probability_matched_K_storage_protection'))''',
                        '''            for label,kind in (('per_step','interval64_vs_per_step_full_kv_w1_matched_K'),('random_pp','interval64_full_kv_w1_vs_official_random_pp_matched_K')):
                reference=root/'reference_previous'/f'{name}_{label}_grades.jsonl'
                comparisons.append(compare(name+'_'+label,name,reference,candidate_path,kind))''')
    p.write_text(text)

    check = '''"""Interval64 validation: new layer vs independent reference, plus v2 regressions."""
import hashlib, importlib.util, json, os, sys, unittest
from pathlib import Path
import torch
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'source/ra_harness'))
suite=unittest.TestSuite()
for name in ('test_w1_full_interval','test_w1_qk_fast','test_full_w1_qk'):
    spec=importlib.util.spec_from_file_location(name,root/'source/tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
with (root/'correctness.log').open('x') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
proof=dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
 tests=result.testsRun,skipped=[str(s) for s in result.skipped],gpu=torch.cuda.get_device_name(),
 node=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID'),K=[1024,2048],mask_interval=64,backend='stable_sort',
 log_sha256=hashlib.sha256((root/'correctness.log').read_bytes()).hexdigest())
(root/'correctness.json').write_text(json.dumps(proof,indent=2)+'\\n')
print(json.dumps(proof),flush=True)
assert proof['status']=='passed'
'''
    (ROOT/'source/scripts/benchmark_w1_qk_fast.py').write_text(check)
    for stage in ('correctness', 'smoke', 'gate', 'formal', 'collect'):
        p = ROOT/'source/scripts'/f'{stage}.sbatch'
        text = p.read_text().replace('hmmt-qk2048-', 'hmmt-full64-')
        if stage == 'smoke':
            text = replace_once(text, '--array=0-0%1', '--array=0-1%2')
        if stage == 'formal':
            text = replace_once(text, '--array=0-19%2', '--array=0-39%8')
        p.write_text(text)
    write(ROOT/'provenance.json', dict(base=str(BASE), references=references,
          score='raw BF16 current-query QK at refresh; stable chronological ties; no selector softmax',
          implementation='full_w1_raw_qk_interval64_v1', refresh_selector='full_w1_raw_qk_fast_v2',
          backend='stable_sort', mask_interval=64,
          implementation_sha256=sha(ROOT/'source/ra_harness/kvcompress/engine/w1_full_interval_cache.py'),
          v2_selector_sha256=sha(ROOT/'source/ra_harness/kvcompress/engine/w1_qk_fast_cache.py'),
          input_sha256=sha(ROOT/'inputs/hmmt.jsonl')))
    (ROOT/'README.md').write_text('''# HMMT Full-KV W1 raw-QK, mask refreshed every64 steps: K1024 / K2048

Algorithm change relative to per-step Full-KV W1, following the official random_pp
cadence. All KV is retained. At decode steps1,65,129,... the frozen v2 raw-QK
selector ranks the complete history with the current representative query and
selects total K (prompt + recent64 inside K). Until the next refresh each new
token is appended to that set, so attention covers K..K+63 positions (random_pp
K1024 covers1024..1087). Any earlier token can return at a later refresh.
Qwen3-4B BF16, 60 canonical questions x2 sampled outputs, batch2, seed42,
representative seed0, temperature0.6, generation top-p0.95, total limit32768.
GPU correctness (new layer vs independent reference + v2 regressions) -> two
3-question smokes -> CPU gate -> 2x20 formal shards (<=8 A10080GB) -> collection.
Comparators: per-step Full-KV W1 and official random_pp at the same K (paired).
''')
    for p in (ROOT/'source').rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    for stage in ('correctness', 'smoke', 'gate', 'formal', 'collect'):
        subprocess.run(['bash', '-n', str(ROOT/'source/scripts'/f'{stage}.sbatch')], check=True)
    assert sha(ROOT/'source/ra_harness/kvcompress/engine/w1_qk_fast_cache.py') == sha(BASE/'source/ra_harness/kvcompress/engine/w1_qk_fast_cache.py')
    write(ROOT/'static_validation.json', dict(status='passed', K=[1024, 2048], cells=2, mask_interval=64,
          questions=60, answers_per_cell=120, formal_shards=40, max_concurrent_gpus=8,
          refresh_selector_identical_to_per_step_v2=True, backend='stable_sort'))
    files = [ROOT/p for p in ('config.json', 'smoke_work.json', 'formal_work.json',
                             'model_assets_sha256.json', 'inputs/hmmt.jsonl', 'data/hmmt/test.jsonl', 'provenance.json')]
    files += [p for folder in ('source', 'reference_previous') for p in (ROOT/folder).rglob('*') if p.is_file()]
    write(ROOT/'manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})
    print(json.dumps(dict(root=str(ROOT), static_validation='passed', manifest_files=len(files))), flush=True)


if __name__ == '__main__':
    main()
