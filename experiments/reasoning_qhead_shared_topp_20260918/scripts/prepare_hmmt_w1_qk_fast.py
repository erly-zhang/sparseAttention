"""Freeze K1024 full/hybrid raw-QK implementation optimization and gated jobs."""
import ast
import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = Path(__file__).resolve().parents[1]
BASE = E/'results/hmmt_w1_raw_qk_k1024_hybrid4096_compact1000_20261011'
FULL = E/'results/hmmt_w1_raw_qk_k1024_full_compact500_20261009'
ROOT = E/'results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012'


def read(p): return json.loads(p.read_text())
def write(p, data): p.write_text(json.dumps(data, indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=ROOT.name)
    parser.add_argument('--previous-failed-run', type=Path)
    args = parser.parse_args()
    assert Path(args.run_name).name == args.run_name
    ROOT = E/'results'/args.run_name
    assert not ROOT.exists(), 'Never overwrite a run'
    ROOT.mkdir()
    for folder in ('source', 'inputs', 'data'):
        files = [Path(s) for s in subprocess.check_output(
            ['rg', '--files', '--hidden', '--no-ignore', str(BASE/folder)], text=True).splitlines()]
        files = [p for p in files if '__pycache__' not in p.parts and p.suffix != '.pyc']
        for parent in {(ROOT/folder/p.relative_to(BASE/folder)).parent for p in files}:
            parent.mkdir(parents=True, exist_ok=True)
        def copy(p):
            with p.open('rb') as src, (ROOT/folder/p.relative_to(BASE/folder)).open('xb') as dst:
                shutil.copyfileobj(src, dst)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy, files))
    for name in ('model_assets_sha256.json', 'prompt_protection_stats.json'):
        shutil.copyfile(BASE/name, ROOT/name)
    for rel in ('ra_harness/kvcompress/engine/w1_qk_fast_cache.py',
                'scripts/run_w1_qk_fast.py', 'scripts/benchmark_w1_qk_fast.py',
                'scripts/run_full_w1_optimized.py', 'tests/test_w1_qk_fast.py',
                'tests/test_full_w1_qk.py', 'tests/test_w1_interval1000.py'):
        shutil.copyfile(E/rel, ROOT/'source'/rel)
    (ROOT/'logs').mkdir()
    (ROOT/'reference_previous').mkdir()
    cfg = read(BASE/'config.json')
    cfg.update(created=datetime.datetime.now().astimezone().isoformat(),
               experiment='hmmt_w1_raw_qk_fast_full_hybrid_k1024_v2',
               scope_authorization='2026-10-12 user requests further implementation optimization then HMMT K1024; explicitly confirms full and random4096 hybrid both run.',
               optimization='Same BF16 QK and representative; exact packed signed BF16+earlier-position TopK; fused index assembly; skip unused short-history scoring; same hybrid Uniform primary/secondary stream with fused key processing and no redundant residual sort.',
               rank_backend_selection='Select per storage from synchronized GPU layer microbenchmark; stable_sort and packed_qk_topk both preserve masks.',
               cells=[])
    for storage in ('full', 'hybrid'):
        cell = dict(name=f'hmmt_k1024_w1_raw_qk_fast_{storage}_protected', task='hmmt',
                    count=60, K=1024, shards=20,
                    method='grid_full_w1_topk_protected' if storage == 'full' else 'grid_hybrid1000_w1_topk_protected',
                    storage=storage, scorer='w1', selector='topk', protection='protected', window=1,
                    compaction_interval=None if storage == 'full' else 1000, score_mode='qk',
                    extra_budget=0 if storage == 'full' else 4096,
                    candidate_budget=None if storage == 'full' else 5120,
                    implementation=f'{storage}_w1_raw_qk_fast_v2')
        cfg['cells'].append(cell)
        old_root = FULL if storage == 'full' else BASE
        old_name = 'hmmt_k1024_w1_raw_qk_full_protected' if storage == 'full' else 'hmmt_k1024_w1_raw_qk_hybrid1000_protected'
        old = old_root/'cells'/old_name/'formal'
        for artifact in ('summary.json', 'grades.jsonl'):
            shutil.copyfile(old/artifact, ROOT/'reference_previous'/f"{cell['name']}_{artifact}")
        completions = []
        for shard in range(4):
            for run in (0, 1):
                files = list((old/f'shard_{shard}'/f'run_{run}').glob('completions*.jsonl'))
                assert len(files) == 1
                for line in files[0].read_text().splitlines():
                    record = json.loads(line)
                    completions.append(dict(idx=record['idx'], run=run, completion=record['completion']))
        assert len(completions) == len({(r['idx'], r['run']) for r in completions}) == 120
        write(ROOT/'reference_previous'/f"{cell['name']}_completions.json", completions)
    cfg['caveats'] = [s for s in cfg['caveats'] if not s.startswith('Compact1000')]
    cfg['caveats'] += ['Full KV retains all historical keys; attention selects protected K1024 each step.',
                       'Implementation equivalence is to the existing raw-QK versions, not historical probability scoring.',
                       'GPU benchmark selects exact backend per storage; formal timings may use a different A10080GB allocation.',
                       '20 shards/cell x3 questions: at most2 additional GPUs; shorter reservations fit before maintenance when resources allow.']
    write(ROOT/'config.json', cfg)
    write(ROOT/'smoke_work.json', [dict(cell=i, shard=0, num_shards=1) for i in range(2)])
    write(ROOT/'formal_work.json', [dict(cell=i, shard=s, num_shards=20) for s in range(20) for i in range(2)])

    gen = ROOT/'source/ra_harness/kvcompress/harness/generation_utils.py'
    text = gen.read_text().replace('sum(l.compaction_events for l in layers)',
                                   "sum(getattr(l,'compaction_events',0) for l in layers)")
    gen.write_text(text)
    script = ROOT/'source/scripts/selector_grid.py'
    text = script.read_text().replace('run_w1_interval1000.py', 'run_w1_qk_fast.py')
    text = text.replace("with (out/'runner.log').open('x') as f:proc=subprocess.run(args,env=env_for(root,cell['task']),stdout=f,stderr=subprocess.STDOUT)",
        "env=env_for(root,cell['task'])\n    env.update(RA_W1_QK_FAST_BACKEND=read(root/'microbenchmark.json')['selected_backends'][cell['storage']],RA_W1_TOKEN_OUTPUT=str(out/'token_ids.json'))\n    with (out/'runner.log').open('x') as f:proc=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)")
    start = text.index("        assert r['w1_score_mode']=='qk'")
    end = text.index("        ts=r['generated_tokens_per_rollout']", start)
    text = text[:start]+"""        assert r['w1_score_mode']=='qk'
        assert r['w1_rank_backend']==read(root/'microbenchmark.json')['selected_backends'][cell['storage']]
        assert r['w1_implementation']==cell['implementation']
        if cell['storage']=='hybrid':
            assert r['compaction_interval']==r['eviction_interval']==1000
            assert r['compaction_phase']=='after_attention'
            assert r['cache_semantics']=='bounded_candidates_temporary_attention_mask'
            assert r['candidate_budget']==5120 and r['extra_candidate_budget']==4096
            assert r['max_capacity']==6120
            assert r['compaction_core_token_sum']==1024*r['compaction_token_observations']
            assert r['compaction_residual_token_sum']==4096*r['compaction_token_observations']
        else:
            assert r['compaction_interval'] is None and r['eviction_interval']==1
            assert r['cache_semantics']=='full_history_temporary_mask' and r['compaction_events']==0
"""+text[end:]
    start = text.index('        total=0;events=0')
    end = text.index("        assert all(c==P+max(ts)-1", start)
    text = text[:start]+"""        total=0;events=0
        for t in ts:
            cached=P
            for step in range(1,t):
                cached+=1;total+=min(cell['K'],cached)
                if cell['storage']=='hybrid' and step%1000==0 and cached>5120:
                    cached=5120;events+=1
        assert r['decode_selected_pairs']==factor*total
        longest=max(ts);cached=P;batch_events=0
        for step in range(1,longest):
            cached+=1
            if cell['storage']=='hybrid' and step%1000==0 and cached>5120:
                cached=5120;batch_events+=1
        assert r['compaction_events']==layers*batch_events
        assert r['selection_events']==layers*(longest-1)
        if cell['storage']=='hybrid':assert r['compaction_token_observations']==layers*events
        assert r['max_selected_tokens']<=max(P,1024)
        for row in r['final_cache_seqlens']:assert all(n==cached for n in row)
"""+text[end:]
    needle = '    return metrics,comps'
    replacement = """    tokens=read(d/'token_ids.json')
    assert len(tokens)==len(ids) and {t['idx'] for t in tokens}==ids
    refs={(r['run'],r['idx']):r['completion'] for r in read(root/'reference_previous'/f"{cell['name']}_completions.json")}
    byid={r['idx']:r for r in metrics}
    for t in tokens:
        assert len(t['token_ids'])==2
        for run, sequence in enumerate(t['token_ids']):
            assert len(sequence)==byid[t['idx']]['input_tokens']+max(byid[t['idx']]['generated_tokens_per_rollout'])
    for (run,idx),text in comps.items():
        assert expected[idx]['prompt_text']+text==refs[run,idx], ('raw-QK output divergence',cell['name'],idx,run)
    return metrics,comps"""
    assert needle in text
    text = text.replace(needle, replacement)
    text = text.replace("'passed/interval1000_analytic' if cell['storage']=='compact' else 'passed/bounded_hybrid_fixed_attention_analytic'",
        "'passed/full_kv_fixed_budget_analytic' if cell['storage']=='full' else 'passed/bounded_hybrid_fixed_attention_analytic'")
    text = text.replace("'compact1000_vs_compact500_raw_QK' if cell['storage']=='compact' else 'hybrid_random4096_interval1000_vs_full_raw_QK'", "'implementation_only_same_raw_QK_algorithm'")
    text = text.replace("name+'_previous_compact500' if cell['storage']=='compact' else name+'_previous_full'", "name+'_previous_same_storage'")
    script.write_text(text)
    for stage in ('correctness', 'smoke', 'gate', 'formal', 'collect'):
        p = ROOT/'source/scripts'/f'{stage}.sbatch'
        text = p.read_text().replace('hmmt-k1024-i1000-', 'hmmt-qk-fast-')
        text = re.sub(r'(?m)^#SBATCH --job-name=.*$', f'#SBATCH --job-name=hmmt-qk-fast-{stage}', text)
        if stage == 'correctness':
            text = text.replace('--time=01:00:00', '--time=00:45:00').replace('check_budgets.py','benchmark_w1_qk_fast.py')
        if stage == 'smoke':
            text = text.replace('--time=04:00:00', '--time=03:00:00')
        if stage == 'formal':
            text = text.replace('--array=0-7%2','--array=0-39%2').replace('--time=20:00:00','--time=03:00:00')
        p.write_text(text)
    write(ROOT/'provenance.json', dict(base=str(BASE), full_reference=str(FULL),
          input_sha256=sha(ROOT/'inputs/hmmt.jsonl'),
          implementation_sha256=sha(E/'ra_harness/kvcompress/engine/w1_qk_fast_cache.py'),
          previous_failed_run=str(args.previous_failed_run.resolve()) if args.previous_failed_run else None))
    (ROOT/'README.md').write_text('''# HMMT raw-QK W1 implementation optimization: K1024

Full KV and bounded random4096 candidates, with the existing1000-step update.
Qwen3-4B BF16,60 canonical questions x2 sampled outputs, batch2, seed42,
temperature0.6, generation top-p0.95, prompt+output cap32768. Protect prompt+recent64 inside K.
Only implementation changes: exact raw BF16 stable-order TopK, fused index assembly,
skip unused short-history scoring; same random draws, fused residual key processing.

GPU correctness and timing precede full generation on the same3 inputs per method.
Every output must match the completed same-storage raw-QK reference text exactly.
Formal inference runs60 inputs per cell in20 shards, with at most2 extra A10080GB GPUs.
Each3-question shard requests3 hours to permit scheduling before maintenance.
Independent output directory; source/config/input/model hashes checked by every job.
Collector requires60 unique inputs/120 outputs per cell, full input alignment and
analytic pair accounting. Decode includes all scoring/selection/random/gather/attention/sampling;
loading, warmup, correctness tests and serialization are excluded.
''')
    for p in (ROOT/'source').rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    for stage in ('correctness','smoke','gate','formal','collect'):
        subprocess.run(['bash','-n',str(ROOT/'source/scripts'/f'{stage}.sbatch')],check=True)
    write(ROOT/'static_validation.json', dict(status='passed', cells=2, questions_per_cell=60,
          rollouts_per_cell=120, smoke_ids=[0,1,2], formal_shards=40, AST='passed',shell='passed'))
    files = [ROOT/p for p in ('config.json','smoke_work.json','formal_work.json',
                             'model_assets_sha256.json','inputs/hmmt.jsonl','data/hmmt/test.jsonl','provenance.json')]
    files += [p for folder in ('source','reference_previous') for p in (ROOT/folder).rglob('*') if p.is_file()]
    write(ROOT/'manifest.json', {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    print(json.dumps(dict(root=str(ROOT), static_validation='passed', manifest_files=len(files))))


if __name__ == '__main__':
    main()
