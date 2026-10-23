"""Freeze the multi-dataset hybrid v6 / official random_pp / Dense comparison.

One source tree, data directory and input reference set for all three methods.
Waves: k1024 (hybrid K1024, random_pp K1024, Dense) then k2048 (hybrid, random_pp).
"""
import ast, datetime, hashlib, json, os, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

E = Path(__file__).resolve().parents[1]
R = E/'results'
BASE = R/'hmmt_w1_raw_qk_hybrid_variants_k1024_k2048_20261016'
DENSE_SRC = R/'hmmt_dense_and_timing_diagnosis_20261008/source/ra_harness/kvcompress/harness/generation_utils.py'
MULTI = R/'compact_multibench_20260926'
MATH_DATA = R/'representative_w1_compact_math_20260926/data/math/test.jsonl'
MATH_INPUTS = R/'official_random_pp_k1024_r64_20260924/inputs.jsonl'
ROOT = R/'multibench_v6_randompp_dense_20261016'
NEW = ['ra_harness/kvcompress/engine/w1_hybrid_gpu_select_cache.py',
       'ra_harness/kvcompress/engine/w1_hybrid_variants_cache.py',
       'ra_harness/kvcompress/engine/w1_hybrid_v5_cache.py',
       'ra_harness/kvcompress/engine/w1_hybrid_v6_cache.py',
       'tests/test_w1_hybrid_gpu_select.py', 'tests/test_w1_hybrid_variants.py',
       'tests/test_w1_hybrid_v5.py', 'tests/test_w1_hybrid_v6.py', 'tests/test_multibench_dense_metrics.py',
       'scripts/multibench_v6.py']
# (name, questions, questions per shard): small tasks first so early results are complete.
TASKS = [('aime25', 30, 1), ('aime26', 30, 1), ('hmmt', 60, 1), ('gpqa', 198, 2),
         ('math', 500, 4), ('livecodebench', 383, 2)]


def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:70], text.count(old))
    return text.replace(old, new)


def main():
    assert not ROOT.exists(), 'Never overwrite a run'
    ROOT.mkdir()
    files = [p for p in (BASE/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for parent in {(ROOT/'source'/p.relative_to(BASE/'source')).parent for p in files}:
        parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda p: shutil.copyfile(p, ROOT/'source'/p.relative_to(BASE/'source')), files))
    for rel in NEW:
        shutil.copyfile(E/rel, ROOT/'source'/rel)
    runner = (E/'scripts/run_w1_hybrid_variants.py').read_text()
    runner = replace(runner, 'w1_hybrid_variants_cache import install', 'w1_hybrid_v6_cache import install')
    runner = replace(runner, '[W1 hybrid v4 variants]', '[W1 hybrid v6 single launch]')
    (ROOT/'source/scripts/run_w1_hybrid_v6.py').write_text(runner)

    # Dense timing/metric branch: identical phase boundaries for every method.
    h = ROOT/'source/ra_harness/kvcompress/harness'
    gen = (h/'generation_utils.py').read_text()
    dense = DENSE_SRC.read_text()
    gen = replace(gen, "    measure_phases = metrics is not None and isinstance(current_cache, EvictCache) and (getattr(current_cache.layers[0], 'uses_temporary_mask', False) or getattr(current_cache.layers[0], 'records_attention_metrics', False))",
                  "    measure_phases = metrics is not None")
    start_marker = '    if measure_phases:\n        torch.cuda.synchronize(device)\n'
    end_marker = "\n    if metrics is not None and isinstance(current_cache, EvictCache) and all("
    a, b = gen.index(start_marker), gen.index(end_marker)
    da, db = dense.index(start_marker), dense.index(end_marker)
    old_block, new_block = gen[a:b], dense[da:db]
    assert 'standard_dense_DynamicCache' in new_block and 'isinstance(current_cache, EvictCache)' in new_block
    # The sparse branch of the new block must equal the old block (re-indented).
    for line in old_block.splitlines()[2:]:
        assert ('    '+line).rstrip() in new_block or line.strip() in new_block, line
    gen = gen[:a]+new_block+gen[b:]
    (h/'generation_utils.py').write_text(gen)
    ev = (h/'eval_hf.py').read_text()
    # Only the two timing/memory guards (as in the Dense run); line 194 (warmup sample_id) stays eviction-only.
    ev = replace(ev, "        if args.sparsity_method == 'eviction':\n            torch.cuda.synchronize()\n            torch.cuda.reset_peak_memory_stats()",
                 "        if args.sparsity_method in ('eviction', 'dense'):\n            torch.cuda.synchronize()\n            torch.cuda.reset_peak_memory_stats()")
    ev = replace(ev, "        if args.sparsity_method == 'eviction':\n            torch.cuda.synchronize()\n        end = time.time()",
                 "        if args.sparsity_method in ('eviction', 'dense'):\n            torch.cuda.synchronize()\n        end = time.time()")
    (h/'eval_hf.py').write_text(ev)

    # Data and per-question input references shared by every method.
    for name, _, _ in TASKS:
        (ROOT/'data'/name).mkdir(parents=True)
        src = MATH_DATA if name == 'math' else MULTI/'data'/name/'test.jsonl'
        shutil.copyfile(src, ROOT/'data'/name/'test.jsonl')
    os.symlink(os.readlink(MULTI/'data/livecodebench/livecodebench_tests'), ROOT/'data/livecodebench/livecodebench_tests')
    shutil.copyfile(MULTI/'test_assets_sha256.json', ROOT/'data/lcb_test_assets_sha256.json')
    (ROOT/'inputs').mkdir()
    for name, count, _ in TASKS:
        src = MATH_INPUTS if name == 'math' else MULTI/'inputs'/f'{name}.jsonl'
        rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
        assert len(rows) == count and {r['idx'] for r in rows} == set(range(count)), name
        keep = [{k: r[k] for k in ('idx', 'input_tokens', 'input_ids_sha256', 'prompt_text', 'gold')} for r in rows]
        (ROOT/'inputs'/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in keep))
    shutil.copyfile(BASE/'model_assets_sha256.json', ROOT/'model_assets_sha256.json')
    for d in ('logs', 'cells'):
        (ROOT/d).mkdir()

    base_cfg = read(BASE/'config.json')
    tasks = [dict(name=n, count=c, per_shard=q, shards=-(-c//q)) for n, c, q in TASKS]
    cells = []
    for task in tasks:
        for method, K, wave in (('hybrid', 1024, 'k1024'), ('random_pp', 1024, 'k1024'), ('dense', None, 'k1024'),
                                ('hybrid', 2048, 'k2048'), ('random_pp', 2048, 'k2048')):
            cells.append(dict(name=f"{task['name']}_{method}" + (f'_k{K}' if K else ''), task=task['name'],
                              method=method, K=K, wave=wave))
    index = {(c['task'], c['method'], c['K']): i for i, c in enumerate(cells)}
    for wave, methods in (('k1024', (('hybrid', 1024), ('random_pp', 1024), ('dense', None))),
                          ('k2048', (('hybrid', 2048), ('random_pp', 2048)))):
        work = [dict(cell=index[(t['name'], m, k)], shard=s, num_shards=t['shards'])
                for t in tasks for s in range(t['shards']) for m, k in methods]
        write(ROOT/f'work_{wave}.json', work)
    cfg = dict(model=base_cfg['model'], model_geometry=base_cfg['model_geometry'], packages=base_cfg['packages'],
               max_tokens=32768, seed=42, batch_size=2, rollouts=2, generation_temperature=0.6, generation_top_p=0.95,
               tasks=tasks, cells=cells,
               created=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(),
               fairness='One frozen source/data/input-reference set for all methods; runtime input SHA check per question; '
                        'identical generation settings, warmup, timing scope and graders; only the attention/cache algorithm differs; dense prefill for all.',
               methods=dict(
                   hybrid='W1 v6 (bitwise = v4/v2 masks): every decode step raw-QK representative-head TopK of total K (prompt + recent64 inside K; '
                          'prompts longer than K-64 protect the first K-65 tokens); bounded pool: every 1024 steps keep W1 core + full prompt + Uniform random to K+4096.',
                   random_pp='Official per-KV-head Uniform TopK, prompt protected up to K-65, recent64 residual, eviction every 64 steps.',
                   dense='Standard DynamicCache + FlashAttention2.'),
               grading=dict(math='last boxed answer, math equivalence (timeout=False, 30 s process bound)',
                            gpqa='strict final boxed A-D', livecodebench='all tests pass, 20 s/test, 2048 MiB, release_v6 medium subset'))
    write(ROOT/'config.json', cfg)

    s = ROOT/'source/scripts'
    head = '''#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
'''
    common = '''set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RUN_ROOT="${1:?}"
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
export PYTHONPATH="$RUN_ROOT/source/ra_harness"
'''
    (s/'mb_check.sbatch').write_text(head+'''#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=01:00:00
#SBATCH --job-name=mb6-check
'''+common+'"$PY" -u "$RUN_ROOT/source/scripts/multibench_v6.py" check "$RUN_ROOT"\n')
    (s/'mb_infer.sbatch').write_text(head+'''#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=n0068,n0090,n0049
#SBATCH --time=03:00:00
#SBATCH --job-name=mb6-infer
'''+common+'"$PY" -u "$RUN_ROOT/source/scripts/multibench_v6.py" infer "$RUN_ROOT" "${2:?}" "${SLURM_ARRAY_TASK_ID:?}"\n')
    (s/'mb_collect.sbatch').write_text(head+'''#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --job-name=mb6-collect
'''+common+'export MB_LCB_WORKERS=28\n"$PY" -u "$RUN_ROOT/source/scripts/multibench_v6.py" collect "$RUN_ROOT" "${2:?}"\n')
    for p in (ROOT/'source').rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    for name in ('mb_check', 'mb_infer', 'mb_collect'):
        subprocess.run(['bash', '-n', str(s/f'{name}.sbatch')], check=True)
    files = [ROOT/p for p in ('config.json', 'work_k1024.json', 'work_k2048.json', 'model_assets_sha256.json')]
    files += sorted(p for p in (ROOT/'inputs').rglob('*') if p.is_file())
    files += sorted(p for p in (ROOT/'data').glob('*/test.jsonl'))+[ROOT/'data/lcb_test_assets_sha256.json']
    files += sorted(p for p in (ROOT/'source').rglob('*') if p.is_file())
    write(ROOT/'manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in files})
    print(json.dumps(dict(root=str(ROOT), cells=len(cells),
                          k1024_tasks=len(read(ROOT/'work_k1024.json')), k2048_tasks=len(read(ROOT/'work_k2048.json')))), flush=True)


if __name__ == '__main__':
    main()
