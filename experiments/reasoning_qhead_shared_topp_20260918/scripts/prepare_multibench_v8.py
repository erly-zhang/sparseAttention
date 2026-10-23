"""Freeze the v8 pool-update ablation run (Qwen3-4B, MATH500 + HMMT, K=1024).

usage: prepare_multibench_v8.py {stage1|finalize}

Source: the v7 Qwen3-4B root's frozen source (= v6 frozen source with its recorded fixes + the paper
baseline adapters), plus the pool-ablation engine module, its runner, its tests and multibench_v8.py.
Data, input references, gold answers and model asset hashes are copied unchanged from that root, so the
prompts are byte-identical to the v6 hybrid / Dense cells these cells are compared against.
"""
import datetime, hashlib, json, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

E = Path(__file__).resolve().parents[1]
R = E/'results'
V7 = R/'multibench_v7_baselines_qwen3_4b_20261018'
ROOT = R/'multibench_v8_pool_ablation_k1024_20261022'
NEW = ['ra_harness/kvcompress/engine/w1_hybrid_pool_ablation_cache.py', 'scripts/run_w1_hybrid_pool_ablation.py',
       'tests/test_w1_hybrid_pool_ablation.py', 'scripts/multibench_v8.py']
TASKS = ('math', 'hmmt')
METHODS = ('pool_i', 'pool_ii', 'pool_iii')
TESTS = ['test_w1_hybrid_pool_ablation', 'test_w1_hybrid_v6', 'test_w1_hybrid_variants']


def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2)+'\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stage1():
    assert not ROOT.exists(), 'Never overwrite a run'
    ROOT.mkdir()
    files = [p for p in (V7/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for parent in {(ROOT/'source'/p.relative_to(V7/'source')).parent for p in files}:
        parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda p: shutil.copyfile(p, ROOT/'source'/p.relative_to(V7/'source')), files))
    for rel in NEW:
        assert not (ROOT/'source'/rel).exists(), rel
        shutil.copyfile(E/rel, ROOT/'source'/rel)
    for sb in ('mb_check', 'mb_infer', 'mb_collect'):
        p = ROOT/'source/scripts'/f'{sb}.sbatch'
        p.write_text(p.read_text().replace('multibench_v7.py', 'multibench_v8.py').replace('mb7-', 'mb8-'))
    (ROOT/'data').mkdir()
    for d in (V7/'data').iterdir():
        if d.is_dir():
            (ROOT/'data'/d.name).mkdir()
            shutil.copyfile(d/'test.jsonl', ROOT/'data'/d.name/'test.jsonl')
    os.symlink(os.readlink(V7/'data/livecodebench/livecodebench_tests'), ROOT/'data/livecodebench/livecodebench_tests')
    for f in ('lcb_test_assets_sha256.json', 'triattn_stats.pt'):
        shutil.copyfile(V7/'data'/f, ROOT/'data'/f)
    for d in ('inputs', 'gold'):
        shutil.copytree(V7/d, ROOT/d)
    for d in ('logs', 'cells'):
        (ROOT/d).mkdir()
    shutil.copyfile(V7/'model_assets_sha256.json', ROOT/'model_assets_sha256.json')
    v7 = read(V7/'config.json')
    tasks = [t for t in v7['tasks'] if t['name'] in TASKS]
    cells = [dict(name=f"{t['name']}_{m}_k1024", task=t['name'], method=m, K=1024, wave='k1024')
             for t in tasks for m in METHODS]
    index = {(c['task'], c['method']): i for i, c in enumerate(cells)}
    work = [dict(cell=index[(t['name'], m)], shard=s, num_shards=t['shards'])
            for t in tasks for s in range(t['shards']) for m in METHODS]
    write(ROOT/'work_k1024.json', work)
    write(ROOT/'work_k2048.json', [])
    cfg = {k: v7[k] for k in ('model', 'model_geometry', 'packages', 'max_tokens', 'seed', 'batch_size', 'rollouts',
                              'generation', 'fairness', 'grading')}
    cfg.update(tasks=tasks, cells=cells, correctness_tests=TESTS, source_base=str(V7),
               created=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(),
               methods=dict(pool_i='Hybrid W1 (v6 attention), total pool 4096; after every decode step one Uniform '
                                   'token outside that step\'s W1 core is removed (v6 compaction, interval 1).',
                            pool_ii='Hybrid W1 v6, pool K+4096, compaction every 64 decode steps.',
                            pool_iii='Hybrid W1 (v6 attention), total pool 4096; before every decode step one Uniform '
                                     'token outside the full prompt and the 63 newest slots is removed (W1 core not '
                                     'protected), then W1 selects K.'),
               comparators='v6 hybrid K1024 and Dense cells of multibench_v6_randompp_dense_20261016 (same inputs)')
    write(ROOT/'config.json', cfg)
    print(json.dumps(dict(root=str(ROOT), cells=len(cells), work=len(work))))


def write_manifest():
    files = [ROOT/p for p in ('config.json', 'work_k1024.json', 'work_k2048.json', 'model_assets_sha256.json')]
    files += sorted((ROOT/'inputs').glob('*.jsonl'))+sorted((ROOT/'gold').glob('*.jsonl'))
    files += sorted((ROOT/'data').glob('*/test.jsonl'))+[ROOT/'data/lcb_test_assets_sha256.json', ROOT/'data/triattn_stats.pt']
    files += sorted(p for p in (ROOT/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    write(ROOT/'manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in files})


if __name__ == '__main__':
    stage1() if sys.argv[1] == 'stage1' else write_manifest()
