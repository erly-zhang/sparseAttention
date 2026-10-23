"""Freeze a v7 multi-dataset run.  usage: prepare_multibench_v7.py {qwen4b_baselines|phi4_all} {stage1|finalize}

Both runs reuse the v6 run's frozen source (with its recorded fixes), data directory
and graders, plus the paper-baseline adapters and the v6 kernel generalisation.
stage1 writes everything but the input references; for a new model the references are
then generated with the frozen harness (multibench_v7.py references) and finalize
writes the manifest.
"""
import ast, datetime, hashlib, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

E = Path(__file__).resolve().parents[1]
R = E/'results'
V6 = R/'multibench_v6_randompp_dense_20261016'
RA = Path('/mmfs1/project/yl258/jz648/Random-Attention')
NEW = ['ra_harness/kvcompress/engine/official_baselines_cache.py', 'ra_harness/kvcompress/engine/cache_utils.py',
       'ra_harness/kvcompress/engine/w1_hybrid_v6_cache.py',
       'tests/test_official_baselines.py', 'tests/test_w1_hybrid_v6_phi.py', 'scripts/multibench_v7.py']
BASELINES = ('snapkv', 'rkv', 'vase', 'triattn')
SPECS = {
    'qwen4b_baselines': dict(root=R/'multibench_v7_baselines_qwen3_4b_20261018', model='/scratch/yl258/jz648/Qwen3-4B',
                             stats='qwen3-4b.pt', reuse_inputs=True,
                             k1024=BASELINES, k2048=BASELINES,
                             tests=['test_official_baselines', 'test_official_random_pp']),
    'phi4_all': dict(root=R/'multibench_v7_phi4_reasoning_20261018', model='/scratch/yl258/jz648/Phi-4-reasoning',
                     stats='phi-4.pt', reuse_inputs=False,
                     k1024=('dense', 'hybrid', 'random_pp')+BASELINES, k2048=('hybrid', 'random_pp')+BASELINES,
                     tests=['test_w1_hybrid_v6', 'test_w1_hybrid_v6_phi', 'test_w1_hybrid_variants',
                            'test_w1_hybrid_gpu_select', 'test_official_baselines', 'test_official_random_pp',
                            'test_multibench_dense_metrics']),
}


def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2)+'\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:70], text.count(old))
    return text.replace(old, new)


def stage1(spec):
    root = spec['root']
    assert not root.exists(), 'Never overwrite a run'
    root.mkdir()
    files = [p for p in (V6/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for parent in {(root/'source'/p.relative_to(V6/'source')).parent for p in files}:
        parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda p: shutil.copyfile(p, root/'source'/p.relative_to(V6/'source')), files))
    # The v6 working-tree engine equals the frozen v6 one except the documented edits.
    for rel in NEW:
        shutil.copyfile(E/rel, root/'source'/rel)
    ev = root/'source/ra_harness/kvcompress/harness/eval_hf.py'
    text = ev.read_text()
    text = replace(text, "            if getattr(args,'eviction_mode','dense') == 'official_random_pp':",
                   """            if getattr(args,'eviction_mode','dense') in ('official_snapkv','official_rkv','official_vase','official_triattn'):
                generation_metrics.update(selection_strategy=args.eviction_mode, residual_length=args.residual_length,
                    n_large=args.n_large, rkv_lambda=args.rkv_lambda, smooth=args.smooth,
                    upstream_commit='64db9688a12f0926db45fec039370f6ceb1ab4fe')
            if getattr(args,'eviction_mode','dense') == 'official_random_pp':""")
    ev.write_text(text)
    # Data: identical files to the v6 run (LCB tests by the same symlink and hash list).
    (root/'data').mkdir()
    for d in (V6/'data').iterdir():
        if d.is_dir():
            (root/'data'/d.name).mkdir()
            shutil.copyfile(d/'test.jsonl', root/'data'/d.name/'test.jsonl')
    os.symlink(os.readlink(V6/'data/livecodebench/livecodebench_tests'), root/'data/livecodebench/livecodebench_tests')
    shutil.copyfile(V6/'data/lcb_test_assets_sha256.json', root/'data/lcb_test_assets_sha256.json')
    shutil.copyfile(RA/'kvcompress/harness/stats_pl'/spec['stats'], root/'data/triattn_stats.pt')
    (root/'inputs').mkdir()
    (root/'gold').mkdir()
    for p in (V6/'inputs').glob('*.jsonl'):
        if spec['reuse_inputs']:
            shutil.copyfile(p, root/'inputs'/p.name)
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        (root/'gold'/p.name).write_text(''.join(json.dumps(dict(idx=r['idx'], gold=r['gold']))+'\n' for r in rows))
    for d in ('logs', 'cells'):
        (root/d).mkdir()
    model = Path(spec['model'])
    mc = read(model/'config.json')
    geometry = dict(num_hidden_layers=mc['num_hidden_layers'], num_attention_heads=mc['num_attention_heads'],
                    num_key_value_heads=mc['num_key_value_heads'],
                    head_dim=mc.get('head_dim') or mc['hidden_size']//mc['num_attention_heads'])
    assets = sorted(p for p in model.iterdir() if p.is_file() and p.suffix in ('.json', '.safetensors', '.model', '.txt', '.py'))
    with ThreadPoolExecutor(6) as pool:
        hashes = dict(zip([p.name for p in assets], pool.map(sha, assets)))
    write(root/'model_assets_sha256.json', hashes)
    v6 = read(V6/'config.json')
    tasks = v6['tasks']
    cells = []
    for task in tasks:
        for wave, K in (('k1024', 1024), ('k2048', 2048)):
            for method in spec[wave]:
                cells.append(dict(name=f"{task['name']}_{method}"+('' if method == 'dense' else f'_k{K}'),
                                  task=task['name'], method=method, K=None if method == 'dense' else K, wave=wave))
    index = {(c['task'], c['method'], c['K']): i for i, c in enumerate(cells)}
    for wave, K in (('k1024', 1024), ('k2048', 2048)):
        work = [dict(cell=index[(t['name'], m, None if m == 'dense' else K)], shard=s, num_shards=t['shards'])
                for t in tasks for s in range(t['shards']) for m in spec[wave]]
        write(root/f'work_{wave}.json', work)
    cfg = dict(model=str(model), model_geometry=geometry, packages=v6['packages'], max_tokens=32768, seed=42,
               batch_size=2, rollouts=2, generation='model generation_config temperature and top-p (top-k unused by the harness, as in the paper)',
               tasks=tasks, cells=cells, correctness_tests=spec['tests'],
               created=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(),
               fairness=v6['fairness'], methods=v6['methods'] | dict(
                   snapkv='Paper port (upstream 64db9688): attn --smooth, K incl. recent64, eviction every 64 steps.',
                   rkv='Paper port: attn_rkv rkv_lambda=0.5 --smooth.',
                   vase='Paper port: range_sink_sample_attn n_large=K/4 --smooth (private RNG stream).',
                   triattn='Paper port: triattn_ph n_large=256 with the model calibration stats, TRIATTN_FIX_MEMO=1.'),
               grading=v6['grading'], source_base=str(V6))
    write(root/'config.json', cfg)
    s = root/'source/scripts'
    for name in ('mb_check', 'mb_infer', 'mb_collect'):
        text = (s/f'{name}.sbatch').read_text().replace('multibench_v6.py', 'multibench_v7.py').replace('mb6-', 'mb7-')
        if name == 'mb_infer':
            text = replace(text, '#SBATCH --time=03:00:00', '#SBATCH --time=04:00:00')
        (s/f'{name}.sbatch').write_text(text)
    for p in (root/'source').rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    write_manifest(root)
    print(json.dumps(dict(root=str(root), cells=len(cells), geometry=geometry,
                          k1024=len(read(root/'work_k1024.json')), k2048=len(read(root/'work_k2048.json')))))


def write_manifest(root):
    files = [root/p for p in ('config.json', 'work_k1024.json', 'work_k2048.json', 'model_assets_sha256.json')]
    files += sorted(p for p in (root/'inputs').glob('*.jsonl'))+sorted((root/'gold').glob('*.jsonl'))
    files += sorted((root/'data').glob('*/test.jsonl'))+[root/'data/lcb_test_assets_sha256.json', root/'data/triattn_stats.pt']
    files += sorted(p for p in (root/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    write(root/'manifest.json', {str(p.relative_to(root)): sha(p) for p in files})


if __name__ == '__main__':
    spec = SPECS[sys.argv[1]]
    stage1(spec) if sys.argv[2] == 'stage1' else write_manifest(spec['root'])
