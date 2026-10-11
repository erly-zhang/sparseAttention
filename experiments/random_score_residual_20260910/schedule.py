"""Fail-closed, resumable single-method 3 smoke -> 497 formal -> 3 profile."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
METHOD = 'probmean_shared6144_randomscore4096_allmembers_seed42'
SHARDS = [(0, 125), (125, 250), (250, 375), (375, 497)]
REQUIRED = ('model', 'data', 'group_config', 'reference_metrics', 'old_random_metrics',
            'shared_metrics', 'smoke_data', 'task_config_dir', 'dataset_loader', 'flexprefill_root')


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(row): return row['input_ids_sha256'], int(row['input_tokens'])


def save(path, value):
    """Immutable results; identical resume writes are harmless."""
    path = Path(path)
    if path.exists():
        if read(path) != value: raise RuntimeError(f'Refusing to overwrite {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2)


def source_hashes():
    result = {}
    for sub in ('random_score_residual_20260910', 'residual6040_20260908',
                'probmean_compact_20260906', 'random_residual_20260908', 'rep_residual_20260907'):
        for p in (WORK / 'experiments' / sub).glob('*.py'):
            result[str(p.relative_to(WORK))] = sha(p)
    for name in ('token_compacted_sparse.py', 'benchmark_shareprefill_ae3.py',
                 'run_dense_multimodel_infinitebench.py', 'run_kv_oracle_infinitebench.py', 'jsd_grouped_sparse.py'):
        p = WORK / 'experiments' / name
        result[str(p.relative_to(WORK))] = sha(p)
    return result


def preflight(config):
    missing = [k for k in REQUIRED if not config.get(k) or not Path(config[k]).exists()]
    if missing: return {'ready': False, 'missing': missing}
    c = {k: Path(config[k]).resolve() for k in REQUIRED}
    model = read(c['model'] / 'config.json')
    assert (model['num_attention_heads'], model['num_key_value_heads'], model['num_hidden_layers']) == (32, 8, 32)
    assert model['hidden_size'] == 4096 and model['model_type'] == 'llama'
    index = c['model'] / 'model.safetensors.index.json'
    weights = sorted(set(read(index)['weight_map'].values()))
    assert all((c['model'] / p).is_file() for p in weights)
    assert (c['model'] / 'tokenizer_config.json').is_file()
    assert (c['model'] / 'tokenizer.json').is_file()
    data, ref, smoke = rows(c['data']), rows(c['reference_metrics']), rows(c['smoke_data'])
    assert len(data) == len(ref) == 497
    assert len({json.dumps(x, sort_keys=True) for x in data}) == 497
    assert len(set(map(identity, ref))) == 497
    assert len({r['input_ids_sha256'] for r in ref}) == 497
    assert smoke == data[:3] and len(smoke) == 3
    for key in ('old_random_metrics', 'shared_metrics'):
        other = rows(c[key])
        assert len(other) == 497 and set(map(identity, other)) == set(map(identity, ref))
        assert all(isinstance(r['selected_token_pairs'], int) and isinstance(r['causal_token_pairs'], int) for r in other)
    old = {identity(r): r for r in rows(c['old_random_metrics'])}
    for r in rows(c['shared_metrics']):
        assert all(r[k] == old[identity(r)][k] for k in ('selected_token_pairs', 'causal_token_pairs'))
    groups = read(c['group_config'])['layers']
    assert set(map(int, groups)) == set(range(32))
    for g in groups.values():
        assert len(g) == len({x['representative'] for x in g}) == 3
        assert sorted(h for x in g for h in x['members']) == list(range(32))
        assert all(x['representative'] in x['members'] for x in g)
    assert (c['task_config_dir'] / 'kv_retrieval.yaml').is_file()
    assert (c['flexprefill_root'] / 'experiments/benchmark/utils.py').is_file()
    return {'ready': True, 'count': 497, 'smoke_count': 3, 'config': {k: str(v) for k, v in c.items()},
        'artifact_sha256': {k: sha(v) for k, v in c.items() if v.is_file()},
        'model_files': {str(p.relative_to(c['model'])): sha(p) for p in c['model'].glob('*.json')},
        'weight_files': [{'name': w, 'bytes': (c['model']/w).stat().st_size} for w in weights],
        'task_config_sha256': {p.name: sha(p) for p in c['task_config_dir'].iterdir() if p.suffix in ('.py', '.yaml')},
        'flexprefill_utils_sha256': sha(c['flexprefill_root'] / 'experiments/benchmark/utils.py')}


def check_correctness(local_path, cross_path):
    local, cross = read(local_path), read(cross_path)
    assert local['passed'] and local['fingerprint'] == cross['fingerprint']
    local_uuids = {s.split(',')[0].strip() for s in local['gpu_inventory'].splitlines()}
    cross_uuids = {s.split(',')[0].strip() for s in cross['gpu_inventory'].splitlines()}
    assert local_uuids.isdisjoint(cross_uuids), 'Cross-GPU test must use a different physical GPU'
    assert local['torch'] == cross['torch'], 'Use the same test environment on both GPUs'
    for name in ('sampler.py', 'selector.py', 'diagnostics.py', 'test_correctness.py'):
        assert local['source_sha256'][name] == cross['source_sha256'][name] == sha(HERE / name)
    for name, digest in local['legacy_source_sha256'].items():
        assert digest == cross['legacy_source_sha256'][name] == sha(WORK/name)
    return {'passed': True, 'local': str(local_path), 'cross': str(cross_path),
            'fingerprint': local['fingerprint'], 'distinct_physical_gpus': True}


def task_config(c, dest, records):
    datadir, cfgdir = dest / 'filtered_data', dest / 'task_configs'
    datadir.mkdir(parents=True, exist_ok=True)
    cfgdir.mkdir(exist_ok=True)
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    path = datadir / 'kv_retrieval.jsonl'
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    node = yaml.compose((Path(c['task_config_dir']) / 'kv_retrieval.yaml').read_text())
    # Preserve function tags, templates, filters and metric rules.
    top = {k.value: v for k, v in node.value}
    top['dataset_path'].value = c['dataset_loader']
    kwargs = {k.value: v for k, v in top['dataset_kwargs'].value}
    kwargs['data_dir'].value = str(datadir)
    text = yaml.serialize(node)
    path = cfgdir / 'kv_retrieval.yaml'
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    for p in Path(c['task_config_dir']).glob('*.py'):
        target = cfgdir / p.name
        if target.exists(): assert sha(target) == sha(p)
        else: shutil.copy2(p, target)
    return cfgdir


def validate(path, c, a, b, phase):
    s = read(path / 'summary.json')
    rr = rows(path / 'online_metrics.jsonl')
    expected = list(map(identity, rows(c['reference_metrics'])[a:b]))
    ids = list(map(identity, rr))
    assert len(ids) == len(set(ids)) == b-a and set(ids) == set(expected)
    align = s['input_alignment']
    assert align['status'] == 'passed' and align['count'] == b-a and align['reference_count'] == 497
    assert s['method'] == METHOD and s['run_args']['max_new_tokens'] == 128
    assert s['run_args']['max_length'] == 131072 and s['run_args']['seed'] == 42 and s['run_args']['chat']
    assert s['method_metadata']['profile_only'] == (phase == 'profile')
    old = {identity(x): x for x in rows(c['old_random_metrics'])}
    shared = {identity(x): x for x in rows(c['shared_metrics'])}
    for r in rr:
        assert r['profile_only'] == (phase == 'profile')
        for field in ('selected_token_pairs', 'causal_token_pairs'):
            assert r[field] == old[identity(r)][field] == shared[identity(r)][field]
        for field in ('prefill_latency_sec', 'total_latency_sec'):
            assert np.isfinite(r[field]) and r[field] > 0
        assert np.isfinite(r['decode_latency_sec']) and r['decode_latency_sec'] >= 0
        assert r['peak_memory_bytes'] > 0
        if phase == 'smoke': assert r['smoke_indices_and_finite_logits_passed']
        if phase == 'profile': assert Path(r['profiling_file']).is_file()
    samples = read(path / 'lm_eval_results.json')['samples']['kv_retrieval']
    assert len(samples) == len({x['doc_hash'] for x in samples}) == b-a
    assert all(x['get_score_one_kv_retrieval'] in (0, 1) for x in samples)
    accuracy = sum(x['get_score_one_kv_retrieval'] for x in samples)/(b-a)
    assert abs(accuracy - s['official_lm_eval_results']['kv_retrieval']['get_score_one_kv_retrieval,none']) < 1e-9
    return s, rr, samples


def free_gpu_inventory():
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True)
    # This scheduler intentionally uses one allocated, visible GPU serially.
    assert len(inventory.strip().splitlines()) == 1, 'Expose exactly one allocated GPU before launch'
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    uuid = inventory.split(',')[0].strip()
    if uuid in active: raise RuntimeError('Allocated GPU is occupied; existing processes were not modified')
    return inventory


def job(root, c, phase, a, b):
    directory = root / 'attempts' / phase / f'{a}_{b}'
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in sorted(directory.glob('attempt*')):
        path = attempt / METHOD / 'kv_retrieval'
        try:
            validate(path, c, a, b, phase)
            return path
        except (FileNotFoundError, AssertionError, KeyError, json.JSONDecodeError): pass
    inventory = free_gpu_inventory()
    attempt = directory / f'attempt{len(list(directory.glob("attempt*")))+1}'
    attempt.mkdir()
    cfgdir = task_config(c, root/'data'/f'{a}_{b}', rows(c['data'])[a:b])
    cmd = [sys.executable, str(HERE/'online.py'), '--model', c['model'], '--method', METHOD,
        '--task', 'kv_retrieval', '--max_length', '131072', '--batch_size', '1', '--seed', '42', '--chat',
        '--group_config', c['group_config'], '--flexprefill_root', c['flexprefill_root'],
        '--task_config_dir', str(cfgdir), '--record_sparsity', '--fixed_topk_budget', '10240',
        '--output_dir', str(attempt), '--reference_metrics', c['reference_metrics']]
    env = {**os.environ, 'PYTHONPATH': str(WORK), 'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'RANDOM_SCORE_PROFILE': str(int(phase == 'profile')), 'RANDOM_SCORE_VALIDATE': str(int(phase == 'smoke')),
        'TRITON_CACHE_DIR': str(root/'cache/triton'), 'HF_HOME': str(root/'cache/huggingface')}
    expected_path=attempt/'expected_inputs.json'
    save(expected_path,[list(identity(r)) for r in rows(c['reference_metrics'])[a:b]])
    env['RANDOM_SCORE_EXPECTED_INPUTS']=str(expected_path)
    save(attempt/'command.json', {'command': cmd, 'phase': phase, 'gpu': inventory, 'started': time.time()})
    with (attempt/'run.log').open('x') as log:
        subprocess.run(cmd, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    path = attempt/METHOD/'kv_retrieval'
    validate(path, c, a, b, phase)
    return path


def merge(root, c, paths):
    chunks = [validate(p, c, a, b, 'formal') for p, (a,b) in zip(paths, SHARDS)]
    rr = [r for _, part, _ in chunks for r in part]
    ref = rows(c['reference_metrics'])
    assert len(rr) == len({identity(r) for r in rr}) == 497
    by_id = {identity(r): r for r in rr}
    assert set(by_id) == set(map(identity, ref))
    rr = [dict(by_id[identity(r)], call_index=i) for i,r in enumerate(ref)]
    samples = [dict(x, source_shard=str(p)) for p, (_,_,part) in zip(paths, chunks) for x in part]
    assert len(samples) == len({x['doc_hash'] for x in samples}) == 497
    correct = int(sum(x['get_score_one_kv_retrieval'] for x in samples))
    pre = [r['prefill_latency_sec'] for r in rr]
    selected = sum(r['selected_token_pairs'] for r in rr)
    causal = sum(r['causal_token_pairs'] for r in rr)
    runtime = {'count': 497, 'correct': correct, 'accuracy': correct/497,
        'avg_prefill_latency_sec': float(np.mean(pre)), 'median_prefill_latency_sec': float(np.median(pre)),
        'p90_prefill_latency_sec': float(np.quantile(pre,.9)),
        'avg_decode_latency_sec': float(np.mean([r['decode_latency_sec'] for r in rr])),
        'avg_total_latency_sec': float(np.mean([r['total_latency_sec'] for r in rr])),
        'max_peak_memory_bytes': max(r['peak_memory_bytes'] for r in rr),
        'selected_token_pairs': selected, 'causal_token_pairs': causal, 'global_pair_sparsity': 1-selected/causal}
    for key in ('old_random_metrics','shared_metrics'):
        other = rows(c[key])
        assert sum(x['selected_token_pairs'] for x in other) == selected
        assert sum(x['causal_token_pairs'] for x in other) == causal
    summary = {**chunks[0][0], 'runtime': runtime,
        'timing_scope': chunks[0][0]['method_metadata']['timing_scope'],
        'source_shards': list(map(str, paths)),
        'input_alignment': {'status':'passed', 'alignment_scope':'full', 'count':497, 'reference_count':497,
                            'compared_fields':['input_ids_sha256','input_tokens']},
        'pair_budget_alignment': {'passed':True, 'per_sample_exact':True, 'total_exact':True},
        'official_lm_eval_results': {'kv_retrieval':{'get_score_one_kv_retrieval,none':correct/497}}}
    dest = root/'merged'/METHOD/'kv_retrieval'
    dest.mkdir(parents=True,exist_ok=True)
    for name, data in [('online_metrics.jsonl',rr),('predictions.jsonl',samples)]:
        text = ''.join(json.dumps(r)+'\n' for r in data)
        path = dest/name
        if path.exists(): assert path.read_text() == text
        else: path.write_text(text)
    save(dest/'summary.json',summary)
    return dest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--correctness',type=Path)
    ap.add_argument('--cross-gpu',type=Path)
    args=ap.parse_args()
    root=args.output_root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(args.config)
    ready=preflight(cfg)
    print(json.dumps(ready,indent=2),flush=True)
    if not ready['ready']:
        save(root/f'preflight_missing_{time.time_ns()}.json',ready)
        return 2
    if args.preflight_only:return 0
    if not args.correctness or not args.cross_gpu:
        raise RuntimeError('Both local correctness and independent physical GPU fingerprint are required')
    gate=check_correctness(args.correctness,args.cross_gpu)
    c=ready['config']
    manifest={'method':METHOD,'seed':42,'formal_count':497,'smoke_indices':[0,1,2],
              'profile_indices':[0,1,2],'shards':SHARDS,'artifacts':ready,'sources':source_hashes()}
    # Convert tuples before equality checks on resume.
    manifest=json.loads(json.dumps(manifest))
    save(root/'manifest.json',manifest)
    versions=subprocess.check_output([sys.executable,'-s','-m','pip','freeze'],text=True)
    version_file=root/'environment.freeze.txt'
    if version_file.exists():assert version_file.read_text()==versions,'Environment changed since start'
    else:version_file.write_text(versions)
    save(root/'correctness_passed.json',gate)
    frozen=root/'source'
    for name,digest in manifest['sources'].items():
        target=frozen/name
        if target.exists():assert sha(target)==digest
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(WORK/name,target)
    smoke=job(root,c,'smoke',0,3)
    save(root/'smoke_passed.json',{'passed':True,'count':3,'path':str(smoke),'max_new_tokens':128})
    paths=[]
    for a,b in SHARDS:
        assert source_hashes()==manifest['sources'],'Source changed during run'
        paths.append(job(root,c,'formal',a,b))
    merged=merge(root,c,paths)
    profile=job(root,c,'profile',0,3)
    from experiments.random_score_residual_20260910.report import summarize
    summarize(root,c,merged,profile)
    save(root/'complete.json',{'complete':True,'formal_count':497,'input_alignment':'passed/full',
         'smoke_count':3,'profiling_count':3,'merged':str(merged)})
    return 0


if __name__=='__main__': sys.exit(main())
