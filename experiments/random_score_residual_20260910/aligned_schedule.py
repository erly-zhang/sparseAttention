"""Data-order/reference-order adapter for the isolated random-score scheduler.

The evaluator sorts requests by length. Dataset slices therefore cannot be
validated against positional slices of the full generation-order metrics.
This adapter resolves every dataset record to its exact canonical token hash.
"""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml
from jinja2 import Environment, StrictUndefined
from transformers import AutoTokenizer

from experiments.random_score_residual_20260910 import schedule as base

_preflight = base.preflight
_identities = []


def resolve_identities(config):
    data = base.rows(config['data'])
    ref = base.rows(config['reference_metrics'])
    node = yaml.compose((Path(config['task_config_dir']) / 'kv_retrieval.yaml').read_text())
    template_text = {k.value: v for k, v in node.value}['doc_to_text'].value
    template = Environment(undefined=StrictUndefined).from_string(template_text)
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    lookup = {base.identity(row): i for i, row in enumerate(ref)}
    identities = []
    for index, record in enumerate(data):
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': template.render(**record)}],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        limit = 131072 - 128
        if len(ids) > limit:
            left = limit // 2
            ids = ids[:left] + ids[-(limit-left):]
        identity = (hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(), len(ids))
        if identity not in lookup:
            raise RuntimeError(f'Dataset record {index} does not match any canonical input: {identity}')
        identities.append(identity)
    assert len(identities) == len(set(identities)) == 497
    assert set(identities) == set(lookup)
    return identities, [lookup[x] for x in identities]


def preflight(config):
    global _identities
    ready = _preflight(config)
    if ready['ready']:
        _identities, positions = resolve_identities(ready['config'])
        ready['data_order_alignment'] = {
            'passed': True, 'count': 497, 'input_identities': _identities,
            'reference_indices': positions,
            'meaning': 'Each dataset record maps to an exact input hash and token count; no positional assumption.'}
    return ready


def expected(a, b):
    assert len(_identities) == 497
    return _identities[a:b]


def validate(path, c, a, b, phase):
    s = base.read(path / 'summary.json')
    rr = base.rows(path / 'online_metrics.jsonl')
    ids = list(map(base.identity, rr))
    assert len(ids) == len(set(ids)) == b-a and set(ids) == set(expected(a, b))
    align = s['input_alignment']
    assert align['status'] == 'passed' and align['count'] == b-a and align['reference_count'] == 497
    assert s['method'] == base.METHOD and s['run_args']['max_new_tokens'] == 128
    assert s['run_args']['max_length'] == 131072 and s['run_args']['seed'] == 42 and s['run_args']['chat']
    assert s['method_metadata']['profile_only'] == (phase == 'profile')
    old = {base.identity(x): x for x in base.rows(c['old_random_metrics'])}
    shared = {base.identity(x): x for x in base.rows(c['shared_metrics'])}
    for r in rr:
        assert r['profile_only'] == (phase == 'profile')
        for field in ('selected_token_pairs', 'causal_token_pairs'):
            assert r[field] == old[base.identity(r)][field] == shared[base.identity(r)][field]
        for field in ('prefill_latency_sec', 'total_latency_sec'):
            assert np.isfinite(r[field]) and r[field] > 0
        assert np.isfinite(r['decode_latency_sec']) and r['decode_latency_sec'] >= 0
        assert r['peak_memory_bytes'] > 0
        if phase == 'smoke':
            assert r['smoke_indices_and_finite_logits_passed']
        if phase == 'profile':
            assert Path(r['profiling_file']).is_file()
    samples = base.read(path / 'lm_eval_results.json')['samples']['kv_retrieval']
    assert len(samples) == len({x['doc_hash'] for x in samples}) == b-a
    assert all(x['get_score_one_kv_retrieval'] in (0, 1) for x in samples)
    accuracy = sum(x['get_score_one_kv_retrieval'] for x in samples)/(b-a)
    assert abs(accuracy - s['official_lm_eval_results']['kv_retrieval']['get_score_one_kv_retrieval,none']) < 1e-9
    return s, rr, samples


def job(root, c, phase, a, b):
    directory = root / 'attempts' / phase / f'{a}_{b}'
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in sorted(directory.glob('attempt*')):
        path = attempt / base.METHOD / 'kv_retrieval'
        try:
            validate(path, c, a, b, phase)
            return path
        except (FileNotFoundError, AssertionError, KeyError, json.JSONDecodeError):
            pass
    inventory = base.free_gpu_inventory()
    attempt = directory / f'attempt{len(list(directory.glob("attempt*")))+1}'
    attempt.mkdir()
    cfgdir = base.task_config(c, root/'data'/f'{a}_{b}', base.rows(c['data'])[a:b])
    cmd = [sys.executable, str(base.HERE/'online.py'), '--model', c['model'], '--method', base.METHOD,
           '--task', 'kv_retrieval', '--max_length', '131072', '--batch_size', '1', '--seed', '42', '--chat',
           '--group_config', c['group_config'], '--flexprefill_root', c['flexprefill_root'],
           '--task_config_dir', str(cfgdir), '--record_sparsity', '--fixed_topk_budget', '10240',
           '--output_dir', str(attempt), '--reference_metrics', c['reference_metrics']]
    env = {**os.environ, 'PYTHONPATH': str(base.WORK), 'PYTHONNOUSERSITE': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'RANDOM_SCORE_PROFILE': str(int(phase == 'profile')),
           'RANDOM_SCORE_VALIDATE': str(int(phase == 'smoke')),
           'TRITON_CACHE_DIR': str(root/'cache/triton'), 'HF_HOME': str(root/'cache/huggingface')}
    expected_path = attempt/'expected_inputs.json'
    base.save(expected_path, [list(x) for x in expected(a, b)])
    env['RANDOM_SCORE_EXPECTED_INPUTS'] = str(expected_path)
    base.save(attempt/'command.json', {'command': cmd, 'phase': phase, 'gpu': inventory,
                                     'started': base.time.time(), 'identity_mapping': 'data-order exact hashes'})
    with (attempt/'run.log').open('x') as log:
        subprocess.run(cmd, cwd=base.WORK, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    path = attempt/base.METHOD/'kv_retrieval'
    validate(path, c, a, b, phase)
    return path


# Reuse the original fail-closed gates, immutable manifest, merge and reporting.
# Only dataset-slice identity resolution changes; online inference is untouched.
base.preflight = preflight
base.validate = validate
base.job = job

if __name__ == '__main__':
    sys.exit(base.main())
