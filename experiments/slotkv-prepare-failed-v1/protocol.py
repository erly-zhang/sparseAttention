"""Frozen design and validation for a 2x2 output assembly ablation."""
import ast
import hashlib
import random
from pathlib import Path

METHODS = ('dense', 'group_pair', 'group_native', 'all_pair', 'all_native')
TASKS = ('kv_retrieval', 'code_debug')
CELLS = {'group_pair': ('q_kv', 'pair'), 'group_native': ('q_kv', 'native'),
         'all_pair': ('q_only', 'pair'), 'all_native': ('q_only', 'native')}
LIMIT = 131072 - 128
CODE_REV = '90f0394333616266d9fe85824ceaf505093cbaa5'
CODE_SHA = '481cc7d6315af24086fd57e2ed764f113a844f3ad9d66ab1828f547fccb31f2e'
CODE_URL = 'https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/' + CODE_REV + '/code_debug.jsonl'
CODE_TEMPLATE = ('Following is a Python code where exactly one of the functions/methods '
    'has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\n'
    'A. {a}\nB. {b}\nC. {c}\nD. {d}\n\nThe correct option is:')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def code_prompt(row):
    a, b, c, d = row['options']
    assert all(isinstance(x, str) and x for x in (a, b, c, d))
    return CODE_TEMPLATE.format(context=row['context'], a=a, b=b, c=c, d=d)


def code_indices(size):
    assert size >= 54
    return random.Random(20260929).sample(list(range(4, size)), 50)


def scorers(path):
    names = ['get_score_one_' + task for task in TASKS]
    funcs = [f for f in ast.parse(Path(path).read_text()).body
             if isinstance(f, ast.FunctionDef) and f.name in names]
    assert {f.name for f in funcs} == set(names)
    ns = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), ns)
    return {task: ns['get_score_one_' + task] for task in TASKS}


def pair_count(n, mode, heads=32):
    if mode == 'q_only':
        return n * (n + 1) // 2
    assert mode == 'q_kv'
    sizes = [(n + heads - 1 - h) // heads for h in range(heads)]
    return sum(size * (size + 1) // 2 for size in sizes)


def validate(rows, inputs, dense_layers):
    expected = {(r['task'], r['source_index']): r for r in inputs}
    assert len(expected) == len(inputs)
    assert len(rows) == len(inputs) * len(METHODS)
    seen = set()
    for row in rows:
        key = (row['task'], row['source_index'], row['method'])
        assert key not in seen and row['method'] in METHODS
        seen.add(key)
        item = expected[key[:2]]
        for field in ('input_tokens', 'input_ids_sha256', 'answers', 'raw_row_sha256'):
            assert row[field] == item[field], (key, field)
        assert 1 <= len(row['generated_ids']) == row['generated_tokens'] <= 128
        assert row['full_kv_cache'] and row['dense_decode'] and row['finite_all_steps']
        if row['method'] == 'dense':
            assert row['routed_layer_eligible_pairs'] == {}
        else:
            counts = {int(k): v for k, v in row['routed_layer_eligible_pairs'].items()}
            assert set(counts) == set(range(32)) - set(dense_layers)
            assert set(counts.values()) == {pair_count(row['input_tokens'], CELLS[row['method']][0])}
        if row['task'] == 'kv_retrieval' and row['source_index'] in (331, 61, 16) and row['method'] in ('dense', 'group_pair'):
            assert row['historical_token_ids_equal'] is True
    assert seen == {(task, index, method) for task, index in expected for method in METHODS}
    return True
