"""Dependency-free, deterministic synthetic retrieval protocol."""
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

SPLITS = {'train': 32, 'validation': 8, 'heldout': 8}
METHODS = ('dense', 'qonly', 'linear', 'mlp')
MAX_NEW = 128


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    # Exclusive creation intentionally preserves incomplete files on interruption.
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def specs():
    return [dict(id=f'{split}-{i:02d}', split=split, index=i,
                 seed=202609280000 + s * 1000 + i,
                 length_bin=(4096, 16384)[i % 2],
                 depth=(0.1, 0.35, 0.65, 0.9)[(i // 2) % 4])
            for s, (split, count) in enumerate(SPLITS.items()) for i in range(count)]


def make_prompt(spec, encode_chat):
    rng = random.Random(spec['seed'])
    pool = [(str(uuid.UUID(int=rng.getrandbits(128))),
             str(uuid.UUID(int=rng.getrandbits(128))))
            for _ in range(spec['length_bin'] // 20)]

    def render(count):
        pairs = pool[1:count]
        target_row = int((count - 1) * spec['depth'])
        pairs.insert(target_row, pool[0])
        body = '\n'.join(f'{k}: {v}' for k, v in pairs)
        prompt = ('Read the key-value records below. Each key occurs exactly once.\n'
                  + body + '\n\nReturn only the value associated with key '
                  + pool[0][0] + '. Do not include an explanation.')
        return prompt, encode_chat(prompt), pairs, target_row

    lo, hi = 2, len(pool)
    assert len(render(lo)[1]) < spec['length_bin'] < len(render(hi)[1])
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        if len(render(middle)[1]) <= spec['length_bin']:
            lo = middle
        else:
            hi = middle
    prompt, ids, pairs, target_row = render(lo)
    assert spec['length_bin'] - 256 <= len(ids) <= spec['length_bin']
    return dict(**spec, prompt=prompt, ids=ids, pairs=pairs,
                target_key=pool[0][0], answer=pool[0][1], target_row=target_row)


def validate_inputs(records, excluded_hashes=()):
    assert {r['id'] for r in records} == {r['id'] for r in specs()}
    assert len(records) == 48
    assert len({r['input_ids_sha256'] for r in records}) == 48
    assert not set(excluded_hashes).intersection(r['input_ids_sha256'] for r in records)
    seen_keys, seen_values = set(), set()
    expected = {s['id']: s for s in specs()}
    for row in records:
        assert all(row[k] == v for k, v in expected[row['id']].items())
        assert row['length_bin'] - 256 <= row['input_tokens'] <= row['length_bin']
        keys, values = zip(*row['pairs'])
        assert len(set(keys)) == len(keys) and len(set(values)) == len(values)
        assert seen_keys.isdisjoint(keys) and seen_values.isdisjoint(values)
        seen_keys.update(keys)
        seen_values.update(values)
        assert row['pairs'][row['target_row']] == [row['target_key'], row['answer']]
    return True


def exact_match(prediction, answer):
    return int(prediction.strip().strip('"\'`').strip() == answer)


def validate_results(rows, records):
    expected = {r['id']: r for r in records if r['split'] == 'heldout'}
    assert len(rows) == 32
    assert {(r['id'], r['method']) for r in rows} == {
        (identity, method) for identity in expected for method in METHODS}
    for row in rows:
        assert row['input_ids_sha256'] == expected[row['id']]['input_ids_sha256']
        assert row['correct'] == exact_match(row['prediction'], expected[row['id']]['answer'])
        assert row['finite_all_steps'] and row['full_kv_cache'] and row['dense_decode']
        assert 1 <= len(row['generated_ids']) <= MAX_NEW
    return True
