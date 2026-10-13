"""Fresh synthetic pilot, separate from all previously evaluated inputs."""
import hashlib
import json
import os
from pathlib import Path

LAYERS = tuple(range(0, 32, 2))
METHODS = ('dense', 'repeat16', 'mlp16')
SPLITS = {'train': 32, 'validation': 8, 'heldout': 8}
RANK = 128
EPOCHS = 5
POSITIONS = 256
DEADLINE = 13800


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def specs():
    return [dict(id=f'{split}-{i:02d}', split=split, index=i,
                 seed=202609290000 + s * 1000 + i,
                 length_bin=(4096, 16384)[i % 2],
                 depth=(0.1, 0.35, 0.65, 0.9)[(i // 2) % 4])
            for s, (split, count) in enumerate(SPLITS.items()) for i in range(count)]


def positions(n):
    if n < POSITIONS:
        raise ValueError('prompt too short')
    # Uniform context coverage plus all final64 prompt positions, chosen a priori.
    prefix = [i * (n - 65) // 191 for i in range(192)]
    result = prefix + list(range(n - 64, n))
    assert len(set(result)) == POSITIONS
    return result


def validate_inputs(rows, excluded_hashes=(), excluded_keys=(), excluded_values=()):
    expected = {s['id']: s for s in specs()}
    assert len(rows) == len(expected) == 48
    assert {r['id'] for r in rows} == set(expected)
    assert len({r['input_ids_sha256'] for r in rows}) == 48
    assert not set(excluded_hashes).intersection(r['input_ids_sha256'] for r in rows)
    keys_seen, values_seen = set(excluded_keys), set(excluded_values)
    for row in rows:
        assert all(row[k] == v for k, v in expected[row['id']].items())
        assert row['length_bin'] - 256 <= row['input_tokens'] <= row['length_bin']
        keys, values = zip(*row['pairs'])
        assert len(set(keys)) == len(keys) and len(set(values)) == len(values)
        assert keys_seen.isdisjoint(keys) and values_seen.isdisjoint(values)
        keys_seen.update(keys)
        values_seen.update(values)
        assert row['pairs'][row['target_row']] == [row['target_key'], row['answer']]
        assert row['feature_positions'] == positions(row['input_tokens'])


def exact_match(prediction, answer):
    return int(prediction.strip().strip('\"\'`').strip() == answer)


def pick_checkpoint(history):
    if not history or any(not __import__('math').isfinite(r['validation_loss']) for r in history):
        raise ValueError('invalid validation history')
    return min(history, key=lambda r: (r['validation_loss'], r['epoch']))


def projection(elapsed, dense_cost, sparse_cost, generation_cost, layer_fit_cost):
    # Conservative full-length prefix collection for every layer and every prompt.
    return elapsed + 1.5 * (40 * dense_cost + 16 * 40 * sparse_cost
                           + 24 * generation_cost + 16 * layer_fit_cost) + 1200


def validate_results(rows, manifest):
    heldout = {r['id']: r for r in manifest if r['split'] == 'heldout'}
    assert len(rows) == 24
    assert {(r['id'], r['method']) for r in rows} == {(i, m) for i in heldout for m in METHODS}
    for row in rows:
        item = heldout[row['id']]
        assert row['input_ids_sha256'] == item['input_ids_sha256']
        assert row['correct'] == exact_match(row['prediction'], item['answer'])
        assert row['full_kv_cache'] and row['finite_all_steps'] and row['dense_decode']
        assert 1 <= len(row['generated_ids']) <= 128
        pairs = {int(k): v for k, v in row['pairs'].items()}
        assert pairs == ({} if row['method'] == 'dense' else
                         {i: item['input_tokens'] * (item['input_tokens'] + 1) // 2 for i in LAYERS})
