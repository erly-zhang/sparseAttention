"""Deterministic sampling and descriptive output diagnostics, without tuning."""
import random


def select_indices(previous_indices):
    regression = [331, 61, 16]
    excluded = set(previous_indices) | {0, 1, 2, 3} | set(regression)
    fresh = random.Random(20260928).sample([i for i in range(500) if i not in excluded], 47)
    return regression + fresh


def repetition(ids):
    n = len(ids)
    run = longest = 0
    last = None
    for token in ids:
        run = run + 1 if token == last else 1
        longest = max(longest, run)
        last = token
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, n - 3))]
    repeated_fraction = 1 - len(set(grams)) / len(grams) if grams else 0.0
    suffix_period = None
    if n >= 32:
        tail = ids[-32:]
        suffix_period = next((p for p in range(1, 9)
                              if all(tail[i] == tail[i % p] for i in range(32))), None)
    return dict(unique_tokens=len(set(ids)), longest_identical_token_run=longest,
                repeated_4gram_fraction=repeated_fraction,
                last_32_token_period=suffix_period,
                repetitive_flag=n >= 32 and (suffix_period is not None or repeated_fraction >= 0.8))


def validate_rows(rows, manifest, methods):
    indices = manifest['indices']
    assert len(indices) == len(set(indices)) == 50
    assert len(rows) == len(methods) * 50
    expected = {r['source_index']: r for r in manifest['records']}
    assert set(expected) == set(indices)
    assert len({r['input_ids_sha256'] for r in expected.values()}) == 50
    found = set()
    for r in rows:
        key = (r['method'], r['source_index'])
        assert key not in found
        found.add(key)
        assert r['method'] in methods and r['source_index'] in expected
        for field in ('input_tokens', 'input_ids_sha256', 'answers'):
            assert r[field] == expected[r['source_index']][field]
        assert r['generated_tokens'] == len(r['generated_ids']) <= 128
        assert r['full_kv_cache'] and r['dense_decode']
    for index in indices:
        group = {r['method']: r for r in rows if r['source_index'] == index}
        assert set(group) == set(methods)
        assert group['repeat']['routed_layer_eligible_pairs'] == group['pair']['routed_layer_eligible_pairs']
        for method in ('dense', 'repeat'):
            if index in (331, 61, 16):
                assert group[method]['historical_token_ids_equal'] is True
    return True
