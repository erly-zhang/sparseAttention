"""Independent CPU review of completed outputs; does not modify frozen inputs.

Average layers within each prompt before resampling prompts. Paired contrasts
resample the same prompt IDs for both methods. Intervals are pointwise, not
adjusted for multiple comparisons.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'review'
OUT.mkdir(exist_ok=True)
METHODS = ['dense_full', 'coverage_token_vertical',
           'coverage_token_vertical_slash', 'random_token_vertical',
           'random_token_vertical_slash', 'flexprefill', 'minference']
TASKS = ['kv_retrieval', 'longbook_choice_eng', 'longbook_qa_eng', 'code_debug']
METRICS = {'energy_reconstructed', 'random_energy_reconstructed',
           'excess_energy_reconstructed', 'active_count',
           'reconstruction_relative_l2', 'remainder_relative_l2',
           'dense_active_support_recall', 'delta_energy_in_dense_active_span'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    torch.set_num_threads(2)
    manifest = read(HERE / 'manifest.json')
    locked = read(HERE / 'lock.json')
    for path, expected in locked['files'].items():
        assert sha(Path(path)) == expected, path
    expected_keys = {t: {r['key'] for r in manifest['records'] if r['task'] == t}
                     for t in TASKS}
    assert all(len(keys) == 50 for keys in expected_keys.values())
    source_rows = {(r['key'], r['method']): r for r in manifest['records']}
    cells = defaultdict(list)
    prompt_layers = defaultdict(dict)
    seen = set()
    for index in range(128):
        layer, task = index // 4, TASKS[index % 4]
        directory = HERE / 'outputs' / f'worker-{index}'
        completion = read(directory / 'complete.json')
        assert completion == dict(status='complete', layer=layer,
                                  task=task, records=50)
        assert read(directory / 'gpu-check.json')['status'] == 'PASS'
        paths = [p for p in directory.glob('*.json')
                 if p.name not in {'complete.json', 'progress.json', 'gpu-check.json'}]
        assert {p.stem for p in paths} == expected_keys[task]
        for path in paths:
            row = read(path)
            key = row['key']
            assert row['task'] == task and row['layer'] == layer
            assert (layer, key) not in seen
            seen.add((layer, key))
            assert path.with_suffix('.pt').is_file()
            assert set(row['methods']) == set(METHODS)
            for method in METHODS:
                assert set(row['methods'][method]) == {'16', '25'}
                for k_text, metrics in row['methods'][method].items():
                    assert set(metrics) == METRICS
                    k = int(k_text)
                    for metric, groups in metrics.items():
                        if method == 'dense_full' and metric == 'delta_energy_in_dense_active_span':
                            assert not groups
                            continue
                        assert 'tail' in groups
                        for group, value in groups.items():
                            assert np.isfinite(value)
                            cells[(task, method, k, layer, group, metric)].append(value)
                        identity = (task, method, k, key, metric)
                        prompt_layers[identity][layer] = groups['tail']
        if (index + 1) % 16 == 0:
            print('REVIEWED WORKERS', index + 1, flush=True)
    assert len(seen) == 6400
    summary = read(HERE / 'outputs/jspace-summary.json')
    maximum_error = 0.0
    assert len(summary['layer_means']) == len(cells)
    for row in summary['layer_means']:
        identity = tuple(row[name] for name in
                         ('task', 'method', 'k', 'layer', 'group', 'metric'))
        values = cells[identity]
        assert len(values) == row['n'] == 50
        maximum_error = max(maximum_error, abs(np.mean(values) - row['mean']))
    assert maximum_error < 1e-12

    values_by_group = defaultdict(dict)
    for (task, method, k, key, metric), layers in prompt_layers.items():
        assert set(layers) == set(range(32))
        values_by_group[(task, method, k, metric)][key] = float(np.mean(list(layers.values())))
    rows, contrasts = [], []
    for task in TASKS:
        keys = sorted(expected_keys[task])
        # Reuse the same draws for all methods and contrasts within a task.
        draws = np.random.default_rng(42 + TASKS.index(task)).integers(0, 50, (5000, 50))
        def stats(values):
            boot = values[draws].mean(axis=1)
            return dict(mean=float(values.mean()), n=50,
                        bootstrap_95_low=float(np.quantile(boot, .025)),
                        bootstrap_95_high=float(np.quantile(boot, .975)))
        for k in (16, 25):
            for method in METHODS:
                for metric in sorted(METRICS):
                    if method == 'dense_full' and metric == 'delta_energy_in_dense_active_span':
                        continue
                    group = values_by_group[(task, method, k, metric)]
                    assert set(group) == set(keys)
                    values = np.array([group[key] for key in keys])
                    rows.append(dict(task=task, method=method, k=k, group='tail',
                                     metric=metric, **stats(values)))
            for name, minuend, subtrahend in [
                ('calibrated_slash_minus_vertical', METHODS[2], METHODS[1]),
                ('random_slash_minus_vertical', METHODS[4], METHODS[3]),
                ('calibrated_slash_minus_flexprefill', METHODS[2], METHODS[5]),
                ('calibrated_slash_minus_dense', METHODS[2], METHODS[0]),
            ]:
                for metric in ('dense_active_support_recall', 'energy_reconstructed',
                               'reconstruction_relative_l2'):
                    a = values_by_group[(task, minuend, k, metric)]
                    b = values_by_group[(task, subtrahend, k, metric)]
                    difference = np.array([a[key] - b[key] for key in keys])
                    contrasts.append(dict(task=task, k=k, group='tail', contrast=name,
                                          metric=metric, **stats(difference)))

    # Recompute support overlaps directly from saved token IDs and coefficients
    # in four representative layer-16 files, without the worker's vectorization.
    checks = []
    for task, key in zip(TASKS, ['kv_retrieval-359', 'longbook_choice_eng-49',
                               'longbook_qa_eng-310', 'code_debug-149']):
        layer = 16
        index = layer * 4 + TASKS.index(task)
        path = HERE / 'outputs' / f'worker-{index}' / (key + '.pt')
        data = torch.load(path, map_location='cpu', weights_only=True)
        row = read(path.with_suffix('.json'))
        n = len(data['positions'])
        assert data['methods'] == METHODS
        source = source_rows[(key, 'dense_full')]
        assert data['positions'] == source['positions']
        tail = set(source['position_groups']['tail'])
        indices = [i for i, p in enumerate(data['positions']) if p in tail]
        assert len(indices) == 128
        norms = []
        for method in METHODS:
            src = torch.load(source_rows[(key, method)]['artifact'], map_location='cpu',
                             weights_only=True, mmap=True)
            assert src['positions'] == data['positions']
            norms.append(src['residuals'][layer].float().norm(dim=-1).numpy())
            del src
        for k, fit in data['k'].items():
            sup = fit['support'].reshape(7, n, k).numpy()
            coef = fit['coefficients'].reshape(7, n, k).numpy()
            assert np.isfinite(coef).all() and (coef >= 0).all()
            assert ((sup >= -1) & (sup < 128256)).all()
            for mi, method in enumerate(METHODS):
                overlaps = []
                active_counts = []
                for i in range(n):
                    chosen = sup[mi, i][sup[mi, i] >= 0]
                    assert len(set(chosen.tolist())) == len(chosen)
                    active = {int(t) for t, c in zip(sup[mi, i], coef[mi, i])
                              if t >= 0 and c > 1e-7 * norms[mi][i]}
                    active_counts.append(len(active))
                    if i in indices:
                        dense = {int(t) for t, c in zip(sup[0, i], coef[0, i])
                                 if t >= 0 and c > 1e-7 * norms[0][i]}
                        assert dense
                        overlaps.append(len(dense & active) / len(dense))
                metrics = row['methods'][method][str(k)]
                assert abs(np.mean(overlaps) - metrics['dense_active_support_recall']['tail']) < 1e-6
                energy = fit['energy_reconstructed'].reshape(7, n)[mi, indices].double().mean().item()
                assert abs(energy - metrics['energy_reconstructed']['tail']) < 1e-6
                stored_active = fit['paired_position_metrics'][method]['active_count'].numpy()
                assert np.array_equal(np.array(active_counts), stored_active)
        checks.append(dict(task=task, key=key, layer=layer, status='PASS'))
    result = dict(status='PASS', scope='Tail positions; prompt mean over 32 layers; 50 prompts/task',
                  bootstrap=dict(unit='prompt', paired=True, repetitions=5000,
                                 seeds=[42, 43, 44, 45], intervals='pointwise percentile 95%, unadjusted'),
                  workers=128, sample_layer_pairs=6400,
                  maximum_per_layer_mean_reconciliation_error=maximum_error,
                  tensor_spot_checks=checks, task_means=rows, paired_contrasts=contrasts,
                  inputs={name: sha(HERE / name) for name in
                          ['manifest.json', 'lock.json', 'outputs/jspace-summary.json',
                           'outputs/audit.json', 'outputs/readout-summary.json']},
                  review_script_sha256=sha(Path(__file__)))
    (OUT / 'reviewed-task-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('REVIEW PASS', 'max mean error', maximum_error, flush=True)
    for r in contrasts:
        if r['k'] == 25 and r['metric'] == 'dense_active_support_recall':
            print(r['task'], r['contrast'], *(round(r[x] * 100, 3) for x in
                  ['mean', 'bootstrap_95_low', 'bootstrap_95_high']), flush=True)


if __name__ == '__main__':
    main()
