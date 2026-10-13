"""Independently validate observed head partitions and export per-layer counts."""
import collections
import csv
import datetime
import fcntl
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'representative_patterns_fast_retry03_20261001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition(values):
    vs, qa = values['vertical_slash_heads'], values['query_aware_heads']
    assert sorted(vs + qa) == list(range(32)) and not set(vs) & set(qa)
    assert len(vs) == values['vertical_slash_count']
    assert len(qa) == values['query_aware_count']
    assert len(qa) == values['actual_query_aware_branch_calls']
    assert values['tau'] == .1


def summarize(rows):
    layers, heads = {}, {}
    for layer in range(32):
        key = str(layer)
        records = [row['layers'][key] for row in rows]
        for record in records:
            partition(record)
        vs = [x['vertical_slash_count'] for x in records]
        qa = [x['query_aware_count'] for x in records]
        layers[key] = {'samples': len(rows), 'vertical_slash_mean': sum(vs) / len(vs),
            'query_aware_mean': sum(qa) / len(qa),
            'vertical_slash_min': min(vs), 'vertical_slash_max': max(vs),
            'query_aware_min': min(qa), 'query_aware_max': max(qa),
            'vertical_slash_total': sum(vs), 'query_aware_total': sum(qa)}
        heads[key] = {str(head): {
            'query_aware_samples': sum(head in x['query_aware_heads'] for x in records),
            'vertical_slash_samples': sum(head in x['vertical_slash_heads'] for x in records)}
            for head in range(32)}
    return {'layers': layers, 'heads': heads}


def aggregate_if_ready():
    folders = [HERE / 'outputs' / ('worker-' + str(i)) for i in range(10)]
    if not all((p / 'complete.json').exists() for p in folders):
        return False
    with (HERE / 'outputs' / 'aggregate.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        audit_path = HERE / 'outputs' / 'independent-audit.json'
        if audit_path.exists():
            return True
        manifest = json.loads((ORIGINAL / 'inputs/manifest.json').read_text())
        items = {x['key']: x for x in manifest['evaluation']}
        source_hash = sha(HERE / 'source-lock.json')
        rows, hashes = [], {}
        for folder in folders:
            assert not (folder / 'failure.json').exists()
            completion = json.loads((folder / 'complete.json').read_text())
            assert completion['status'] == 'complete' and completion['records'] == 20
            assert completion['source_lock_sha256'] == source_hash
            for filename, digest in completion['files'].items():
                assert sha(folder / filename) == digest
                if not filename.endswith('.json'):
                    continue
                row = json.loads((folder / filename).read_text())
                if 'layers' not in row:
                    continue
                assert row['key'] in items and row['task'] == items[row['key']]['task']
                assert row['input_tokens'] == items[row['key']]['input_tokens']
                assert set(row['layers']) == set(map(str, range(32)))
                assert set(row['repeat_layers']) == set(map(str, range(32)))
                for repeat, observed in enumerate(row['passes']):
                    assert observed['layers'] == (row['layers'] if repeat == 0 else row['repeat_layers'])
                    assert sha(folder / observed['logits_file']) == observed['logits_sha256']
                    for values in observed['layers'].values():
                        partition(values)
                    import numpy as np
                    actual = np.load(folder / observed['logits_file'], allow_pickle=False)
                    original_folder = ORIGINAL / 'outputs' / ('formal-' + str(5 + 7 * completion['shard']))
                    original = json.loads((original_folder / (row['key'] + '-flexprefill.json')).read_text())
                    assert sha(original_folder / (row['key'] + '-flexprefill.json')) == row['original_record_sha256']
                    assert sha(original_folder / original['logits_file']) == row['original_logits_sha256'] == original['logits_sha256']
                    historical = np.load(original_folder / original['logits_file'], allow_pickle=False)
                    assert actual.shape == historical.shape == (128256,) and np.isfinite(actual).all()
                    assert bool(np.array_equal(actual, historical)) == observed['historical_logits_bitwise_equal']
                    assert float(np.abs(actual - historical).max()) == observed['historical_logits_max_abs_error']
                    assert (int(actual.argmax()) == int(historical.argmax())) == observed['historical_first_token_equal']
                changed = {str(layer): sorted(set(row['layers'][str(layer)]['query_aware_heads']) ^
                            set(row['repeat_layers'][str(layer)]['query_aware_heads'])) for layer in range(32)}
                assert row['changed_heads_by_layer'] == changed
                assert row['changed_head_decisions'] == sum(len(v) for v in changed.values())
                rows.append(row)
            hashes[folder.name] = sha(folder / 'complete.json')
        assert len(rows) == len({row['key'] for row in rows}) == 200
        assert {row['key'] for row in rows} == set(items)
        counts = collections.Counter(row['task'] for row in rows)
        assert len(counts) == 4 and all(n == 50 for n in counts.values())
        report = {'status': 'PASS', 'samples': len(rows), 'head_decisions': len(rows) * 32 * 32,
            'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'statistical_scope': 'Same-configuration replay, not recovered historical head decisions',
            'source_lock_sha256': source_hash, 'completion_hashes': hashes,
            'classification': 'sqrt_JS < 0.1: query-aware; otherwise vertical-slash',
            'query_aware_keeps_minimum_vertical_slash_blocks': True,
            'all': summarize(rows),
            'by_task': {task: summarize([r for r in rows if r['task'] == task]) for task in sorted(counts)}}
        repeats = [dict(row, layers=row['repeat_layers']) for row in rows]
        report['repeat'] = summarize(repeats)
        changed = sum(row['changed_head_decisions'] for row in rows)
        report['stability'] = {'changed_decisions': changed,
            'total_compared_decisions': report['head_decisions'],
            'agreement_fraction': 1 - changed / report['head_decisions'],
            'samples_with_any_change': sum(row['changed_head_decisions'] > 0 for row in rows),
            'by_layer_changed': {str(layer): sum(len(row['changed_heads_by_layer'][str(layer)]) for row in rows)
                                 for layer in range(32)}}
        observations = [value for row in rows for value in row['passes']]
        report['historical_logits_comparison'] = {
            'passes': len(observations),
            'bitwise_equal': sum(x['historical_logits_bitwise_equal'] for x in observations),
            'first_token_equal': sum(x['historical_first_token_equal'] for x in observations),
            'maximum_absolute_logit_difference': max(x['historical_logits_max_abs_error'] for x in observations)}
        report['limitations'] = ['Historical runs did not save head classification or intermediate Q/K.',
            'A diagnostic plain-prefill repeat also differed bitwise; the numerical cause remains unlocalized.',
            'No historical benchmark scores are changed; this is a descriptive replay usage statistic.']
        total_qa = sum(x['query_aware_total'] for x in report['all']['layers'].values())
        report['total_query_aware_decisions'] = total_qa
        report['total_vertical_slash_decisions'] = report['head_decisions'] - total_qa
        csv_path = HERE / 'outputs' / 'layer-head-counts.csv'
        with csv_path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['layer'] + list(report['all']['layers']['0']))
            writer.writeheader()
            for layer, stats in report['all']['layers'].items():
                writer.writerow(dict(layer=layer, **stats))
        task_csv = HERE / 'outputs' / 'task-layer-head-counts.csv'
        with task_csv.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['task', 'layer'] + list(report['all']['layers']['0']))
            writer.writeheader()
            for task, summary in report['by_task'].items():
                for layer, stats in summary['layers'].items():
                    writer.writerow(dict(task=task, layer=layer, **stats))
        temporary = audit_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(audit_path)
        print('HEAD_USAGE_AUDIT PASS', len(rows), 'samples', total_qa, 'query-aware decisions', flush=True)
        return True


if __name__ == '__main__':
    print('ready:', aggregate_if_ready())
