"""Read-only source analysis of HMMT correct-answer lengths and decode timing."""
import csv
import hashlib
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results'
OUT = ROOT / 'hmmt_correct_output_lengths_20261011'
OLD = ROOT / 'hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells'
C500 = ROOT / 'hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells'
RAW = ROOT / 'hmmt_w1_raw_qk_k1024_full_compact500_20261009/cells'
METHODS = {
    'dense': (ROOT / 'hmmt_dense_and_timing_diagnosis_20261008/formal', None),
    'w1_probability_compact64_k1024': (OLD / 'hmmt_k1024_grid_compact_w1_topk_protected/formal', None),
    'w1_probability_compact500_k1024': (C500 / 'hmmt_k1024_grid_compact500_w1_topk_protected/formal', None),
    'w1_probability_compact64_k2048': (OLD / 'hmmt_k2048_grid_compact_w1_topk_protected/formal', None),
    'w1_probability_compact500_k2048': (C500 / 'hmmt_k2048_grid_compact500_w1_topk_protected/formal', None),
    'w1_rawqk_compact500_k1024': (RAW / 'hmmt_k1024_w1_raw_qk_compact_protected/formal', None),
    'w1_rawqk_full_k1024': (RAW / 'hmmt_k1024_w1_raw_qk_full_protected/formal', None),
    'w1_probability_full_k1024_historical': (OLD / 'hmmt_k1024_grid_full_w1_topk_protected/formal', None),
    'w1_probability_full_k2048_optimized': (ROOT / 'hmmt_full_w1_optimized_k2048_protected_20261009/formal', ROOT / 'hmmt_full_w1_optimized_k2048_protected_20261009/grades.jsonl'),
}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    ordered = sorted(values)
    return {'count': len(values), 'mean': st.mean(values), 'median': st.median(values),
            'min': ordered[0], 'max': ordered[-1]}


def main():
    OUT.mkdir(exist_ok=True)
    source_files = set()
    inputs, methods, rows, records = {}, {}, [], {}
    for name, (formal, grade_path) in METHODS.items():
        metrics = {}
        for path in sorted(formal.glob('shard_*/cache_metrics.jsonl')):
            source_files.add(path)
            for metric in jsonl(path):
                idx = metric['idx']
                assert idx not in metrics, (name, 'duplicate metric', idx)
                metrics[idx] = metric
                identity = (metric['input_tokens'], metric['input_ids_sha256'])
                if name == 'dense':
                    inputs[idx] = identity
                else:
                    assert inputs[idx] == identity, (name, 'input mismatch', idx)
        assert len(metrics) == 60
        grade_path = grade_path or formal / 'grades.jsonl'
        source_files.add(grade_path)
        grades = {}
        for grade in jsonl(grade_path):
            key = (grade['idx'], grade['run'])
            assert key not in grades, (name, 'duplicate grade', key)
            metric = metrics[key[0]]
            tokens = metric['generated_tokens_per_rollout'][key[1]]
            eos = metric['eos_reached'][key[1]]
            assert grade.get('generated_tokens', tokens) == tokens
            assert grade.get('eos_reached', eos) == eos
            cap = tokens == 32768 - metric['input_tokens']
            assert cap or eos, (name, key, 'non-EOS below cap')
            grades[key] = {'correct': bool(grade['correct']), 'tokens': tokens, 'cap_hit': cap}
            rows.append({'method': name, 'idx': key[0], 'run': key[1],
                         'correct': bool(grade['correct']), 'generated_tokens': tokens,
                         'eos_reached': eos, 'cap_hit': cap,
                         'input_tokens': metric['input_tokens'], 'input_ids_sha256': metric['input_ids_sha256']})
        assert len(grades) == 120
        assert set(grades) == {(idx, run) for idx in inputs for run in range(2)}
        correct = [g['tokens'] for g in grades.values() if g['correct']]
        seconds = sum(m['decode_seconds'] for m in metrics.values())
        calls = sum(m['decode_forward_calls'] for m in metrics.values())
        methods[name] = {
            'questions': 60, 'answers': 120, 'correct': len(correct),
            'all_output_tokens': distribution([g['tokens'] for g in grades.values()]),
            'correct_output_tokens': distribution(correct),
            'correct_cap_hits': sum(g['correct'] and g['cap_hit'] for g in grades.values()),
            'all_cap_hits': sum(g['cap_hit'] for g in grades.values()),
            'decode_seconds_per_question_batch_of_two': seconds / 60,
            'decode_ms_per_batch_forward': seconds / calls * 1000,
            'decode_forward_calls_per_question': calls / 60,
        }
        summary_path = formal / 'summary.json'
        if not summary_path.exists():
            summary_path = formal.parent / 'summary.json'
        source_files.add(summary_path)
        summary = json.loads(summary_path.read_text())
        assert summary['correct'] == len(correct)
        records[name] = grades
    paired = {}
    dense = records['dense']
    for name, grades in records.items():
        if name == 'dense':
            continue
        keys = sorted(k for k in dense if dense[k]['correct'] and grades[k]['correct'])
        dl = [dense[k]['tokens'] for k in keys]
        sl = [grades[k]['tokens'] for k in keys]
        paired[name] = {
            'both_correct_count': len(keys), 'dense_tokens': distribution(dl),
            'sparse_tokens': distribution(sl), 'mean_difference_tokens': st.mean([s-d for d, s in zip(dl, sl)]),
            'relative_mean_difference_percent': (st.mean(sl) / st.mean(dl)-1)*100,
            'sparse_longer': sum(s>d for d, s in zip(dl, sl)),
            'sparse_shorter': sum(s<d for d, s in zip(dl, sl)),
            'equal': sum(s==d for d, s in zip(dl, sl)),
            'idx_run_keys': keys,
        }
    result = {
        'executed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'HMMT, 60 questions x 2 sampled rollouts, W1 protected prompt + recent64; generation length excludes prompt and includes reasoning and answer.',
        'timing_definition': 'GPU-synchronized wall-clock decode seconds per question generating a batch of two rollouts, not per individual rollout. Per-step latency uses total wall seconds / total batch decode forward calls.',
        'comparison_caveat': 'Probability compact64 vs compact500 compares the same selector. Raw-QK compact500 changes the selector arithmetic too; no raw-QK compact64 or K2048 control exists here. Both-correct subsets describe conditioned outcomes, not causal effects.',
        'alignment': 'passed/full: 60 matching input lengths and input_ids_sha256 for every method; 120 unique idx/run grades per method; all stored grade lengths/EOS matched raw metrics.',
        'methods': methods, 'paired_with_dense_both_correct': paired,
        'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(source_files)],
    }
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    with (OUT / 'per_rollout.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, m in methods.items():
        print(name, json.dumps(m))
        if name in paired:
            print('PAIRED', json.dumps({k:v for k,v in paired[name].items() if k != 'idx_run_keys'}))


if __name__ == '__main__':
    main()
