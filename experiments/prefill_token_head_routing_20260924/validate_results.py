"""Reject partial/duplicate/misaligned InfiniteBench cells before reporting.
Review revision: not yet executed; historical validation applies to frozen code.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from audit import constant, LLAMA


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def input_hash(ids):
    # Same little-endian int64 bytes used by canonical input_ids_sha256.
    return hashlib.sha256(struct.pack('<' + 'q' * len(ids), *ids)).hexdigest()


def validate(root):
    counts = constant(LLAMA, 'TASK_COUNTS')
    report = []
    for task, expected in counts.items():
        reference_inputs = reference_docs = None
        for mode in ('dense','q_only','q_kv'):
            cell = root/mode/task
            results = json.loads((cell/'lm_eval_results.json').read_text())
            samples = lines(cell/'math_find_predictions.jsonl') if task == 'math_find' else results['samples'][task]
            docs = [row['doc_id'] for row in samples]
            if len(docs) != expected or len(set(docs)) != expected:
                raise ValueError(f'{mode}/{task}: missing or duplicate document IDs')
            if set(docs) != set(range(expected)):
                raise ValueError(f'{mode}/{task}: document index coverage is not 0..{expected-1}')
            identity = {}
            for row in samples:
                hashes = tuple(row.get(key) for key in ('doc_hash','prompt_hash','target_hash'))
                if not all(isinstance(value, str) and len(value) == 64 for value in hashes):
                    raise ValueError(f'{mode}/{task}: missing document/prompt/target identity')
                identity[row['doc_id']] = hashes
            metrics = lines(cell/'online_metrics.jsonl')
            if len(metrics) != expected:
                raise ValueError(f'{mode}/{task}: incomplete telemetry')
            inputs = [r['input_ids_sha256'] for r in metrics]
            # Stream full-token sidecars instead of materializing all long prompts.
            eos_count = non_limit_count = detail_count = 0
            with (cell/'routing_generations.jsonl').open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if detail_count >= expected:
                        raise ValueError(f'{mode}/{task}: extra generation row')
                    metric = metrics[detail_count]
                    if row['mode'] != mode or row['input_ids_sha256'] != inputs[detail_count]:
                        raise ValueError(f'{mode}/{task}: telemetry/input identity mismatch')
                    if input_hash(row['input_ids']) != row['input_ids_sha256']:
                        raise ValueError(f'{mode}/{task}: input hash does not match saved token IDs')
                    if len(row['input_ids']) != metric['input_tokens']:
                        raise ValueError(f'{mode}/{task}: input length mismatch')
                    if row['generated_length'] != len(row['generated_token_ids']) or row['generated_length'] != metric['generated_tokens']:
                        raise ValueError(f'{mode}/{task}: generation length mismatch')
                    if row['stop_reason'] not in ('eos','length_limit','stopping_criteria'):
                        raise ValueError(f'{mode}/{task}: invalid stop classification')
                    for key in ('prefill_seconds','decode_forward_seconds','total_seconds'):
                        if not math.isfinite(row[key]) or row[key] < 0:
                            raise ValueError(f'{mode}/{task}: invalid {key}')
                    if row['total_seconds'] + 1e-6 < row['prefill_seconds'] + row['decode_forward_seconds']:
                        raise ValueError(f'{mode}/{task}: inconsistent timing')
                    eos_count += row['stop_reason'] == 'eos'
                    non_limit_count += row['stop_reason'] != 'length_limit'
                    detail_count += 1
            if detail_count != expected:
                raise ValueError(f'{mode}/{task}: incomplete generation sidecar')
            if mode == 'dense':
                reference_inputs, reference_docs = inputs, identity
            elif inputs != reference_inputs or identity != reference_docs:
                raise ValueError(f'{mode}/{task}: cross-condition input/document mismatch')
            report.append({'mode':mode,'task':task,'count':expected,
                'official_score':results['results'][task],
                'eos_rate':eos_count/expected, 'non_length_stop_rate':non_limit_count/expected})
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(validate(a.root),indent=2)+'\n')
