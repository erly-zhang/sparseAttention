"""Gated three-shard rerun with immutable old provenance and prediction replay."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml
from experiments.learned_mask_eval_20260914 import launch as old
from experiments.learned_mask_eval_20260914.gpu_preflight import allocation
from .journal import digest, identity, read_record, shard_indices

ROOT = old.ROOT
SOURCE = Path(__file__).parent
OUT = ROOT/'experiments/outputs/learned_mask_recovery_20260916'
PREVIOUS = ROOT/'experiments/outputs/learned_mask_eval_20260914'
VARIANTS = ('per_head', 'per_layer')
read, save, sha = old.read, old.save, old.sha


def source_hashes():
    return {p.name: sha(p) for p in SOURCE.iterdir() if p.suffix in ('.py', '.sbatch')}


def provenance():
    smoke = read(PREVIOUS/'smoke_passed.json')
    assert smoke['status'] == 'passed'
    proof_path = Path(smoke['correctness_path'])
    assert sha(proof_path) == smoke['correctness_sha256']
    proof = old.checked_proof(proof_path)
    for variant in VARIANTS:
        item = proof['training_audit']['selected'][variant]
        assert sha(item['checkpoint']) == item['checkpoint_sha256']
        assert sha(old.GROUPS) == item['groups_sha256']
        assert read(PREVIOUS/'formal'/variant/'command.json')['selected'] == item
    return proof_path, proof


def fingerprint(variant, proof_path, proof):
    return digest(dict(variant=variant, selected=proof['training_audit']['selected'][variant],
        old_correctness_sha256=sha(proof_path), recovery_source_sha256=source_hashes(),
        reference_sha256=sha(old.REFERENCE), data_sha256=sha(old.DATA/'filtered_data/kv_retrieval.jsonl'),
        model=old.MODEL, batch=1, seed=42, greedy=True, dtype='bf16', max_context=131072,
        max_new_tokens=128, chat=True))


def prepare(destination, indices):
    records = old.read_rows(old.DATA/'filtered_data/kv_retrieval.jsonl')
    reference = old.read_rows(old.REFERENCE)
    assert len(records) == len(reference) == len({identity(r) for r in reference}) == 497
    assert len(indices) == len(set(indices)) and all(0 <= i < 497 for i in indices)
    config, data = destination/'task_configs', destination/'filtered_data'
    config.mkdir()
    data.mkdir()
    data_path = data/'kv_retrieval.jsonl'
    with data_path.open('x') as stream:
        for i in indices:
            stream.write(json.dumps(records[i], ensure_ascii=False)+'\n')
    for path in old.TEMPLATE.glob('*.py'):
        shutil.copy2(path, config/path.name)
    node = yaml.compose((old.TEMPLATE/'kv_retrieval.yaml').read_text())
    fields = {key.value: value for key, value in node.value}
    values = {key.value: value for key, value in fields['dataset_kwargs'].value}
    values['data_dir'].value = str(data)
    fields['dataset_path'].value = str(old.DATA/'task_configs/InfiniteBench.py')
    with (config/'kv_retrieval.yaml').open('x') as stream:
        stream.write(yaml.serialize(node))
    save(destination/'input_manifest.json', dict(original_doc_indices=indices,
        original_data_sha256=sha(old.DATA/'filtered_data/kv_retrieval.jsonl'),
        shard_data_sha256=sha(data_path), source_doc_sha256=[digest(records[i]) for i in indices]))
    return config, data_path


def verify_journal(destination, variant, count, fp):
    folder = destination/f'nnonly_{variant}_compact10240'/'kv_retrieval'
    metrics = old.read_rows(folder/'online_metrics.jsonl')
    predictions = old.read_rows(folder/'predictions.jsonl')
    files = list((destination/'generation_journal').glob('*.json'))
    assert len(files) == len(metrics) == len(predictions) == count
    by_id = {identity(r): r for r in metrics}
    pred_ids = {identity(r) for r in predictions}
    assert len(by_id) == len(pred_ids) == count and set(by_id) == pred_ids
    for path in files:
        record = read_record(path, fp)
        metric = by_id.pop(identity(record['metric']))
        assert digest(record['metric']) == digest(metric)
        assert len(record['generated_ids']) == metric['generated_tokens']
    assert not by_id
    save(destination/'journal_verified.json', dict(status='passed', count=count, fingerprint=fp,
        format='atomic fsynced per-input generated token IDs and timed metric record'))


def compare_predictions(left, right, variant):
    def index(path):
        rows = old.read_rows(path/f'nnonly_{variant}_compact10240'/'kv_retrieval/predictions.jsonl')
        return {identity(r): r for r in rows}
    a, b = index(left), index(right)
    assert set(a) == set(b) and len(a) == 3
    for key in a:
        for field in ('target', 'raw_prediction', 'filtered_prediction', 'score', 'generated_tokens'):
            assert a[key][field] == b[key][field], (variant, field, key)


def run(variant, phase, shard=None):
    assert variant in VARIANTS
    access = allocation()
    proof_path, proof = provenance()
    fp = fingerprint(variant, proof_path, proof)
    if phase == 'formal':
        gate = read(OUT/'smoke_passed.json')
        assert gate['status'] == 'passed' and gate['source_sha256'] == source_hashes()
        assert gate['fingerprints'][variant] == fp
        assert os.environ.get('SLURM_JOB_ID')
        assert os.environ.get('SLURM_JOB_NAME') == f'nnrec-{variant}-{shard}'
        indices = shard_indices(shard)
        destination = OUT/'formal'/variant/f'shard{shard}'
    else:
        assert phase in ('smoke', 'replay') and shard is None
        indices = [0, 1, 2]
        destination = OUT/phase/variant
    destination.mkdir(parents=True, exist_ok=False)
    code, error = 1, None
    try:
        config, data = prepare(destination, indices)
        audit_path = destination/'training_audit.json'
        save(audit_path, proof['training_audit'])
        shutil.copytree(SOURCE, destination/'recovery_source', ignore=shutil.ignore_patterns('__pycache__'))
        command = [sys.executable, '-u', '-m', 'experiments.learned_mask_recovery_20260916.online',
            '--model', old.MODEL, '--method', f'nnonly_{variant}_compact10240', '--task', 'kv_retrieval',
            '--limit', '0', '--max_length', '131072', '--batch_size', '1', '--seed', '42', '--chat',
            '--group_config', str(old.GROUPS), '--calibration_scope', 'benchmark_specific',
            '--flexprefill_root', str(ROOT/'FlexPrefill'), '--task_config_dir', str(config),
            '--reference_metrics', str(old.REFERENCE), '--record_sparsity', '--output_dir', str(destination)]
        env = dict(os.environ, PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
            PYTHONPATH=str(ROOT)+':'+str(ROOT/'FlexPrefill'), LEARNED_MASK_SMOKE=str(int(phase != 'formal')),
            LEARNED_MASK_TRAINING_AUDIT=str(audit_path), LEARNED_RECOVERY_FINGERPRINT=fp,
            LEARNED_RECOVERY_JOURNAL=str(destination/'generation_journal'))
        env.pop('LEARNED_RECOVERY_REPLAY', None)
        if phase == 'replay':
            env['LEARNED_RECOVERY_REPLAY'] = str(OUT/'smoke'/variant/'generation_journal')
        save(destination/'command.json', dict(command=command, phase=phase, shard=shard,
            allocation=access, fingerprint=fp, recovery_source_sha256=source_hashes(),
            correctness_path=str(proof_path), correctness_sha256=sha(proof_path),
            selected=proof['training_audit']['selected'][variant], data_sha256=sha(data),
            reference_sha256=sha(old.REFERENCE), start_time=time.time()))
        with (destination/'runner.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = old.verify(destination, variant, len(indices), data)
        verify_journal(destination, variant, len(indices), fp)
        if phase == 'smoke':
            compare_predictions(PREVIOUS/'smoke'/variant, destination, variant)
        if phase == 'replay':
            compare_predictions(OUT/'smoke'/variant, destination, variant)
        print('VERIFIED', variant, phase, shard, result, flush=True)
        code = 0
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        save(destination/'exit.json', dict(returncode=code, error=error, end_time=time.time()))


def submit():
    access = allocation()
    path, proof = provenance()
    gate = read(OUT/'smoke_passed.json')
    assert gate['status'] == 'passed' and gate['source_sha256'] == source_hashes()
    for variant in VARIANTS:
        assert gate['fingerprints'][variant] == fingerprint(variant, path, proof)
        for phase in ('smoke', 'replay'):
            folder = OUT/phase/variant
            assert read(folder/'exit.json')['returncode'] == 0
            assert read(folder/'verified_complete.json')['count'] == 3
            assert read(folder/'journal_verified.json')['status'] == 'passed'
    lock = (OUT/'submission.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue = subprocess.check_output(['squeue', '-u', 'jz648', '-h', '-o', '%j'], text=True)
    assert not any(name.startswith(('nnrec-', 'nnkv-per_head', 'nnkv-per_layer')) for name in queue.splitlines())
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', '1286401,1286403',
                                         '--format=JobID,State'], text=True)
    assert {tuple(x.split('|')[:2]) for x in accounting.splitlines() if x.strip()} == {
        ('1286401', 'TIMEOUT'), ('1286403', 'TIMEOUT')}
    submissions = OUT/'submissions'
    submissions.mkdir(exist_ok=False)
    jobs = [(v, i) for v in VARIANTS for i in range(3)]
    script = SOURCE/'formal.sbatch'
    for variant, shard in jobs:
        assert not (OUT/'formal'/variant/f'shard{shard}').exists()
        command = ['sbatch', '--test-only', f'--job-name=nnrec-{variant}-{shard}', str(script), variant, str(shard)]
        result = subprocess.run(command, capture_output=True, text=True)
        save(submissions/f'{variant}_{shard}_test_only.json', dict(command=command,
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, 'Site rejected test-only; no bypass'
    for variant, shard in jobs:
        command = ['sbatch', '--parsable', f'--job-name=nnrec-{variant}-{shard}', str(script), variant, str(shard)]
        save(submissions/f'{variant}_{shard}_intent.json', dict(command=command, allocation=access, time=time.time()))
        result = subprocess.run(command, capture_output=True, text=True)
        save(submissions/f'{variant}_{shard}.json', dict(command=command, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr, timestamp=time.time()))
        assert result.returncode == 0 and result.stdout.strip().split(';')[0].isdigit()
        print('SUBMITTED', variant, shard, result.stdout.strip(), flush=True)


def controller():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'controller.lock').open('x'):
        pass
    code, error = 1, None
    try:
        allocation()
        path, proof = provenance()
        subprocess.run([sys.executable, '-m', 'unittest', 'experiments.learned_mask_recovery_20260916.test_journal', '-v'], check=True)
        save(OUT/'preflight.json', dict(status='passed', source_sha256=source_hashes(),
            old_correctness_sha256=sha(path), partitions=[shard_indices(i) for i in range(3)],
            count_per_variant=497, calibration_unchanged=True, timestamp=time.time()))
        for variant in VARIANTS:
            for phase in ('smoke', 'replay'):
                print('START', variant, phase, flush=True)
                run(variant, phase)
        save(OUT/'smoke_passed.json', dict(status='passed', source_sha256=source_hashes(),
            fingerprints={v: fingerprint(v, path, proof) for v in VARIANTS},
            count_per_variant=3, historical_smoke_predictions_equal=True,
            durable_token_replay_predictions_equal=True, timestamp=time.time()))
        submit()
        code = 0
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        save(OUT/'controller_exit.json', dict(returncode=code, error=error, timestamp=time.time()))


def merge(variant):
    import numpy as np
    assert variant in VARIANTS
    path, proof = provenance()
    fp = fingerprint(variant, path, proof)
    reference = old.read_rows(old.REFERENCE)
    expected = {identity(r) for r in reference}
    metrics, predictions, manifests, evidence = {}, {}, [], []
    scorer = old.official_scorer()
    for shard in range(3):
        submission = read(OUT/'submissions'/f'{variant}_{shard}.json')
        job = submission['stdout'].strip().split(';')[0]
        accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job,
            '--format=JobID,State,ExitCode'], text=True)
        assert any(row.split('|')[:3] == [job, 'COMPLETED', '0:0'] for row in accounting.splitlines()), accounting
        destination = OUT/'formal'/variant/f'shard{shard}'
        assert read(destination/'exit.json')['returncode'] == 0
        assert read(destination/'verified_complete.json')['status'] == 'passed'
        assert read(destination/'journal_verified.json')['fingerprint'] == fp
        command = read(destination/'command.json')
        assert command['fingerprint'] == fp and command['selected'] == proof['training_audit']['selected'][variant]
        manifest = read(destination/'input_manifest.json')
        assert manifest['original_doc_indices'] == shard_indices(shard)
        assert sha(destination/'filtered_data/kv_retrieval.jsonl') == manifest['shard_data_sha256']
        manifests.extend(manifest['original_doc_indices'])
        folder = destination/f'nnonly_{variant}_compact10240'/'kv_retrieval'
        local_metrics = old.read_rows(folder/'online_metrics.jsonl')
        local_predictions = old.read_rows(folder/'predictions.jsonl')
        assert len(local_metrics) == len(local_predictions) == len(shard_indices(shard))
        for record in local_metrics:
            key = identity(record)
            assert key in expected and key not in metrics
            journal = read_record(destination/'generation_journal'/(key[1]+'.json'), fp)
            assert digest(journal['metric']) == digest(record)
            metrics[key] = record
        for record in local_predictions:
            key = identity(record)
            assert key not in predictions and scorer((record['target'], record['filtered_prediction'])) == record['score']
            predictions[key] = record
        evidence.append(dict(shard=shard, job=job, accounting=accounting,
            metrics_sha256=sha(folder/'online_metrics.jsonl'), predictions_sha256=sha(folder/'predictions.jsonl'),
            verification_sha256=sha(destination/'verified_complete.json')))
    assert sorted(manifests) == list(range(497))
    assert len(metrics) == len(predictions) == 497 and set(metrics) == set(predictions) == expected
    ordered_metrics = [metrics[identity(r)] for r in reference]
    ordered_predictions = [predictions[identity(r)] for r in reference]
    alignment = old.validate_input_alignment(old.REFERENCE, ordered_metrics) if hasattr(old, 'validate_input_alignment') else None
    if alignment is None:
        from experiments.benchmark_shareprefill_ae3 import validate_input_alignment
        alignment = validate_input_alignment(old.REFERENCE, ordered_metrics)
    assert alignment['status'] == 'passed' and alignment['alignment_scope'] == 'full'
    selected = sum(r['selected_token_pairs'] for r in ordered_metrics)
    causal = sum(r['causal_token_pairs'] for r in ordered_metrics)
    prefill = [r['prefill_latency_sec'] for r in ordered_metrics]
    correct = sum(r['score'] for r in ordered_predictions)
    report = dict(status='passed', count=497, reference_count=497, input_alignment=alignment,
        correct=correct, accuracy=correct/497, selected_fold=proof['training_audit']['selected'][variant]['fold'],
        selected_checkpoint=proof['training_audit']['selected'][variant], fingerprint=fp,
        prefill_mean=float(np.mean(prefill)), prefill_median=float(np.median(prefill)), prefill_p90=float(np.quantile(prefill,.9)),
        decode_mean=float(np.mean([r['decode_latency_sec'] for r in ordered_metrics])),
        total_mean=float(np.mean([r['total_latency_sec'] for r in ordered_metrics])),
        selected_token_pairs=selected, causal_token_pairs=causal, global_pair_sparsity=1-selected/causal,
        peak_memory_bytes=max(r['peak_memory_bytes'] for r in ordered_metrics), evidence=evidence,
        prediction_audit='Official scorer reapplied to 497 saved filtered predictions joined by input identity',
        note='Historical Shared10240 actual pair comparison still required; not replaced by analytical budget check')
    destination = OUT/'merged'/variant
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in [('online_metrics', ordered_metrics), ('predictions', ordered_predictions)]:
        with (destination/(name+'.jsonl')).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    save(destination/'verified_complete.json', report)
    print('MERGED_VERIFIED', variant, report, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('controller', 'formal', 'merge'))
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--shard', type=int, choices=range(3))
    args = parser.parse_args()
    if args.phase == 'controller':
        controller()
    elif args.phase == 'merge':
        merge(args.variant)
    else:
        run(args.variant, args.phase, args.shard)


if __name__ == '__main__':
    main()
