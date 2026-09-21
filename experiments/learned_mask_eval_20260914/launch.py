"""Full-generation smoke/formal execution with identity-joined prediction audit."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import yaml
from .checkpoints import read
from .gpu_preflight import allocation, OUT
from experiments.learned_mask_nnonly_20260914.pipeline import ROOT, GROUPS, REFERENCE, sha
from experiments.learned_mask_nnonly_20260914.model import VARIANTS
from experiments.shared_flexprefill_predictions_20260912.predictions import export, read_rows

SOURCE = Path(__file__).parent
DATA = ROOT/'experiments/data/kv_randomscore_20260910'
MODEL = '/mmfs1/scratch/yl258/jz648/Llama-3.1-8B-Instruct'
TEMPLATE = ROOT/'experiments/outputs/random_score_residual_20260910/aligned_recovery/data/0_3/task_configs'


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2)


def checked_proof(path):
    proof = read(path)
    assert proof['status'] == 'passed'
    current = {str(p.resolve().relative_to(ROOT.resolve())): sha(p)
               for p in SOURCE.iterdir() if p.suffix in ('.py', '.sbatch')}
    assert current == proof['source_sha256'], 'Online package changed after correctness check'
    for name, digest in proof['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in proof['runtime_dependency_sha256'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in proof['training_audit']['dependency_sha256'].items():
        assert sha(name) == digest, name
    return proof


def prepare_config(destination, count):
    records = read_rows(DATA/'filtered_data/kv_retrieval.jsonl')
    reference = read_rows(REFERENCE)
    assert len(records) == len(reference) == 497
    assert len({(r['input_tokens'], r['input_ids_sha256']) for r in reference}) == 497
    config = destination/'task_configs'
    data = destination/'filtered_data'
    config.mkdir()
    data.mkdir()
    with (data/'kv_retrieval.jsonl').open('x') as stream:
        for row in records[:count]:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    for path in TEMPLATE.glob('*.py'):
        shutil.copy2(path, config/path.name)
    node = yaml.compose((TEMPLATE/'kv_retrieval.yaml').read_text())
    fields = {key.value: value for key, value in node.value}
    values = {key.value: value for key, value in fields['dataset_kwargs'].value}
    values['data_dir'].value = str(data)
    fields['dataset_path'].value = str(DATA/'task_configs/InfiniteBench.py')
    (config/'kv_retrieval.yaml').write_text(yaml.serialize(node))
    return config, data/'kv_retrieval.jsonl'


def official_scorer():
    path = TEMPLATE/'metrics.py'
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == 'get_score_one_kv_retrieval']
    assert len(nodes) == 1
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['get_score_one_kv_retrieval']


def verify(destination, variant, count, data):
    method = f'nnonly_{variant}_compact10240'
    folder = destination/method/'kv_retrieval'
    prediction = export(folder, data, official_scorer(), count)
    summary = read(folder/'summary.json')
    metrics = read_rows(folder/'online_metrics.jsonl')
    expected = {(r['input_tokens'], r['input_ids_sha256']) for r in read_rows(REFERENCE)}
    got = [(r['input_tokens'], r['input_ids_sha256']) for r in metrics]
    assert len(got) == len(set(got)) == count and set(got) <= expected
    alignment = summary['input_alignment']
    assert alignment['status'] == 'passed' and alignment['count'] == count
    assert alignment['reference_count'] == 497
    if count == 497:
        assert set(got) == expected and alignment['alignment_scope'] == 'full'
    assert summary['method'] == method and summary['run_args']['max_new_tokens'] == 128
    assert summary['method_metadata']['selected_checkpoint']['fold'] in (0, 1, 2)
    for r in metrics:
        n = r['input_tokens']
        # All nonlocal selected keys are before a; therefore at fixed final
        # budget and the same protected local tile, exact pair count is fixed.
        pairs = sum((min(10240, min(a+128, n))-(min(a+128, n)-a))*(min(a+128, n)-a)
                    + (min(a+128, n)-a)*(min(a+128, n)-a+1)//2 for a in range(0, n, 128))
        assert r['selected_token_pairs'] == 32*32*pairs
        assert r['causal_token_pairs'] == 32*32*n*(n+1)//2
        assert 0 < r['generated_tokens'] <= 128
        assert 0 < r['prefill_latency_sec'] < r['total_latency_sec']
        if count == 3:
            assert r['smoke_indices_and_finite_logits_passed']
    prefill = [r['prefill_latency_sec'] for r in metrics]
    selected = sum(r['selected_token_pairs'] for r in metrics)
    causal = sum(r['causal_token_pairs'] for r in metrics)
    verified = dict(status='passed', count=count, prediction_audit=prediction, input_alignment=alignment,
        selected_fold=summary['method_metadata']['selected_checkpoint']['fold'],
        prefill_mean=float(np.mean(prefill)), prefill_median=float(np.median(prefill)),
        prefill_p90=float(np.quantile(prefill, .9)),
        decode_mean=float(np.mean([r['decode_latency_sec'] for r in metrics])),
        total_mean=float(np.mean([r['total_latency_sec'] for r in metrics])),
        selected_token_pairs=selected, causal_token_pairs=causal, global_pair_sparsity=1-selected/causal,
        peak_memory_bytes=max(r['peak_memory_bytes'] for r in metrics),
        note='Exact analytical pair budget checked; historical Shared rows still require identity join in final report')
    save(destination/'verified_complete.json', verified)
    return verified


def run(variant, phase, proof_path):
    access = allocation()
    proof = checked_proof(proof_path)
    if phase == 'formal':
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NAME') == 'nnkv-'+variant
        smoke = read(OUT/'smoke_passed.json')
        assert smoke['status'] == 'passed' and smoke['correctness_sha256'] == sha(proof_path)
        for v in VARIANTS:
            assert read(OUT/'smoke'/v/'verified_complete.json')['count'] == 3
            assert read(OUT/'smoke'/v/'exit.json')['returncode'] == 0
    destination = OUT/phase/variant
    destination.mkdir(parents=True, exist_ok=False)
    count = 3 if phase == 'smoke' else 497
    code, error = 1, None
    try:
        config, data = prepare_config(destination, count)
        training_path = destination/'training_audit.json'
        save(training_path, proof['training_audit'])
        shutil.copytree(SOURCE, destination/'source', ignore=shutil.ignore_patterns('__pycache__', '._*'))
        method = f'nnonly_{variant}_compact10240'
        command = [sys.executable, '-u', '-m', 'experiments.learned_mask_eval_20260914.online',
            '--model', MODEL, '--method', method, '--task', 'kv_retrieval', '--limit', '0',
            '--max_length', '131072', '--batch_size', '1', '--seed', '42', '--chat',
            '--group_config', str(GROUPS), '--calibration_scope', 'benchmark_specific',
            '--flexprefill_root', str(ROOT/'FlexPrefill'), '--task_config_dir', str(config),
            '--reference_metrics', str(REFERENCE), '--record_sparsity', '--output_dir', str(destination)]
        env = dict(os.environ, PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
            PYTHONPATH=str(ROOT)+':'+str(ROOT/'FlexPrefill'), LEARNED_MASK_SMOKE=str(int(phase == 'smoke')),
            LEARNED_MASK_TRAINING_AUDIT=str(training_path))
        save(destination/'command.json', dict(command=command, phase=phase, allocation=access,
            correctness_sha256=sha(proof_path), selected=proof['training_audit']['selected'][variant],
            data_sha256=sha(data), reference_sha256=sha(REFERENCE), start_time=time.time()))
        with (destination/'runner.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = verify(destination, variant, count, data)
        print('GENERATION_VERIFIED', variant, phase, result, flush=True)
        code = 0
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        save(destination/'exit.json', dict(returncode=code, error=error, end_time=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'formal'))
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--proof', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'formal':
        assert args.variant
        run(args.variant, args.phase, args.proof)
    else:
        assert args.variant is None
        for variant in VARIANTS:
            existing = OUT/'smoke'/variant
            if existing.exists():
                assert read(existing/'verified_complete.json')['count'] == 3
                assert read(existing/'exit.json')['returncode'] == 0
                assert read(existing/'command.json')['correctness_sha256'] == sha(args.proof)
            else:
                run(variant, 'smoke', args.proof)
        save(OUT/'smoke_passed.json', dict(status='passed', variants=list(VARIANTS), count_per_variant=3,
            correctness_path=str(args.proof), correctness_sha256=sha(args.proof), timestamp=time.time()))


if __name__ == '__main__':
    main()
