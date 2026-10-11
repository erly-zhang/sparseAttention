"""Full HMMT launch companion for the already validated full-KV W1 implementation."""
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(ROOT/'source/scripts'), str(ROOT/'source/ra_harness'),
                str(ROOT/'source/ra_harness/kvcompress/harness')]
from selector_grid import env_for, grade_bounded, read, rows, sha, write


def verify():
    for rel, digest in read(ROOT/'manifest.json').items():
        assert sha(ROOT/rel) == digest, rel
    cfg = read(ROOT/'config.json')
    gate = read(ROOT/'pilot_gate.json')
    assert gate['status'] == 'passed' and gate['source_identity'] == 'passed'
    assert (cfg['K'], cfg['method'], cfg['rank_backend'], cfg['max_tokens']) == (
        2048, 'grid_full_w1_topk_protected', 'stable_sort', 32768)
    assert cfg['questions'] == 60 and cfg['rollouts'] == cfg['batch_size'] == 2
    return cfg


def validate(shard):
    cfg = read(ROOT/'config.json')
    out = ROOT/'formal'/f'shard_{shard}'
    assert read(out/'exit.json')['exit_code'] == 0
    expected = set(range(shard, 60, cfg['shards']))
    refs = {r['idx']: r for r in rows(ROOT/'inputs/hmmt.jsonl')}
    metrics = rows(out/'cache_metrics.jsonl')
    assert len(metrics) == len(expected) and {m['idx'] for m in metrics} == expected
    factor = cfg['model_geometry']['num_attention_heads'] * cfg['model_geometry']['num_hidden_layers']
    for m in metrics:
        ref = refs[m['idx']]
        p = m['input_tokens']
        assert (p, m['input_ids_sha256']) == (ref['input_tokens'], ref['input_ids_sha256'])
        assert m['full_w1_implementation'] == 'optimized'
        assert m['full_w1_rank_backend'] == cfg['rank_backend']
        assert m['grid_method'] == cfg['method'] and m['grid_storage'] == 'full'
        assert m['grid_scorer'] == 'w1' and m['grid_selector'] == 'topk'
        assert m['grid_protection'] == 'protected' and m['query_window'] == 1
        assert m['token_budget'] == cfg['K'] and m['eviction_interval'] == 1
        assert m['cache_semantics'] == 'full_history_temporary_mask' and m['compaction_interval'] is None
        lengths = m['generated_tokens_per_rollout']
        assert len(lengths) == len(m['eos_reached']) == 2
        assert m['decode_forward_calls'] == max(lengths)-1 and m['decode_query_tokens'] == sum(lengths)-2
        selected = 0
        for t, eos in zip(lengths, m['eos_reached']):
            assert 1 <= t <= cfg['max_tokens']-p
            if not eos:
                assert t == cfg['max_tokens']-p
            forwards = t-1
            early = min(forwards, max(0, cfg['K']-p))
            selected += early*p + early*(early+1)//2 + (forwards-early)*cfg['K']
        prefill = 2*factor*p*(p+1)//2
        assert m['prefill_selected_pairs'] == m['prefill_causal_pairs'] == prefill
        assert m['selected_pairs'] == prefill + factor*selected
        assert m['causal_pairs'] == factor*sum((p+t-1)*(p+t)//2 for t in lengths)
        assert sum(m['per_layer_selected_pairs']) == m['selected_pairs']
        assert sum(m['per_layer_causal_pairs']) == m['causal_pairs']
        assert m['selection_events'] == cfg['model_geometry']['num_hidden_layers']*(max(lengths)-1)
        assert all(all(n == p+max(lengths)-1 for n in layer) for layer in m['final_cache_seqlens'])
        assert all(n == p+max(lengths)-1 for n in m['final_cumulative_length'])
    token_rows = read(out/'token_ids.json')
    assert len(token_rows) == len(expected) and {t['idx'] for t in token_rows} == expected
    assert all(len(t['token_ids']) == 2 for t in token_rows)
    for t in token_rows:
        if t['idx'] in (0, 1, 2):
            pilot_tokens = read(ROOT/'reference_pilot'/f"input_{t['idx']}_optimized_token_ids.json")[0]
            assert t['token_ids'] == pilot_tokens['token_ids'], ('pilot token divergence', t['idx'])
    completions = {}
    for run in (0, 1):
        files = list((out/f'run_{run}').glob('completions*.jsonl'))
        assert len(files) == 1
        rr = rows(files[0])
        assert len(rr) == len(expected) and {r['idx'] for r in rr} == expected
        for r in rr:
            prefix = refs[r['idx']]['prompt_text']
            assert r['completion'].startswith(prefix)
            completions[run, r['idx']] = r['completion'][len(prefix):]
    return metrics, completions


def audit():
    cfg = verify()
    refs = rows(ROOT/'inputs/hmmt.jsonl')
    assert len(refs) == 60 and {r['idx'] for r in refs} == set(range(60))
    assert max(r['input_tokens'] for r in refs)+64 <= cfg['K']
    shard_ids = [i for shard in range(cfg['shards']) for i in range(shard, 60, cfg['shards'])]
    assert len(shard_ids) == len(set(shard_ids)) == 60
    write(ROOT/'static_validation.json', dict(status='passed', questions=60, answers=120,
        source_identity='passed/frozen_pilot', shard_coverage='passed/exact',
        prior_small_tensor_and_full_generation_correctness='passed/reused'))


def infer(shard):
    cfg = verify()
    assert 0 <= shard < cfg['shards']
    for rel, digest in read(ROOT/'model_assets_sha256.json').items():
        assert sha(Path(cfg['model'])/rel) == digest, rel
    import torch
    assert torch.cuda.device_count() == 1 and os.environ.get('SLURM_JOB_ID')
    gpu = torch.cuda.get_device_properties(0)
    assert 'A100' in gpu.name and gpu.total_memory > 75*1024**3
    packages = {k: importlib.metadata.version(k) for k in cfg['packages']}
    assert packages == cfg['packages']
    out = ROOT/'formal'/f'shard_{shard}'
    out.mkdir(parents=True)
    write(out/'hardware.json', dict(gpu=gpu.name, memory=gpu.total_memory,
        node=platform.node(), job=os.environ['SLURM_JOB_ID'], packages=packages))
    env = env_for(ROOT, 'hmmt')
    env.update(RA_FULL_W1_TRIAL_IMPL='optimized', RA_FULL_W1_RANK_BACKEND=cfg['rank_backend'],
        RA_FULL_W1_TOKEN_OUTPUT=str(out/'token_ids.json'))
    args = [sys.executable, '-u', str(ROOT/'source/scripts/run_full_w1_trial.py'),
        '--model_name_or_path', cfg['model'], '--data_name', 'hmmt', '--split', 'test',
        '--prompt_type', 'qwen-instruct', '--surround_with_messages',
        '--attention_implementation', 'flash_attention_2', '--sparsity_method', 'eviction',
        '--eviction_mode', cfg['method'], '--token_budget', str(cfg['K']),
        '--residual_length', '0', '--top_p', '.95', '--query_window', '1',
        '--mask_seed', '42', '--rep_seed', '0', '--seed', '42', '--eviction_interval', '1',
        '--batch_size', '2', '--run_id', '0', '--max_tokens', str(cfg['max_tokens']),
        '--warmup_new_tokens', '2', '--use_batch_exist', '--output_dir', str(out),
        '--num_shards', str(cfg['shards']), '--shard_id', str(shard)]
    write(out/'command.json', args)
    write(out/'status.json', dict(phase='generating', shard=shard))
    with (out/'runner.log').open('x') as log:
        proc = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    write(out/'exit.json', dict(exit_code=proc.returncode))
    assert proc.returncode == 0
    validate(shard)
    write(out/'status.json', dict(phase='completed', input_alignment='passed/full_within_shard',
        pair_accounting='passed/fixed_budget_analytic', pilot_token_ids='passed/where_present'))


def collect():
    cfg = verify()
    refs = {r['idx']: r for r in rows(ROOT/'inputs/hmmt.jsonl')}
    metrics, texts, sources = [], {}, {}
    for shard in range(cfg['shards']):
        mm, cc = validate(shard)
        assert not set(texts)&set(cc)
        metrics += mm
        texts.update(cc)
        out = ROOT/'formal'/f'shard_{shard}'
        for p in [out/'cache_metrics.jsonl', out/'token_ids.json', out/'hardware.json',
                  *out.glob('run_*/completions*.jsonl')]:
            sources[str(p.relative_to(ROOT))] = sha(p)
    assert len(metrics) == 60 and {m['idx'] for m in metrics} == set(range(60))
    assert set(texts) == {(run, i) for run in (0, 1) for i in range(60)}
    # Reuse the established grader, with imports cached before fork.
    from Utils.parser import extract_answer
    from Utils.grader import check_is_correct
    grades = []
    for (run, idx), text in sorted(texts.items()):
        prediction, correct = grade_bounded(text, refs[idx]['gold'])
        grades.append(dict(run=run, idx=idx, prediction=prediction, correct=int(correct)))
    seconds = sum(m['decode_seconds'] for m in metrics)
    selected = sum(m['selected_pairs'] for m in metrics)
    causal = sum(m['causal_pairs'] for m in metrics)
    record = dict(status='completed', method='full_w1_fixed_ranges_batched_exact_v1',
        K=cfg['K'], protection='prompt+recent64', storage='full', questions=60, answers=120,
        correct=sum(g['correct'] for g in grades), accuracy=sum(g['correct'] for g in grades)/120,
        mean_decode_seconds=seconds/60, median_decode_seconds=statistics.median(m['decode_seconds'] for m in metrics),
        aggregate_ms_per_decode_forward=1000*seconds/sum(m['decode_forward_calls'] for m in metrics),
        mean_generated_tokens=statistics.mean(t for m in metrics for t in m['generated_tokens_per_rollout']),
        median_generated_tokens=statistics.median(t for m in metrics for t in m['generated_tokens_per_rollout']),
        cap_hits=sum(not e for m in metrics for e in m['eos_reached']),
        maximum_peak_allocated_gib=max(m['peak_memory_allocated'] for m in metrics)/2**30,
        mean_prefill_seconds=statistics.mean(m['prefill_seconds'] for m in metrics),
        mean_total_seconds=statistics.mean(m['batch_wall_seconds'] for m in metrics),
        selected_pairs=selected, causal_pairs=causal, global_pair_sparsity=1-selected/causal,
        input_alignment='passed/full', pair_accounting='passed/fixed_budget_analytic',
        pilot_generated_token_ids='passed/exact', sources_sha256=sources,
        timing_scope='GPU-synchronized decode wall; loading/warmup and token-ID serialization excluded')
    write(ROOT/'summary.json', record)
    (ROOT/'grades.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in grades))
    comparisons = []
    for label, path in cfg['comparison_sources'].items():
        path = Path(path)
        source_root = path.parent.parent if label == 'Dense' else Path(cfg['historical_grid_root'])
        if label == 'Dense' or label == 'historical_full_W1_K2048':
            rr = rows(source_root/'inputs/hmmt.jsonl')
            assert {(x['idx'],x['input_tokens'],x['input_ids_sha256']) for x in rr} == {
                (x['idx'],x['input_tokens'],x['input_ids_sha256']) for x in refs.values()}
        s = read(path)
        comparisons.append(dict(method=label, source_summary=str(path), source_sha256=sha(path),
            correct=s['correct'], accuracy=s['accuracy'], mean_decode_seconds=s['decode_seconds']['mean'],
            mean_generated_tokens=s['generated_tokens']['mean']))
    baseline_dir = Path(cfg['comparison_sources']['historical_full_W1_K2048']).parent
    mismatches = []
    for run in (0,1):
        old = [r for p in baseline_dir.glob(f'shard_*/run_{run}/completions*.jsonl') for r in rows(p)]
        assert len(old) == 60 and {r['idx'] for r in old} == set(range(60))
        for r in old:
            prefix = refs[r['idx']]['prompt_text']
            assert r['completion'].startswith(prefix)
            if r['completion'][len(prefix):] != texts[run,r['idx']]:
                mismatches.append(dict(run=run, idx=r['idx']))
    result = dict(status='completed', optimized=record, baselines=comparisons,
        historical_full_W1_output_text_alignment=dict(matched=120-len(mismatches), total=120, mismatches=mismatches),
        caveat='Full matched inputs/settings; methods ran on different A10080GB allocations. Speed comparisons are not same-device paired timings.')
    pp = Path(cfg['official_random_pp_root'])/'summary.json'
    if pp.exists():
        assert read(pp)['input_alignment'] == 'passed/full'
        result['official_random_pp'] = read(pp)['cells']
    else:
        result['official_random_pp'] = dict(status='not_yet_collected', source=str(pp))
    write(ROOT/'comparison.json', result)
    (ROOT/'analysis/results_en.txt').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'audit':
        audit()
    elif sys.argv[1] == 'infer':
        infer(int(sys.argv[3]))
    elif sys.argv[1] == 'collect':
        collect()
    else:
        raise ValueError(sys.argv[1])
