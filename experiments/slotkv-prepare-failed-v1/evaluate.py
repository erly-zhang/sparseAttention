"""Isolated 500-generation diagnostic. No retries, tuning, or KV compaction."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
import urllib.request
from protocol import METHODS, TASKS, CELLS, LIMIT, CODE_URL, CODE_REV, CODE_SHA
from protocol import sha, code_prompt, code_indices, scorers, validate
from legacy_checks import select_indices, repetition

HERE = Path(__file__).resolve().parent
ROOT = Path('/ocean/projects/cis260309p/jzhang114/sparseAttention')
PRIOR = ROOT / 'experiments/psc_kl_quality_retry02_20260927'
ROUTING = ROOT / 'experiments/prefill_token_head_routing_20260924/routing.py'
SMOKE = ROOT / 'experiments/paired_head_assembly_20260927/complete.json'
INPUTS = HERE / 'inputs'
PINS = {PRIOR / 'quality.py': '7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61',
        PRIOR / 'env.sh': 'c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9',
        ROUTING: '8abd4dc5bd21a69d6af7c2ef6632c3cd545fe1a0a51b8e2c79dd1e270ee652dd',
        SMOKE: 'c139bafa937ea6d2e88bdf877f39477a6b9dbf7c62e9543e118eedfd95498201',
        PRIOR / 'inputs/manifest.json': 'dc76a7a783ef48c5be21338b3c1c3cb0c8a816470bf7661a232da75057bd35e3'}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(path, value):
    path = Path(path)
    assert not path.exists(), f'Refusing overwrite: {path}'
    with path.with_suffix(path.suffix + '.tmp').open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(handle.name, path)


def preflight():
    for path, expected in PINS.items():
        assert sha(path) == expected, path
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split()
        assert sha(HERE / name) == digest, name
    q = load('ablation_prior', PRIOR / 'quality.py')
    q.verify_sources()
    prior = json.loads((PRIOR / 'inputs/manifest.json').read_text())
    for name, expected in prior['tokenizer_files'].items():
        assert sha(q.MODEL / name) == expected
    metrics = q.FLEX / 'experiments/benchmark/infinitebench/metrics.py'
    assert sha(metrics) == prior['scorer_sha256']
    return q, prior, scorers(metrics)


def prepare():
    from transformers import AutoTokenizer
    q, prior, _ = preflight()
    assert not INPUTS.exists(), 'Frozen inputs already exist'
    data_dir = HERE / 'data'
    data_dir.mkdir(exist_ok=True)
    code = data_dir / 'code_debug.jsonl'
    if not code.exists():
        part = data_dir / 'code_debug.download'
        assert not part.exists(), 'Preserve previous incomplete download for diagnosis'
        with urllib.request.urlopen(CODE_URL, timeout=120) as response, part.open('xb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        assert sha(part) == CODE_SHA
        part.rename(code)
    assert sha(code) == CODE_SHA
    kv = q.BASE / 'data/kv_retrieval.jsonl'
    assert sha(kv) == prior['dataset_sha256']
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    INPUTS.mkdir()
    records = []
    datasets = {}
    for task, data in (('kv_retrieval', kv), ('code_debug', code)):
        raw_lines = data.read_text().splitlines()
        assert len(raw_lines) == (500 if task == 'kv_retrieval' else 394)
        indices = select_indices(prior['indices']) if task == 'kv_retrieval' else code_indices(len(raw_lines))
        datasets[task] = dict(path=str(data), sha256=sha(data), rows=len(raw_lines), indices=indices)
        for index in indices:
            row = json.loads(raw_lines[index])
            if task == 'kv_retrieval':
                ids, before = q.render_ids(tok, row)
            else:
                assert len(row['answer']) == 2 and row['answer'][1] in 'ABCD'
                rendered = tok.apply_chat_template([dict(role='user', content=code_prompt(row))],
                    tokenize=False, add_generation_prompt=True)
                ids = tok.encode(rendered, add_special_tokens=False)
                before = len(ids)
                if before > LIMIT:
                    ids = ids[:LIMIT // 2] + ids[-(LIMIT - LIMIT // 2):]
            raw = struct.pack('<' + 'q' * len(ids), *ids)
            path = INPUTS / f'{task}-{index}.ids'
            with path.open('xb') as handle:
                handle.write(raw)
            if task == 'kv_retrieval' and index in (331, 61, 16):
                assert raw == (PRIOR / f'inputs/{index}.ids').read_bytes()
            records.append(dict(task=task, source_index=index, dataset_id=row.get('id'),
                input_tokens=len(ids), before_truncation=before, input_ids_sha256=sha(path),
                raw_row_sha256=__import__('hashlib').sha256(raw_lines[index].encode()).hexdigest(),
                answers=row['answer'], options=row.get('options', [])))
    assert len(records) == len({r['input_ids_sha256'] for r in records}) == 100
    # Smoke includes fixed KV regressions and the three longest selected code prompts.
    smoke = [['kv_retrieval', i] for i in (331, 61, 16)]
    smoke += [[r['task'], r['source_index']] for r in sorted(
        [r for r in records if r['task'] == 'code_debug'],
        key=lambda r: (-r['input_tokens'], r['source_index']))[:3]]
    save(INPUTS / 'manifest.json', dict(records=records, datasets=datasets, smoke=smoke,
        tokenizer_files=prior['tokenizer_files'], scorer_sha256=prior['scorer_sha256'],
        code_revision=CODE_REV, code_template='InfiniteBench yarn_mistral Code.Debug user content + Llama chat template',
        truncation='head+tail, at 130944 token IDs, same as prior KV diagnostic',
        selection='KV prior fixed 3 + 47; code Random(20260929), exclude indices 0..3',
        generation='greedy128; diagnostic cap, not official task-specific generation caps'))
    print('FROZEN 100 INPUTS', sha(INPUTS / 'manifest.json'), flush=True)


def setup():
    q, _, scores = preflight()
    manifest = json.loads((INPUTS / 'manifest.json').read_text())
    assert len(manifest['records']) == 100
    for item in manifest['records']:
        path = INPUTS / f"{item['task']}-{item['source_index']}.ids"
        assert sha(path) == item['input_ids_sha256'] and path.stat().st_size == item['input_tokens'] * 8
    for dataset in manifest['datasets'].values():
        assert sha(dataset['path']) == dataset['sha256']
    return q, manifest, scores


def selected(manifest, phase):
    smoke = {tuple(x) for x in manifest['smoke']}
    return [r for r in manifest['records'] if ((r['task'], r['source_index']) in smoke) == (phase == 'smoke')]


def stem(method, item):
    return f"{method}-{item['task']}-{item['source_index']}"


def worker(method, phase):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
    from adapter import install
    q, manifest, scores = setup()
    assert transformers.__version__.startswith('4.44.')
    assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = AutoModelForCausalLM.from_pretrained(q.MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    assert (model.config.num_hidden_layers, model.config.num_attention_heads,
            model.config.num_key_value_heads, model.config.hidden_size) == (32, 32, 8, 4096)
    # This only implements the existing FFN in-place; no learned calibration MLP.
    q.install_mlp(model)
    routing = load('ablation_routing', ROUTING)
    state = None
    if method != 'dense':
        kv_mode, assembly = CELLS[method]
        state = install(model, assembly, routing.routed_attention, kv_mode=kv_mode)
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    old = {(r['method'], r['source_index']): r for r in json.loads(SMOKE.read_text())['records']}

    def step(tokens, cache, offset):
        with torch.inference_mode():
            out = model.model(input_ids=tokens,
                position_ids=torch.arange(offset, offset + tokens.shape[1], device='cuda')[None],
                attention_mask=torch.ones((1, offset + tokens.shape[1]), dtype=torch.long, device='cuda'),
                past_key_values=cache, use_cache=True, return_dict=True)
            logits = model.lm_head(out.last_hidden_state[:, -1:]).float()[0, 0]
        assert bool(torch.isfinite(logits).all()), 'Non-finite generation logits'
        return logits

    # Fixed tiny warmup only. No duplicate benchmark generations.
    cache = DynamicCache()
    step(torch.tensor([[1, 2, 3, 4]], device='cuda'), cache, 0)
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    for item in selected(manifest, phase):
        name = stem(method, item)
        assert not (HERE / 'results' / f'{name}.json').exists()
        raw = (INPUTS / f"{item['task']}-{item['source_index']}.ids").read_bytes()
        ids = torch.tensor(np.frombuffer(raw, dtype='<i8').copy(), device='cuda')[None]
        n = ids.shape[1]
        cache = DynamicCache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = step(ids, cache, 0)
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - start
        assert len(cache.key_cache) == len(cache.value_cache) == 32
        assert all(k.shape == v.shape == (1, 8, n, 128) for k, v in zip(cache.key_cache, cache.value_cache))
        last = logits.detach().cpu().double().numpy()
        with (HERE / 'results' / f'{name}.npy').open('xb') as handle:
            np.save(handle, last, allow_pickle=False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        generated = []
        for j in range(128):
            token = int(logits.argmax())
            generated.append(token)
            if token in eos or j == 127:
                break
            logits = step(torch.tensor([[token]], device='cuda'), cache, n + j)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - start
        assert all(k.shape == v.shape == (1, 8, n + len(generated) - 1, 128)
                   for k, v in zip(cache.key_cache, cache.value_cache))
        prediction = tok.decode(generated, skip_special_tokens=True)
        historical = None
        if item['task'] == 'kv_retrieval' and item['source_index'] in (331, 61, 16) and method in ('dense', 'group_pair'):
            reference = old[('dense' if method == 'dense' else 'pair', item['source_index'])]
            assert item['input_ids_sha256'] == reference['input_ids_sha256']
            historical = reference['generated_ids'] == generated
        record = dict(**item, method=method, phase=phase, generated_ids=generated,
            prediction=prediction, generated_tokens=len(generated), eos_ids=eos,
            stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',
            correct=int(scores[item['task']](('===answerspliter==='.join(item['answers']), prediction))),
            prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), repetition=repetition(generated),
            routed_layer_eligible_pairs={} if state is None else dict(state['pairs']),
            historical_token_ids_equal=historical, full_kv_cache=True, dense_decode=True,
            finite_all_steps=True, gpu=torch.cuda.get_device_name(), node=os.uname().nodename,
            job=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__,
            transformers_version=transformers.__version__, logits_sha256=sha(HERE / 'results' / f'{name}.npy'))
        save(HERE / 'results' / f'{name}.json', record)
        print('RESULT', method, item['task'], item['source_index'], record['correct'],
              round(prefill_seconds, 3), len(generated), flush=True)
        assert historical is not False, f'Historical generated-token regression: {name}'
        del cache, logits, ids, last
        gc.collect()
        torch.cuda.empty_cache()


def audit(phase=None):
    import numpy as np
    from adapter import DENSE_LAYERS
    _, manifest, scores = setup()
    inputs = manifest['records'] if phase is None else selected(manifest, phase)
    rows = [json.loads((HERE / 'results' / (stem(method, item) + '.json')).read_text())
            for item in inputs for method in METHODS]
    validate(rows, inputs, DENSE_LAYERS)
    for row in rows:
        assert row['correct'] == scores[row['task']](('===answerspliter==='.join(row['answers']), row['prediction']))
        path = HERE / 'results' / (stem(row['method'], row) + '.npy')
        assert sha(path) == row['logits_sha256']
        a = np.load(HERE / 'results' / (stem('dense', row) + '.npy'), allow_pickle=False)
        b = np.load(path, allow_pickle=False)
        assert a.shape == b.shape == (128256,) and np.isfinite(a).all() and np.isfinite(b).all()
        def logsoftmax(x):
            y = x - x.max()
            return y - np.log(np.exp(y).sum())
        logp, logq = logsoftmax(a), logsoftmax(b)
        kl = float((np.exp(logp) * (logp - logq)).sum())
        assert kl >= -1e-10
        row['final_prompt_vocabulary_kl_dense_to_method'] = max(0., kl)
    summary = {}
    for task in TASKS:
        summary[task] = {}
        for method in METHODS:
            group = [r for r in rows if r['task'] == task and r['method'] == method]
            summary[task][method] = dict(count=len(group), correct=sum(r['correct'] for r in group),
                capped=sum(r['stop_reason'] == 'max_new_tokens' for r in group),
                repetitive=sum(r['repetition']['repetitive_flag'] for r in group),
                mean_prefill_seconds=statistics.mean(r['prefill_seconds'] for r in group),
                mean_decode_seconds=statistics.mean(r['decode_seconds'] for r in group),
                mean_final_prompt_vocabulary_kl=statistics.mean(r['final_prompt_vocabulary_kl_dense_to_method'] for r in group))
    return dict(records=rows, summary=summary, manifest_sha256=sha(INPUTS / 'manifest.json'))


def run():
    _, manifest, _ = setup()
    (HERE / 'results').mkdir()
    save(HERE / 'protocol.json', dict(methods=METHODS, cells=CELLS, max_new_tokens=128, seed=42,
        dense_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31], dtype='BF16',
        model='Llama-3.1-8B-Instruct', scale=1.0, owner='original_position % 32',
        resource=dict(gpu='1 H100 80GB', cpus=8, memory='128GB', walltime='6 hours'),
        manifest_sha256=sha(INPUTS / 'manifest.json'), source_manifest_sha256=sha(HERE / 'SHA256SUMS'),
        backend='q_kv FA2; q_only explicit masked SDPA; dense FA2; all decode dense FA2',
        limitations=['Final-prompt vocabulary KL, not layerwise KL',
            'Pair two occupied slots versus native one: slot identity and duplication jointly change',
            'Unequal attention pairs and kernels: not a compute-matched speed comparison',
            '50 per task, 3 KV regression inputs already observed; no tuning',
            'Same head-tail truncation and greedy128 diagnostic cap; not official full benchmark settings']))
    start = time.monotonic()
    for method in METHODS:
        subprocess.run([sys.executable, __file__, '--method', method, '--phase', 'smoke'], check=True)
    report = audit('smoke')
    elapsed = time.monotonic() - start
    remaining_estimate = 0.
    for task in TASKS:
        count = sum(r['task'] == task for r in selected(manifest, 'remaining'))
        for method in METHODS:
            group = [r for r in report['records'] if r['task'] == task and r['method'] == method]
            remaining_estimate += count * max(r['prefill_seconds'] + r['decode_seconds'] for r in group)
    # Reserve for load/serialization, and never start a stage projected beyond allocation.
    projected = elapsed + remaining_estimate * 1.5 + 1200
    report.update(status='smoke_passed', elapsed_seconds=elapsed, projected_total_seconds=projected)
    save(HERE / 'smoke-complete.json', report)
    print('SMOKE PASSED 30; PROJECTED SECONDS', projected, flush=True)
    assert projected < 5.5 * 3600, 'Budget gate: projected workload exceeds reserved 6h allocation'
    for method in METHODS:
        subprocess.run([sys.executable, __file__, '--method', method, '--phase', 'remaining'], check=True)
    report = audit()
    assert len(list((HERE / 'results').glob('*.json'))) == len(report['records']) == 500
    report['status'] = 'complete'
    save(HERE / 'complete.json', report)
    print('ABLATION COMPLETE 500', json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--phase', choices=('smoke', 'remaining'))
    args = parser.parse_args()
    try:
        if args.prepare:
            prepare()
        elif args.check:
            setup()
            print('FROZEN PREFLIGHT PASS', flush=True)
        elif args.audit:
            print(json.dumps(audit()['summary']))
        elif args.method:
            assert args.phase
            worker(args.method, args.phase)
        else:
            run()
    except BaseException as error:
        if not (HERE / 'failure.json').exists():
            save(HERE / 'failure.json', dict(error=repr(error), job=os.environ.get('SLURM_JOB_ID'), args=vars(args)))
        raise
