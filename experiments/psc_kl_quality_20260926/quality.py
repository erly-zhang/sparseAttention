"""Isolated, fail-closed PSC paired quality diagnostic. No automatic resubmission."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path('/ocean/projects/cis260309p/jzhang114/sparseAttention')
BASE = ROOT / 'experiments/psc_kv_layer_outputs_20260925'
HERE = Path(__file__).resolve().parent
MODEL = ROOT.parent / 'prefill-routing-deps/models/Llama-3.1-8B-Instruct'
FLEX = ROOT.parent / 'prefill-routing-deps/FlexPrefill'
METHODS = ('dense', 'tail_dense', 'interleaved', 'flexprefill', 'minference')
DENSE_LAYERS = dict(tail_dense=list(range(21, 32)),
                    interleaved=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31])
LIMIT = 131072 - 128


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_sources():
    pins = json.loads((HERE / 'source_pins.json').read_text())
    for name, expected in pins.items():
        assert sha(name) == expected, ('Source changed', name)


def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def render_ids(tok, row):
    prompt = ('Extract the value corresponding to the specified key in the JSON object below.\n\n'
              + row['context'] + '\n\n' + row['input'])
    rendered = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                      tokenize=False, add_generation_prompt=True)
    ids = tok.encode(rendered, add_special_tokens=False)
    before = len(ids)
    if before > LIMIT:
        ids = ids[:LIMIT // 2] + ids[-(LIMIT - LIMIT // 2):]
    return ids, before


def scorer():
    path = FLEX / 'experiments/benchmark/infinitebench/metrics.py'
    fn = next(x for x in ast.parse(path.read_text()).body
              if isinstance(x, ast.FunctionDef) and x.name == 'get_score_one_kv_retrieval')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns[fn.name], sha(path)


def previous_indices():
    excluded = {0, 1, 2, 3}
    evidence = {}
    scalar = {'source_index', 'input_source_index', 'input_index', 'test_index'}
    plural = {'calibration_indices', 'test_indices'}
    def visit(obj, found):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in scalar and type(val) is int and 0 <= val < 500:
                    found.add(val)
                elif key in plural and isinstance(val, list):
                    found.update(x for x in val if type(x) is int and 0 <= x < 500)
                elif isinstance(val, (dict, list)):
                    visit(val, found)
        elif isinstance(obj, list):
            for val in obj:
                if isinstance(val, (dict, list)):
                    visit(val, found)
    for top in sorted((ROOT / 'experiments').glob('psc*')):
        if not top.is_dir() or top.resolve() == HERE:
            continue
        for parent, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d not in {'env', 'deps', 'data', '__pycache__',
                       'triton300', 'triton-cache', 'gpu-cache300', 'dense_tensors', 'tensors'}]
            for name in files:
                path = Path(parent) / name
                if path.suffix not in ('.json', '.jsonl') or path.stat().st_size > 32 * 1024**2:
                    continue
                found = set()
                try:
                    if path.suffix == '.jsonl':
                        for line in path.read_text().splitlines():
                            if line.strip():
                                visit(json.loads(line), found)
                    else:
                        visit(json.loads(path.read_text()), found)
                except (ValueError, UnicodeError):
                    raise RuntimeError(f'Unparseable prior provenance: {path}')
                if found:
                    evidence[str(path)] = sorted(found)
                    excluded.update(found)
    return sorted(excluded), evidence


def prepare():
    from transformers import AutoTokenizer
    verify_sources()
    out = HERE / 'inputs'
    assert not out.exists(), 'Refuse to replace a frozen input manifest'
    manifest = json.loads((BASE / 'inputs/manifest.json').read_text())
    data = BASE / 'data/kv_retrieval.jsonl'
    assert sha(data) == manifest['data_sha256']
    for name, expected in manifest['tokenizer_files'].items():
        assert sha(MODEL / name) == expected, name
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    assert len(rows) == 500
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    old, _ = render_ids(tok, rows[3])
    assert hashlib.sha256(struct.pack('<' + 'q' * len(old), *old)).hexdigest() == manifest['records'][3]['input_ids_sha256']
    excluded, evidence = previous_indices()
    pool = [i for i in range(500) if i not in excluded]
    assert len(pool) >= 32, ('Insufficient unseen PSC inputs', len(pool))
    selected = random.Random(42).sample(pool, 32)
    out.mkdir()
    records = []
    for i in [3] + selected:
        ids, before = render_ids(tok, rows[i])
        raw = struct.pack('<' + 'q' * len(ids), *ids)
        (out / f'{i}.ids').write_bytes(raw)
        records.append(dict(source_index=i, input_tokens=len(ids), before_truncation=before,
                            input_ids_sha256=hashlib.sha256(raw).hexdigest(), answers=rows[i]['answer']))
    assert len({r['input_ids_sha256'] for r in records}) == 33
    score, score_sha = scorer()
    assert score(('abc', '"abc"')) == 1 and score(('abc', 'abcd')) == 0
    save(out / 'manifest.json', dict(status='frozen', indices=selected, smoke_index=3,
         excluded=excluded, exclusion_evidence=evidence, records=records,
         dataset_sha256=sha(data), dataset_revision=manifest['dataset_revision'],
         tokenizer_files=manifest['tokenizer_files'], scorer_sha256=score_sha,
         selection='random.Random(42).sample(ascending eligible indices, 32)',
         unseen_scope='PSC provenance with explicit input indices; not a globally unseen benchmark'))
    print('PREPARED', selected, flush=True)


def install_mlp(model):
    import types
    import torch
    for layer in model.model.layers:
        def forward(self, x):
            assert not torch.is_grad_enabled()
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            gate.mul_(up)
            del up
            return self.down_proj(gate)
        layer.mlp.forward = types.MethodType(forward, layer.mlp)


def worker(method, index, smoke=False):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    verify_sources()
    sys.path[:0] = [str(BASE), str(ROOT), str(ROOT / 'experiments/prefill_token_head_routing_20260924/layer_output_comparison')]
    from adapters import install
    from metrics import compare_outputs
    assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    manifest = json.loads((HERE / 'inputs/manifest.json').read_text())
    item = next(r for r in manifest['records'] if r['source_index'] == index)
    raw = (HERE / 'inputs' / f'{index}.ids').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == item['input_ids_sha256']
    ids = torch.tensor(np.frombuffer(raw, dtype='<i8').copy(), device='cuda')[None]
    n = ids.shape[1]
    assert n == item['input_tokens'] and n <= LIMIT
    rootout = HERE / ('smoke' if smoke else 'formal') / str(index)
    out = rootout / method
    out.mkdir(parents=True, exist_ok=False)
    refs = Path(os.environ['QUALITY_SCRATCH'])
    refs.mkdir(exist_ok=True)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    install_mlp(model)
    originals = {name: module.forward for name, module in model.named_modules()}
    model, state = install(model, 'routing_c' if method in DENSE_LAYERS else method)
    if method in DENSE_LAYERS:
        for i in DENSE_LAYERS[method]:
            model.model.layers[i].self_attn.forward = originals[f'model.layers.{i}.self_attn']
        assert sum(layer.self_attn.forward == originals[f'model.layers.{i}.self_attn']
                   for i, layer in enumerate(model.model.layers)) == 11
    sparse = {name: module.forward for name, module in model.named_modules()}
    def restore(forwards):
        for name, module in model.named_modules():
            module.forward = forwards[name]
        assert all(m.forward == forwards[k] for k, m in model.named_modules())
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    def step(tokens, cache, offset):
        with torch.inference_mode():
            result = model.model(input_ids=tokens,
                position_ids=torch.arange(offset, offset + tokens.shape[1], device='cuda')[None],
                attention_mask=torch.ones((1, offset + tokens.shape[1]), dtype=torch.long, device='cuda'),
                past_key_values=cache, use_cache=True, return_dict=True)
            logits = model.lm_head(result.last_hidden_state[:, -1:]).float()[0, 0]
        assert torch.isfinite(logits).all()
        assert len(cache.key_cache) == 32
        assert all(k.shape == v.shape == (1, 8, offset + tokens.shape[1], 128)
                   for k, v in zip(cache.key_cache, cache.value_cache))
        return logits
    def prefill():
        restore(sparse)
        cache = DynamicCache()
        logits = step(ids, cache, 0)
        restore(originals)
        return cache, logits
    def generate(cache, logits, retain_logits=False):
        generated, distributions = [], []
        for j in range(128):
            if retain_logits:
                distributions.append(logits.detach().cpu())
            token = int(logits.argmax())
            generated.append(token)
            if token in eos or j == 127:
                break
            logits = step(torch.tensor([[token]], device='cuda'), cache, n + j)
        return generated, distributions
    # Warmup never contributes to formal pair counts or metrics.
    restore(sparse)
    warm = DynamicCache()
    step(ids[:, :129], warm, 0)
    del warm
    restore(originals)
    torch.cuda.empty_cache()
    unhooked = None
    if smoke:
        cache, logits = prefill()
        unhooked, _ = generate(cache, logits)
        del cache, logits
        torch.cuda.empty_cache()
    restore(sparse)
    instrumentation = None
    if method in ('flexprefill', 'minference'):
        from experiments.baseline_sparsity import install_baseline_sparsity_instrumentation
        instrumentation = install_baseline_sparsity_instrumentation(model, method)
        sparse = {name: module.forward for name, module in model.named_modules()}
        instrumentation.begin_sample(call_index=0, input_ids_sha256=item['input_ids_sha256'],
                                      input_tokens=n, context={'task': 'kv_retrieval'})
    rows, handles, pairs = [], [], {}
    prior = 0
    if method != 'dense':
        assert json.loads((refs / 'identity.json').read_text()) == item
    def count(layer):
        def hook(module, args, output):
            nonlocal prior
            if instrumentation:
                instrumentation.flush_stats()
                now = instrumentation.stats.selected_token_pairs
                pairs[layer] = now - prior
                prior = now
            elif method == 'dense' or layer in DENSE_LAYERS[method]:
                pairs[layer] = 32 * n * (n + 1) // 2
            else:
                pairs[layer] = int(state['pairs'][layer])
            assert 0 < pairs[layer] <= 32 * n * (n + 1) // 2
        return hook
    def capture(layer, point):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            cpu = value.detach().cpu()
            assert cpu.shape == (1, n, 4096)
            path = refs / f'{point}_{layer:02}.pt'
            if method == 'dense':
                torch.save(cpu, path)
                reference = cpu
            else:
                reference = torch.load(path, map_location='cpu', weights_only=True)
            metric = compare_outputs(reference, cpu, torch.ones((1, n), dtype=torch.bool))
            assert math.isfinite(metric['kl_dense_to_method'])
            if method == 'dense':
                assert abs(metric['kl_dense_to_method']) < 1e-7
            rows.append(dict(method=method, source_index=index, layer=layer, point=point,
                             input_ids_sha256=item['input_ids_sha256'], **metric))
        return hook
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(count(i)))
        handles.append(layer.self_attn.o_proj.register_forward_hook(capture(i, 'attention_post_o_proj')))
        handles.append(layer.register_forward_hook(capture(i, 'block_output')))
    torch.cuda.reset_peak_memory_stats()
    cache, logits = prefill()
    for handle in handles:
        handle.remove()
    assert len(rows) == 64 and len(pairs) == 32
    for row in rows:
        row['retained_causal_pairs'] = pairs[row['layer']]
    free, reference_logits = generate(cache, logits, retain_logits=method == 'dense')
    del cache, logits
    if smoke:
        assert free == unhooked, 'Measurement hooks changed generated token IDs'
    if method == 'dense':
        torch.save(torch.stack(reference_logits), refs / 'vocab.pt')
        save(refs / 'tokens.json', free)
        save(refs / 'identity.json', item)
    reference = torch.load(refs / 'vocab.pt', map_location='cpu', weights_only=True)
    continuation = json.loads((refs / 'tokens.json').read_text())
    cache, logits = prefill()
    vocab_kl = []
    for j, token in enumerate(continuation):
        p = reference[j].to('cuda').float().log_softmax(-1)
        q = logits.float().log_softmax(-1)
        kl = float((p.exp() * (p - q)).sum())
        assert math.isfinite(kl) and kl > -1e-5
        if method == 'dense':
            assert abs(kl) < 1e-6, 'Dense vocabulary self-KL failed'
        vocab_kl.append(kl)
        if j + 1 < len(continuation):
            logits = step(torch.tensor([[token]], device='cuda'), cache, n + j)
    del cache, logits
    score, score_sha = scorer()
    assert score_sha == manifest['scorer_sha256']
    prediction = tok.decode(free, skip_special_tokens=True)
    result = dict(status='complete', method=method, **item, generated_ids=free,
        prediction=prediction, raw_prediction=tok.decode(free, skip_special_tokens=False),
        correct=score(('===answerspliter==='.join(item['answers']), prediction)),
        stop_reason='eos' if free[-1] in eos else 'max_new_tokens', eos_ids=eos,
        vocabulary_kl=vocab_kl, next_token_kl=vocab_kl[0], mean_vocabulary_kl=sum(vocab_kl)/len(vocab_kl),
        reference_continuation_sha256=hashlib.sha256(json.dumps(continuation).encode()).hexdigest(),
        selected_causal_pairs=sum(pairs.values()), dense_causal_pairs=32 * 32 * n * (n+1)//2,
        hook_generation_equal=(free == unhooked) if smoke else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        diagnostic_seconds=time.monotonic()-start, not_inference_timing=True,
        gpu=torch.cuda.get_device_name(), node=os.uname().nodename,
        job=os.environ['SLURM_JOB_ID'], metadata=state.get('metadata'), mlp_buffer_reuse_all_methods=True)
    save(out / 'layers.json', rows)
    save(out / 'result.json', result)
    print('COMPLETE', method, index, result['correct'], flush=True)


def driver():
    start = time.monotonic()
    verify_sources()
    assert os.environ.get('SLURM_JOB_ID')
    assert shutil.disk_usage(os.environ['QUALITY_SCRATCH']).free > 70 * 1024**3, 'Insufficient node-local reference space'
    assert not (HERE / 'run_started.json').exists(), 'Refuse duplicate run'
    save(HERE / 'run_started.json', dict(job=os.environ['SLURM_JOB_ID'], started=time.time()))
    model_prov = json.loads((BASE / 'model_provenance.json').read_text())
    for name, info in model_prov['files'].items():
        assert sha(MODEL / name) == info['sha256'], name
    deadline = start + 11.8 * 3600
    def launch(args, log):
        remaining = deadline - time.monotonic()
        assert remaining > 120, 'Approved time budget exhausted'
        with log.open('x') as stream:
            subprocess.run([sys.executable, *args], stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=remaining)
    # Existing causal / GQA / routing-reference gates run in isolated processes.
    checks = HERE / 'checks'
    checks.mkdir()
    os.environ['PSC_RUN_DIR'] = str(checks)
    for method in ('dense', 'routing_c', 'flexprefill', 'minference'):
        launch([str(BASE / 'correctness.py'), '--method', method], checks / f'{method}.log')
    manifest = json.loads((HERE / 'inputs/manifest.json').read_text())
    smoke_start = time.monotonic()
    for method in METHODS:
        launch([str(Path(__file__)), 'worker', '--method', method, '--index', '3', '--smoke'],
               HERE / f'smoke_{method}.log')
    smoke_seconds = time.monotonic() - smoke_start
    projected = smoke_seconds * 32 * 1.25
    gate = dict(smoke_seconds=smoke_seconds, projected_formal_seconds=projected,
                remaining_seconds=deadline-time.monotonic(), conservative_projection=True)
    save(HERE / 'budget_gate.json', gate)
    assert projected + 300 < gate['remaining_seconds'], 'Budget gate failed; no automatic expansion'
    save(HERE / 'smoke_passed.json', dict(status='passed', methods=list(METHODS), input=3))
    for index in manifest['indices']:
        for method in METHODS:
            launch([str(Path(__file__)), 'worker', '--method', method, '--index', str(index)],
                   HERE / f'formal_{index}_{method}.log')
    results, layers = [], []
    for index in manifest['indices']:
        for method in METHODS:
            path = HERE / 'formal' / str(index) / method
            result = json.loads((path / 'result.json').read_text())
            assert result['status'] == 'complete'
            results.append(result)
            layers.extend(json.loads((path / 'layers.json').read_text()))
    assert len(results) == len({(r['source_index'], r['method']) for r in results}) == 160
    assert len(layers) == len({(r['source_index'],r['method'],r['layer'],r['point']) for r in layers}) == 10240
    score, _ = scorer()
    for r in results:
        assert score(('===answerspliter==='.join(r['answers']), r['prediction'])) == r['correct']
    save(HERE / 'results.json', results)
    save(HERE / 'layers.json', layers)
    save(HERE / 'complete.json', dict(status='complete', inputs=32, results=160, layer_records=10240,
        correct={m: sum(r['correct'] for r in results if r['method']==m) for m in METHODS},
        elapsed_seconds=time.monotonic()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'worker'))
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--index', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare()
        elif args.action == 'run':
            driver()
        else:
            worker(args.method, args.index, args.smoke)
    except BaseException:
        if args.action == 'run':
            save(HERE / 'failure.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
