"""Sequential pre-o_proj calibration, with heldout generation only after training."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

from protocol import (LAYERS, METHODS, EPOCHS, DEADLINE, specs, positions,
                      validate_inputs, validate_results, exact_match, pick_checkpoint,
                      projection, sha, save)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENTS = ROOT / 'experiments'
PRIOR = EXPERIMENTS / 'psc_kl_quality_retry02_20260927'
OLD = EXPERIMENTS / 'qonly_mlp_calibration_20260928'
ABLATION = EXPERIMENTS / 'head_slot_kv_ablation_20260928'
REPEAT = EXPERIMENTS / 'qonly_repeat16_20260929'
PINS = {
    PRIOR / 'quality.py': '7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61',
    PRIOR / 'env.sh': 'c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9',
    PRIOR / 'inputs/manifest.json': 'dc76a7a783ef48c5be21338b3c1c3cb0c8a816470bf7661a232da75057bd35e3',
    OLD / 'protocol.py': '26ac2c894bf63736f309551fd2a9e4832a81e44d938fafc49669c7da21f8be7f',
    OLD / 'inputs/manifest.json': 'cd00fff90f1fe3ba9a5be44c2ad92e7369ca8bd2567f065de7833bb023c94bf1',
    REPEAT / 'config.py': 'bba78593eb26da757f77920d49fd684124f948bf90dda505dce559e39eb0d868',
    ABLATION / 'inputs/manifest.json': 'abdf9a76ce3565cb1ce65c1231b90ec9a1857c0816a2a7146057ddcf69151dbc',
    ABLATION / 'legacy_checks.py': '50011b36274403486429a27bc8f90d48e910e804d034c8a92fae0f55a716b297',
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight():
    for path, digest in PINS.items():
        assert sha(path) == digest, path
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split()
        assert sha(HERE / name) == digest, name
    layout = load('frozen_repeat_layout', REPEAT / 'config.py')
    layout.dependencies(ROOT)
    q = load('preoproj_quality', PRIOR / 'quality.py')
    q.verify_sources()
    previous = json.loads((PRIOR / 'inputs/manifest.json').read_text())
    for name, digest in previous['tokenizer_files'].items():
        assert sha(q.MODEL / name) == digest
    return q, layout, previous


def exclusions():
    old = json.loads((OLD / 'inputs/manifest.json').read_text())['records']
    benchmark = json.loads((ABLATION / 'inputs/manifest.json').read_text())['records']
    return dict(excluded_hashes=sorted({r['input_ids_sha256'] for r in old + benchmark}),
                excluded_keys=sorted(k for r in old for k, _ in r['pairs']),
                excluded_values=sorted(v for r in old for _, v in r['pairs']))


def prepare():
    from transformers import AutoTokenizer
    q, _, previous = preflight()
    generator = load('frozen_synthetic_generator', OLD / 'protocol.py')
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    def encode(prompt):
        return tok.encode(tok.apply_chat_template([dict(role='user', content=prompt)],
            tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    (HERE / 'inputs').mkdir()
    rows = []
    for spec in specs():
        row = generator.make_prompt(spec, encode)
        ids = row.pop('ids')
        path = HERE / 'inputs' / (row['id'] + '.ids')
        with path.open('xb') as handle:
            handle.write(struct.pack('<' + 'q' * len(ids), *ids))
        row.update(input_tokens=len(ids), input_ids_sha256=sha(path), feature_positions=positions(len(ids)))
        rows.append(row)
    rows = json.loads(json.dumps(rows))
    excluded = exclusions()
    validate_inputs(rows, **excluded)
    save(HERE / 'inputs/manifest.json', dict(records=rows, **excluded,
        source_manifest_sha256=sha(HERE / 'SHA256SUMS'), tokenizer_files=previous['tokenizer_files'],
        model=str(q.MODEL), algorithm='Q-only repeat16 pre-o_proj residual SiLU MLP',
        layers=list(LAYERS), rank=128, prefill_only=True, dense_decode=True,
        training='sequential layerwise, previous selected adapters active, no heldout features',
        target='dense pre-o_proj O at same original token positions; no correctness filtering',
        loss='mean(relative O squared error + relative original-o_proj output squared error)',
        selection='per-layer validation-only; identity epoch0 plus5 trained epochs',
        limits='new synthetic4K/16K pilot;8 heldout; not benchmark accuracy',
        hyperparameters=dict(seed=42, epochs=EPOCHS, batch=64, lr=0.001, max_new_tokens=128)))
    print('FROZEN FRESH48', sha(HERE / 'inputs/manifest.json'), flush=True)


def check():
    q, layout, _ = preflight()
    manifest = json.loads((HERE / 'inputs/manifest.json').read_text())
    assert manifest['source_manifest_sha256'] == sha(HERE / 'SHA256SUMS')
    validate_inputs(manifest['records'], **{k: manifest[k] for k in exclusions()})
    for row in manifest['records']:
        path = HERE / 'inputs' / (row['id'] + '.ids')
        assert sha(path) == row['input_ids_sha256'] and path.stat().st_size == row['input_tokens'] * 8
    return q, layout, manifest


class Engine:
    def __init__(self, model, tokenizer, layout, hooks):
        self.model, self.tokenizer, self.layout, self.hooks = model, tokenizer, layout, hooks
        eos = model.generation_config.eos_token_id
        self.eos = [eos] if isinstance(eos, int) else list(eos)

    def ids(self, item):
        import numpy as np
        return np.fromfile(HERE / 'inputs' / (item['id'] + '.ids'), dtype='<i8').tolist()

    def forward(self, tokens, cache, offset):
        import torch
        self.hooks.prefill = offset == 0
        with torch.no_grad():
            output = self.model.model(input_ids=tokens,
                position_ids=torch.arange(offset, offset + tokens.shape[1], device='cuda')[None],
                attention_mask=torch.ones(1, offset + tokens.shape[1], dtype=torch.long, device='cuda'),
                past_key_values=cache, use_cache=True, return_dict=True)
            logits = self.model.lm_head(output.last_hidden_state[:, -1]).float()[0]
        assert bool(torch.isfinite(logits).all())
        self.cache_check(cache, offset + tokens.shape[1], 32)
        return logits

    @staticmethod
    def cache_check(cache, n, layers):
        assert len(cache.key_cache) == len(cache.value_cache) == layers
        assert all(k.shape == v.shape == (1, 8, n, 128)
                   for k, v in zip(cache.key_cache, cache.value_cache))

    def collect(self, item, layer=None):
        import torch
        from transformers import DynamicCache
        from calibrator import CaptureComplete
        cache = DynamicCache()
        hooks = self.hooks
        hooks.capture = set(LAYERS) if layer is None else {layer}
        hooks.indices, hooks.stop, hooks.features = item['feature_positions'], layer, {}
        hooks.enabled, hooks.calls = layer is not None, {}
        state = None if layer is None else self.layout.install(self.model, 'repeat16')
        start = time.monotonic()
        try:
            try:
                self.forward(torch.tensor([self.ids(item)], device='cuda'), cache, 0)
                assert layer is None, 'prefix capture did not stop'
            except CaptureComplete:
                assert layer is not None
                self.cache_check(cache, item['input_tokens'], layer + 1)
            expected = set(LAYERS) if layer is None else {layer}
            assert set(hooks.features) == expected
            assert all(v.shape == (256, 4096) and bool(torch.isfinite(v).all()) for v in hooks.features.values())
            if state:
                assert state['pairs'] == {i: item['input_tokens'] * (item['input_tokens'] + 1) // 2
                                          for i in LAYERS if i <= layer}
                assert set(hooks.calls) == {i for i in hooks.adapters if i < layer}
            return dict(hooks.features), time.monotonic() - start
        finally:
            hooks.capture, hooks.stop, hooks.prefill = set(), None, False
            if state:
                state['close']()
            del cache
            gc.collect()
            torch.cuda.empty_cache()

    def generate(self, item, method):
        import torch
        from transformers import DynamicCache
        hooks = self.hooks
        hooks.capture, hooks.stop, hooks.calls = set(), None, {}
        hooks.enabled = method == 'mlp16'
        state = None if method == 'dense' else self.layout.install(self.model, 'repeat16')
        cache = DynamicCache()
        start = time.monotonic()
        try:
            tokens = torch.tensor([self.ids(item)], device='cuda')
            logits = self.forward(tokens, cache, 0)
            torch.cuda.synchronize()
            prefill = time.monotonic() - start
            first = logits.detach().cpu().numpy().copy()
            calls = dict(hooks.calls)
            generated = []
            for j in range(128):
                token = int(logits.argmax())
                generated.append(token)
                if token in self.eos or j == 127:
                    break
                logits = self.forward(torch.tensor([[token]], device='cuda'), cache, len(tokens[0]) + j)
                assert hooks.calls == calls, 'MLP activated during decode'
            assert calls == ({i: 1 for i in hooks.adapters} if method == 'mlp16' else {})
            if state:
                self.layout.validate_counts('repeat16', state['pairs'], len(tokens[0]))
            return dict(id=item['id'], method=method, input_ids_sha256=item['input_ids_sha256'],
                generated_ids=generated, prediction=self.tokenizer.decode(generated, skip_special_tokens=True),
                stop_reason='eos' if generated[-1] in self.eos else 'max_new_tokens', eos_ids=self.eos,
                prefill_seconds=prefill, total_seconds=time.monotonic() - start,
                full_kv_cache=True, finite_all_steps=True, dense_decode=True,
                adapter_prefill_calls=calls, adapter_decode_calls=0,
                pairs={} if state is None else dict(state['pairs'])), first
        finally:
            hooks.prefill, hooks.enabled = False, False
            if state:
                state['close']()
            del cache
            gc.collect()
            torch.cuda.empty_cache()


def run():
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from calibrator import PreProjectionMLP, ProjectionHooks, loss
    q, layout, manifest = check()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    assert 'H100' in torch.cuda.get_device_name() and transformers.__version__.startswith('4.44.')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    start = time.monotonic()
    for directory in ('teacher', 'features', 'checkpoints', 'results', 'smoke'):
        (HERE / directory).mkdir()
    save(HERE / 'run-start.json', dict(job=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
        gpu=torch.cuda.get_device_name(), manifest_sha256=sha(HERE / 'inputs/manifest.json')))
    def deadline():
        assert time.monotonic() - start < DEADLINE, 'Pilot deadline; no automatic retry'
    def save_tensor(path, value):
        with path.open('xb') as handle:
            torch.save(value, handle)
        return sha(path)
    model = AutoModelForCausalLM.from_pretrained(q.MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    q.install_mlp(model)  # Frozen FFN memory implementation, not the new correctors.
    versions = {n: p._version for n, p in model.named_parameters()}
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    hooks = ProjectionHooks(model, LAYERS)
    engine = Engine(model, tok, layout, hooks)
    diagnostics = load('preoproj_diagnostics', ABLATION / 'legacy_checks.py')
    records = manifest['records']
    calibration = [r for r in records if r['split'] != 'heldout']
    smoke_items = [r for r in calibration if r['id'] in ('train-00', 'train-01')]
    dense_times, sparse_times, generation_times, teacher_meta = [], [], [], {}
    hooks.adapters = {i: PreProjectionMLP().cuda().eval() for i in LAYERS}
    smoke_features = []
    for item in smoke_items:
        deadline()
        features, duration = engine.collect(item)
        dense_times.append(duration)
        digest = save_tensor(HERE / 'teacher' / (item['id'] + '.pt'), features)
        teacher_meta[item['id']] = dict(sha256=digest, input_ids_sha256=item['input_ids_sha256'],
                                      positions=item['feature_positions'], layers=list(LAYERS))
        sparse, duration = engine.collect(item, 30)
        sparse_times.append(duration)
        smoke_features.append((sparse[30], features[30]))
        outputs = {}
        for method in METHODS:
            row, logits = engine.generate(item, method)
            row['correct'] = exact_match(row['prediction'], item['answer'])
            outputs[method] = row
            generation_times.append(row['total_seconds'])
        assert outputs['repeat16']['generated_ids'] == outputs['mlp16']['generated_ids']
        save(HERE / 'smoke' / (item['id'] + '.json'), outputs)
    # Measure a real five-epoch fit, then discard this smoke-only corrector.
    probe = PreProjectionMLP().cuda()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0)
    x = torch.cat([p[0] for p in smoke_features])
    y = torch.cat([p[1] for p in smoke_features])
    tick = time.monotonic()
    for epoch in range(EPOCHS):
        for offset in range(0, len(x), 64):
            optimizer.zero_grad(set_to_none=True)
            value = loss(probe, model.model.layers[30].self_attn.o_proj,
                         x[offset:offset+64].cuda(), y[offset:offset+64].cuda())
            value.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
    torch.cuda.synchronize()
    fit_estimate = (time.monotonic() - tick) * 20 * 1.25
    projected = projection(time.monotonic() - start, max(dense_times), max(sparse_times),
                           max(generation_times), fit_estimate)
    save(HERE / 'budget-projection.json', dict(projected_seconds=projected, limit=DEADLINE,
         dense_seconds=max(dense_times), sparse_seconds=max(sparse_times), fit_seconds_per_layer=fit_estimate))
    assert projected < DEADLINE, f'Budget gate rejected: {projected}'
    save(HERE / 'smoke-complete.json', dict(status='passed', projected_seconds=projected,
         identity_generated_ids_equal=True, identity_prefill_calls=16, identity_decode_calls=0))
    print('PREOPROJ SMOKE PASS', projected, flush=True)
    hooks.adapters = {}
    del probe, optimizer, x, y, smoke_features
    for item in calibration:
        deadline()
        if item['id'] in teacher_meta:
            continue
        features, _ = engine.collect(item)
        digest = save_tensor(HERE / 'teacher' / (item['id'] + '.pt'), features)
        teacher_meta[item['id']] = dict(sha256=digest, input_ids_sha256=item['input_ids_sha256'],
                                      positions=item['feature_positions'], layers=list(LAYERS))
        print('TEACHER', item['id'], flush=True)
    save(HERE / 'teacher/manifest.json', teacher_meta)
    selections = {}
    for layer in LAYERS:
        deadline()
        groups = {'train': [], 'validation': []}
        feature_meta = []
        for item in calibration:
            deadline()
            sparse, _ = engine.collect(item, layer)
            target_path = HERE / 'teacher' / (item['id'] + '.pt')
            assert sha(target_path) == teacher_meta[item['id']]['sha256']
            dense = torch.load(target_path, weights_only=True)[layer]
            groups[item['split']].append((sparse[layer], dense))
            feature_meta.append(dict(id=item['id'], split=item['split'],
                input_ids_sha256=item['input_ids_sha256'], positions=item['feature_positions']))
        batches = {s: tuple(torch.cat([p[j] for p in pairs]) for j in (0, 1)) for s, pairs in groups.items()}
        path = HERE / 'features' / f'layer{layer:02d}.pt'
        digest = save_tensor(path, batches)
        save(HERE / 'features' / f'layer{layer:02d}.json', dict(sha256=digest, rows=feature_meta,
            previous_selected={str(i): s['sha256'] for i, s in selections.items()}))
        torch.manual_seed(42 + layer)
        adapter = PreProjectionMLP().cuda()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=0)
        projection_module = model.model.layers[layer].self_attn.o_proj
        history = []
        for epoch in range(EPOCHS + 1):
            if epoch:
                adapter.train()
                x, y = batches['train']
                order = torch.randperm(len(x), generator=torch.Generator().manual_seed(42 + layer * 10 + epoch))
                for offset in range(0, len(x), 64):
                    deadline()
                    ix = order[offset:offset+64]
                    optimizer.zero_grad(set_to_none=True)
                    value = loss(adapter, projection_module, x[ix].cuda(), y[ix].cuda())
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1, error_if_nonfinite=True)
                    optimizer.step()
            adapter.eval()
            x, y = batches['validation']
            total = 0.0
            with torch.no_grad():
                for offset in range(0, len(x), 64):
                    deadline()
                    value = loss(adapter, projection_module, x[offset:offset+64].cuda(), y[offset:offset+64].cuda())
                    total += value.item() * len(x[offset:offset+64])
            path = HERE / 'checkpoints' / f'layer{layer:02d}-epoch{epoch}.pt'
            digest = save_tensor(path, adapter.state_dict())
            history.append(dict(epoch=epoch, validation_loss=total / len(x), path=path.name, sha256=digest))
            print('TRAIN', layer, epoch, total / len(x), flush=True)
        best = pick_checkpoint(history)
        selected_path = HERE / 'checkpoints' / best['path']
        assert sha(selected_path) == best['sha256']
        adapter.load_state_dict(torch.load(selected_path, weights_only=True))
        assert all(bool(torch.isfinite(p).all()) for p in adapter.parameters())
        hooks.adapters[layer] = adapter.eval().requires_grad_(False)
        selections[layer] = best
        save(HERE / 'checkpoints' / f'layer{layer:02d}-selection.json', dict(best=best, history=history,
            train_ids=[r['id'] for r in calibration if r['split']=='train'],
            validation_ids=[r['id'] for r in calibration if r['split']=='validation']))
        del groups, batches, optimizer, x, y
        gc.collect()
        torch.cuda.empty_cache()
    assert len(hooks.adapters) == 16
    assert sum(p.numel() for a in hooks.adapters.values() for p in a.parameters()) == 16844800
    assert all(not p.requires_grad and p._version == versions[n] for n, p in model.named_parameters())
    save(HERE / 'training-complete.json', dict(status='complete', selections=selections,
        trainable_parameters=16844800, no_heldout_features=True, base_model_unchanged=True))
    print('TRAINING COMPLETE16; HELDOUT START', flush=True)
    rows = []
    for item in records:
        if item['split'] != 'heldout':
            continue
        for method in METHODS:
            deadline()
            row, logits = engine.generate(item, method)
            row.update(correct=exact_match(row['prediction'], item['answer']),
                repetition=diagnostics.repetition(row['generated_ids']),
                job=os.environ['SLURM_JOB_ID'], node=os.uname().nodename)
            path = HERE / 'results' / f'{method}-{item["id"]}.npy'
            with path.open('xb') as handle:
                np.save(handle, logits, allow_pickle=False)
            row['logits_sha256'] = sha(path)
            save(path.with_suffix('.json'), row)
            rows.append(row)
            print('RESULT', method, item['id'], row['correct'], len(row['generated_ids']), flush=True)
    validate_results(rows, records)
    assert len(list((HERE / 'teacher').glob('*.pt'))) == 40
    assert len(list((HERE / 'features').glob('*.pt'))) == 16
    assert len(list((HERE / 'checkpoints').glob('*.pt'))) == 96
    assert len(list((HERE / 'results').glob('*.json'))) == 24
    for row in rows:
        path = HERE / 'results' / f'{row["method"]}-{row["id"]}.npy'
        assert sha(path) == row['logits_sha256']
        a = np.load(HERE / 'results' / f'dense-{row["id"]}.npy').astype('float64')
        b = np.load(path).astype('float64')
        assert a.shape == b.shape == (128256,) and np.isfinite(a).all() and np.isfinite(b).all()
        lp = a - a.max() - np.log(np.exp(a - a.max()).sum())
        lq = b - b.max() - np.log(np.exp(b - b.max()).sum())
        row['final_prompt_vocabulary_kl'] = float((np.exp(lp) * (lp - lq)).sum())
    summary = {m: dict(count=8, correct=sum(r['correct'] for r in rows if r['method']==m),
        capped=sum(r['stop_reason']=='max_new_tokens' for r in rows if r['method']==m),
        repetitive=sum(r['repetition']['repetitive_flag'] for r in rows if r['method']==m),
        mean_final_prompt_vocabulary_kl=statistics.mean(r['final_prompt_vocabulary_kl'] for r in rows if r['method']==m))
        for m in METHODS}
    save(HERE / 'complete.json', dict(status='complete', records=rows, summary=summary,
        elapsed_seconds=time.monotonic()-start, manifest_sha256=sha(HERE/'inputs/manifest.json')))
    hooks.close()
    print('PREOPROJ COMPLETE24', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'run'))
    args = parser.parse_args()
    try:
        {'prepare': prepare, 'check': check, 'run': run}[args.action]()
    except BaseException as error:
        if args.action != 'check' and not (HERE / 'failure.json').exists():
            save(HERE / 'failure.json', dict(error=repr(error), action=args.action, job=os.environ.get('SLURM_JOB_ID')))
        raise
