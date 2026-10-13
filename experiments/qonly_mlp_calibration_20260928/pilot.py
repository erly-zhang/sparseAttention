"""Isolated Q-only/native readout calibration pilot; no automatic retries."""
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

from protocol import specs, make_prompt, validate_inputs, validate_results
from protocol import save, sha, exact_match, METHODS, MAX_NEW

HERE = Path(__file__).resolve().parent
ROOT = Path('/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments')
PRIOR = ROOT / 'psc_kl_quality_retry02_20260927'
ABLATION = ROOT / 'head_slot_kv_ablation_20260928'
ROUTING = ROOT / 'prefill_token_head_routing_20260924/routing.py'
PINS = {
    PRIOR / 'quality.py': '7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61',
    PRIOR / 'env.sh': 'c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9',
    PRIOR / 'inputs/manifest.json': 'dc76a7a783ef48c5be21338b3c1c3cb0c8a816470bf7661a232da75057bd35e3',
    ABLATION / 'adapter.py': 'a14e7a1b11462f9cbfb02e43bb979ef7fc3277df21f186e9a9b4c43b10231397',
    ABLATION / 'assembly.py': '3d49d9cc58117081000e22886f891b07cf91681cbc68ef9ae9043d244587531d',
    ABLATION / 'legacy_checks.py': '50011b36274403486429a27bc8f90d48e910e804d034c8a92fae0f55a716b297',
    ABLATION / 'inputs/manifest.json': 'abdf9a76ce3565cb1ce65c1231b90ec9a1857c0816a2a7146057ddcf69151dbc',
    ROUTING: '8abd4dc5bd21a69d6af7c2ef6632c3cd545fe1a0a51b8e2c79dd1e270ee652dd',
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight():
    for path, digest in PINS.items():
        assert sha(path) == digest, path
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split()
        assert sha(HERE / name) == digest, name
    q = load('calibration_prior', PRIOR / 'quality.py')
    q.verify_sources()
    old = json.loads((PRIOR / 'inputs/manifest.json').read_text())
    for name, digest in old['tokenizer_files'].items():
        assert sha(q.MODEL / name) == digest, name
    return q, old


def prepare():
    from transformers import AutoTokenizer
    q, old = preflight()
    inputs = HERE / 'inputs'
    inputs.mkdir()
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)

    def encode(prompt):
        text = tok.apply_chat_template([dict(role='user', content=prompt)],
                                       tokenize=False, add_generation_prompt=True)
        return tok.encode(text, add_special_tokens=False)

    excluded = {r['input_ids_sha256'] for r in json.loads(
        (ABLATION / 'inputs/manifest.json').read_text())['records']}
    rows = []
    for spec in specs():
        row = make_prompt(spec, encode)
        ids = row.pop('ids')
        path = inputs / (row['id'] + '.ids')
        with path.open('xb') as handle:
            handle.write(struct.pack('<' + 'q' * len(ids), *ids))
        row.update(input_tokens=len(ids), input_ids_sha256=sha(path))
        rows.append(row)
    # JSON is the canonical representation (pairs become lists).
    rows = json.loads(json.dumps(rows))
    validate_inputs(rows, excluded)
    save(inputs / 'manifest.json', dict(records=rows, excluded_hashes=sorted(excluded),
        tokenizer_files=old['tokenizer_files'], model=str(q.MODEL),
        source_manifest_sha256=sha(HERE / 'SHA256SUMS'),
        kv_mode='q_only', assembly='native', seed=42, max_new_tokens=MAX_NEW,
        task='independent synthetic UUID retrieval; strict normalized exact match',
        selection='32 train,8 validation,8 heldout; disjoint prompt seeds/keys/values',
        truncation='none; maximum row count below token bin, within256 tokens',
        teacher='dense greedy completion, no correctness filtering',
        adapter='post-final-RMSNorm residual bottleneck128; SiLU or linear',
        optimization=dict(epochs=5, microbatch=8, lr=0.001, alpha=1, beta=1, temperature=1)))
    print('FROZEN SYNTHETIC48', sha(inputs / 'manifest.json'), flush=True)


def check():
    q, _ = preflight()
    manifest = json.loads((HERE / 'inputs/manifest.json').read_text())
    assert manifest['source_manifest_sha256'] == sha(HERE / 'SHA256SUMS')
    validate_inputs(manifest['records'], manifest['excluded_hashes'])
    for row in manifest['records']:
        path = HERE / 'inputs' / (row['id'] + '.ids')
        assert sha(path) == row['input_ids_sha256']
        assert path.stat().st_size == row['input_tokens'] * 8
    print('FROZEN PREFLIGHT PASS', flush=True)
    return q, manifest


class Engine:
    def __init__(self, model, tokenizer, install, route, dense_layers, device):
        import torch
        self.model, self.tokenizer = model, tokenizer
        self.install, self.route, self.dense_layers = install, route, dense_layers
        self.device = device
        eos = model.generation_config.eos_token_id
        self.eos = [eos] if isinstance(eos, int) else list(eos)
        self.torch = torch

    def step(self, tokens, cache, offset):
        torch = self.torch
        with torch.no_grad():
            output = self.model.model(input_ids=tokens,
                position_ids=torch.arange(offset, offset + tokens.shape[1], device=self.device)[None],
                attention_mask=torch.ones((1, offset + tokens.shape[1]), dtype=torch.long, device=self.device),
                past_key_values=cache, use_cache=True, return_dict=True)
            hidden = output.last_hidden_state[:, -1]
        config = self.model.config
        expected = (1, config.num_key_value_heads, offset + tokens.shape[1],
                    config.hidden_size // config.num_attention_heads)
        assert len(cache.key_cache) == len(cache.value_cache) == config.num_hidden_layers
        assert all(k.shape == v.shape == expected for k, v in zip(cache.key_cache, cache.value_cache))
        assert bool(torch.isfinite(hidden).all())
        return hidden

    def generate(self, ids, method='qonly', adapter=None, teacher_ids=None):
        from transformers import DynamicCache
        torch = self.torch
        cache = DynamicCache()
        n = len(ids)
        if teacher_ids is not None:
            assert 1 <= len(teacher_ids) <= MAX_NEW
            assert not any(token in self.eos for token in teacher_ids[:-1])
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        state = None
        if method != 'dense':
            state = self.install(self.model, 'native', self.route,
                dense_layers=self.dense_layers,
                backend='flash_attention_2' if self.device == 'cuda' else 'sdpa', kv_mode='q_only')
        states, generated = [], []
        start = time.monotonic()
        prefill = None
        try:
            with torch.no_grad():
                tokens = torch.tensor([ids], device=self.device)
                for j in range(MAX_NEW if teacher_ids is None else len(teacher_ids)):
                    hidden = self.step(tokens, cache, 0 if j == 0 else n + j - 1)
                    states.append(hidden[0].detach().float().cpu())
                    corrected = hidden if adapter is None else adapter(hidden, enabled=method != 'dense')
                    logits = self.model.lm_head(corrected).float()[0]
                    assert bool(torch.isfinite(logits).all())
                    if j == 0:
                        first_logits = logits.detach().cpu().numpy().copy()
                        prefill = time.monotonic() - start
                    token = int(logits.argmax()) if teacher_ids is None else int(teacher_ids[j])
                    generated.append(token)
                    if teacher_ids is None and (token in self.eos or j == MAX_NEW - 1):
                        break
                    tokens = torch.tensor([[token]], device=self.device)
            if state:
                routed = set(range(self.model.config.num_hidden_layers)) - set(self.dense_layers)
                assert set(state['pairs']) == routed
                assert set(state['pairs'].values()) == {n * (n + 1) // 2}
            duration = time.monotonic() - start
            return dict(generated_ids=generated, prediction=self.tokenizer.decode(generated, skip_special_tokens=True),
                stop_reason='eos' if generated[-1] in self.eos else 'max_new_tokens',
                prefill_seconds=prefill, decode_seconds=duration - prefill, total_seconds=duration,
                full_kv_cache=True, finite_all_steps=True, dense_decode=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device == 'cuda' else None,
                routed_layer_eligible_pairs={} if state is None else dict(state['pairs'])), torch.stack(states), first_logits
        finally:
            if state:
                state['close']()
            del cache
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()


def run():
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from calibrator import ResidualReadout, paired_loss
    q, manifest = check()
    assert transformers.__version__.startswith('4.44.')
    assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name()
    assert os.environ.get('SLURM_JOB_ID'), 'Independent allocation required'
    start = time.monotonic()
    save(HERE / 'run-start.json', dict(job=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
        manifest_sha256=sha(HERE / 'inputs/manifest.json'), gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__, transformers_version=transformers.__version__))
    for directory in ('features', 'results', 'checkpoints'):
        (HERE / directory).mkdir()

    def deadline():
        assert time.monotonic() - start < 6600, 'Pilot deadline: preserve partial output, no retry'

    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = AutoModelForCausalLM.from_pretrained(q.MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    assert (model.config.num_hidden_layers, model.config.hidden_size,
            model.config.num_attention_heads, model.config.num_key_value_heads) == (32, 4096, 32, 8)
    q.install_mlp(model)  # Existing frozen FFN memory implementation, not calibration.
    load('assembly', ABLATION / 'assembly.py')
    adapter_module = load('qonly_frozen_adapter', ABLATION / 'adapter.py')
    route = load('qonly_frozen_routing', ROUTING).routed_attention
    repetition = load('qonly_diagnostics', ABLATION / 'legacy_checks.py').repetition
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    engine = Engine(model, tok, adapter_module.install, route, adapter_module.DENSE_LAYERS, 'cuda')
    features = {}
    teacher_outputs = {}
    identity = ResidualReadout(4096, 128).cuda().eval()

    def read_ids(item):
        return np.fromfile(HERE / 'inputs' / (item['id'] + '.ids'), dtype='<i8').tolist()

    def collect(item):
        deadline()
        tick = time.monotonic()
        dense, dh, logits = engine.generate(read_ids(item), 'dense')
        sparse, sh, _ = engine.generate(read_ids(item), teacher_ids=dense['generated_ids'])
        assert dense['generated_ids'] == sparse['generated_ids']
        assert dh.shape == sh.shape == (len(dense['generated_ids']), 4096)
        h = sh.to('cuda', dtype=torch.bfloat16)
        assert torch.equal(identity(h, enabled=True), h), 'Initial identity failed'
        path = HERE / 'features' / (item['id'] + '.npz')
        with path.open('xb') as handle:
            np.savez(handle, dense=dh.numpy(), sparse=sh.numpy(),
                     teacher_ids=np.asarray(dense['generated_ids'], dtype=np.int64))
        with (HERE / 'features' / (item['id'] + '-logits.npy')).open('xb') as handle:
            np.save(handle, logits, allow_pickle=False)
        metadata = dict(id=item['id'], split=item['split'], input_ids_sha256=item['input_ids_sha256'],
            features_sha256=sha(path), dense=dense, sparse=sparse, positions=list(range(len(dh))),
            feature_positions='state j predicts teacher token j; prefix teacher_ids[:j]',
            collection_seconds=time.monotonic() - tick,
            teacher_correct=exact_match(dense['prediction'], item['answer']),
            first_logits_sha256=sha(HERE / 'features' / (item['id'] + '-logits.npy')))
        save(HERE / 'features' / (item['id'] + '.json'), metadata)
        features[item['id']] = (sh, dh)
        teacher_outputs[item['id']] = (dense, logits)
        print('COLLECT', item['id'], metadata['teacher_correct'], len(dh), flush=True)
        return metadata

    records = manifest['records']
    smoke_items = [r for r in records if r['id'] in ('train-00', 'train-01')]
    smoke, generation_costs = [], []
    for item in smoke_items:
        smoke.append(collect(item))
        plain, _, _ = engine.generate(read_ids(item))
        initial, _, _ = engine.generate(read_ids(item), adapter=identity)
        assert plain['generated_ids'] == initial['generated_ids'], 'Initial adapter generation mismatch'
        generation_costs.extend([plain['total_seconds'], initial['total_seconds']])
    projected = (time.monotonic() - start + 1.5 * (46 * max(r['collection_seconds'] for r in smoke)
                 + 24 * max(generation_costs)) + 1800)
    assert projected < 6600, f'Pilot budget gate rejected projected seconds {projected}'
    save(HERE / 'smoke-complete.json', dict(status='passed', projected_seconds=projected,
        samples=[r['id'] for r in smoke], initial_identity_generated_ids_equal=True))
    print('PILOT SMOKE PASS', projected, flush=True)
    for item in records:
        if item['id'] not in features:
            collect(item)

    # Equal prompt weight; validation checkpoint selection never reads heldout pairs.
    def batches(split):
        output = []
        for item in records:
            if item['split'] == split:
                x, y = features[item['id']]
                output.append((x, y, torch.full((len(x),), 1.0 / len(x))))
        return tuple(torch.cat([r[j] for r in output]) for j in range(3))

    train = batches('train')
    valid = batches('validation')
    trained = {}
    for method in ('linear', 'mlp'):
        torch.manual_seed(42)
        calibration = ResidualReadout(4096, 128, nonlinear=method == 'mlp').cuda()
        optimizer = torch.optim.AdamW(calibration.parameters(), lr=0.001, weight_decay=0)
        history = []
        best = None
        for epoch in range(5):
            calibration.train()
            order = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(42 + epoch))
            for offset in range(0, len(order), 8):
                deadline()
                ix = order[offset:offset + 8]
                x, y, w = (value[ix].cuda() for value in train)
                optimizer.zero_grad(set_to_none=True)
                # Global prompt weighting, not per-microbatch normalization.
                loss = paired_loss(calibration, model.lm_head, x.to(torch.bfloat16),
                                   y.to(torch.bfloat16), weights=w)
                loss = loss * w.sum() * (len(train[0]) / (32 * len(ix)))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(calibration.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                assert torch.isfinite(norm)
            calibration.eval()
            total, mass = 0.0, 0.0
            with torch.no_grad():
                for offset in range(0, len(valid[0]), 8):
                    deadline()
                    x, y, w = (value[offset:offset + 8].cuda() for value in valid)
                    value = paired_loss(calibration, model.lm_head, x.to(torch.bfloat16),
                                        y.to(torch.bfloat16), weights=w)
                    total += value.item() * w.sum().item()
                    mass += w.sum().item()
            value = total / mass
            path = HERE / 'checkpoints' / f'{method}-epoch{epoch}.pt'
            with path.open('xb') as handle:
                torch.save(calibration.state_dict(), handle)
            history.append(dict(epoch=epoch, validation_loss=value, path=path.name, sha256=sha(path)))
            if best is None or value < best['validation_loss']:
                best = history[-1]
            print('TRAIN', method, epoch, value, flush=True)
        calibration.load_state_dict(torch.load(HERE / 'checkpoints' / best['path'], weights_only=True))
        save(HERE / 'checkpoints' / (method + '-selection.json'), dict(best=best, history=history,
            training_ids=[r['id'] for r in records if r['split'] == 'train'],
            validation_ids=[r['id'] for r in records if r['split'] == 'validation']))
        trained[method] = calibration.eval()

    result_rows = []
    for item in records:
        if item['split'] != 'heldout':
            continue
        for method in METHODS:
            deadline()
            calibration = trained.get(method)
            if method == 'dense':
                output, first_logits = teacher_outputs[item['id']]
            else:
                output, _, first_logits = engine.generate(read_ids(item), adapter=calibration)
            path = HERE / 'results' / f"{method}-{item['id']}.npy"
            with path.open('xb') as handle:
                np.save(handle, first_logits, allow_pickle=False)
            # Heldout teacher-forced diagnostics on the same prefix, never used for selection.
            x, y = features[item['id']]
            errors, kls = [], []
            with torch.no_grad():
                for offset in range(0, len(x), 8):
                    a = y[offset:offset + 8].cuda().to(torch.bfloat16)
                    b = a if method == 'dense' else x[offset:offset + 8].cuda().to(torch.bfloat16)
                    if calibration is not None:
                        b = calibration(b, enabled=True)
                    errors.extend(((b.float() - a.float()).square().sum(-1) /
                                   a.float().square().sum(-1).clamp_min(1e-8)).cpu().tolist())
                    lp = model.lm_head(a).float().log_softmax(-1)
                    lq = model.lm_head(b).float().log_softmax(-1)
                    kls.extend((lp.exp() * (lp - lq)).sum(-1).cpu().tolist())
            row = dict(**output, id=item['id'], method=method,
                input_ids_sha256=item['input_ids_sha256'], correct=exact_match(output['prediction'], item['answer']),
                repetition=repetition(output['generated_ids']), logits_sha256=sha(path),
                teacher_forced_mean_normalized_hidden_error=statistics.mean(errors),
                teacher_forced_mean_vocabulary_kl=statistics.mean(kls))
            save(HERE / 'results' / f"{method}-{item['id']}.json", row)
            result_rows.append(row)
            print('RESULT', method, item['id'], row['correct'], row['stop_reason'], flush=True)

    validate_results(result_rows, records)
    for row in result_rows:
        path = HERE / 'results' / f"{row['method']}-{row['id']}.npy"
        assert sha(path) == row['logits_sha256']
        a = np.load(HERE / 'results' / f"dense-{row['id']}.npy", allow_pickle=False).astype('float64')
        b = np.load(path, allow_pickle=False).astype('float64')
        assert a.shape == b.shape == (128256,) and np.isfinite(a).all() and np.isfinite(b).all()
        lp = a - a.max() - np.log(np.exp(a - a.max()).sum())
        lq = b - b.max() - np.log(np.exp(b - b.max()).sum())
        row['independent_final_prompt_vocabulary_kl'] = float((np.exp(lp) * (lp - lq)).sum())
    assert len(list((HERE / 'results').glob('*.json'))) == 32
    assert len(list((HERE / 'features').glob('*.npz'))) == 48
    for item in records:
        meta = json.loads((HERE / 'features' / (item['id'] + '.json')).read_text())
        assert meta['features_sha256'] == sha(HERE / 'features' / (item['id'] + '.npz'))
    summary = {}
    for method in METHODS:
        group = [r for r in result_rows if r['method'] == method]
        summary[method] = dict(count=len(group), correct=sum(r['correct'] for r in group),
            repetitive=sum(r['repetition']['repetitive_flag'] for r in group),
            capped=sum(r['stop_reason'] == 'max_new_tokens' for r in group),
            mean_final_prompt_vocabulary_kl=statistics.mean(r['independent_final_prompt_vocabulary_kl'] for r in group))
    save(HERE / 'complete.json', dict(status='complete', records=result_rows, summary=summary,
        elapsed_seconds=time.monotonic() - start, manifest_sha256=sha(HERE / 'inputs/manifest.json')))
    print('PILOT COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'run'))
    args = parser.parse_args()
    try:
        {'prepare': prepare, 'check': check, 'run': run}[args.action]()
    except BaseException as error:
        if args.action != 'check' and not (HERE / 'failure.json').exists():
            save(HERE / 'failure.json', dict(error=repr(error), action=args.action,
                                           job=os.environ.get('SLURM_JOB_ID')))
        raise
