"""Matched three-input smoke; consumes frozen IDs and never submits jobs."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/ocean/projects/cis260309p/jzhang114/sparseAttention')
PRIOR = ROOT / 'experiments/psc_kl_quality_retry02_20260927'
ROUTING = ROOT / 'experiments/prefill_token_head_routing_20260924/routing.py'
METHODS = ('dense', 'repeat', 'pair')
PINS = {
    PRIOR / 'quality.py': '7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61',
    PRIOR / 'env.sh': 'c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9',
    ROUTING: '8abd4dc5bd21a69d6af7c2ef6632c3cd545fe1a0a51b8e2c79dd1e270ee652dd',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def setup():
    for path, expected in PINS.items():
        assert sha(path) == expected, f'Source changed: {path}'
    q = load_module('paired_prior_quality', PRIOR / 'quality.py')
    q.verify_sources()
    manifest = json.loads((PRIOR / 'inputs/manifest.json').read_text())
    for name, expected in manifest['tokenizer_files'].items():
        assert sha(q.MODEL / name) == expected, name
    indices = manifest['indices'][:3]
    assert indices == [331, 61, 16], indices
    for index in indices:
        item = next(r for r in manifest['records'] if r['source_index'] == index)
        assert sha(PRIOR / f'inputs/{index}.ids') == item['input_ids_sha256']
    return q, manifest, indices


def worker(method):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from adapter import install, DENSE_LAYERS

    q, manifest, indices = setup()
    assert transformers.__version__.startswith('4.44.'), transformers.__version__
    assert torch.cuda.device_count() == 1
    assert 'H100' in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    routing = load_module('paired_frozen_routing', ROUTING)
    model = AutoModelForCausalLM.from_pretrained(q.MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    assert (model.config.num_hidden_layers, model.config.num_attention_heads,
            model.config.num_key_value_heads, model.config.hidden_size) == (32, 32, 8, 4096)
    q.install_mlp(model)
    state = None if method == 'dense' else install(model, method, routing.routed_attention)
    assert not any(mod._forward_hooks for mod in model.modules())
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    score, score_sha = q.scorer()
    assert score_sha == manifest['scorer_sha256']

    def ids_for(index):
        raw = (PRIOR / f'inputs/{index}.ids').read_bytes()
        return torch.tensor(np.frombuffer(raw, dtype='<i8').copy(), device='cuda')[None]

    def step(tokens, cache, offset):
        with torch.inference_mode():
            out = model.model(input_ids=tokens,
                position_ids=torch.arange(offset, offset + tokens.shape[1], device='cuda')[None],
                attention_mask=torch.ones((1, offset + tokens.shape[1]), dtype=torch.long, device='cuda'),
                past_key_values=cache, use_cache=True, return_dict=True)
            return model.lm_head(out.last_hidden_state[:, -1:]).float()[0, 0]

    def prefill(ids):
        cache = DynamicCache()
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = step(ids, cache, 0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert torch.isfinite(logits).all()
        assert len(cache.key_cache) == 32
        assert all(k.shape == v.shape == (1, 8, ids.shape[1], 128)
                   for k, v in zip(cache.key_cache, cache.value_cache))
        return cache, logits, elapsed

    warm_ids = ids_for(indices[0])
    cache, logits, _ = prefill(warm_ids)
    for j in range(2):
        logits = step(logits.argmax().reshape(1, 1), cache, warm_ids.shape[1] + j)
    torch.cuda.synchronize()
    del cache, logits, warm_ids
    gc.collect()
    torch.cuda.empty_cache()

    for index in indices:
        item = next(r for r in manifest['records'] if r['source_index'] == index)
        ids = ids_for(index)
        n = ids.shape[1]
        torch.cuda.reset_peak_memory_stats()
        cache, logits, prefill_seconds = prefill(ids)
        # Diagnostics run outside timing, only at the final prompt position.
        last_logits = logits.detach().cpu().double()
        dense_path = HERE / 'results' / f'dense-{index}.npy'
        kl = 0.0
        if method == 'dense':
            np.save(dense_path, last_logits.numpy(), allow_pickle=False)
        else:
            dense_logits = torch.from_numpy(np.load(dense_path, allow_pickle=False))
            logp, logq = dense_logits.log_softmax(-1), last_logits.log_softmax(-1)
            kl = float((logp.exp() * (logp - logq)).sum())
            assert kl >= -1e-10
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated = []
        for j in range(128):
            token = int(logits.argmax())
            generated.append(token)
            if token in eos or j == 127:
                break
            logits = step(torch.tensor([[token]], device='cuda'), cache, n + j)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started
        assert torch.isfinite(logits).all()
        assert all(k.shape[2] == n + len(generated) - 1 for k in cache.key_cache)
        peak = torch.cuda.max_memory_allocated()
        prediction = tok.decode(generated, skip_special_tokens=True)
        historical_equal = None
        if method != 'pair':
            previous_method = 'dense' if method == 'dense' else 'interleaved'
            previous = json.loads((PRIOR / f'formal/{index}/{previous_method}/result.json').read_text())
            assert previous['input_ids_sha256'] == item['input_ids_sha256']
            historical_equal = generated == previous['generated_ids']
        counts = {} if state is None else dict(state['pairs'])
        if state is not None:
            assert set(counts) == set(range(32)) - set(DENSE_LAYERS)
        record = dict(method=method, **item, generated_ids=generated, prediction=prediction,
            correct=score(('===answerspliter==='.join(item['answers']), prediction)),
            generated_tokens=len(generated), stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',
            prefill_seconds=prefill_seconds, natural_decode_seconds=decode_seconds,
            natural_total_seconds=prefill_seconds + decode_seconds,
            peak_allocated_bytes=peak, prompt_last_logits_kl_dense_to_method=kl,
            historical_token_ids_equal=historical_equal, routed_layer_eligible_pairs=counts,
            gpu=torch.cuda.get_device_name(), node=os.uname().nodename,
            job=os.environ['SLURM_JOB_ID'], full_kv_cache=True, dense_decode=True,
            torch_version=torch.__version__, transformers_version=transformers.__version__)
        save(HERE / 'results' / f'{method}-{index}.json', record)
        print('RESULT', json.dumps(record), flush=True)
        if historical_equal is False:
            raise RuntimeError(f'Historical prediction regression: {method}/{index}; stop before pair')
        del cache, logits, ids
        gc.collect()
        torch.cuda.empty_cache()


def run():
    from adapter import DENSE_LAYERS
    from assembly import head_pairs
    _, manifest, indices = setup()
    (HERE / 'results').mkdir(exist_ok=False)
    save(HERE / 'protocol.json', dict(methods=METHODS, indices=indices,
        source_sha256={name: sha(HERE / name) for name in
                       ('assembly.py', 'adapter.py', 'smoke.py', 'test_assembly.py', 'run.sbatch')},
        frozen_pins={str(path): value for path, value in PINS.items()},
        manifest_sha256=sha(PRIOR / 'inputs/manifest.json'),
        tokenizer_files=manifest['tokenizer_files'], dense_layers=DENSE_LAYERS,
        head_pairs=head_pairs(32), scale=1.0, owner='original_position % 32',
        semantics='Assemble O_h after A_h V_h; zero missing output rows, not attention logits',
        precision='BF16', backend='flash_attention_2', seed=42, max_new_tokens=128,
        kl='KL(dense || method), final prompt next-token vocabulary distribution, float64 nats',
        timing='CUDA-synchronized; excludes load, warmup, diagnostics, scoring, serialization',
        limitation='Three inputs, no full benchmark or layerwise KL; output-length-dependent decode time'))
    for method in METHODS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--method', method], check=True)
    rows = [json.loads(p.read_text()) for p in sorted((HERE / 'results').glob('*.json'))]
    assert len(rows) == 9
    for index in indices:
        group = {r['method']: r for r in rows if r['source_index'] == index}
        assert set(group) == set(METHODS)
        assert len({r['input_ids_sha256'] for r in group.values()}) == 1
        assert group['repeat']['routed_layer_eligible_pairs'] == group['pair']['routed_layer_eligible_pairs']
        assert group['dense']['historical_token_ids_equal'] is True
        assert group['repeat']['historical_token_ids_equal'] is True
    save(HERE / 'complete.json', dict(status='complete', records=rows))
    print('SMOKE COMPLETE: 9 aligned results', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        print('PREFLIGHT', setup()[2])
    elif args.method:
        worker(args.method)
    else:
        try:
            run()
        except BaseException as error:
            # Never overwrite an earlier attempt's result on an accidental restart.
            failure = HERE / 'failure.json'
            if not failure.exists() and not (HERE / 'complete.json').exists():
                save(failure, dict(error=repr(error), job=os.environ.get('SLURM_JOB_ID')))
            raise
