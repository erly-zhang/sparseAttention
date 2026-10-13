"""Three-input PSC inference smoke; no KL hooks and no automatic resubmission."""
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
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/ocean/projects/cis260309p/jzhang114/sparseAttention')
PRIOR = ROOT / 'experiments/psc_kl_quality_retry02_20260927'
METHODS = ('dense', 'tail_dense', 'interleaved')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def setup():
    spec = importlib.util.spec_from_file_location('prior_quality', PRIOR / 'quality.py')
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    q.verify_sources()
    manifest = json.loads((PRIOR / 'inputs/manifest.json').read_text())
    for name, expected in manifest['tokenizer_files'].items():
        assert sha(q.MODEL / name) == expected, name
    indices = manifest['indices'][:3]
    assert len(set(indices)) == 3
    for i in indices:
        item = next(r for r in manifest['records'] if r['source_index'] == i)
        assert sha(PRIOR / f'inputs/{i}.ids') == item['input_ids_sha256']
    return q, manifest, indices


def worker(method):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    q, manifest, indices = setup()
    assert torch.cuda.device_count() == 1
    assert 'H100' in torch.cuda.get_device_name()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    sys.path[:0] = [str(q.BASE), str(ROOT)]
    from adapters import install
    model = AutoModelForCausalLM.from_pretrained(q.MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    q.install_mlp(model)
    original = {name: mod.forward for name, mod in model.named_modules()}
    model, state = install(model, 'routing_c' if method != 'dense' else 'dense')
    if method != 'dense':
        for i in q.DENSE_LAYERS[method]:
            model.model.layers[i].self_attn.forward = original[f'model.layers.{i}.self_attn']
    sparse = {name: mod.forward for name, mod in model.named_modules()}
    assert not any(mod._forward_hooks for mod in model.modules())
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    score, score_sha = q.scorer()
    assert score_sha == manifest['scorer_sha256']

    def restore(forwards):
        for name, mod in model.named_modules():
            mod.forward = forwards[name]

    def step(tokens, cache, offset):
        with torch.inference_mode():
            result = model.model(input_ids=tokens,
                position_ids=torch.arange(offset, offset + tokens.shape[1], device='cuda')[None],
                attention_mask=torch.ones((1, offset + tokens.shape[1]), dtype=torch.long, device='cuda'),
                past_key_values=cache, use_cache=True, return_dict=True)
            return model.lm_head(result.last_hidden_state[:, -1:]).float()[0, 0]

    def prefill(ids):
        restore(sparse)
        cache = DynamicCache()
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = step(ids, cache, 0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        restore(original)
        assert torch.isfinite(logits).all()
        assert len(cache.key_cache) == 32
        assert all(k.shape == v.shape == (1, 8, ids.shape[1], 128)
                   for k, v in zip(cache.key_cache, cache.value_cache))
        return cache, logits, elapsed

    def ids_for(i):
        raw = (PRIOR / f'inputs/{i}.ids').read_bytes()
        return torch.tensor(np.frombuffer(raw, dtype='<i8').copy(), device='cuda')[None]

    # Full-length warmup covers the real attention shapes, outside all timings.
    warm_ids = ids_for(indices[0])
    warm_cache, warm_logits, _ = prefill(warm_ids)
    for j in range(2):
        warm_logits = step(warm_logits.argmax().reshape(1, 1), warm_cache, warm_ids.shape[1] + j)
    torch.cuda.synchronize()
    del warm_ids, warm_cache, warm_logits
    gc.collect()
    torch.cuda.empty_cache()

    for index in indices:
        item = next(r for r in manifest['records'] if r['source_index'] == index)
        ids = ids_for(index)
        n = ids.shape[1]
        torch.cuda.reset_peak_memory_stats()
        cache, logits, prefill_s = prefill(ids)
        torch.cuda.synchronize()
        start = time.perf_counter()
        free = []
        for j in range(128):
            token = int(logits.argmax())
            free.append(token)
            if token in eos or j == 127:
                break
            logits = step(torch.tensor([[token]], device='cuda'), cache, n + j)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - start
        assert torch.isfinite(logits).all()
        peak = torch.cuda.max_memory_allocated()
        prediction = tok.decode(free, skip_special_tokens=True)
        prior_result = json.loads((PRIOR / f'formal/{index}/{method}/result.json').read_text())
        assert prior_result['input_ids_sha256'] == item['input_ids_sha256']
        record = dict(method=method, **item, generated_ids=free, prediction=prediction,
            correct=score(('===answerspliter==='.join(item['answers']), prediction)),
            generated_tokens=len(free), decode_forward_steps=len(free)-1,
            stop_reason='eos' if free[-1] in eos else 'max_new_tokens',
            prefill_seconds=prefill_s, natural_decode_seconds=decode_s,
            natural_total_seconds=prefill_s+decode_s, peak_allocated_bytes=peak,
            historical_token_ids_equal=(free == prior_result['generated_ids']))
        del cache, logits
        gc.collect()
        torch.cuda.empty_cache()
        dense = json.loads((PRIOR / f'formal/{index}/dense/result.json').read_text())
        assert dense['input_ids_sha256'] == item['input_ids_sha256']
        forced = (dense['generated_ids'] + [eos[0]] * 32)[:32]
        forced_ids = torch.tensor([forced], device='cuda')
        cache, logits, second_prefill_s = prefill(ids)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for j in range(32):
            logits = step(forced_ids[:, j:j+1], cache, n + j)
        torch.cuda.synchronize()
        fixed_s = time.perf_counter() - start
        assert torch.isfinite(logits).all()
        record.update(fixed_decode_steps=32, fixed_decode_seconds=fixed_s,
            fixed_decode_tokens_per_second=32/fixed_s,
            fixed_continuation_ids=forced, fixed_eos_padding=max(0, 32-len(dense['generated_ids'])),
            second_prefill_seconds=second_prefill_s,
            gpu=torch.cuda.get_device_name(), node=os.uname().nodename,
            job=os.environ['SLURM_JOB_ID'], kl_hooks=False)
        save(HERE / 'results' / f'{method}-{index}.json', record)
        print('RESULT', json.dumps(record), flush=True)
        del ids, cache, logits, forced_ids
        gc.collect()
        torch.cuda.empty_cache()


def run():
    q, manifest, indices = setup()
    (HERE / 'results').mkdir(exist_ok=False)
    save(HERE / 'protocol.json', dict(indices=indices, methods=METHODS,
        manifest_sha256=sha(PRIOR / 'inputs/manifest.json'),
        driver_sha256=sha(__file__), prior_driver_sha256=sha(PRIOR / 'quality.py'),
        precision='BF16', max_new_tokens=128, fixed_decode_steps=32,
        timing='CUDA-synchronized wall time; no model load, warmup, KL hooks, scoring or serialization',
        fixed_decode='Same saved dense continuation; EOS-pad to 32 only for timing, not quality',
        limitation='Three inputs, one natural and one fixed-decode pass; not a full benchmark'))
    for method in METHODS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--method', method], check=True)
    rows = [json.loads(p.read_text()) for p in sorted((HERE / 'results').glob('*.json'))]
    assert len(rows) == 9
    for i in indices:
        group = [r for r in rows if r['source_index'] == i]
        assert {r['method'] for r in group} == set(METHODS)
        assert len({r['input_ids_sha256'] for r in group}) == 1
        assert len({tuple(r['fixed_continuation_ids']) for r in group}) == 1
    save(HERE / 'complete.json', dict(status='complete', results=rows,
        all_historical_predictions_equal=all(r['historical_token_ids_equal'] for r in rows)))
    print('SMOKE COMPLETE: 9 aligned results', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        if args.check:
            print('PREFLIGHT', setup()[2])
        elif args.method:
            worker(args.method)
        else:
            run()
    except BaseException:
        traceback.print_exc()
        raise
