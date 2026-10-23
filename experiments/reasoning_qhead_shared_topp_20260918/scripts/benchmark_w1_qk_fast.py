"""GPU correctness and same-device synchronized timings before HMMT inference."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import unittest
import torch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root/'source/ra_harness'))
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer
from kvcompress.engine.w1_qk_fast_cache import FastFullW1Layer, fast_uniform_residual_ids
from kvcompress.engine.w1_interval1000_cache import uniform_residual_ids


def write(name, data):
    (root/name).write_text(json.dumps(data, indent=2)+'\n')


suite = unittest.TestSuite()
for name in ('test_w1_qk_fast', 'test_full_w1_qk', 'test_w1_interval1000'):
    spec = importlib.util.spec_from_file_location(name, root/'source/tests'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    if name == 'test_w1_qk_fast':
        make_layer = module.layer
with (root/'correctness.log').open('x') as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
proof = dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
             tests=result.testsRun, skipped=result.skipped, gpu=torch.cuda.get_device_name(),
             node=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
             mask_equivalence='stable raw-BF16-QK sort, including all65536 encodings',
             random_equivalence='same random draws, priority keys and RNG state',
             tiny_model='raw-QK baseline/stable fast/packed fast sampled token IDs')
import hashlib
proof['log_sha256'] = hashlib.sha256((root/'correctness.log').read_bytes()).hexdigest()
write('correctness.json', proof)
print(json.dumps(proof), flush=True)
assert proof['status'] == 'passed'


def timed(fn, repetitions=30):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    torch.cuda.synchronize()
    return 1000*(time.perf_counter()-start)/repetitions


@torch.inference_mode()
def benchmark(n):
    layers = {}
    for name in ('baseline', 'stable_sort', 'packed_qk_topk', 'packed_qk32_topk'):
        cls = FullW1OptimizedLayer if name == 'baseline' else FastFullW1Layer
        l = make_layer(cls, 'cuda', n, hkv=8, dim=128)
        l.trace = False
        if name != 'baseline':
            l.rank_backend = name
        layers[name] = l
    reference = layers['baseline'].select_indices()[0].clone()
    assert all(torch.equal(reference, l.select_indices()[0]) for l in layers.values())
    base = layers['baseline']
    k, v = base.k_cache[:, n-1:n].contiguous(), base.v_cache[:, n-1:n].contiguous()
    q = base.queries.transpose(1, 2)

    def attention(l):
        l.cumulative_length = n-1
        l._lengths = [n-1]*2
        return l.attention(q, k, v, 128**-.5)

    out = attention(base)
    assert all(torch.equal(out, attention(l)) for l in layers.values())
    measurements = {name: dict(selector_ms=[], full_layer_ms=[]) for name in layers}
    for trial in range(4):
        order = list(layers) if trial%2 == 0 else list(reversed(layers))
        for name in order:
            l = layers[name]
            measurements[name]['selector_ms'].append(timed(l.select_indices))
            measurements[name]['full_layer_ms'].append(timed(lambda: attention(l)))
    for data in measurements.values():
        data['selector_median_ms'] = statistics.median(data['selector_ms'])
        data['full_layer_median_ms'] = statistics.median(data['full_layer_ms'])
    record = dict(N=n, K=1024, batch=2, Q_heads=32, KV_heads=8, D=128,
                  masks_exact=True, attention_bitwise_equal=True, measurements=measurements)
    print(json.dumps(record), flush=True)
    return record


records = [benchmark(n) for n in (1024, 5120, 6120, 8192, 32700)]
backends = ('stable_sort', 'packed_qk_topk', 'packed_qk32_topk')
# Full history spends most time beyond8192; hybrid live candidates are5120..6120.
selected = {storage: min(backends, key=lambda b: sum(
    r['measurements'][b]['full_layer_median_ms'] for r in records if r['N'] in ns))
    for storage, ns in (('full', (8192, 32700)), ('hybrid', (5120, 6120)))}
core = torch.arange(1024, device='cuda')
a = torch.Generator(device='cuda').manual_seed(42)
b = torch.Generator(device='cuda').manual_seed(42)
workspace = torch.empty(6120, device='cuda', dtype=torch.long)
random_times = dict(
    baseline_ms=timed(lambda: uniform_residual_ids(6120, core, 4096, a)),
    fast_ms=timed(lambda: fast_uniform_residual_ids(6120, core, 4096, b, workspace)))
write('microbenchmark.json', dict(status='completed', gpu=torch.cuda.get_device_name(),
    node=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), results=records,
    selected_backends=selected, residual_sampler=random_times,
    scope='K1024 same-GPU synthetic layer: synchronized wall time;4 alternating rounds x30 repeats, warmup excluded. Includes identical Python state reset. No inference loading, grading or serialization.',
    caveat='Microbenchmark speedups are not full-model speedups. Formal HMMT timing includes sampling and all attention operations.'))
print(json.dumps(dict(selected_backends=selected, residual_sampler=random_times)), flush=True)
