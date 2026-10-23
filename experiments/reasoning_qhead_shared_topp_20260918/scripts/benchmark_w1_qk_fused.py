"""Bounded same-GPU v2/v3 correctness and layer timings; no dataset rerun."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
import unittest

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root/'source/ra_harness'), str(root/'source/tests')]
print('Importing validation dependencies', flush=True)
import torch
from test_w1_qk_fast import layer
from test_w1_qk_fused import GPUChecks
from kvcompress.engine.w1_qk_fast_cache import FastFullW1Layer, FastHybridW1Layer
from kvcompress.engine.w1_qk_fused_cache import FusedFullW1Layer, FusedHybridW1Layer

torch.set_num_threads(1)


def write(name, data):
    (root/name).write_text(json.dumps(data, indent=2)+'\n')


with (root/'correctness.log').open('x') as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GPUChecks))
proof = dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
             tests=result.testsRun, skipped=result.skipped, gpu=torch.cuda.get_device_name(),
             node=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'))
write('correctness.json', proof)
print(json.dumps(proof), flush=True)
assert proof['status'] == 'passed'


def timed(fn, repeats=80):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return 1000*(time.perf_counter()-start)/repeats


@torch.inference_mode()
def benchmark(storage, budget, n):
    old_cls, new_cls = ((FastFullW1Layer, FusedFullW1Layer) if storage == 'full'
                       else (FastHybridW1Layer, FusedHybridW1Layer))
    caches = {name: layer(cls, 'cuda', n, budget=budget, hkv=8, dim=128)
              for name, cls in (('v2', old_cls), ('v3', new_cls))}
    for cache in caches.values():
        cache.trace = False
        cache.rank_backend = 'stable_sort' if storage == 'full' else 'packed_qk_topk'
    old = caches['v2']
    q = old.queries.transpose(1, 2)
    k, v = old.k_cache[:, n-1:n].contiguous(), old.v_cache[:, n-1:n].contiguous()

    def pipeline(cache):
        ids, counts = cache.select_indices()
        cache.gather(ids)
        return ids

    def attention(cache):
        cache.cumulative_length = n-1
        cache._lengths = [n-1]*2
        return cache.attention(q, k, v, 128**-.5)

    assert torch.equal(pipeline(caches['v2']), pipeline(caches['v3']))
    assert torch.equal(attention(caches['v2']), attention(caches['v3']))
    measurements = {name: dict(selection_and_gather_ms=[], complete_layer_ms=[]) for name in caches}
    for trial in range(6):
        for name in (('v2', 'v3') if trial%2 == 0 else ('v3', 'v2')):
            cache = caches[name]
            # Fix the same live state before both measurements.
            cache.cumulative_length = n
            cache._lengths = [n]*2
            measurements[name]['selection_and_gather_ms'].append(timed(lambda: pipeline(cache)))
            measurements[name]['complete_layer_ms'].append(timed(lambda: attention(cache)))
    for record in measurements.values():
        for key in ('selection_and_gather_ms', 'complete_layer_ms'):
            record[key.replace('_ms', '_median_ms')] = statistics.median(record[key])
    v2 = measurements['v2']['complete_layer_median_ms']
    v3 = measurements['v3']['complete_layer_median_ms']
    record = dict(storage=storage, K=budget, N=n, batch=2, Q_heads=32, KV_heads=8,
                  D=128, masks_exact=True, attention_bitwise_equal=True,
                  measurements=measurements, complete_layer_reduction_percent=100*(1-v3/v2))
    print(json.dumps(record), flush=True)
    return record


records = [benchmark(storage, k, n) for storage, k, n in (
    ('full', 1024, 1024), ('full', 1024, 8192), ('full', 1024, 32700),
    ('full', 2048, 32700), ('hybrid', 1024, 5120), ('hybrid', 1024, 6120),
    ('hybrid', 2048, 6144), ('hybrid', 2048, 7144))]
write('microbenchmark.json', dict(status='completed', gpu=torch.cuda.get_device_name(),
    node=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), results=records,
    scope='Six alternating rounds x80 repeats, synchronized wall time, same GPU, warmup excluded. '
          'Includes Python representative draw, raw-QK selection, index handling, KV append/gather '
          'and attention. Physical compaction events are tested for correctness separately.',
    caveat='Synthetic layer timings are not full-Qwen3-4B or HMMT end-to-end speedups. '
           'No full-dataset accuracy experiment was launched.'))
