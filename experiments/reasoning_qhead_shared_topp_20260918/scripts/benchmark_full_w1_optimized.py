"""Same-GPU exact-mask validation and synchronized implementation timings."""
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
sys.path.insert(0, str(root / 'source/ra_harness'))
from kvcompress.engine.selector_grid_cache import SelectorGridLayer
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer


def save(name, obj):
    (root / name).write_text(json.dumps(obj, indent=2) + '\n')


spec = importlib.util.spec_from_file_location('test_full_w1_optimized', root / 'source/tests/test_full_w1_optimized.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with (root / 'correctness.log').open('x') as stream:
    check = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
proof = dict(status='passed' if check.wasSuccessful() and not check.skipped else 'failed',
             tests=check.testsRun, skipped=check.skipped, gpu=torch.cuda.get_device_name(),
             node=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'))
save('correctness.json', proof);print(json.dumps(proof), flush=True)
assert proof['status'] == 'passed'


def timed(fn, repeats=40):
    for _ in range(5):
        fn()
    torch.cuda.synchronize();start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter()-start) / repeats


@torch.inference_mode()
def bench(n, k):
    gen = torch.Generator(device='cuda').manual_seed(7473)
    q = torch.randn(2, 1, 32, 128, device='cuda', dtype=torch.bfloat16, generator=gen)
    layers = {}
    for name, cls, backend in [('baseline', SelectorGridLayer, None),
                               ('stable_sort', FullW1OptimizedLayer, 'stable_sort'),
                               ('packed_topk', FullW1OptimizedLayer, 'packed_topk')]:
        kw = dict(top_p=.95, query_window=1, eviction_interval=1, rep_seed=0,
                  sample_id=19, max_cache_length=32768)
        if backend:
            kw['rank_backend'] = backend
        layer = cls(3, k, 0, 'grid_full_w1_topk_protected', 0., False, False, **kw)
        layer.lazy_initialization(torch.zeros(2, 137, 8, 128, device='cuda', dtype=torch.bfloat16))
        if not layers:
            layer.k_cache[:, :n].normal_(generator=gen);layer.v_cache[:, :n].normal_(generator=gen)
        else:
            layer.k_cache[:, :n].copy_(layers['baseline'].k_cache[:, :n])
            layer.v_cache[:, :n].copy_(layers['baseline'].v_cache[:, :n])
        layer.cumulative_length=n;layer._lengths=[n, n];layer.cache_seqlens.fill_(n)
        layer.slot_positions[:, :n] = torch.arange(n, device='cuda')
        layer.update_queries(q.transpose(1, 2))
        layers[name] = layer
    old = layers['baseline'].select_indices()[0].clone()
    for name in ['stable_sort', 'packed_topk']:
        assert torch.equal(old, layers[name].select_indices()[0]), name
    k_new = layers['baseline'].k_cache[:, n-1:n].contiguous()
    v_new = layers['baseline'].v_cache[:, n-1:n].contiguous()
    def combined(layer):
        # Equal synthetic state reset is included for both implementations.
        layer.cumulative_length=n-1;layer._lengths=[n-1, n-1]
        layer.cache_seqlens.fill_(n-1)
        layer.update_queries(q.transpose(1, 2))
        return layer.attention(q, k_new, v_new, 128**-.5)
    old_out=combined(layers['baseline'])
    for name in ['stable_sort', 'packed_topk']:
        assert torch.equal(old_out, combined(layers[name])), name
    measurements={name:dict(selector_ms=[],full_layer_ms=[]) for name in layers}
    # Alternate traversal to reduce a consistent first/last timing bias.
    for trial in range(4):
        order=list(layers) if trial%2==0 else list(reversed(layers))
        for name in order:
            layer=layers[name]
            measurements[name]['selector_ms'].append(timed(layer.select_indices))
            measurements[name]['full_layer_ms'].append(timed(lambda:combined(layer)))
    result=dict(N=n,K=k,batch=2,Q_heads=32,KV_heads=8,D=128,mask_exact=True,attention_bitwise_equal=True,
                measurements={name:{**v,'selector_median_ms':statistics.median(v['selector_ms']),
                                    'full_layer_median_ms':statistics.median(v['full_layer_ms'])}
                              for name,v in measurements.items()})
    print(json.dumps(result),flush=True)
    return result


results=[bench(n,k) for n in [8192,32700] for k in [1024,2048]]
total={name:sum(r['measurements'][name]['full_layer_median_ms'] for r in results)
       for name in ['stable_sort','packed_topk']}
winner=min(total,key=total.get)
proof=dict(status='completed',gpu=torch.cuda.get_device_name(),node=os.uname().nodename,
           job=os.environ.get('SLURM_JOB_ID'),results=results,selected_backend=winner,
           timing_scope='Synthetic same-GPU layer timing; loading, correctness, warmups and serialization excluded. Includes equal state reset. Four alternating rounds, synchronized40-call wall per round; no profiler in wall measurements.',
           caveat='Layer timings are not full-model or full-dataset speedups; BF16 scores, stable-tie masks and attention outputs match exactly on this GPU.')
save('microbenchmark.json',proof)
