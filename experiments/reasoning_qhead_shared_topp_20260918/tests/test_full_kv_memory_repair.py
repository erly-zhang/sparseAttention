"""Regression tests: immediate release, bounded storage, unchanged attention/masks."""
import gc
import unittest
import weakref
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.random_fixed_recent_shared_mask_cache import RandomFixedRecentSharedMaskLayer
from test_shared_mask_cache import step
MODE='qhead_shared_mask_random_fixed_recent_gpu'

def cache(layers=2, limit=32):
    return EvictCache(Qwen3Config(num_hidden_layers=layers),0,0,MODE,
        rkv_lambda=0.,smooth=False,verbose=False,reasoning_budget=1024,
        max_cache_length=limit)

def initialize(c,k):
    for layer in c.layers:layer.lazy_initialization(k)
    return [weakref.ref(x) for layer in c.layers for x in (layer,layer.k_cache,layer.v_cache)]

class MemoryRepairTests(unittest.TestCase):
    def test_release_without_cyclic_gc(self):
        enabled=gc.isenabled();gc.disable()
        try:
            c=cache();refs=initialize(c,torch.zeros(2,3,2,8));del c
            self.assertTrue(all(r() is None for r in refs))
        finally:
            if enabled:gc.enable()

    def test_storage_bound_and_overflow(self):
        l=cache(limit=8).layers[0];k=torch.zeros(2,3,2,8)
        l.prepare(k,k);pointer=l.k_cache.data_ptr();self.assertEqual(l.k_cache.shape[1],8)
        l.cumulative_length=7;l.prepare(k[:,:1],k[:,:1])
        self.assertEqual(pointer,l.k_cache.data_ptr())
        with self.assertRaises(ValueError):l.prepare(k,k)
        with self.assertRaises(ValueError):cache(limit=2).layers[0].prepare(k,k)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_cuda_exact_against_dynamic_storage(self):
        def make(limit):
            return RandomFixedRecentSharedMaskLayer(0,0,0,MODE,0.,False,False,
                reasoning_budget=8,recent_window=3,sample_id=19,max_cache_length=limit)
        a,b=make(None),make(64)
        g=torch.Generator(device='cuda').manual_seed(829)
        for t in range(40):
            n=3 if t==0 else 1
            q=torch.randn(2,n,4,8,device='cuda',dtype=torch.bfloat16,generator=g)
            k=torch.randn(2,n,2,8,device='cuda',dtype=q.dtype,generator=g)
            v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
            torch.testing.assert_close(step(a,q,k,v),step(b,q,k,v),rtol=0,atol=0)
            if t:
                for x,y in zip(a._last_selected_ids,b._last_selected_ids):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)
            torch.testing.assert_close(a.k_cache[:,:a.cumulative_length],b.k_cache[:,:b.cumulative_length],rtol=0,atol=0)
            self.assertEqual(a.selected_pairs,b.selected_pairs)
            self.assertEqual(a.causal_pairs,b.causal_pairs)
        for layer in (a,b):layer.batch_select_indices(torch.tensor([1],device='cuda'))
        torch.testing.assert_close(a.select_indices()[0],b.select_indices()[0],rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_cuda_three_full_geometry_lifetimes(self):
        # The actual 36-layer, two-rollout, 32k BF16 storage, without model weights.
        gc.collect();enabled=gc.isenabled();gc.disable()
        try:
            k=torch.zeros(2,3,8,128,device='cuda',dtype=torch.bfloat16)
            baseline=torch.cuda.memory_allocated()
            measurements=[]
            for _ in range(3):
                c=cache(layers=36,limit=32768);refs=initialize(c,k)
                live=torch.cuda.memory_allocated()
                self.assertGreaterEqual(live-baseline,9*1024**3)
                del c;torch.cuda.synchronize()
                after=torch.cuda.memory_allocated();measurements.append({'live':live,'after':after})
                self.assertTrue(all(r() is None for r in refs))
                self.assertEqual(after,baseline)
            print('FULL_GEOMETRY_MEMORY',measurements,flush=True)
        finally:
            if enabled:gc.enable()

if __name__=='__main__':unittest.main(verbosity=2)
