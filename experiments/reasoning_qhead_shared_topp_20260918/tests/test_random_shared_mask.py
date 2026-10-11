"""Correctness of random positions, common per-step head IDs and full KV semantics."""
import math
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.random_shared_mask_cache import RandomSharedMaskLayer,keep_count,nominal_head,uniform_positions
from test_shared_mask_cache import step,reference

MODE='qhead_shared_mask_random'

def make(layer=0,device='cpu',length=17,prompt=2,ratio=.4):
    obj=RandomSharedMaskLayer(layer,0,0,MODE,0.,False,False,random_keep_ratio=ratio,mask_seed=42,sample_id=19)
    key=torch.zeros(2,prompt,2,8,device=device)
    obj.lazy_initialization(key)
    obj.prepare(torch.zeros(2,length,2,8,device=device),key)
    obj.cache_seqlens.fill_(length);obj.prompt_len=prompt;obj.cumulative_length=length
    obj.update_queries(torch.zeros(2,4,1,8,device=device))
    return obj

class RandomSharedMaskTests(unittest.TestCase):
    def test_factory_and_no_query_scoring(self):
        c=EvictCache(Qwen3Config(num_hidden_layers=3),0,0,MODE,rkv_lambda=0.,smooth=False,verbose=False,random_keep_ratio=.4)
        self.assertTrue(all(isinstance(l,RandomSharedMaskLayer) for l in c.layers))
        l=make();self.assertIsNone(l.query_window);self.assertIsNone(l.top_p)
        with patch('torch.softmax',side_effect=AssertionError('random selector must not score QK')):
            before=[x.clone() for x in l.select_indices()]
            l.update_queries(torch.full((2,4,100,8),float('nan')))
            after=l.select_indices()
        self.assertFalse(hasattr(l,'queries'))
        for a,b in zip(before,after):self.assertTrue(torch.equal(a,b))
        with self.assertRaises(RuntimeError):l._run_eviction(17)

    def test_budget_prompt_uniqueness_short_and_large(self):
        for n in [0,1,2,4,5,6,10,11,1001,32700]:
            l=make(length=n+2);old=l.k_cache.clone();ids=l.select_indices()
            for x in ids:
                self.assertEqual(x.numel(),2+(2*n+4)//5)
                self.assertEqual(x[:2].tolist(),[0,1])
                self.assertEqual(x.tolist(),sorted(set(x.tolist())))
                self.assertLess(int(x[-1]),n+2)
            torch.testing.assert_close(l.k_cache,old,rtol=0,atol=0)
        self.assertEqual(keep_count(100,.07),7)
        self.assertEqual(make(ratio=1.).select_indices()[0].tolist(),list(range(17)))

    def test_common_head_independent_layers_reproducible_rng(self):
        cpu=torch.random.get_rng_state().clone()
        cuda=torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        masks=[];heads=[]
        for layer in range(5):
            l=make(layer=layer,length=102);masks.append(l.select_indices()[0].tolist());heads.append(l._rep_head)
        self.assertEqual(len(set(heads)),1);self.assertEqual(len(set(map(tuple,masks))),5)
        self.assertGreater(len({nominal_head(0,19,t,32) for t in range(30)}),1)
        for layer in reversed(range(5)):
            self.assertEqual(make(layer=layer,length=102).select_indices()[0].tolist(),masks[layer])
        torch.testing.assert_close(torch.random.get_rng_state(),cpu,rtol=0,atol=0)
        if cuda is not None:torch.testing.assert_close(torch.cuda.get_rng_state(),cuda,rtol=0,atol=0)

    def test_batch_filter_preserves_survivor_stream(self):
        a=make(length=102);b=make(length=102)
        b.batch_select_indices(torch.tensor([1]))
        self.assertEqual(a.select_indices()[1].tolist(),b.select_indices()[0].tolist())
        self.assertEqual(b._batch_ids,[1])

    def test_uniform_marginal_frequency(self):
        counts=torch.zeros(10)
        for seed in range(3000):counts[uniform_positions(10,4,seed,19,15,0,0,'cpu')]+=1
        self.assertLess(float((counts/3000-.4).abs().max()),.035)

    @unittest.skipUnless(torch.cuda.is_available(),'requires allocated CUDA GPU')
    def test_cpu_gpu_identical_subsets(self):
        a=make(length=1002);b=make(device='cuda',length=1002)
        self.assertEqual([x.tolist() for x in a.select_indices()],[x.tolist() for x in b.select_indices()])
        self.assertEqual(a._rep_head,b._rep_head)

    @unittest.skipUnless(torch.cuda.is_available(),'requires allocated CUDA GPU')
    def test_kernel_reference_append_only_long_history_and_pairs(self):
        for ratio in (.4,1.):
            l=RandomSharedMaskLayer(0,0,0,MODE,0.,False,False,random_keep_ratio=ratio,sample_id=19)
            gen=torch.Generator(device='cuda').manual_seed(313)
            ks=[];vs=[];sp=0;cp=0
            for t in range(12):
                n=3 if t==0 else 1
                q=torch.randn(2,n,4,8,device='cuda',dtype=torch.bfloat16,generator=gen)
                k=torch.randn(2,n,2,8,device='cuda',dtype=torch.bfloat16,generator=gen)
                v=torch.randn(2,n,2,8,device='cuda',dtype=torch.bfloat16,generator=gen)
                ks.append(k);vs.append(v);out=step(l,q,k,v)
                full_k=torch.cat(ks,1);full_v=torch.cat(vs,1);length=full_k.shape[1]
                torch.testing.assert_close(l.k_cache[:,:length],full_k,rtol=0,atol=0)
                torch.testing.assert_close(l.v_cache[:,:length],full_v,rtol=0,atol=0)
                self.assertEqual(l.cache_seqlens.tolist(),[length,length]);self.assertFalse(hasattr(l,'queries'))
                if t==0:
                    for i in range(n):torch.testing.assert_close(out[:,i:i+1].float(),reference(q[:,i:i+1],full_k,full_v,[list(range(i+1))]*2),rtol=.02,atol=.012)
                    sp+=2*4*6;cp+=2*4*6
                else:
                    torch.testing.assert_close(out.float(),reference(q,full_k,full_v,[x.tolist() for x in l._last_selected_ids]),rtol=.02,atol=.012)
                    sp+=2*4*(3+keep_count(t,ratio));cp+=2*4*length
            self.assertEqual(l.selected_pairs,sp);self.assertEqual(l.causal_pairs,cp)
        l=make(device='cuda',length=1100,prompt=16)
        l.k_cache=l.k_cache.bfloat16();l.v_cache=l.v_cache.bfloat16()
        q=torch.ones(2,1,4,8,device='cuda',dtype=torch.bfloat16);k=torch.ones(2,1,2,8,device='cuda',dtype=torch.bfloat16)
        old=l.k_cache[:,:1100].clone();step(l,q,k,k)
        torch.testing.assert_close(l.k_cache[:,:1100],old,rtol=0,atol=0)
        self.assertEqual(l.cache_seqlens.tolist(),[1101,1101])
        tiny=RandomSharedMaskLayer(0,0,0,MODE,0.,False,False,random_keep_ratio=.4)
        step(tiny,q,k,k);step(tiny,q,k,k)
        self.assertEqual(tiny.prompt_len,1);self.assertEqual(tiny.cache_seqlens.tolist(),[2,2])

if __name__=='__main__':unittest.main(verbosity=2)
