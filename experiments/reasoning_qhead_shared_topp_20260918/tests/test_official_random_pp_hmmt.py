"""HMMT budget guard and exact upstream parity at the matched total budgets."""
import unittest
from types import SimpleNamespace
import torch
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.official_random_pp_cache import OfficialRandomPPLayer
from kvcompress.engine.official_random_attention.cache_utils import EvictLayer as Upstream
from kvcompress.harness.my_utils import init_evict_configs


class HMMTRandomPPTests(unittest.TestCase):
    def test_hmmt_budgets_and_historical_math_guard(self):
        def args(task, k):
            return SimpleNamespace(data_name=task,token_budget=k,residual_length=64,query_window=1,
                eviction_mode='official_random_pp',eviction_interval=1,rkv_lambda=0.,smooth=False,
                n_large=-1,temperature=1.,top_p=.95,rep_seed=0,verbose=False,mask_seed=42)
        for k in [1024,2048]:
            result=init_evict_configs(args('hmmt',k))
            self.assertEqual(result['cache_config']['token_budget'],k)
        self.assertEqual(init_evict_configs(args('math',1024))['cache_config']['token_budget'],1024)
        for task,k in [('math',2048),('hmmt',4096)]:
            with self.assertRaises(ValueError):init_evict_configs(args(task,k))

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_cuda_exact_upstream_cache_and_max_hmmt_prompt(self):
        for k in [1024,2048]:
            a=OfficialRandomPPLayer(2,k,64,'official_random_pp',0.,False,False,sample_id=13,mask_seed=42)
            b=Upstream(2,k,64,'random_pp',0.,False,False)
            x=torch.arange(k+64.,device='cuda').view(1,k+64,1,1).expand(2,k+64,8,16).clone()
            for layer in [a,b]:
                layer.lazy_initialization(x[:,:648]);layer.k_cache.copy_(x);layer.v_cache.copy_(x+3)
                layer.cache_seqlens.fill_(k+64);layer.cumulative_length=k+64;layer.prompt_len=648
            before_cpu=torch.random.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state().clone()
            a._run_eviction(k+64)
            self.assertTrue(torch.equal(before_cpu,torch.random.get_rng_state()))
            self.assertTrue(torch.equal(before_cuda,torch.cuda.get_rng_state()))
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.cuda.manual_seed(_stable_hash(42,13,2));b._run_eviction(k+64)
            self.assertTrue(torch.equal(a.k_cache,b.k_cache));self.assertTrue(torch.equal(a.v_cache,b.v_cache))
            self.assertEqual(a.cache_seqlens.tolist(),[k,k])
            for row in range(2):
                for head in range(8):
                    ids=a.k_cache[row,:k,head,0].long().tolist()
                    self.assertEqual(ids,sorted(set(ids)))
                    self.assertEqual(ids[:648],list(range(648)))
                    self.assertEqual(ids[-64:],list(range(k,k+64)))
            self.assertFalse(torch.equal(a.k_cache[0,:k-64,0],a.k_cache[0,:k-64,1]))


if __name__=='__main__':unittest.main(verbosity=2)
