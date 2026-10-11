import unittest
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache,_stable_hash
from kvcompress.engine.random_compact_precomputed_cache import CompactRandomMaskPlan
from kvcompress.engine.random_recent_shared_mask_cache import gpu_uniform_positions
from test_shared_mask_cache import step
OLD='qhead_shared_random_compact_recent_gpu'
NEW='qhead_shared_random_compact_precomputed_gpu'

def cache(mode,budget=8,interval=4,recent=2,layers=2):
    return EvictCache(Qwen3Config(num_hidden_layers=layers),0,0,mode,rkv_lambda=0.,smooth=False,verbose=False,
        reasoning_budget=budget,eviction_interval=interval,recent_window=recent,mask_seed=42,sample_id=19,track_positions=True)

def expected(p,step,layer,b,device,budget=1024,interval=64,recent=64):
    rng=torch.Generator(device=device).manual_seed(_stable_hash(0x475055524543,42,19,step,layer,b))
    older=budget+interval-recent;count=budget-recent
    chosen=gpu_uniform_positions(older,count,rng,device)
    return torch.cat((torch.arange(p,device=device),p+chosen,torch.arange(p+budget+interval-recent,p+budget+interval,device=device)))

class CompactPrecomputedTests(unittest.TestCase):
    def test_all_event_masks_and_private_rng(self):
        state=torch.random.get_rng_state().clone();plan=CompactRandomMaskPlan(3,8,4,2,42,19,chunk_events=3)
        plan.precompute_all(5,59,2,'cpu')
        self.assertEqual(plan.steps,tuple(range(12,60,4)))
        for t in plan.steps:
            for l in range(3):
                for b in range(2):torch.testing.assert_close(plan.indices(l,t,5,[b])[0],expected(5,t,l,b,'cpu',8,4,2),rtol=0,atol=0)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        with self.assertRaises(ValueError):plan.indices(0,60,5,[0])
        with self.assertRaises(RuntimeError):plan.precompute_all(5,59,2,'cpu')

    def test_empty_plan_short_generation(self):
        plan=CompactRandomMaskPlan(2,1024,64);plan.precompute_all(71,1087,2,'cpu')
        self.assertEqual(plan.ids.numel(),0);self.assertEqual(plan.prepared_masks,0)

    def test_cpu_exact_persistent_positions_after_filter(self):
        a,b=cache(OLD),cache(NEW);b.compact_mask_plan.precompute_all(3,55,2,'cpu')
        for c in (a,b):
            for l in c.layers:
                l.lazy_initialization(torch.zeros(2,3,2,8));l.k_cache[:,:3]=1;l.v_cache[:,:3]=2
                l._cached_tokens=3;l.cumulative_length=3;l.cache_seqlens.fill_(3);l.compact_positions[:,:3]=torch.arange(3)
        for t in range(1,56):
            if t==17:
                a.batch_select_indices(torch.tensor([1]));b.batch_select_indices(torch.tensor([1]))
            for la,lb in zip(a.layers,b.layers):
                for l in (la,lb):
                    n=l._cached_tokens;l.k_cache[:,n]=t;l.v_cache[:,n]=t+1
                    l.compact_positions[:,n]=l.cumulative_length
                    l._cached_tokens+=1;l.cumulative_length+=1;l.cache_seqlens+=1
                    if l._cached_tokens==l.capacity:l.compact()
                torch.testing.assert_close(la.k_cache[:,:la._cached_tokens],lb.k_cache[:,:lb._cached_tokens],rtol=0,atol=0)
                torch.testing.assert_close(la.compact_positions,lb.compact_positions,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_cuda_all36layers_real_budget_and_rng(self):
        plan=CompactRandomMaskPlan(36,1024,64,64,42,19)
        cpu=torch.random.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        plan.precompute_all(71,32768-71-1,2,'cuda')
        self.assertEqual(plan.steps[0],1088);self.assertLessEqual(plan.steps[-1],32696)
        for event in (0,1,16,len(plan.steps)-1):
            for l in range(36):
                for b in (0,1):torch.testing.assert_close(plan.ids[event,l,b],expected(71,plan.steps[event],l,b,'cuda'),rtol=0,atol=0)
        self.assertTrue(torch.equal(cpu,torch.random.get_rng_state()));self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
        print('FULL_COMPACT_PLAN_BYTES',plan.ids.numel()*8,'EVENTS',len(plan.steps),flush=True)

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_cuda_full_attention_and_eviction_exactness(self):
        a,b=cache(OLD),cache(NEW);b.compact_mask_plan.precompute_all(3,55,2,'cuda')
        gen=torch.Generator(device='cuda').manual_seed(121);batch=2
        for t in range(56):
            if t==17:
                ix=torch.tensor([1],device='cuda');a.batch_select_indices(ix);b.batch_select_indices(ix);batch=1
            n=3 if t==0 else 1
            for la,lb in zip(a.layers,b.layers):
                q=torch.randn(batch,n,4,8,device='cuda',dtype=torch.bfloat16,generator=gen)
                k=torch.randn(batch,n,2,8,device='cuda',dtype=q.dtype,generator=gen);v=k+2
                torch.testing.assert_close(step(la,q,k,v),step(lb,q,k,v),rtol=0,atol=0)
                torch.testing.assert_close(la.k_cache[:,:la._cached_tokens],lb.k_cache[:,:lb._cached_tokens],rtol=0,atol=0)
                torch.testing.assert_close(la.compact_positions,lb.compact_positions,rtol=0,atol=0)
                self.assertEqual(la.selected_pairs,lb.selected_pairs);self.assertEqual(la.causal_pairs,lb.causal_pairs)

if __name__=='__main__':unittest.main(verbosity=2)
