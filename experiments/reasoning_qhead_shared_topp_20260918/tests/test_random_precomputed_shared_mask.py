import gc
import unittest
import weakref
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache,_stable_hash
from kvcompress.engine.random_recent_shared_mask_cache import gpu_uniform_positions
from kvcompress.engine.random_precomputed_shared_mask_cache import RandomMaskPlan
from test_shared_mask_cache import step
MODE='qhead_shared_mask_random_precomputed_recent_gpu'
OLD='qhead_shared_mask_random_recent_gpu'

def cache(mode=MODE,window=16,layers=2,recent=3):
    kw=dict(rkv_lambda=0.,smooth=False,verbose=False,random_keep_ratio=.4,
        recent_window=recent,mask_seed=42,sample_id=19)
    if mode==MODE:kw['mask_precompute_window']=window
    return EvictCache(Qwen3Config(num_hidden_layers=layers),0,0,mode,**kw)

def oracle(p,step,layer,rollout,device,recent=64):
    k=(2*step+4)//5;r=min(recent,k);n=step-r
    rng=torch.Generator(device=device).manual_seed(_stable_hash(0x475055524543,42,19,step,layer,rollout))
    chosen=gpu_uniform_positions(n,k-r,rng,device)
    return torch.cat((torch.arange(p,device=device),p+chosen,torch.arange(p+step-r,p+step,device=device)))

class PrecomputedMaskTests(unittest.TestCase):
    def exact_masks(self,device,layers):
        plan=RandomMaskPlan(layers,.4,64,42,19,window=4)
        cpu=torch.random.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        for t in (1,2,63,64,159,160,161,191,192,193,255,256,257,1024,8192,32760):
            ids=plan.build_step(t,71,(0,1),device)
            self.assertEqual(ids.shape,(layers,2,71+(2*t+4)//5))
            for l in range(layers):
                for b in (0,1):torch.testing.assert_close(ids[l,b].long(),oracle(71,t,l,b,device),rtol=0,atol=0)
        self.assertTrue(torch.equal(cpu,torch.random.get_rng_state()))
        if gpu is not None:self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))

    def test_cpu_exact_masks_and_rng(self):self.exact_masks('cpu',2)

    def test_window_bound_and_original_survivor(self):
        p=RandomMaskPlan(2,.4,64,42,19,window=4);p.max_step=170
        for t in range(161,171):
            active=(0,1) if t<163 else (1,)
            for l in range(2):
                ids=p.indices(l,t,71,active,'cpu')
                for row,original in enumerate(active):torch.testing.assert_close(ids[row].long(),oracle(71,t,l,original,'cpu'),rtol=0,atol=0)
            self.assertLessEqual(len(p.blocks),4)
        self.assertEqual(p.prepared_steps,10)
        self.assertTrue(max(p.blocks)<=170)
        with self.assertRaises(ValueError):p.indices(0,171,71,(1,),'cpu')

    def test_factory_no_reference_cycle(self):
        enabled=gc.isenabled();gc.disable()
        try:
            c=cache();self.assertIs(c.mask_plan,c.layers[0].mask_plan)
            refs=[weakref.ref(c.mask_plan),weakref.ref(c.layers[0])];del c
            self.assertTrue(all(r() is None for r in refs))
        finally:
            if enabled:gc.enable()

    def test_invalid_window(self):
        for w in (0,65,-1):
            with self.assertRaises(ValueError):cache(window=w)

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_cuda_all36_layers_exact_masks_and_rng(self):self.exact_masks('cuda',36)

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_cuda_attention_gqa_filter_and_full_history(self):
        a,b=cache(OLD),cache();b.mask_plan.max_step=39
        gen=torch.Generator(device='cuda').manual_seed(173);batch=2
        for t in range(40):
            if t==20:
                ix=torch.tensor([1],device='cuda');a.batch_select_indices(ix);b.batch_select_indices(ix);batch=1
            n=3 if t==0 else 1
            for la,lb in zip(a.layers,b.layers):
                q=torch.randn(batch,n,4,8,device='cuda',dtype=torch.bfloat16,generator=gen)
                k=torch.randn(batch,n,2,8,device='cuda',dtype=q.dtype,generator=gen)
                v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=gen)
                torch.testing.assert_close(step(la,q,k,v),step(lb,q,k,v),rtol=0,atol=0)
                self.assertEqual(la.selected_pairs,lb.selected_pairs);self.assertEqual(la.causal_pairs,lb.causal_pairs)
                torch.testing.assert_close(la.k_cache[:,:la.cumulative_length],lb.k_cache[:,:lb.cumulative_length],rtol=0,atol=0)
                if t:
                    for x,y in zip(la._last_selected_ids,lb._last_selected_ids):torch.testing.assert_close(x,y.long(),rtol=0,atol=0)
        self.assertLessEqual(b.mask_plan.max_plan_bytes,16*2*2*(3+16)*4)

if __name__=='__main__':unittest.main(verbosity=2)
