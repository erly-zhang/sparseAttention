import unittest
import torch
torch.set_num_threads(2)
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.random_fixed_recent_shared_mask_cache import RandomFixedRecentSharedMaskLayer
from kvcompress.engine.random_recent_shared_mask_cache import RandomRecentSharedMaskLayer
from test_shared_mask_cache import step,reference
MODE='qhead_shared_mask_random_fixed_recent_gpu'


def make(device='cpu', reasoning=2048, prompt=71, budget=1024, recent=64):
    l=RandomFixedRecentSharedMaskLayer(0,0,0,MODE,0.,False,False,
        reasoning_budget=budget,recent_window=recent,sample_id=19)
    k=torch.zeros(2,prompt,2,8,device=device)
    l.lazy_initialization(k);l.prepare(torch.zeros(2,prompt+reasoning,2,8,device=device),k)
    n=prompt+reasoning
    l.k_cache[:,:n].zero_();l.v_cache[:,:n].zero_()
    l.cache_seqlens.fill_(n);l.cumulative_length=n;l.prompt_len=prompt
    l.update_queries(torch.zeros(2,4,1,8,device=device))
    return l


class FixedReasoningTests(unittest.TestCase):
    def test_factory_and_exact_budget(self):
        c=EvictCache(Qwen3Config(num_hidden_layers=2),0,0,MODE,rkv_lambda=0.,smooth=False,verbose=False,reasoning_budget=1024)
        self.assertTrue(all(isinstance(l,RandomFixedRecentSharedMaskLayer) for l in c.layers))
        for p in (1,71,814,1300):
            for r in (0,1,63,64,65,1023,1024,1025,2048):
                l=make(prompt=p,reasoning=r)
                for ids in l.select_indices():
                    self.assertEqual(len(ids),p+min(r,1024))
                    self.assertEqual(ids[:p].tolist(),list(range(p)))
                    self.assertEqual(ids.tolist(),sorted(set(ids.tolist())))
                    self.assertTrue(bool((ids<p+r).all()))
                    if r:self.assertEqual(ids[-min(64,r):].tolist(),list(range(p+r-min(64,r),p+r)))
                    if r<=1024:self.assertEqual(ids.tolist(),list(range(p+r)))
                self.assertEqual(l.cache_seqlens.tolist(),[p+r]*2)
                self.assertFalse(hasattr(l,'queries'));self.assertIsNone(l.top_p)

    def test_validation(self):
        for budget,recent in [(0,64),(1024,1025),(-1,0)]:
            with self.assertRaises(ValueError):make(budget=budget,recent=recent)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_cuda_rng_filter_and_inherited_sampler(self):
        a=make('cuda',reasoning=2560,prompt=71);b=make('cuda',reasoning=2560,prompt=71)
        cpu=torch.random.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        ids=a.select_indices();again=a.select_indices()
        for x,y in zip(ids,again):self.assertTrue(torch.equal(x,y))
        b.batch_select_indices(torch.tensor([1],device='cuda'))
        self.assertTrue(torch.equal(ids[1],b.select_indices()[0]))
        self.assertTrue(torch.equal(cpu,torch.random.get_rng_state()));self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
        # 40% of 2560 is exactly1024: identical stream and counts imply identical masks.
        old=RandomRecentSharedMaskLayer(0,0,0,'qhead_shared_mask_random_recent_gpu',0.,False,False,random_keep_ratio=.4,recent_window=64,sample_id=19)
        old.__dict__.update(a.__dict__);old.random_keep_ratio=.4
        for x,y in zip(ids,old.select_indices()):self.assertTrue(torch.equal(x,y))
        n=a.cumulative_length;k=a.k_cache[:,:n].clone();v=a.v_cache[:,:n].clone()
        missed=next(i for i in range(71,n-64) if i not in set(ids[0].tolist()))
        a.prepare(torch.zeros(2,24,2,8,device='cuda'),None)
        a.k_cache[:,n:n+24].zero_();a.v_cache[:,n:n+24].zero_()
        recovered=False
        for extra in range(1,25):
            a.cumulative_length=n+extra;a.cache_seqlens.fill_(n+extra)
            if missed in a.select_indices()[0].tolist():recovered=True;break
        self.assertTrue(recovered)
        torch.testing.assert_close(a.k_cache[:,:n],k,rtol=0,atol=0);torch.testing.assert_close(a.v_cache[:,:n],v,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_cuda_attention_and_exact_pair_counts(self):
        l=RandomFixedRecentSharedMaskLayer(0,0,0,MODE,0.,False,False,reasoning_budget=8,recent_window=3,sample_id=19)
        g=torch.Generator(device='cuda').manual_seed(829);ks=[];vs=[];selected=0;causal=0;ptr=[]
        for t in range(20):
            n=3 if t==0 else 1
            q=torch.randn(2,n,4,8,device='cuda',dtype=torch.bfloat16,generator=g)
            k=torch.randn(2,n,2,8,device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
            ks.append(k);vs.append(v);result=step(l,q,k,v);fk=torch.cat(ks,1);fv=torch.cat(vs,1)
            torch.testing.assert_close(l.k_cache[:,:fk.shape[1]],fk,rtol=0,atol=0)
            if t:
                torch.testing.assert_close(result.float(),reference(q,fk,fv,[x.tolist() for x in l._last_selected_ids]),rtol=.02,atol=.012)
                selected+=8*(3+min(t,8));causal+=8*(3+t);ptr.append(l._temp_k.data_ptr())
            else:
                for j in range(3):torch.testing.assert_close(result[:,j:j+1].float(),reference(q[:,j:j+1],fk,fv,[list(range(j+1))]*2),rtol=.02,atol=.012)
                selected+=48;causal+=48
        self.assertEqual(l.selected_pairs,selected);self.assertEqual(l.causal_pairs,causal);self.assertEqual(len(set(ptr)),1)
        # Real1024 budget boundary and full history beyond it.
        long=make('cuda',prompt=71,reasoning=1100);long.k_cache=long.k_cache.bfloat16();long.v_cache=long.v_cache.bfloat16()
        q=torch.ones(2,1,4,8,device='cuda',dtype=torch.bfloat16);k=torch.ones(2,1,2,8,device='cuda',dtype=q.dtype)
        step(long,q,k,k)
        self.assertEqual(long.cache_seqlens.tolist(),[1172,1172])
        self.assertEqual([len(x) for x in long._last_selected_ids],[1095,1095])
        torch.testing.assert_close(long._last_selected_ids[0][-64:],torch.arange(1108,1172,device='cuda'))

if __name__=='__main__':unittest.main(verbosity=2)
