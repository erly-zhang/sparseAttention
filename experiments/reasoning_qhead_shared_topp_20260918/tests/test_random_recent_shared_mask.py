"""Budgeted recent protection, GPU sampler, and fused gather/reference regression."""
import unittest
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.random_shared_mask_cache import keep_count
from kvcompress.engine.random_recent_shared_mask_cache import RandomRecentSharedMaskLayer,gpu_uniform_positions,random_priority_keys,fused_gather_kv,_gather_kv
from test_shared_mask_cache import step,reference
MODE='qhead_shared_mask_random_recent_gpu'

def make(device='cpu',length=200,prompt=2,layer=0,recent=64):
    l=RandomRecentSharedMaskLayer(layer,0,0,MODE,0.,False,False,random_keep_ratio=.4,recent_window=recent,sample_id=19)
    k=torch.zeros(2,prompt,2,8,device=device)
    l.lazy_initialization(k);l.prepare(torch.zeros(2,length,2,8,device=device),k)
    l.k_cache[:,:length].zero_();l.v_cache[:,:length].zero_()
    l.cache_seqlens.fill_(length);l.cumulative_length=length;l.prompt_len=prompt
    l.update_queries(torch.zeros(2,4,1,8,device=device))
    return l

class RandomRecentTests(unittest.TestCase):
    def test_factory_budget_short_and_long(self):
        cache=EvictCache(Qwen3Config(num_hidden_layers=2),0,0,MODE,rkv_lambda=0.,smooth=False,verbose=False,random_keep_ratio=.4,recent_window=64)
        self.assertTrue(all(isinstance(l,RandomRecentSharedMaskLayer) for l in cache.layers))
        for r in (0,1,2,5,63,64,159,160,161,999,1100):
            l=make(length=2+r);ids=l.select_indices();k=keep_count(r,.4);recent=min(64,k)
            for x in ids:
                self.assertEqual(x.numel(),2+k);self.assertEqual(x[:2].tolist(),[0,1])
                self.assertEqual(x.tolist(),sorted(set(x.tolist())))
                if recent:self.assertEqual(x[-recent:].tolist(),list(range(2+r-recent,2+r)))
            self.assertFalse(hasattr(l,'queries'));self.assertIsNone(l.query_window)

    def test_priority_tie_break(self):
        # Equal primary scores use independent secondary keys; exact duplicate
        # pairs finally use positions, without overflowing signed int64.
        a=torch.tensor([1,1,2,1],dtype=torch.int64);b=torch.tensor([3,5,0,5],dtype=torch.int64)
        keys=random_priority_keys(a,b,torch.arange(4),2,29)
        self.assertEqual(keys.argsort(descending=True).tolist(),[2,3,1,0])
        maximal=random_priority_keys(torch.tensor([2**32-1]),torch.tensor([2**16-1]),torch.tensor([2**15-1]),15,16)
        self.assertEqual(int(maximal[0]),2**63-1)

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_cuda_rng_private_and_batch_reproducible(self):
        cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
        a=make('cuda',layer=0);b=make('cuda',layer=1);c=make('cuda',layer=0)
        ai=a.select_indices();bi=b.select_indices()
        self.assertEqual(a._rep_head,b._rep_head)
        self.assertFalse(torch.equal(ai[0],bi[0]))
        c.batch_select_indices(torch.tensor([1],device='cuda'))
        self.assertTrue(torch.equal(ai[1],c.select_indices()[0]))
        self.assertTrue(torch.equal(a.select_indices()[0],ai[0]))
        torch.testing.assert_close(torch.random.get_rng_state(),cpu,rtol=0,atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(),cuda,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_cuda_uniform_marginals(self):
        counts=torch.zeros(10,device='cuda');gen=torch.Generator(device='cuda')
        for seed in range(2000):
            gen.manual_seed(seed)
            ids=gpu_uniform_positions(10,4,gen,'cuda');counts[ids]+=1
        self.assertLess(float((counts/2000-.4).abs().max()),.05)

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_fused_gather_and_runtime_count(self):
        gen=torch.Generator(device='cuda').manual_seed(821)
        k=torch.randn(2,600,2,8,device='cuda',dtype=torch.bfloat16,generator=gen);v=k+3
        tk=torch.full_like(k,float('nan'));tv=torch.full_like(k,float('nan'))
        for n in (1,31,57,63,83,127,199):
            ids=torch.stack([torch.randperm(600,device='cuda',generator=gen)[:n].sort().values for _ in range(2)])
            fused_gather_kv(k,v,ids,tk,tv)
            for b in range(2):
                torch.testing.assert_close(tk[b,:n],k[b,ids[b]],rtol=0,atol=0)
                torch.testing.assert_close(tv[b,:n],v[b,ids[b]],rtol=0,atol=0)
        # N is a runtime argument, preventing a fresh compilation for every size.
        self.assertFalse(_gather_kv.params[5].is_constexpr)

    @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
    def test_attention_reference_append_reuse_and_filter(self):
        l=RandomRecentSharedMaskLayer(0,0,0,MODE,0.,False,False,random_keep_ratio=.4,recent_window=3,sample_id=19)
        gen=torch.Generator(device='cuda').manual_seed(821);ks=[];vs=[];selected=0;causal=0
        addresses=[]
        for t in range(20):
            n=3 if t==0 else 1
            q=torch.randn(2,n,4,8,device='cuda',dtype=torch.bfloat16,generator=gen)
            k=torch.randn(2,n,2,8,device='cuda',dtype=torch.bfloat16,generator=gen);v=torch.randn(k.shape,device='cuda',dtype=k.dtype,generator=gen)
            ks.append(k);vs.append(v);out=step(l,q,k,v);fk=torch.cat(ks,1);fv=torch.cat(vs,1);length=fk.shape[1]
            torch.testing.assert_close(l.k_cache[:,:length],fk,rtol=0,atol=0);torch.testing.assert_close(l.v_cache[:,:length],fv,rtol=0,atol=0)
            self.assertEqual(l.cache_seqlens.tolist(),[length,length]);self.assertFalse(hasattr(l,'queries'))
            if t:
                torch.testing.assert_close(out.float(),reference(q,fk,fv,[x.tolist() for x in l._last_selected_ids]),rtol=.02,atol=.012)
                selected+=8*(3+keep_count(t,.4));causal+=8*length;addresses.append(l._temp_k.data_ptr())
            else:
                for j in range(3):torch.testing.assert_close(out[:,j:j+1].float(),reference(q[:,j:j+1],fk,fv,[list(range(j+1))]*2),rtol=.02,atol=.012)
                selected+=48;causal+=48
        self.assertEqual(len(set(addresses)),1);self.assertEqual(l.gather_allocations,1)
        self.assertEqual(l.selected_pairs,selected);self.assertEqual(l.causal_pairs,causal)
        old=l.k_cache[1,:length].clone();l.batch_select_indices(torch.tensor([1],device='cuda'))
        torch.testing.assert_close(l.k_cache[0,:length],old,rtol=0,atol=0)
        q=q[1:];k=k[1:];v=v[1:];out=step(l,q,k,v)
        torch.testing.assert_close(out.float(),reference(q,torch.cat((fk[1:],k),1),torch.cat((fv[1:],v),1),[l._last_selected_ids[0].tolist()]),rtol=.02,atol=.012)
        # Force growth of persistent and temporary storage; still no 1024 cap.
        long=make('cuda',length=1100,prompt=16);long.k_cache=long.k_cache.bfloat16();long.v_cache=long.v_cache.bfloat16()
        q=torch.ones(2,1,4,8,device='cuda',dtype=torch.bfloat16);k=torch.ones(2,1,2,8,device='cuda',dtype=torch.bfloat16)
        step(long,q,k,k);self.assertEqual(long.cache_seqlens.tolist(),[1101,1101])
        tiny=RandomRecentSharedMaskLayer(0,0,0,MODE,0.,False,False,random_keep_ratio=.4,recent_window=64)
        step(tiny,q,k,k);step(tiny,q,k,k);self.assertEqual(tiny.cache_seqlens.tolist(),[2,2])

if __name__=='__main__':unittest.main(verbosity=2)
