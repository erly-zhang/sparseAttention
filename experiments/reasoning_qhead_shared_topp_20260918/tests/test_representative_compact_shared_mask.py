import unittest
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.representative_compact_recent_cache import RepresentativeCompactRecentLayer
from kvcompress.engine.random_recent_shared_mask_cache import gpu_uniform_positions
from test_shared_mask_cache import step,reference
MODE='qhead_shared_representative_compact_recent_gpu'
torch.set_num_threads(2)

def make(budget=8,recent=3,interval=4,device='cpu',prompt=5):
    l=RepresentativeCompactRecentLayer(0,0,0,MODE,0.,False,False,reasoning_budget=budget,
        recent_window=recent,eviction_interval=interval,sample_id=19,track_positions=True)
    l.lazy_initialization(torch.zeros(2,prompt,2,8,device=device))
    l.update_queries(torch.ones(2,4,1,8,device=device))
    return l

class RepresentativeCompactTests(unittest.TestCase):
    def test_factory_validation(self):
        c=EvictCache(Qwen3Config(num_hidden_layers=2),0,0,MODE,rkv_lambda=0.,smooth=False,verbose=False,reasoning_budget=1024,eviction_interval=64,query_window=1)
        self.assertTrue(all(isinstance(l,RepresentativeCompactRecentLayer) for l in c.layers))
        for b,r,i in [(0,0,4),(8,9,4),(8,3,0)]:
            with self.assertRaises(ValueError):make(b,r,i)

    def test_compaction_budget_preservation_rng(self):
        for p in (1,71,1300):
            a=make(1024,64,64,prompt=p);b=make(1024,64,64,prompt=p)
            n=a.capacity
            for l in (a,b):
                l.k_cache.copy_(torch.arange(n)[None,:,None,None]);l.v_cache.copy_(l.k_cache*2)
                l.compact_positions[:]=torch.arange(n);l._cached_tokens=n;l.cumulative_length=n;l.cache_seqlens.fill_(n)
            rng=torch.random.get_rng_state().clone();ptr=a.k_cache.data_ptr()
            a.compact();b.compact()
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertTrue(torch.equal(a.k_cache[:,:p+1024],b.k_cache[:,:p+1024]))
            self.assertEqual(a.k_cache.data_ptr(),ptr);self.assertEqual(a.capacity,p+1088)
            self.assertEqual(a.cache_seqlens.tolist(),[p+1024]*2)
            ids=a.compact_positions[:,:p+1024].clone()
            for row in ids:
                self.assertEqual(row[:p].tolist(),list(range(p)))
                self.assertEqual(row[-64:].tolist(),list(range(n-64,n)))
                self.assertTrue(bool((row[1:]>row[:-1]).all()))
            torch.testing.assert_close(a.k_cache[:,:p+1024,0,0],ids.float())
            torch.testing.assert_close(a.v_cache[:,:p+1024,0,0],ids.float()*2)
            # Finished-row filtering preserves original rollout identity and sampling.
            b.batch_select_indices(torch.tensor([1]))
            for l in (a,b):
                l.k_cache[:,p+1024:n]=0;l.v_cache[:,p+1024:n]=0
                l.compact_positions[:,p+1024:n]=torch.arange(n,n+64)
                l._cached_tokens=n;l.cumulative_length+=64;l.cache_seqlens.fill_(n)
                l.compact()
            self.assertTrue(torch.equal(a.compact_positions[1],b.compact_positions[0]))
            self.assertTrue(set(a.compact_positions[0,:p+1024].tolist())<=set(ids[0].tolist())|set(range(n,n+64)))

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_cuda_attention_compaction_and_filter(self):
        l=make(device='cuda');l.k_cache=l.k_cache.bfloat16();l.v_cache=l.v_cache.bfloat16()
        l._temp_k=l._temp_k.bfloat16();l._temp_v=l._temp_v.bfloat16();l.queries=l.queries.bfloat16()
        g=torch.Generator(device='cuda').manual_seed(913);fk=fv=None;selected=causal=0;prior=None
        for t in range(43):
            n=5 if t==0 else 1;b=len(l._batch_ids)
            q=torch.randn(b,n,4,8,device='cuda',dtype=torch.bfloat16,generator=g)
            k=torch.randn(b,n,2,8,device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
            fk=k if fk is None else torch.cat((fk,k),1);fv=v if fv is None else torch.cat((fv,v),1)
            before=l._cached_tokens
            ids=[list(range(5)) for _ in range(b)] if t==0 else [row[:before].tolist()+[4+t] for row in l.compact_positions]
            state=torch.cuda.get_rng_state().clone();res=step(l,q,k,v)
            self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))
            if t==0:
                for j in range(5):torch.testing.assert_close(res[:,j:j+1].float(),reference(q[:,j:j+1],fk,fv,[list(range(j+1))]*b),rtol=.02,atol=.012)
                selected+=b*4*15;causal+=b*4*15
            else:
                torch.testing.assert_close(res.float(),reference(q,fk,fv,ids),rtol=.02,atol=.012)
                selected+=b*4*(before+1);causal+=b*4*(5+t)
            self.assertLessEqual(l._cached_tokens,17);self.assertEqual(l.k_cache.shape[1],17)
            self.assertEqual(l.get_seq_length(),5+t)
            for row in l.compact_positions[:,:l._cached_tokens]:
                self.assertEqual(row[:5].tolist(),list(range(5)))
                self.assertTrue(bool((row[1:]>row[:-1]).all()))
            for j in range(b):
                torch.testing.assert_close(l.k_cache[j,:l._cached_tokens],fk[j,l.compact_positions[j,:l._cached_tokens]],rtol=0,atol=0)
                torch.testing.assert_close(l.v_cache[j,:l._cached_tokens],fv[j,l.compact_positions[j,:l._cached_tokens]],rtol=0,atol=0)
            if t==20:
                l.batch_select_indices(torch.tensor([1],device='cuda'));fk=fk[1:];fv=fv[1:]
        self.assertEqual(l.selected_pairs,selected);self.assertEqual(l.causal_pairs,causal)
        self.assertEqual(l.selection_events,8)

    def test_topk_probability_gqa_and_ties(self):
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            l=make(device=device);n=l.capacity;p=l.prompt_len
            g=torch.Generator(device=device).manual_seed(711)
            l.k_cache.copy_(torch.randn(l.k_cache.shape,device=device,generator=g))
            l.update_queries(torch.randn(2,4,1,8,device=device,generator=g))
            l._cached_tokens=n;l.cumulative_length=n;l.cache_seqlens.fill_(n)
            state=torch.random.get_rng_state().clone()
            chosen=l.select_indices();self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
            for b in range(2):
                q=l.queries[b,l._rep_head,0];kv=l._rep_head//2
                logits=torch.stack([torch.dot(q,l.k_cache[b,j,kv]) for j in range(n)])/(8**.5)
                probs=torch.softmax(logits.float(),0)
                ranked=sorted(range(p,n-3),key=lambda j:(-float(probs[j]),j))[:5]
                expected=list(range(p))+sorted(ranked)+list(range(n-3,n))
                self.assertEqual(chosen[b].tolist(),expected)
            # All equal scores: tie resolution uses earlier original position.
            l.queries.zero_();chosen=l.select_indices()
            self.assertEqual(chosen[0].tolist(),list(range(p+5))+list(range(n-3,n)))
            # Protection does not consume older-reasoning slots, even when
            # protected locations have the highest score.
            l.queries.fill_(.01);l.k_cache.zero_();l.k_cache[:,:p]=100;l.k_cache[:,n-3:n]=100
            chosen=l.select_indices()
            self.assertEqual(chosen[0].tolist(),list(range(p+5))+list(range(n-3,n)))

    def test_only_current_query_and_private_representative(self):
        l=make();q=torch.arange(2*4*9*8,dtype=torch.float32).reshape(2,4,9,8)
        l.update_queries(q);self.assertEqual(tuple(l.queries.shape),(2,4,1,8))
        torch.testing.assert_close(l.queries,q[:,:,-1:]);q.zero_()
        self.assertTrue(bool(l.queries.any()))
        l.update_queries(torch.ones(2,4,1,8));self.assertTrue(bool((l.queries==1).all()))
        with self.assertRaises(ValueError):
            RepresentativeCompactRecentLayer(0,0,0,MODE,0.,False,False,reasoning_budget=8,query_window=16)

if __name__=='__main__':unittest.main(verbosity=2)
