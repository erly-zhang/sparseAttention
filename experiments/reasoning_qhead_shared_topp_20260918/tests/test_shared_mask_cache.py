"""Tests for the requested full-history/shared-mask semantics, not cache eviction."""
import math
import os
import unittest
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.shared_mask_cache import SharedMaskLayer

os.environ['QHEAD_DEBUG']='1'
os.environ.pop('QHEAD_LOG',None)
torch.set_num_threads(2)


def make(device='cpu',p=.95,batch=2,prompt=2,length=6):
    layer=SharedMaskLayer(0,0,0,'qhead_shared_mask_topp',0.,False,False,top_p=p,sample_id=17)
    k=torch.zeros(batch,prompt,2,8,device=device)
    layer.lazy_initialization(k)
    layer.prepare(torch.zeros(batch,length,2,8,device=device),k)
    layer.cache_seqlens.fill_(length);layer.prompt_len=prompt;layer.cumulative_length=length
    layer.update_queries(torch.zeros(batch,4,1,8,device=device))
    return layer


def step(layer,q,k,v):
    from kvcompress.engine.modify_forward import cache_attention
    layer.update_queries(q.transpose(1,2))
    return cache_attention(layer,q,k,v,1/math.sqrt(q.shape[-1]))


def reference(q,k,v,ids):
    out=torch.empty_like(q,dtype=torch.float32)
    for b,idx in enumerate(ids):
        for h in range(q.shape[2]):
            logits=q[b,0,h].float() @ k[b,idx,h//2].float().T/math.sqrt(q.shape[-1])
            out[b,0,h]=torch.softmax(logits,-1) @ v[b,idx,h//2].float()
    return out


class SharedMaskTests(unittest.TestCase):
    def test_factory_and_forbidden_eviction(self):
        for name in ('qhead_shared_topp','qhead_shared_mask_topp'):
            c=EvictCache(Qwen3Config(num_hidden_layers=2),0,0,name,rkv_lambda=0.,smooth=False,verbose=False)
            self.assertTrue(all(isinstance(l,SharedMaskLayer) for l in c.layers))
            with self.assertRaises(RuntimeError):c.layers[0]._run_eviction(10)
            with self.assertRaises(RuntimeError):c.layers[0].post_attention_update(1)
        with self.assertRaises(ValueError):
            SharedMaskLayer(0,1024,0,'qhead_shared_topp',0.,False,False)

    def test_unselected_key_can_return_without_cache_mutation(self):
        l=make()
        l.k_cache[:,2,:,0]=6.;l.k_cache[:,3,:,0]=-6.
        l.queries[:,:,:,0]=math.sqrt(8)
        original_k=l.k_cache.clone();original_v=l.v_cache.clone();original_lengths=l.cache_seqlens.clone()
        first=l.select_indices()
        self.assertIn(2,first[0].tolist());self.assertNotIn(3,first[0].tolist())
        l.queries.neg_()
        second=l.select_indices()
        self.assertIn(3,second[0].tolist())
        torch.testing.assert_close(l.k_cache,original_k,rtol=0,atol=0)
        torch.testing.assert_close(l.v_cache,original_v,rtol=0,atol=0)
        torch.testing.assert_close(l.cache_seqlens,original_lengths,rtol=0,atol=0)
        self.assertEqual(l.get_seq_length(),6)

    def test_independent_selector_and_invalid_slots(self):
        l=make(p=.95)
        weights=torch.tensor([[.01,.01,.8,.1,.07,.03],[.01,.01,.25,.25,.25,.25]])
        l.k_cache[:,:6,:,0]=weights.log()[:,:,None]
        l.k_cache[:,6:]=float('nan')
        l.queries[:,:,:,0]=math.sqrt(8)
        ids=l.select_indices()
        for b in range(2):
            probs=weights[b,2:].double();probs/=probs.sum()
            order=sorted(range(4),key=lambda i:-float(probs[i]))
            mass=0.;selected=[]
            for i in order:
                selected.append(i+2);mass+=float(probs[i])
                if mass>=.95:break
            self.assertEqual(ids[b].tolist(),sorted([0,1]+selected))
        self.assertNotEqual(ids[0].numel(),ids[1].numel())
        self.assertEqual(l.cache_seqlens.tolist(),[6,6])

    def test_query_window_real_counts_and_ring_wrap(self):
        for window in (16,64):
            l=SharedMaskLayer(0,0,0,'qhead_shared_mask_topp',0.,False,False,query_window=window)
            l.update_queries(torch.arange(3.).view(1,1,3,1).expand(1,4,3,8))
            self.assertEqual(l._q_valid,3)
            for position in range(3,window+24):
                l.update_queries(torch.full((1,4,1,8),float(position)))
                expected=list(range(max(0,position-window+1),position+1))
                actual=sorted(l.queries[0,0,:l._q_valid,0].tolist())
                self.assertEqual(actual,expected)
            l=SharedMaskLayer(0,0,0,'qhead_shared_mask_topp',0.,False,False,query_window=window)
            l.update_queries(torch.arange(100.).view(1,1,100,1).expand(1,4,100,8))
            self.assertEqual(l._q_valid,window)
            self.assertEqual(l.queries[0,0,:,0].tolist(),list(range(100-window,100)))

    def test_rep_rng_is_local_and_reproducible(self):
        a=make();b=make()
        state=torch.random.get_rng_state().clone()
        for _ in range(8):
            a.select_indices();b.select_indices()
            self.assertEqual(a._rep_head,b._rep_head)
        torch.testing.assert_close(torch.random.get_rng_state(),state)

    @unittest.skipUnless(torch.cuda.is_available(),'requires allocated CUDA GPU')
    def test_current_step_sparse_output_and_append_only_history(self):
        for p in (.5,1.):
            l=SharedMaskLayer(0,0,0,'qhead_shared_mask_topp',0.,False,False,top_p=p)
            gen=torch.Generator(device='cuda').manual_seed(821)
            ks=[];vs=[]
            saw_sparse=False
            for t in range(10):
                n=3 if t==0 else 1
                q=torch.randn(2,n,4,8,device='cuda',dtype=torch.bfloat16,generator=gen)
                k=torch.randn(2,n,2,8,device='cuda',dtype=torch.bfloat16,generator=gen)
                v=torch.randn(2,n,2,8,device='cuda',dtype=torch.bfloat16,generator=gen)
                ks.append(k);vs.append(v)
                out=step(l,q,k,v)
                full_k=torch.cat(ks,1);full_v=torch.cat(vs,1);length=full_k.shape[1]
                torch.testing.assert_close(l.k_cache[:,:length],full_k,rtol=0,atol=0)
                torch.testing.assert_close(l.v_cache[:,:length],full_v,rtol=0,atol=0)
                self.assertEqual(l.cache_seqlens.tolist(),[length,length])
                self.assertEqual(l.get_seq_length(),length)
                if t==0:
                    for i in range(n):
                        expected=reference(q[:,i:i+1],full_k,full_v,[list(range(i+1))]*2)
                        torch.testing.assert_close(out[:,i:i+1].float(),expected,rtol=.02,atol=.012)
                else:
                    ids=[x.tolist() for x in l._last_selected_ids]
                    expected=reference(q,full_k,full_v,ids)
                    torch.testing.assert_close(out.float(),expected,rtol=.02,atol=.012)
                    if any(len(x)<length for x in ids):
                        saw_sparse=True
                        dense=reference(q,full_k,full_v,[list(range(length))]*2)
                        self.assertGreater(float((out.float()-dense).abs().max()),.001)
            if p<1:
                self.assertTrue(saw_sparse)
                self.assertLess(l.selected_pairs,l.causal_pairs)
            else:self.assertEqual(l.selected_pairs,l.causal_pairs)
            # Batch filtering removes finished sequences, not token positions of survivors.
            original=l.k_cache[1,:length].clone()
            l.batch_select_indices(torch.tensor([1],device='cuda'))
            self.assertEqual(l._batch_ids,[1]);self.assertEqual(l.cache_seqlens.tolist(),[length])
            torch.testing.assert_close(l.k_cache[0,:length],original,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'requires allocated CUDA GPU')
    def test_long_full_history_and_single_token_prefill(self):
        l=make('cuda',prompt=16,length=1100)
        l.k_cache=l.k_cache.bfloat16();l.v_cache=l.v_cache.bfloat16();l.queries=l.queries.bfloat16()
        old=l.k_cache[:,:1100].clone()
        q=torch.ones(2,1,4,8,device='cuda',dtype=torch.bfloat16)
        k=torch.ones(2,1,2,8,device='cuda',dtype=torch.bfloat16)
        step(l,q,k,k)
        self.assertEqual(l.cache_seqlens.tolist(),[1101,1101])
        torch.testing.assert_close(l.k_cache[:,:1100],old,rtol=0,atol=0)
        self.assertGreater(l.k_cache.shape[1],1024)
        tiny=SharedMaskLayer(0,0,0,'qhead_shared_mask_topp',0.,False,False)
        step(tiny,q,k,k);step(tiny,q,k,k)
        self.assertEqual(tiny.prompt_len,1);self.assertEqual(tiny.cache_seqlens.tolist(),[2,2])
        self.assertEqual(tiny.selection_events,1)

if __name__=='__main__':unittest.main(verbosity=2)
