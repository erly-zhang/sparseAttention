"""W1 probability core parity, recent union, full KV and ragged fused attention."""
import math
import unittest
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.shared_mask_cache import SharedMaskLayer
from kvcompress.engine.w1_recent_shared_mask_cache import W1RecentSharedMaskLayer
from test_shared_mask_cache import step, reference
MODE='qhead_shared_mask_w1_recent_gpu'


def layer(recent=64, **kw):
    return W1RecentSharedMaskLayer(0,0,0,MODE,0.,False,False,
        query_window=1,top_p=.95,recent_window=recent,sample_id=17,**kw)


def setup(l,n,device='cpu'):
    z=torch.zeros(2,n,2,8,device=device)
    l.prepare(z,z);l.k_cache[:,:n].zero_();l.v_cache[:,:n].zero_()
    l.cache_seqlens.fill_(n);l.cumulative_length=n;l.prompt_len=2
    q=torch.zeros(2,4,1,8,device=device);q[:,:,:,0]=math.sqrt(8)
    l.update_queries(q)
    return l


class W1RecentTests(unittest.TestCase):
    def test_factory_and_reject_other_windows(self):
        c=EvictCache(Qwen3Config(num_hidden_layers=2),0,0,MODE,rkv_lambda=0.,smooth=False,
            verbose=False,query_window=1,top_p=.95,recent_window=64)
        self.assertTrue(all(isinstance(x,W1RecentSharedMaskLayer) for x in c.layers))
        with self.assertRaises(ValueError):
            W1RecentSharedMaskLayer(0,0,0,MODE,0.,False,False,query_window=16)
        with self.assertRaises(RuntimeError):c.layers[0]._run_eviction(100)
        with self.assertRaises(RuntimeError):c.layers[0].post_attention_update(1)
        q=torch.randn(2,4,13,8);c.layers[0].update_queries(q)
        torch.testing.assert_close(c.layers[0].queries,q[:,:,-1:])
        c.layers[0].update_queries(-q[:,:,:1])
        torch.testing.assert_close(c.layers[0].queries,-q[:,:,:1])

    def test_core_exactly_old_w1_and_union_short_long_ties_private_rng(self):
        for n in (2,3,65,66,67,100,1103):
            a=setup(layer(),n)
            b=setup(SharedMaskLayer(0,0,0,'qhead_shared_mask_topp',0.,False,False,
                query_window=1,top_p=.95,sample_id=17),n)
            rng=torch.Generator().manual_seed(n)
            values=torch.randn((2,n,2,8),generator=rng)
            if n==100:values.zero_() # exact score ties: use unchanged torch.sort order
            a.k_cache[:,:n]=values;b.k_cache[:,:n]=values
            a.k_cache[:,n:]=float('nan');b.k_cache[:,n:]=float('nan')
            before=a.k_cache[:,:n].clone();state=torch.random.get_rng_state().clone()
            for _ in range(3):
                final=a.select_indices();core=b.select_indices()
                self.assertEqual(a._rep_head,b._rep_head)
                for row in range(2):
                    torch.testing.assert_close(a._last_topp_ids[row],core[row],rtol=0,atol=0)
                    self.assertEqual(final[row].tolist(),sorted(set(core[row].tolist())|set(range(max(2,n-64),n))))
                    if n<=66:self.assertEqual(final[row].tolist(),list(range(n)))
            torch.testing.assert_close(a.k_cache[:,:n],before,rtol=0,atol=0)
            torch.testing.assert_close(torch.random.get_rng_state(),state,rtol=0,atol=0)

    def test_independent_probabilities_and_history_reselection(self):
        a=setup(layer(recent=2),7)
        weights=torch.tensor([[.3,.2,.4,.07,.02,.008,.002],[.01,.01,.45,.3,.15,.06,.02]])
        a.k_cache[:,:7,:,0]=weights.log()[:,:,None]
        ids=a.select_indices()
        for row in range(2):
            p=weights[row,2:].double();p/=p.sum()
            order=sorted(range(5),key=lambda j:-float(p[j]));core=[];mass=0.
            for j in order:
                core.append(j+2);mass+=float(p[j])
                if mass>=.95:break
            self.assertEqual(ids[row].tolist(),sorted(set([0,1,5,6]+core)))
        # A middle historical position initially excluded can return next step.
        a=setup(layer(recent=2),100)
        a.k_cache[:,2,:,0]=20;a.k_cache[:,50,:,0]=-20
        self.assertNotIn(50,a.select_indices()[0].tolist())
        a.update_queries(-a.queries.clone())
        self.assertIn(50,a.select_indices()[0].tolist())
        self.assertEqual(a.cache_seqlens.tolist(),[100,100])

    @unittest.skipUnless(torch.cuda.is_available(),'allocated CUDA GPU required')
    def test_cuda_ragged_padding_growth_full_history_pair_count_and_filter(self):
        l=layer();n=1100
        k=torch.zeros(2,n,2,8,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k);k[0,2,:,0]=50
        l.prepare(k,v);l.k_cache[:,:n].copy_(k);l.v_cache[:,:n].copy_(v)
        l.prompt_len=2;l.cache_seqlens.fill_(n);l.cumulative_length=n
        q=torch.zeros(2,1,4,8,device='cuda',dtype=torch.bfloat16);q[:,:,:,0]=math.sqrt(8)
        nk=torch.zeros(2,1,2,8,device='cuda',dtype=torch.bfloat16);nv=torch.randn_like(nk)
        out=step(l,q,nk,nv);fk=torch.cat((k,nk),1);fv=torch.cat((v,nv),1)
        ids=[x.tolist() for x in l._last_selected_ids]
        self.assertNotEqual(len(ids[0]),len(ids[1]))
        torch.testing.assert_close(out.float(),reference(q,fk,fv,ids),rtol=.02,atol=.012)
        torch.testing.assert_close(l.k_cache[:,:n+1],fk,rtol=0,atol=0)
        torch.testing.assert_close(l.v_cache[:,:n+1],fv,rtol=0,atol=0)
        self.assertEqual(l.selected_pairs,4*sum(map(len,ids)))
        self.assertEqual(l.causal_pairs,8*(n+1))
        self.assertEqual(l.selected_pairs,l.topp_core_selected_pairs+l.recent_added_pairs)
        self.assertEqual(l.cache_seqlens.tolist(),[n+1,n+1])
        ptr=l._temp_k.data_ptr();capacity=l._gather_capacity;allocations=l.gather_allocations;l.batch_select_indices(torch.tensor([1],device='cuda'))
        out=step(l,q[1:],nk[1:],nv[1:])
        torch.testing.assert_close(out.float(),reference(q[1:],torch.cat((fk[1:],nk[1:]),1),
            torch.cat((fv[1:],nv[1:]),1),[l._last_selected_ids[0].tolist()]),rtol=.02,atol=.012)
        if l._last_selected_ids[0].numel()<=capacity:
            self.assertEqual(ptr,l._temp_k.data_ptr());self.assertEqual(l.gather_allocations,allocations)
        else:
            self.assertGreaterEqual(l._gather_capacity,2*capacity)
            self.assertEqual(l.gather_allocations,allocations+1)
        self.assertEqual(l._batch_ids,[1])

    @unittest.skipUnless(torch.cuda.is_available(),'allocated CUDA GPU required')
    def test_cuda_recent_zero_matches_old_attention_and_dense_prefill(self):
        a=layer(recent=0)
        b=SharedMaskLayer(0,0,0,'qhead_shared_mask_topp',0.,False,False,
            query_window=1,top_p=.95,sample_id=17)
        gen=torch.Generator(device='cuda').manual_seed(1934);ks=[];vs=[]
        for t in range(12):
            n=4 if t==0 else 1
            q=torch.randn(2,n,4,8,device='cuda',dtype=torch.bfloat16,generator=gen)
            k=torch.randn(2,n,2,8,device='cuda',dtype=torch.bfloat16,generator=gen)
            v=torch.randn(k.shape,device='cuda',dtype=k.dtype,generator=gen)
            oa,ob=step(a,q,k,v),step(b,q,k,v)
            torch.testing.assert_close(oa,ob,rtol=0,atol=0)
            ks.append(k);vs.append(v)
            if t==0:
                for j in range(n):
                    torch.testing.assert_close(oa[:,j:j+1].float(),reference(q[:,j:j+1],k,v,
                        [list(range(j+1))]*2),rtol=.02,atol=.012)
            else:
                for x,y in zip(a._last_selected_ids,b._last_selected_ids):torch.testing.assert_close(x,y,rtol=0,atol=0)
                torch.testing.assert_close(oa.float(),reference(q,torch.cat(ks,1),torch.cat(vs,1),
                    [x.tolist() for x in a._last_selected_ids]),rtol=.02,atol=.012)
        self.assertEqual(a.selected_pairs,b.selected_pairs)
        self.assertEqual(a.recent_added_pairs,0);self.assertEqual(a.gather_allocations,1)

if __name__=='__main__':unittest.main(verbosity=2)
