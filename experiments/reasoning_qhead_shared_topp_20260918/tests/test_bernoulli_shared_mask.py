import gc
import math
import unittest
import weakref
from unittest.mock import patch
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache, _stable_hash
from kvcompress.engine.bernoulli_recent_shared_mask_cache import BernoulliMaskPlan
from test_shared_mask_cache import step, reference
MODE='qhead_shared_mask_bernoulli_recent_gpu'


def make(layers=2,window=4,recent=3):
    return EvictCache(Qwen3Config(num_hidden_layers=layers),0,0,MODE,
        rkv_lambda=0.,smooth=False,verbose=False,random_keep_ratio=.2,
        recent_window=recent,mask_seed=42,sample_id=19,
        mask_precompute_window=window,max_cache_length=512)


class BernoulliTests(unittest.TestCase):
    def protection(self,device):
        plan=BernoulliMaskPlan(3,.2,64,42,19,window=4)
        state=torch.random.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        for t in (1,2,63,64,319,320,321,1024,8192,32760):
            ids=plan.build_step(t,71,(0,1),device)
            lengths,counts,target,_=plan.info[t]
            recent=min(64,target);older=t-recent
            for l in range(3):
                for b in range(2):
                    selected=ids[l,b,:counts[l][b]].long()
                    self.assertTrue(torch.equal(selected[:71],torch.arange(71,device=device)))
                    self.assertTrue(torch.equal(selected[-recent:],torch.arange(71+t-recent,71+t,device=device)))
                    self.assertTrue(bool((selected[1:]>selected[:-1]).all()))
                    self.assertEqual(int(lengths[l,b]),len(selected))
                    rng=torch.Generator(device=device).manual_seed(_stable_hash(0x4245524E,42,19,t,l,b))
                    mask=torch.empty(older,device=device,dtype=torch.bool).bernoulli_((target-recent)/older if older else 0,generator=rng)
                    oracle=torch.cat((torch.arange(71,device=device),71+mask.nonzero().flatten(),torch.arange(71+t-recent,71+t,device=device)))
                    torch.testing.assert_close(selected,oracle,rtol=0,atol=0)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        if gpu is not None:self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))

    def test_cpu_protection_boundaries_and_private_rng(self):self.protection('cpu')

    def test_binomial_marginals_count_variance_and_no_topk(self):
        # 2000 independent layer streams, N=100, p=.2, fixed seed.
        p=BernoulliMaskPlan(2000,.2,0,42,19)
        with patch.object(torch.Tensor,'topk',side_effect=AssertionError('TopK forbidden')):
            ids=p.build_step(100,1,(0,), 'cpu')
        counts=torch.tensor(p.info[100][1]).flatten()-1
        hits=torch.zeros(100)
        for row,k in zip(ids[:,0],counts):hits[row[1:int(k)+1].long()-1]+=1
        self.assertLess(abs(float(counts.float().mean())-20),.5)
        self.assertTrue(13 < float(counts.float().var()) < 19)
        self.assertLess(float((hits/2000-.2).abs().max()),.04)
        self.assertGreater(len(set(counts.tolist())),10)

    def test_window_sharding_replay_and_rollout_survivor(self):
        a=BernoulliMaskPlan(3,.2,3,42,19,window=4)
        b=BernoulliMaskPlan(3,.2,3,42,19,window=1)
        a.max_step=b.max_step=55
        for t in range(17,56):
            active=(0,1) if t<20 else (1,)
            for layer in range(3):
                x=a.bundle(layer,t,5,active,'cpu');y=b.bundle(layer,t,5,active,'cpu')
                self.assertEqual(x[2],y[2])
                for row,n in enumerate(x[2]):
                    torch.testing.assert_close(x[0][row,:n],y[0][row,:n],rtol=0,atol=0)
            self.assertLessEqual(len(a.blocks),4)
        self.assertLessEqual(max(a.blocks),55)
        with self.assertRaises(ValueError):a.bundle(0,56,5,(1,),'cpu')

    def test_no_cache_reference_cycle(self):
        enabled=gc.isenabled();gc.disable()
        try:
            c=make();refs=[weakref.ref(c.mask_plan),weakref.ref(c.layers[0])];del c
            self.assertTrue(all(r() is None for r in refs))
        finally:
            if enabled:gc.enable()

    @unittest.skipUnless(torch.cuda.is_available(),'GPU allocation required')
    def test_cuda_protection_boundaries_and_private_rng(self):self.protection('cuda')

    @unittest.skipUnless(torch.cuda.is_available(),'GPU allocation required')
    def test_gqa_ragged_attention_full_kv_survivor_and_pair_counts(self):
        c=make();c.mask_plan.max_step=80
        rng=torch.Generator(device='cuda').manual_seed(17)
        ks=[[],[]];vs=[[],[]];batch=2;saw_variable=False
        counted=[0,0];causal=[0,0]
        for t in range(81):
            if t==50:
                c.batch_select_indices(torch.tensor([1],device='cuda'));batch=1
                ks=[[x[1:] for x in xs] for xs in ks];vs=[[x[1:] for x in xs] for xs in vs]
            n=5 if t==0 else 1
            for i,l in enumerate(c.layers):
                q=torch.randn(batch,n,4,8,device='cuda',dtype=torch.bfloat16,generator=rng)
                k=torch.randn(batch,n,2,8,device='cuda',dtype=q.dtype,generator=rng)
                v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=rng)
                ks[i].append(k.clone());vs[i].append(v.clone())
                out=step(l,q,k,v)
                fk=torch.cat(ks[i],1);fv=torch.cat(vs[i],1)
                torch.testing.assert_close(l.k_cache[:,:fk.shape[1]],fk,rtol=0,atol=0)
                torch.testing.assert_close(l.v_cache[:,:fv.shape[1]],fv,rtol=0,atol=0)
                self.assertEqual(l.k_cache.shape[1],512)
                if t:
                    ids=l._last_selected_ids
                    expected=reference(q,fk,fv,ids)
                    torch.testing.assert_close(out.float(),expected,atol=.025,rtol=.025)
                    saw_variable |= len({len(x) for x in ids})>1
                    counted[i]+=4*sum(len(x) for x in ids);causal[i]+=4*batch*fk.shape[1]
                else:
                    counted[i]+=batch*4*n*(n+1)//2;causal[i]=counted[i]
                self.assertEqual(l.selected_pairs,counted[i]);self.assertEqual(l.causal_pairs,causal[i])
        self.assertTrue(saw_variable)

if __name__=='__main__':unittest.main(verbosity=2)
