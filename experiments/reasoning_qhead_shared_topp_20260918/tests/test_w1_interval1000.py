"""Independent mask, cache, attention, RNG and cadence checks for new methods."""
import math
import unittest
from unittest.mock import patch

import torch
from kvcompress.engine.selector_grid_cache import SelectorGridLayer, representative
from kvcompress.engine.w1_interval1000_cache import (
    COMPACT_MODE, HYBRID_MODE, W1Compact1000Layer, W1Hybrid1000Layer,
    uniform_residual_ids, install)


def make(hybrid, budget=1024, trace=True):
    cls = W1Hybrid1000Layer if hybrid else W1Compact1000Layer
    mode = HYBRID_MODE if hybrid else COMPACT_MODE
    return cls(3, budget, 0, mode, 0., False, False, query_window=1,
               eviction_interval=1000, rep_seed=0, sample_id=19, track_positions=trace)


def oracle_ids(scores, positions, prompt, absolute, budget):
    protected = (positions < prompt) | (positions >= max(prompt, absolute-64))
    fixed = protected.nonzero().flatten()
    eligible = (~protected).nonzero().flatten()
    count = min(budget, len(positions))-len(fixed)
    order = torch.argsort(scores[eligible], descending=True, stable=True)
    return torch.cat((fixed, eligible[order[:count]])).sort().values


class CPUChecks(unittest.TestCase):
    def test_factory_is_opt_in_and_old_modes_unchanged(self):
        import kvcompress.engine.selector_grid_cache as grid
        old_factory, old_decode = grid.SelectorGridLayer, grid.decode_mode
        try:
            install()
            for hybrid in (False, True):
                mode = HYBRID_MODE if hybrid else COMPACT_MODE
                l = grid.SelectorGridLayer(0,1024,0,mode,0.,False,False,
                                           query_window=1, eviction_interval=1000)
                self.assertEqual(l.storage, 'hybrid' if hybrid else 'compact')
                self.assertEqual(grid.decode_mode(mode)[0], l.storage)
                self.assertEqual(l.score_mode, 'qk')
            old = grid.SelectorGridLayer(0,1024,0,'grid_compact500_w1_topk_protected',
                                        0.,False,False,query_window=1,eviction_interval=500)
            self.assertIs(type(old), SelectorGridLayer)
            self.assertEqual(old.score_mode, 'probability')
        finally:
            grid.SelectorGridLayer, grid.decode_mode = old_factory, old_decode

    def test_query_cadence_current_query_and_capacity(self):
        for hybrid in (False, True):
            l = make(hybrid)
            l.lazy_initialization(torch.zeros(2,640,2,16,dtype=torch.bfloat16))
            self.assertEqual(l.capacity, 6120 if hybrid else 2024)
            q = torch.ones(2,8,1,16,dtype=torch.bfloat16)
            for step in (1,64,999,1000,1001,1999,2000):
                l.cumulative_length = 640+step-1
                l.update_queries(q*step)
                if hybrid:
                    self.assertTrue(torch.equal(l.queries, q*step))
                elif step < 1000:
                    self.assertFalse(hasattr(l,'queries'))
                else:
                    self.assertTrue(torch.equal(l.queries, q*(step//1000*1000)))

    def test_hybrid_absolute_positions_gqa_masks_short_history_and_rng(self):
        g = torch.Generator().manual_seed(918)
        for n in (80,96,97,320,5100):
            l = make(True, budget=96)
            l.lazy_initialization(torch.zeros(2,17,2,16,dtype=torch.bfloat16))
            absolute = n if n<320 else n+4000
            positions = torch.cat((torch.arange(17),torch.arange(absolute-(n-17),absolute)))
            l.slot_positions[:,:n] = positions
            l.k_cache[:,:n] = torch.randn(2,n,2,16,generator=g).bfloat16()
            l.cumulative_length = absolute
            l._lengths = [n]*2
            l.cache_seqlens.fill_(n)
            q = torch.randn(2,8,1,16,generator=g).bfloat16()
            l.update_queries(q)
            before = torch.random.get_rng_state().clone()
            with patch.object(torch.Tensor,'softmax',side_effect=AssertionError('selector softmax')):
                ids, counts = l.select_indices()
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            rep = representative(0,19,3,absolute-17,8)
            self.assertEqual(l._rep_head,rep)
            self.assertEqual(l._rep_kv,rep//4)
            raw = (q[:,rep,:1]@l.k_cache[:,:n,rep//4].transpose(1,2)).squeeze(1)
            for row in range(2):
                expected=oracle_ids(raw[row],positions,17,absolute,96)
                self.assertTrue(torch.equal(ids[row],expected))
                self.assertEqual(counts[row],min(96,n))

    def test_uniform_residual_exclusion_replay_and_marginal_frequency(self):
        n,core,count=9,torch.tensor([0,2,5]),2
        hits=torch.zeros(n)
        before=torch.random.get_rng_state().clone()
        for seed in range(2400):
            g=torch.Generator().manual_seed(seed)
            ids=uniform_residual_ids(n,core,count,g)
            self.assertEqual(len(ids.unique()),count)
            self.assertFalse(torch.isin(ids,core).any())
            self.assertTrue(torch.all(ids[1:]>ids[:-1]))
            hits[ids]+=1
        eligible=~torch.isin(torch.arange(n),core)
        self.assertTrue(torch.all((hits[eligible]/2400-count/int(eligible.sum())).abs()<.035))
        for _ in range(2):
            replay=uniform_residual_ids(n,core,count,torch.Generator().manual_seed(42))
            if _==0:first=replay
            else:self.assertTrue(torch.equal(first,replay))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        all_ids=uniform_residual_ids(n,core,4096,torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(all_ids,eligible.nonzero().flatten()))

    def test_pool_union_exact_budget_and_batch_shrink(self):
        l=make(True);g=torch.Generator().manual_seed(87)
        l.lazy_initialization(torch.zeros(2,640,2,16,dtype=torch.bfloat16))
        n=5640;l.cumulative_length=5640;l._lengths=[n]*2;l.cache_seqlens.fill_(n)
        l.slot_positions[:,:n]=torch.arange(n)
        l.k_cache[:,:n]=torch.randn(2,n,2,16,generator=g).bfloat16()
        l.v_cache[:,:n]=torch.randn(2,n,2,16,generator=g).bfloat16()
        l.update_queries(torch.randn(2,8,1,16,generator=g).bfloat16())
        ids,_=l.select_indices();core_positions=l.slot_positions.gather(1,ids).clone()
        oldk,oldv=l.k_cache[:,:n].clone(),l.v_cache[:,:n].clone()
        before=torch.random.get_rng_state().clone()
        l.compact_candidate_pool(ids)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(l._lengths,[5120,5120])
        for row in range(2):
            pos=l.slot_positions[row,:5120]
            self.assertEqual(len(pos.unique()),5120)
            self.assertTrue(torch.all(torch.isin(core_positions[row],pos)))
            self.assertTrue(torch.equal(l.k_cache[row,:5120],oldk[row,pos]))
            self.assertTrue(torch.equal(l.v_cache[row,:5120],oldv[row,pos]))
        self.assertEqual(l.compaction_core_token_sum,2048)
        self.assertEqual(l.compaction_residual_token_sum,8192)
        # Unused capacity is allocated with empty() and may contain NaNs.
        # Only live KV slots have a defined value after a batch shrink.
        expected=l.k_cache[1,:5120].clone()
        l.batch_select_indices(torch.tensor([1]))
        self.assertEqual(l._batch_ids,[1]);self.assertEqual(l._lengths,[5120])
        self.assertEqual(l._compact_k.shape[0],1)
        self.assertTrue(torch.equal(l.k_cache[0,:5120],expected))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GPUChecks(unittest.TestCase):
    def test_k2048_real_model_geometry_at_compaction_boundaries(self):
        g=torch.Generator(device='cuda').manual_seed(903)
        for hybrid in (False,True):
            for step in ((6000,7000) if hybrid else (2000,3000)):
                l=make(hybrid,budget=2048,trace=True)
                l.lazy_initialization(torch.zeros(2,640,8,128,device='cuda',dtype=torch.bfloat16))
                oldn=(640+step-1) if step==(6000 if hybrid else 2000) else (6144 if hybrid else 2048)+999
                absolute=640+step-1
                pos=torch.cat((torch.arange(640,device='cuda'),torch.arange(absolute-(oldn-640),absolute,device='cuda')))
                keys=torch.randn(2,oldn+1,8,128,device='cuda',dtype=torch.bfloat16,generator=g)
                vals=torch.randn(keys.shape,device='cuda',dtype=keys.dtype,generator=g)
                l.k_cache[:,:oldn]=keys[:,:oldn];l.v_cache[:,:oldn]=vals[:,:oldn]
                l.slot_positions[:,:oldn]=pos;l._lengths=[oldn]*2;l.cache_seqlens.fill_(oldn)
                l.cumulative_length=absolute
                q=torch.randn(2,1,32,128,device='cuda',dtype=keys.dtype,generator=g)
                l.update_queries(q.transpose(1,2))
                out=l.attention(q,keys[:,-1:].contiguous(),vals[:,-1:].contiguous(),1/math.sqrt(128))
                pos=torch.cat((pos,torch.tensor([absolute],device='cuda')))
                for row in range(2):
                    raw=(q[row,0,l._rep_head:l._rep_head+1]@keys[row,:,l._rep_kv].T).squeeze(0)
                    core=oracle_ids(raw,pos,640,absolute+1,2048)
                    if hybrid:self.assertTrue(torch.equal(l._last_selected_ids[row],core))
                    used=core if hybrid else torch.arange(oldn+1,device='cuda')
                    ks,vs=keys[row,used].repeat_interleave(4,dim=1),vals[row,used].repeat_interleave(4,dim=1)
                    logits=torch.einsum('qhd,khd->hqk',q[row].float(),ks.float())/math.sqrt(128)
                    ref=torch.einsum('hqk,khd->qhd',logits.softmax(-1),vs.float())
                    torch.testing.assert_close(out[row].float(),ref,atol=.018,rtol=.035)
                    retained=l.slot_positions[row,:l._lengths[row]]
                    self.assertEqual(len(retained),6144 if hybrid else 2048)
                    self.assertTrue(torch.isin(pos[core],retained).all())
                    slots=torch.searchsorted(pos,retained)
                    self.assertTrue(torch.equal(l.k_cache[row,:len(retained)],keys[row,slots]))
                    self.assertTrue(torch.equal(l.v_cache[row,:len(retained)],vals[row,slots]))
                self.assertEqual(l.compaction_events,1)
                self.assertEqual(l.selected_pairs,32*2*(2048 if hybrid else oldn+1))

    def test_attention_and_two_compactions_with_batch_shrink(self):
        from flash_attn import flash_attn_with_kvcache
        del flash_attn_with_kvcache
        torch.set_num_threads(1)
        for hybrid in (False,True):
            l=make(hybrid,trace=False)
            g=torch.Generator(device='cuda').manual_seed(773)
            expected_pairs=0;events=[]
            final=6001 if hybrid else 2001
            check_steps={0,1,999,1000,1001,1999,2000,2001,4999,5000,5001,5999,6000,6001}
            for step in range(final+1):
                b=2 if step<final else 1
                if step==final:l.batch_select_indices(torch.tensor([1],device='cuda'))
                length=640 if step==0 else 1
                q=torch.randn(b,length,8,16,device='cuda',dtype=torch.bfloat16,generator=g)
                k=torch.randn(b,length,2,16,device='cuda',dtype=q.dtype,generator=g)
                v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
                check=step in check_steps
                if check:
                    if step:
                        live=l._lengths[0]
                        keys=torch.cat((l.k_cache[:,:live].clone(),k),1)
                        vals=torch.cat((l.v_cache[:,:live].clone(),v),1)
                        pos=torch.cat((l.slot_positions[:,:live].clone(),torch.full((b,1),l.cumulative_length,device='cuda')),1)
                    else:keys,vals=k,v;pos=torch.arange(length,device='cuda').expand(b,-1)
                old_events=l.compaction_events
                l.update_queries(q.transpose(1,2))
                out=l.attention(q,k,v,1/4)
                count=640+step if step==0 else (min(1024,640+step) if hybrid else before_length+1)
                expected_pairs+=8*b*(640*641//2 if step==0 else count)
                self.assertEqual(l.selected_pairs,expected_pairs)
                if check:
                    refs=[]
                    for row in range(b):
                        ks,vs=keys[row],vals[row]
                        if step and hybrid:
                            raw=(q[row,0,l._rep_head:l._rep_head+1]@ks[:,l._rep_kv].T).squeeze(0)
                            expected=oracle_ids(raw,pos[row],640,640+step,1024)
                            self.assertTrue(torch.equal(l._last_selected_ids[row],expected))
                            ks,vs=ks[expected],vs[expected]
                        logits=torch.einsum('qhd,khd->hqk',q[row].float(),ks.repeat_interleave(4,dim=1).float())/4
                        if step==0:logits.masked_fill_(torch.triu(torch.ones(length,length,device='cuda',dtype=torch.bool),1)[None],-float('inf'))
                        refs.append(torch.einsum('hqk,khd->qhd',logits.softmax(-1),vs.repeat_interleave(4,dim=1).float()))
                    torch.testing.assert_close(out.float(),torch.stack(refs),atol=.018,rtol=.035)
                    if l.compaction_events>old_events:
                        for row in range(b):
                            positions=l.slot_positions[row,:l._lengths[row]]
                            self.assertTrue(torch.all(positions[1:]>positions[:-1]))
                            original_slots=torch.searchsorted(pos[row],positions)
                            self.assertTrue(torch.equal(l.k_cache[row,:len(positions)],keys[row,original_slots]))
                            self.assertTrue(torch.equal(l.v_cache[row,:len(positions)],vals[row,original_slots]))
                            if hybrid:
                                core=pos[row,l._last_selected_ids[row]]
                                self.assertTrue(torch.isin(core,positions).all())
                                self.assertEqual(len(positions),5120)
                            else:
                                raw=(q[row,0,l._rep_head:l._rep_head+1]@keys[row,:,l._rep_kv].T).squeeze(0)
                                expected=oracle_ids(raw,pos[row],640,640+step,1024)
                                self.assertTrue(torch.equal(positions,pos[row,expected]))
                                self.assertEqual(len(positions),1024)
                if l.compaction_events>old_events:events.append(step)
                before_length=l._lengths[0]
            self.assertEqual(events,[5000,6000] if hybrid else [1000,2000])
            self.assertEqual(l.capacity,6120 if hybrid else 2024)
            self.assertEqual(l._batch_ids,[1])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
