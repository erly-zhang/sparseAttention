"""Independent raw-QK masks and attention oracles for full KV / compact500."""
import math
import unittest
from unittest.mock import patch
import torch
from kvcompress.engine.selector_grid_cache import SelectorGridLayer, representative
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer


def make(storage,device='cpu',prompt=640,batch=2,heads=8,kvheads=2,d=16):
    cls=FullW1OptimizedLayer if storage=='full' else SelectorGridLayer
    mode='grid_full_w1_topk_protected' if storage=='full' else 'grid_compact500_w1_topk_protected'
    l=cls(3,1024,0,mode,0.,False,False,query_window=1,
          eviction_interval=1 if storage=='full' else 500,rep_seed=0,
          sample_id=19,max_cache_length=32768,track_positions=True,score_mode='qk')
    return l


def oracle_ids(scores,positions,prompt,absolute,K=1024):
    # Independent specification: remove mandatory positions before stable ranking.
    required=(positions<prompt)|(positions>=max(prompt,absolute-64))
    fixed=required.nonzero().flatten();eligible=(~required).nonzero().flatten()
    count=min(K,len(positions))-len(fixed)
    ranked=torch.argsort(scores[eligible],descending=True,stable=True)[:count]
    return torch.cat((fixed,eligible[ranked])).sort().values


class RawGridCPUTests(unittest.TestCase):
    def test_compact_current_query_cadence(self):
        l=make('compact');l.prompt_len=640
        q=torch.randn(2,8,1,16)
        for step in (1,64,499,500,501,999,1000):
            l.cumulative_length=640+step-1;l.update_queries(q*step)
            if step<500:self.assertFalse(hasattr(l,'queries'))
            else:self.assertTrue(torch.equal(l.queries,q*(step//500*500)))
        self.assertEqual(l._q_valid,1)
    def test_compact_raw_scores_ragged_gqa_protection_and_rng(self):
        l=make('compact');g=torch.Generator().manual_seed(83)
        l.lazy_initialization(torch.zeros(2,640,2,16,dtype=torch.bfloat16))
        l.cumulative_length=1140;l._lengths=[1140,1127];l.cache_seqlens[:]=torch.tensor(l._lengths)
        l.k_cache.normal_(generator=g)
        for row,n in enumerate(l._lengths):l.slot_positions[row,:n]=torch.arange(n)
        l.queries=torch.randn(2,8,1,16,generator=g).bfloat16();l._q_valid=1;l._num_query_heads=8
        before=torch.random.get_rng_state().clone()
        with patch.object(torch.Tensor,'softmax',side_effect=AssertionError('selector softmax')):
            ids,counts=l.select_indices()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(l._rep_head,representative(0,19,3,500,8))
        raw=(l.queries[:,l._rep_head]@l.k_cache[:,:1140,l._rep_head//4].transpose(1,2)).squeeze(1)
        for row,n in enumerate(l._lengths):
            self.assertTrue(torch.equal(l._last_scores[row,:n],raw[row,:n]))
            ref=oracle_ids(raw[row,:n],l.slot_positions[row,:n],640,1140)
            self.assertTrue(torch.equal(ids[row],ref));self.assertEqual(counts[row],1024)
        self.assertTrue(torch.isneginf(l._last_scores[1,1127:]).all())
    def test_raw_mode_rejects_probability_mean_or_topp(self):
        for mode,window in [('grid_compact500_w16_topk_protected',16),('grid_compact500_w1_topp_none',1)]:
            with self.assertRaises(ValueError):
                SelectorGridLayer(0,1024,0,mode,0.,False,False,query_window=window,eviction_interval=500,score_mode='qk')


@unittest.skipUnless(torch.cuda.is_available(),'GPU required')
class RawGridGPUTests(unittest.TestCase):
    def test_real_geometry_raw_masks(self):
        gen=torch.Generator(device='cuda').manual_seed(71)
        for storage in ('full','compact'):
            for n in (1000,1024,1140,1524) if storage=='compact' else (1000,1024,1140,8192,32700):
                l=make(storage);l.lazy_initialization(torch.zeros(2,640,8,128,device='cuda',dtype=torch.bfloat16))
                l.k_cache[:,:n].normal_(generator=gen);l._lengths=[n,n];l.cache_seqlens.fill_(n)
                l.slot_positions[:,:n]=torch.arange(n,device='cuda');l.cumulative_length=n
                l.queries=torch.randn(2,32,1,128,device='cuda',dtype=torch.bfloat16,generator=gen)
                l._num_query_heads=32;l._q_valid=1
                cpu_state=torch.random.get_rng_state().clone();cuda_state=torch.cuda.get_rng_state().clone()
                with patch.object(torch.Tensor,'softmax',side_effect=AssertionError('selector softmax')):
                    ids,counts=l.select_indices()
                self.assertTrue(torch.equal(cpu_state,torch.random.get_rng_state()))
                self.assertTrue(torch.equal(cuda_state,torch.cuda.get_rng_state()))
                raw=(l.queries[:,l._rep_head]@l.k_cache[:,:n,l._rep_head//4].transpose(1,2)).squeeze(1)
                self.assertTrue(torch.equal(l._last_scores,raw))
                for row in range(2):
                    ref=oracle_ids(raw[row],torch.arange(n,device='cuda'),640,n)
                    self.assertTrue(torch.equal(ids[row],ref));self.assertEqual(counts[row],min(n,1024))
    def test_attention_two_compactions_absolute_positions_and_batch_shrink(self):
        for storage in ('full','compact'):
            l=make(storage);gen=torch.Generator(device='cuda').manual_seed(137)
            expected_pairs=0;events=[]
            for step in range(1002):
                b=2 if step<990 else 1;n=640 if step==0 else 1
                q=torch.randn(b,n,8,16,device='cuda',dtype=torch.bfloat16,generator=gen)
                k=torch.randn(b,n,2,16,device='cuda',dtype=q.dtype,generator=gen)
                v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=gen)
                if step:
                    old_lengths=list(l._lengths)
                    keys=[torch.cat((l.k_cache[row,:length].clone(),k[row]),0) for row,length in enumerate(old_lengths)]
                    vals=[torch.cat((l.v_cache[row,:length].clone(),v[row]),0) for row,length in enumerate(old_lengths)]
                    pos=[torch.cat((l.slot_positions[row,:length].clone(),torch.tensor([l.cumulative_length],device='cuda'))) for row,length in enumerate(old_lengths)]
                else:keys=list(k);vals=list(v);pos=[torch.arange(n,device='cuda')]*b
                previous_events=l.selection_events
                l.update_queries(q.transpose(1,2))
                out=l.attention(q,k,v,1/math.sqrt(16))
                refs=[]
                for row in range(b):
                    ks=keys[row];vs=vals[row]
                    if step and storage=='full':
                        scores=(q[row,0,l._rep_head:l._rep_head+1]@ks[:,l._rep_kv].T).squeeze(0)
                        ids=oracle_ids(scores,pos[row],640,640+step)
                        self.assertTrue(torch.equal(l._last_selected_ids[row],ids))
                        ks=ks[ids];vs=vs[ids]
                    logits=torch.einsum('qhd,khd->hqk',q[row].float(),ks.repeat_interleave(4,dim=1).float())/4
                    if step==0:logits.masked_fill_(torch.triu(torch.ones(n,n,device='cuda',dtype=torch.bool),1)[None],-float('inf'))
                    refs.append(torch.einsum('hqk,khd->qhd',logits.softmax(-1),vs.repeat_interleave(4,dim=1).float()))
                    expected_pairs+=8*(n*(n+1)//2 if step==0 else len(ks))
                    if storage=='compact' and l.selection_events>previous_events:
                        scores=(q[row,0,l._rep_head:l._rep_head+1]@keys[row][:,l._rep_kv].T).squeeze(0)
                        ids=oracle_ids(scores,pos[row],640,640+step)
                        self.assertTrue(torch.equal(l.slot_positions[row,:1024],pos[row][ids]))
                        self.assertTrue(torch.equal(l.k_cache[row,:1024],keys[row][ids]))
                        self.assertTrue(torch.equal(l.v_cache[row,:1024],vals[row][ids]))
                torch.testing.assert_close(out.float(),torch.stack(refs),atol=.018,rtol=.035)
                self.assertEqual(l.selected_pairs,expected_pairs)
                if storage=='compact' and l.selection_events>previous_events:events.append(step)
                if step==989:l.batch_select_indices(torch.tensor([1],device='cuda'))
            if storage=='compact':self.assertEqual(events,[500,1000]);self.assertEqual(l.capacity,1524)
            else:self.assertEqual(l._lengths,[1641])


if __name__=='__main__':unittest.main(verbosity=2)
