"""CPU mask oracles: per-head source, causal support, density and regressions."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from per_head import shared_base, head_index, literal_mask, reference_head, MODES
from patterns import allowed_mask
from selection import build_index

class CPUEvent:
    def __init__(self, **kwargs): pass
    def record(self): pass
    def elapsed_time(self, other): return 0.0

class Masks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(102008)
        torch.set_num_threads(2)
        self.options = json.loads((Path(__file__).parent/'config.json').read_text())['selection']
        self.options = dict(self.options, target_token_budget=128, autoblock_final_token_budget=128,
                            vertical_max_tokens=3, slash_max_offsets=5)
        self.q = torch.randn(1,257,4,8)
        self.k = torch.randn(1,257,2,8)
        self.v = torch.randn_like(self.k)
        with patch('torch.cuda.Event', CPUEvent):
            self.base, _ = shared_base(self.q,self.k,self.v,3,0,self.options)

    def materialize(self, index):
        out = torch.zeros(257,257,dtype=torch.bool)
        for qb in range(index.num_query_blocks):
            start,end=qb*128,min((qb+1)*128,257)
            a,b=int(index.row_starts[0,0,qb]),int(index.row_ends[0,0,qb])
            ids=index.token_indices[a:b].long()
            out[start:end,ids] = allowed_mask(index,qb,torch.arange(start,end),ids)
        return out

    def test_base_unchanged(self):
        with patch('torch.cuda.Event', CPUEvent):
            old,_=build_index(self.q,self.k,self.v,3,0,'representative_token',self.options)
        for name in ['row_starts','row_ends','token_indices','head_to_group']:
            self.assertTrue(torch.equal(getattr(self.base,name),getattr(old,name)))

    def test_independent_sources_exact_mask_and_density(self):
        for mode in MODES:
            masks=[]
            for h in range(4):
                idx,s,columns,offsets=head_index(self.base,self.q,self.k,h,mode,self.options)
                exact=literal_mask(self.base,columns,offsets,257)
                self.assertTrue(torch.equal(self.materialize(idx),exact))
                self.assertTrue(exact.any(-1).all())
                self.assertFalse(exact.triu(1).any())
                self.assertEqual(s['mask_selected_pairs'],int(exact.sum()))
                self.assertEqual(s['vertical_source_head'],h)
                if mode.endswith('_slash'): self.assertEqual(s['slash_source_head'],h)
                self.assertEqual(s['pattern_shared_with_heads'],[h])
                self.assertTrue((exact | literal_mask(self.base,torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long),257)).equal(exact))
                masks.append(exact)
            self.assertTrue(any(not torch.equal(masks[0],m) for m in masks[1:]))

    def test_representative_head_matches_previous_full_mask(self):
        for mode in MODES:
            idx,_,_,_=head_index(self.base,self.q,self.k,3,mode,self.options)
            with patch('torch.cuda.Event',CPUEvent): old,_=build_index(self.q,self.k,self.v,3,0,mode,self.options)
            self.assertTrue(torch.equal(self.materialize(idx),self.materialize(old)))

    def test_reference_preserves_gqa_values(self):
        self.v[:,:,0,:]=3;self.v[:,:,1,:]=7
        for h in range(4):
            _,_,columns,offsets=head_index(self.base,self.q,self.k,h,MODES[1],self.options)
            out=reference_head(self.q,self.k,self.v,h,literal_mask(self.base,columns,offsets,257))
            torch.testing.assert_close(out,torch.full_like(out,3 if h<2 else 7),atol=1e-5,rtol=0)

    def test_uncapped_gamma_coverage(self):
        options=dict(self.options,vertical_max_tokens=None,slash_max_offsets=None)
        for h in range(4):
            _,s,_,_=head_index(self.base,self.q,self.k,h,MODES[1],options)
            self.assertGreaterEqual(s['vertical_retained_mass'],.95-1e-6)
            self.assertGreaterEqual(s['slash_retained_mass'],.95-1e-6)
            self.assertLess(s['vertical_prefix_before_last'],.95)
            self.assertLess(s['slash_prefix_before_last'],.95)

if __name__=='__main__': unittest.main(verbosity=2)
