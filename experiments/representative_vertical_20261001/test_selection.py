"""CPU invariants using the actual repository selector; never load a checkpoint."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from selection import (build_index,make_selector,merge_vertical,vertical_columns,causal_pairs,
    reference_attention,directional_coverage,vote_representatives,normalized_attention,sampled_member_coverage)
from experiments.shared_topp_mask_bench_20260930.attention import nucleus_mask
from experiments.token_compacted_sparse import TokenCompactedIndex


class CPUEvent:
    def __init__(self,**kwargs): pass
    def record(self): pass


def index_rows(index):
    return [index.token_indices[int(index.row_starts[0,0,i]):int(index.row_ends[0,0,i])].long()
        for i in range(index.num_query_blocks)]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.options=json.loads((Path(__file__).parent/'config.json').read_text())['selection']
        self.options.update(target_token_budget=256,autoblock_final_token_budget=256)
        self.q=torch.randn(1,513,4,8)
        self.k=torch.randn(1,513,2,8)
        self.v=torch.randn_like(self.k)

    def build(self,method):
        with patch('torch.cuda.Event',CPUEvent):
            return build_index(self.q,self.k,self.v,3,0,method,self.options)

    def test_original_token_index_is_unchanged(self):
        actual,_=self.build('representative_token')
        original=make_selector({'0':3},'representative_token',self.options,4)
        original.current_layer.set(0)
        with patch('torch.cuda.Event',CPUEvent):
            expected=original(self.q,self.k,self.v,128,.95,1,513)
        for a,b in zip(index_rows(actual),index_rows(expected)):
            self.assertTrue(torch.equal(a,b))

    def test_original_autoblock_index_is_unchanged(self):
        actual,_=self.build('representative_autoblock')
        original=make_selector({'0':3},'representative_autoblock',self.options,4)
        original.current_layer.set(0)
        with patch('torch.cuda.Event',CPUEvent):
            expected=original(self.q,self.k,self.v,128,.95,1,513)
        for a,b in zip(index_rows(actual),index_rows(expected)):
            self.assertTrue(torch.equal(a,b))

    def test_all_heads_share_one_index(self):
        index,stats=self.build('representative_token')
        self.assertEqual(index.num_groups,1)
        self.assertTrue((index.head_to_group==0).all())
        self.assertEqual(stats['shared_with_heads'],list(range(4)))

    def test_token_targets_are_not_expanded(self):
        index,_=self.build('representative_token')
        self.assertEqual([len(x) for x in index_rows(index)],[128,256,256,256,256])

    def test_short_causal_tiles_keep_every_valid_key(self):
        index,_=self.build('representative_token')
        rows=index_rows(index)
        self.assertTrue(torch.equal(rows[0],torch.arange(128)))
        self.assertTrue(torch.equal(rows[1],torch.arange(256)))

    def test_vertical_preserves_base_tokens(self):
        base,_=self.build('representative_token')
        expanded,stats=self.build('representative_token_vertical')
        for a,b in zip(index_rows(base),index_rows(expanded)):
            self.assertTrue(torch.isin(a,b).all())
        self.assertGreaterEqual(stats['mask_selected_pairs'],stats['base_selected_pairs'])
        self.assertGreaterEqual(stats['vertical_retained_mass'],.95-2e-5)
        self.assertFalse(stats['slash_added'])

    def test_causal_pair_density_matches_literal(self):
        index,stats=self.build('representative_token_vertical')
        total=0
        for qb,ids in enumerate(index_rows(index)):
            start=qb*128;end=min(start+128,513)
            allowed=torch.arange(start,end)[:,None]>=ids[None]
            total+=int(allowed.sum())
            self.assertTrue(allowed.any(-1).all())
        self.assertEqual(total,stats['mask_selected_pairs'])
        self.assertEqual(stats['mask_density'],total/(513*514//2))

    def test_deduplication_and_future_column_clipping(self):
        index,_=self.build('representative_token')
        columns=torch.tensor([0,0,200,512])
        merged,stats=merge_vertical(index,columns,513)
        for qb,ids in enumerate(index_rows(merged)):
            self.assertEqual(len(ids),len(ids.unique()))
            self.assertTrue((ids<min((qb+1)*128,513)).all())
        again,_=merge_vertical(merged,columns,513)
        self.assertTrue(torch.equal(merged.token_indices,again.token_indices))

    def test_invalid_vertical_positions_rejected(self):
        index,_=self.build('representative_token')
        with self.assertRaises(ValueError): merge_vertical(index,torch.tensor([-1]),513)
        with self.assertRaises(ValueError): merge_vertical(index,torch.tensor([513]),513)

    def test_vertical_normalization_and_minimal_prefix(self):
        columns,stats=vertical_columns(self.q,self.k,3,self.options)
        probs,_=normalized_attention(self.q[0,-128:,3][None],self.k[0,:,1][None],torch.arange(385,513))
        scores=probs[0].double().mean(0);scores/=scores.sum()
        values= scores.sort(descending=True,stable=True).values
        self.assertGreaterEqual(float(scores[columns].sum()),.95-2e-5)
        self.assertLess(float(values[:len(columns)-1].sum()),.95)
        self.assertEqual(stats['vertical_source_head'],3)

    def test_vertical_cap_records_coverage_loss(self):
        options=dict(self.options,vertical_max_tokens=1)
        columns,stats=vertical_columns(self.q,self.k,3,options)
        self.assertEqual(len(columns),1)
        self.assertTrue(stats['vertical_budget_capped'])
        self.assertLess(stats['vertical_retained_mass'],.95)

    def test_short_input_query_count(self):
        columns,stats=vertical_columns(self.q[:,:7],self.k[:,:7],3,self.options)
        self.assertEqual(stats['vertical_query_count'],7)
        self.assertTrue((columns<7).all())

    def test_threshold_crossing_and_stable_ties(self):
        p=torch.tensor([[.4,.3,.3]],dtype=torch.float64)
        mask,mass,before=nucleus_mask(p,torch.ones_like(p,dtype=torch.bool),.7)
        self.assertEqual(mask.tolist(),[[True,True,False]])
        self.assertAlmostEqual(float(mass[0]),.7)
        self.assertAlmostEqual(float(before[0]),.4)

    def test_reference_uses_each_heads_own_gqa_values(self):
        index,_=self.build('representative_token')
        q=torch.zeros_like(self.q,dtype=torch.float64);k=torch.zeros_like(self.k,dtype=torch.float64)
        v=torch.ones_like(self.v,dtype=torch.float64);v[:,:,1]=10
        out=reference_attention(q,k,v,index)
        torch.testing.assert_close(out[:,:,:2],torch.ones_like(out[:,:,:2]))
        torch.testing.assert_close(out[:,:,2:],torch.full_like(out[:,:,2:],10))

    def test_p1_dense_reference_with_own_probabilities(self):
        # Full support on a short input must match literal, separate-head dense attention.
        q,k,v=self.q[:,:35],self.k[:,:35],self.v[:,:35]
        with patch('torch.cuda.Event',CPUEvent):
            index,_=build_index(q,k,v,3,0,'representative_token',self.options)
        out=reference_attention(q,k,v,index)
        for head in range(4):
            scores=q[0,:,head]@k[0,:,head//2].T/math.sqrt(8)
            scores.masked_fill_(~torch.ones(35,35,dtype=torch.bool).tril(),float('-inf'))
            torch.testing.assert_close(out[0,:,head],scores.softmax(-1)@v[0,:,head//2])

    def test_sampled_member_coverage_scope(self):
        index,_=self.build('representative_token')
        stat=sampled_member_coverage(self.q,self.k,index,count=7,chunk=3)
        self.assertEqual(stat['member_coverage_queries'],7)
        self.assertEqual(len(stat['member_mean_retained_mass']),4)
        self.assertTrue(all(0<=x<=1.00001 for x in stat['member_mean_retained_mass']))

    def test_directional_coverage_matches_literal_provider_target(self):
        q,k=self.q[:,:9],self.k[:,:9]
        result=directional_coverage(q,k,top_p=.6,last_queries=5,chunk=2)
        probs,allowed=normalized_attention(q[0,-5:].transpose(0,1),
            k[0].transpose(0,1).repeat_interleave(2,0),torch.arange(4,9))
        masks,_,_=nucleus_mask(probs,allowed[None].expand_as(probs),.6)
        expected=torch.tensor([[float((probs[target]*masks[provider]).sum(-1).mean())
            for target in range(4)] for provider in range(4)],dtype=torch.float64)
        torch.testing.assert_close(torch.tensor(result['similarity']),expected.float())
        self.assertEqual(result['head'],int(expected.mean(-1).argmax()))

    def test_vote_majority_then_voter_score(self):
        def sample(head,scores): return {'0':dict(head=head,outgoing_scores=scores)}
        self.assertEqual(vote_representatives([sample(1,[0,.6,.9]),sample(1,[0,.6,.9]),sample(2,[0,0,1])],[0]),{'0':1})
        self.assertEqual(vote_representatives([sample(1,[0,.6,0]),sample(2,[0,0,.9])],[0]),{'0':2})

    def test_vote_final_tie_lowest_head(self):
        samples=[{'0':dict(head=h,outgoing_scores=[0,.5,.5])} for h in (2,1)]
        self.assertEqual(vote_representatives(samples,[0]),{'0':1})

    def test_no_samples_rejected(self):
        with self.assertRaises(ValueError): vote_representatives([],[0])

    def test_experiment_requires_approval_before_preflight(self):
        import run
        with patch('run.require_approval',side_effect=PermissionError('pending')),patch('run.preflight') as check:
            with self.assertRaises(PermissionError): run.run('pilot')
            check.assert_not_called()


if __name__=='__main__':
    unittest.main()
