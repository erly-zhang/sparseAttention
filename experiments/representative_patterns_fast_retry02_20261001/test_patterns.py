"""Literal per-query mask oracle catches slash-to-column expansion and overlap bugs."""
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from selection import build_index, reference_attention, sampled_member_coverage, normalized_attention
from patterns import slash_scores, slash_offsets, merge_slash, allowed_mask
from experiments.token_compacted_sparse import TokenCompactedIndex
from test_selection import CPUEvent


def compact(rows,block,h=4):
    counts=torch.tensor([len(r) for r in rows],dtype=torch.int32)
    ends=counts.cumsum(0,dtype=torch.int32)
    return TokenCompactedIndex(row_starts=(ends-counts).view(1,1,-1),row_ends=ends.view(1,1,-1),
        token_indices=torch.tensor(sum(rows,[]),dtype=torch.int32),head_to_group=torch.zeros(h,dtype=torch.int32),
        num_groups=1,num_query_blocks=len(rows),query_block_size=block)


class PatternTests(unittest.TestCase):
    def test_relative_distance_scores_match_literal(self):
        probabilities=torch.tensor([[.1,.2,.3,.4,0],[.1,.1,.1,.2,.5]],dtype=torch.float64)
        positions=torch.tensor([3,4])
        actual=slash_scores(probabilities,positions,5)
        expected=torch.zeros(5,dtype=torch.float64)
        for row,i in enumerate(positions):
            for j in range(int(i)+1): expected[int(i)-j]+=probabilities[row,j]/2
        torch.testing.assert_close(actual,expected)
        self.assertAlmostEqual(float(actual.sum()),1)

    def test_slash_does_not_become_shared_columns(self):
        index=compact([[0],[0],[0]],4)
        expanded,stats=merge_slash(index,torch.tensor([2]),10)
        dense=torch.zeros(10,10,dtype=torch.bool)
        for qb in range(3):
            start=qb*4;end=min(start+4,10)
            begin,stop=int(expanded.row_starts[0,0,qb]),int(expanded.row_ends[0,0,qb])
            ids=expanded.token_indices[begin:stop].long()
            dense[start:end,ids]=allowed_mask(expanded,qb,torch.arange(start,end),ids)
        oracle=torch.eye(10,dtype=torch.bool).roll(-2,1).tril() # replaced with direct positions below
        oracle.zero_();oracle[:,0]=True
        for i in range(2,10): oracle[i,i-2]=True
        self.assertTrue(torch.equal(dense,oracle))
        self.assertEqual(stats['mask_selected_pairs'],int(oracle.sum()))
        self.assertGreater(stats['candidate_causal_pairs'],stats['mask_selected_pairs'])
        self.assertFalse(bool(dense[7,2]))
        self.assertTrue(bool(dense[7,5]))

    def test_overlap_counts_and_partial_tile_match_literal(self):
        for offsets in [[],[0],[1,2,9],[2,2,5],[0,1,2,3,4,5,6,7,8,9]]:
            rows=[[0,1,2],[0,2,4,6],[0,1,8,9]]
            index=compact(rows,4)
            expanded,stats=merge_slash(index,torch.tensor(offsets,dtype=torch.long),10)
            expected=0;before=0
            for i in range(10):
                base={j for j in rows[i//4] if j<=i}
                final=base|{i-d for d in offsets if d<=i}
                qb=i//4
                ids=expanded.token_indices[int(expanded.row_starts[0,0,qb]):int(expanded.row_ends[0,0,qb])].long()
                actual=set(ids[allowed_mask(expanded,qb,torch.tensor([i]),ids)[0]].tolist())
                self.assertEqual(actual,final)
                expected+=len(final);before+=len(base)
            self.assertEqual(stats['mask_selected_pairs'],expected)
            self.assertEqual(stats['slash_added_pairs'],expected-before)

    def test_gamma_minimal_prefix_and_short_queries(self):
        torch.manual_seed(31)
        q=torch.randn(1,9,4,8);k=torch.randn(1,9,2,8)
        options={'vertical_queries':128,'slash_gamma':.7,'slash_max_offsets':None}
        offsets,stats=slash_offsets(q,k,3,options)
        self.assertEqual(stats['slash_query_count'],9)
        self.assertGreaterEqual(stats['slash_retained_mass'],.7-1e-8)
        self.assertLess(stats['slash_prefix_before_last'],.7)
        self.assertTrue((offsets>=0).all() and (offsets<9).all())

    def test_cap_and_invalid_offsets(self):
        q=torch.zeros(1,9,4,8);k=torch.zeros(1,9,2,8)
        offsets,stats=slash_offsets(q,k,3,{'vertical_queries':128,'slash_gamma':.95,'slash_max_offsets':1})
        self.assertEqual(len(offsets),1);self.assertTrue(stats['slash_budget_capped'])
        self.assertLess(stats['slash_retained_mass'],.95)
        for bad in [-1,10]:
            with self.assertRaises(ValueError): merge_slash(compact([[0]],128),torch.tensor([bad]),10)

    def test_gqa_attention_and_coverage_use_exact_mask(self):
        torch.manual_seed(912)
        n=17;q=torch.randn(1,n,4,8,dtype=torch.float64)
        k=torch.randn(1,n,2,8,dtype=torch.float64);v=torch.randn_like(k)
        rows=[[0],[0],[0]];base=compact(rows,8)
        index,_=merge_slash(base,torch.tensor([2,6]),n)
        actual=reference_attention(q,k,v,index)
        oracle=torch.zeros(n,n,dtype=torch.bool);oracle[:,0]=True
        for i in range(n):
            for d in [2,6]:
                if d<=i:oracle[i,i-d]=True
        expected=[];coverage=[]
        for head in range(4):
            scores=q[0,:,head]@k[0,:,head//2].T/math.sqrt(8)
            sparse=scores.masked_fill(~oracle,float('-inf')).softmax(-1)
            expected.append(sparse@v[0,:,head//2])
            dense=scores.masked_fill(~torch.ones(n,n,dtype=torch.bool).tril(),float('-inf')).softmax(-1)
            coverage.append((dense*oracle).sum(-1).mean())
        torch.testing.assert_close(actual[0],torch.stack(expected,1))
        measured=sampled_member_coverage(q,k,index,count=n,chunk=3)
        torch.testing.assert_close(torch.tensor(measured['member_mean_retained_mass'],dtype=torch.float64),torch.stack(coverage))

    def test_full_build_preserves_base_and_vertical_and_pair_counts(self):
        torch.manual_seed(313)
        q=torch.randn(1,513,4,8);k=torch.randn(1,513,2,8);v=torch.randn_like(k)
        options=json.loads((Path(__file__).parent/'config.json').read_text())['selection']
        options=dict(options,target_token_budget=256,autoblock_final_token_budget=256,
            vertical_max_tokens=3,slash_max_offsets=5)
        with patch('torch.cuda.Event',CPUEvent):
            base,bst=build_index(q,k,v,3,0,'representative_token_vertical',options)
            final,stats=build_index(q,k,v,3,0,'representative_token_vertical_slash',options)
        pairs=0
        for qb in range(final.num_query_blocks):
            start=qb*128;end=min(start+128,513)
            ids=final.token_indices[int(final.row_starts[0,0,qb]):int(final.row_ends[0,0,qb])].long()
            valid=allowed_mask(final,qb,torch.arange(start,end),ids)
            old=base.token_indices[int(base.row_starts[0,0,qb]):int(base.row_ends[0,0,qb])].long()
            self.assertTrue(torch.isin(old,ids).all())
            self.assertTrue(valid.any(-1).all());pairs+=int(valid.sum())
        self.assertEqual(pairs,stats['mask_selected_pairs'])
        self.assertEqual(stats['mask_selected_pairs']-stats['base_selected_pairs'],
            stats['vertical_added_pairs']+stats['slash_added_pairs'])
        self.assertTrue(stats['slash_added'])


if __name__=='__main__':
    unittest.main()
