"""CPU formula and end-to-end selector checks; no model experiment is run."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from experiments.masa_token_scoring import (
    RepresentativeMASATokenSelector, masa_pair_scores, value_probe,
)


GROUPS = {"layers": {"0": [
    {"representative": 0, "members": [0, 1]},
    {"representative": 2, "members": [2, 3]},
    {"representative": 4, "members": [4, 5]},
]}}


def selector(**overrides):
    kwargs = dict(
        logical_key_block_size=(32, 64, 128), target_token_budget=19,
        target_token_top_p=None, target_token_min_budget=None,
        target_top_p_start_layer=None, final_token_budget=None,
        maximum_selected_blocks=None, fill_final_token_budget=False,
        minimum_block_coverage_ratio=0.125,
        minimum_block_target_probability_ratio=None, block_probability_top_p=None,
        block_target_probability_top_p=None, block_target_count_coverage_ratio=None,
        block_target_cost_aware=False, block_f_beta=None,
        measure_fixed_target_probability_mass=False,
        force_sink_block=False, force_diagonal_block=False,
        project_target_to_blocks=False,
    )
    kwargs.update(overrides)
    return RepresentativeMASATokenSelector(GROUPS, **kwargs)


class FakeEvent:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass


class MASATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_equation_7_matches_pair_score_and_vertical_unit(self):
        a = torch.randn(2, 3, 4, 7).softmax(-1)
        x = torch.randn(2, 3, 7, 5)
        d = masa_pair_scores(a, x)
        output = a @ x
        for j in range(7):
            contribution = a[..., j, None] * x[..., j, :].unsqueeze(-2)
            direct = output.square().sum(-1) - (output - contribution).square().sum(-1)
            torch.testing.assert_close(d[..., j], direct, atol=2e-6, rtol=2e-5)
            weights = torch.tensor([0.2, 0.3, 0.4, 0.1])
            torch.testing.assert_close((d[..., j] * weights).sum(-1),
                                       (direct * weights).sum(-1), atol=2e-6, rtol=2e-5)

    def test_small_mass_can_win_and_negative_score_is_preserved(self):
        a = torch.tensor([[[[0.98, 0.02]]]])
        x = torch.tensor([[[[0.0], [100.0]]]])
        d = masa_pair_scores(a, x)
        self.assertGreater(d[..., 1].item(), d[..., 0].item())
        a = torch.tensor([[[[0.6, 0.4]]]])
        x = torch.tensor([[[[1.0], [-1.0]]]])
        self.assertLess(masa_pair_scores(a, x)[..., 1].item(), 0)

    def test_probe_controls_do_not_modify_values(self):
        v = torch.tensor([[[[0., 0.], [3., 4.], [0., 100.]]]])
        original = v.clone()
        balanced = value_probe(v, norm_alpha=1)
        torch.testing.assert_close(balanced.norm(dim=-1), torch.tensor([[[0., 1., 1.]]]))
        clipped = value_probe(v, clip_percentile=50)
        torch.testing.assert_close(clipped.norm(dim=-1), torch.tensor([[[0., 5., 5.]]]))
        torch.testing.assert_close(v, original)
        for kwargs in ({"norm_alpha": -1}, {"clip_percentile": 101}):
            with self.assertRaises(ValueError):
                value_probe(v, **kwargs)

    def test_profile_score_matches_explicit_causal_probe_reductions(self):
        q, k, v = [torch.randn(1, 259, 3, 4) for _ in range(3)]
        s = selector(masa_query_chunk_size=1)
        token = s._profile_values.set(v)
        try:
            scores, starts, ends, lengths = s._score_representative_keys(q, k, block_size=128)
            s.masa_query_chunk_size = 16
            other = s._score_representative_keys(q, k, block_size=128)[0]
        finally:
            s._profile_values.reset(token)
        torch.testing.assert_close(scores, other, atol=2e-6, rtol=2e-5)
        for tile, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            query_rows = [q[:, start + (end-start-1)//3],
                          q[:, start + 2*(end-start-1)//3],
                          q[:, end-1], q[:, start:end].mean(1)]
            cutoffs = [start+(end-start-1)//3+1, start+2*(end-start-1)//3+1, end, end]
            expected = torch.zeros(1, 3, 259)
            for row, cutoff, weight in zip(query_rows, cutoffs, s.probe_weights):
                logits = torch.einsum('bgd,bngd->bgn', row, k) / 2
                logits[..., cutoff:] = -torch.inf
                a = logits.softmax(-1)
                output = torch.einsum('bgn,bngd->bgd', a, v)
                for j in range(end):
                    c = a[..., j, None] * v[:, j]
                    expected[..., j] += weight * (
                        output.square().sum(-1) - (output-c).square().sum(-1))
            torch.testing.assert_close(scores[:, :, tile, :end], expected[..., :end],
                                       atol=3e-6, rtol=3e-4)
            self.assertTrue(torch.isneginf(scores[:, :, tile, end:]).all())

    def test_real_selector_gqa_budget_group_broadcast_and_value_dependence(self):
        q = torch.randn(1, 259, 6, 4)
        k, v = [torch.randn(1, 259, 3, 4) for _ in range(2)]
        for interleave in (False, True):
            s = selector()
            context = s.current_layer.set(0)
            with patch('torch.cuda.Event', FakeEvent):
                index = s(q, k, v, 128, 0.9, 1, 100, gqa_interleave=interleave)
            s.current_layer.reset(context)
            self.assertIsNone(s._profile_values.get())
            self.assertEqual(index.head_to_group.tolist(), [0, 0, 1, 1, 2, 2])
            kv_indices = [0, 2, 1] if interleave else [0, 1, 2]
            token = s._profile_values.set(v[:, :, kv_indices])
            scores = s._score_representative_keys(q[:, :, [0, 2, 4]], k[:, :, kv_indices],
                                                   block_size=128)[0]
            changed = s._profile_values.set(v[:, :, kv_indices].flip(1))
            changed_scores = s._score_representative_keys(q[:, :, [0, 2, 4]], k[:, :, kv_indices],
                                                           block_size=128)[0]
            s._profile_values.reset(changed)
            s._profile_values.reset(token)
            self.assertFalse(torch.allclose(scores[..., :128], changed_scores[..., :128]))
            for group in range(3):
                for tile, end in enumerate((128, 256, 259)):
                    begin = index.row_starts[0, group, tile].item()
                    stop = index.row_ends[0, group, tile].item()
                    actual = index.token_indices[begin:stop].long().sort().values
                    expected = scores[0, group, tile].topk(19).indices.sort().values
                    torch.testing.assert_close(actual, expected)
                    self.assertTrue((actual < end).all())

    def test_incompatible_probability_and_block_modes_rejected(self):
        with self.assertRaises(ValueError):
            selector(project_target_to_blocks=True)
        with self.assertRaises(ValueError):
            selector(target_token_budget=None, target_token_top_p=0.95)

    def test_all_queries_matches_full_matrix_error_reduction_and_partial_tile(self):
        q,k,v=[torch.randn(1,259,3,4) for _ in range(3)]
        logits=torch.einsum('bigd,bjgd->bgij',q,k)/2
        logits.masked_fill_(~torch.ones(259,259,dtype=torch.bool).tril(),-torch.inf)
        a=logits.softmax(-1);x=v.permute(0,2,1,3);output=a@x
        contribution=a[...,None]*x.unsqueeze(-3)
        # Independent Eq. 7 subtraction for every single-key column unit.
        direct=output.square().sum(-1).unsqueeze(-1)-(output.unsqueeze(-2)-contribution).square().sum(-1)
        previous=None
        for chunk in (1,7,128):
            s=selector(masa_query_mode='all_queries',masa_full_query_chunk_size=chunk)
            token=s._profile_values.set(v)
            scores,starts,ends,lengths=s._score_representative_keys(q,k,block_size=128)
            s._profile_values.reset(token)
            self.assertEqual(lengths.tolist(),[128,128,3])
            for tile,(start,end) in enumerate(zip(starts.tolist(),ends.tolist())):
                expected=direct[:,:,start:end,:end].mean(-2)
                torch.testing.assert_close(scores[:,:,tile,:end],expected,atol=2e-6,rtol=3e-4)
                self.assertTrue(torch.isneginf(scores[:,:,tile,end:]).all())
            if previous is not None:
                torch.testing.assert_close(scores,previous,atol=2e-6,rtol=3e-4)
            previous=scores

    def test_benchmark_entry_parameters_and_metadata(self):
        from experiments import benchmark_shareprefill_ae3 as benchmark
        config = {"classification_metric": "shareprefill_attention_map_autoencoder",
                  "num_groups_per_layer": 3}
        model = object()
        with patch.object(benchmark, 'patch_transformers_flash_window_flag'), patch.object(
            benchmark, 'install_shareprefill_ae3_token_block_cover', return_value=(None, config)
        ) as install:
            _, _, metadata = benchmark.install_benchmark_method(
                model, 'shareprefill_ae3_representative_token_masa', model_name='llama-test',
                masa_norm_alpha=0.5, masa_query_chunk_size=2,
            )
            self.assertEqual(install.call_args.kwargs['token_ranking'], 'masa')
            self.assertEqual(install.call_args.kwargs['probe_weights'], (0.25,)*4)
            self.assertEqual(install.call_args.kwargs['target_token_budget'], 8192)
            self.assertFalse(install.call_args.kwargs['project_target_to_blocks'])
            self.assertEqual(metadata['online_config']['masa']['norm_alpha'], 0.5)
            self.assertEqual(list(metadata['online_config']['probe_weights'].values()), [0.25]*4)
            self.assertTrue(metadata['online_config']['group_mask_broadcast'])
            _,_,full_metadata=benchmark.install_benchmark_method(
                model,'shareprefill_ae3_representative_token_masa',model_name='llama-test',
                masa_query_mode='all_queries',masa_full_query_chunk_size=128,
            )
            self.assertEqual(install.call_args.kwargs['masa_query_mode'],'all_queries')
            self.assertEqual(full_metadata['online_config']['query_probes'],['all_actual_queries'])
            self.assertIsNone(full_metadata['online_config']['probe_weights'])
            benchmark.install_benchmark_method(
                model, 'shareprefill_ae3_representative_token_topk', model_name='llama-test'
            )
            self.assertEqual(install.call_args.kwargs['token_ranking'], 'qk')
            self.assertEqual(install.call_args.kwargs['probe_weights'], (0.2, 0.3, 0.4, 0.1))
            self.assertEqual(install.call_args.kwargs['target_token_budget'], 8192)


if __name__ == '__main__':
    unittest.main()
