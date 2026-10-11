"""CPU checks for raw-QK selection and the unchanged probability default.

These validate masks/scoring, not GPU attention kernels or generation speed.
"""
import math
import unittest
from unittest.mock import patch

import torch

from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer, chronological_topk, install
from kvcompress.engine.selector_grid_cache import SelectorGridLayer, choose, representative
import kvcompress.engine.selector_grid_cache as grid


def make(mode='qk', n=128, budget=96, protection='protected', dtype=torch.bfloat16,
         batch=2, prompt=17):
    layer = FullW1OptimizedLayer(
        3, budget, 0, f'grid_full_w1_topk_{protection}', 0., False, False,
        query_window=1, eviction_interval=1, rep_seed=0, sample_id=19,
        max_cache_length=256, track_positions=True, score_mode=mode)
    generator = torch.Generator().manual_seed(671)
    layer.lazy_initialization(torch.zeros(batch, prompt, 2, 16, dtype=dtype))
    layer.k_cache[:, :n].copy_(torch.randn(batch, n, 2, 16, generator=generator).to(dtype))
    layer.cumulative_length = n
    layer._lengths = [n] * batch
    layer.cache_seqlens.fill_(n)
    layer.slot_positions[:, :n] = torch.arange(n)
    layer.update_queries(torch.randn(batch, 8, 1, 16, generator=generator).to(dtype))
    return layer


class FullW1QKTests(unittest.TestCase):
    def test_single_query_softmax_rank_equivalence_without_extra_rounding_ties(self):
        scores = torch.tensor([[-9., -2., -2., 0., .5, .5, 3.],
                               [7., 5., 5., 3., 2., 1., -4.]], dtype=torch.float64)
        probabilities = (scores / math.sqrt(16)).softmax(-1)
        for count in range(scores.shape[1] + 1):
            self.assertTrue(torch.equal(chronological_topk(scores, count),
                                        chronological_topk(probabilities, count)))

    def test_raw_scores_masks_protection_short_history_gqa_and_rng(self):
        for protection in ['protected', 'none']:
            for n in [81, 95, 96, 97, 128, 255]:
                layer = make(n=n, protection=protection)
                before = torch.random.get_rng_state().clone()
                # Fail if raw-QK selection accidentally calls softmax.
                with patch.object(torch.Tensor, 'softmax', side_effect=AssertionError('selector softmax')):
                    ids, counts = layer.select_indices()
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                rep = representative(0, 19, 3, n - 17, 8)
                self.assertEqual(layer._rep_head, rep)
                self.assertEqual(layer._rep_kv, rep // 4)
                raw = (layer.queries[:, rep, :1] @
                       layer.k_cache[:, :n, rep // 4].transpose(1, 2)).squeeze(1)
                self.assertTrue(torch.equal(layer._last_scores, raw))
                self.assertEqual(raw.dtype, torch.bfloat16)
                protected = torch.zeros(n, dtype=torch.bool)
                if protection == 'protected':
                    protected[:17] = True
                    protected[max(17, n - 64):] = True
                for row in range(2):
                    expected = choose(raw[row], protected, 'topk', 96, .95)
                    self.assertTrue(torch.equal(ids[row], expected))
                    self.assertEqual(ids[row].unique().numel(), min(n, 96))
                    self.assertTrue(torch.all(ids[row][1:] > ids[row][:-1]))
                    self.assertTrue(torch.all(torch.isin(torch.where(protected)[0], ids[row])))
                self.assertEqual(counts, [min(n, 96)] * 2)

    def test_default_probability_scores_and_masks_match_historical_oracle(self):
        layer = make(mode='probability')
        ids, _ = layer.select_indices()
        logits = layer.queries[:, layer._rep_head, :1] @ layer.k_cache[:, :128, layer._rep_kv].transpose(1, 2)
        expected = (logits / (math.sqrt(16) * layer.temperature)).softmax(-1, dtype=torch.float32).mean(1)
        self.assertTrue(torch.equal(layer._last_scores, expected))
        protected = torch.zeros(128, dtype=torch.bool)
        protected[:17] = True
        protected[64:] = True
        for row in range(2):
            self.assertTrue(torch.equal(ids[row], choose(expected[row], protected, 'topk', 96, .95)))
        self.assertEqual(layer.implementation, 'full_w1_fixed_ranges_batched_exact_v1')

    def test_softmax_underflow_is_a_documented_mask_difference(self):
        modes = [make(mode=mode, n=3, budget=2, protection='none', batch=1, prompt=1)
                 for mode in ['probability', 'qk']]
        for layer in modes:
            layer.queries.zero_()
            layer.queries[:, :, :, 0] = 1
            layer.k_cache[:, :3].zero_()
            layer.k_cache[:, :3, :, 0] = torch.tensor([-800., -796., 0.], dtype=torch.bfloat16)[None, :, None]
        probability_ids, _ = modes[0].select_indices()
        qk_ids, _ = modes[1].select_indices()
        self.assertEqual(probability_ids.tolist(), [[0, 2]])
        self.assertEqual(qk_ids.tolist(), [[1, 2]])
        self.assertEqual(modes[1].implementation, 'full_w1_fixed_ranges_batched_qk_v1')

    def test_batch_shrink_keeps_correct_raw_selection(self):
        layer = make()
        ids, _ = layer.select_indices()
        expected = ids[1].clone()
        layer.batch_select_indices(torch.tensor([1]))
        selected, counts = layer.select_indices()
        self.assertEqual(counts, [96])
        self.assertTrue(torch.equal(selected[0], expected))

    def test_factory_opt_in_restrictions_and_unchanged_default(self):
        original = grid.SelectorGridLayer
        args = (0, 96, 0, 'grid_full_w1_topk_protected', 0., False, False)
        kw = dict(query_window=1, eviction_interval=1)
        try:
            install()
            self.assertEqual(grid.SelectorGridLayer(*args, **kw).score_mode, 'probability')
            install(score_mode='qk')
            self.assertEqual(grid.SelectorGridLayer(*args, **kw).score_mode, 'qk')
            random_args = (0, 96, 0, 'grid_full_random_topk_protected', 0., False, False)
            self.assertIs(type(grid.SelectorGridLayer(*random_args, **kw)), SelectorGridLayer)
            with self.assertRaises(ValueError):
                install('packed_topk', score_mode='qk')
            with self.assertRaises(ValueError):
                FullW1OptimizedLayer(*args, rank_backend='packed_topk', score_mode='qk', **kw)
        finally:
            grid.SelectorGridLayer = original


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
